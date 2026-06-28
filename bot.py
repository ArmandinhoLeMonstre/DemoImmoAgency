"""Phase 4 — router d'intention (branching, dynamic flow).

Même pipeline voix qu'avant (SmallWebRTC dev local : Gladia fr -> OpenAI ->
Cartesia Sonic, VAD Silero), piloté par un `FlowManager` (Pipecat Flows 1.0).

Graphe de conversation :

    greeting --(enregistrer_nom)--> router --+--(router_vendeur)----> vendeur
                                             +--(router_estimation)-> vendeur
                                             +--(router_acheteur)---> acheteur
                                             +--(router_location)---> location

`router` expose quatre edge functions ; le LLM appelle celle qui correspond à
l'intention de l'appelant (branching). Vendre et faire estimer atterrissent tous
deux dans le tunnel « vendeur » ; seul `state["sous_intention"]` les distingue
(« vente_directe » vs « estimation »), ce qui servira en Phase 5 à adapter
l'écriture CRM et le type de RDV. Les nodes d'intention restent des placeholders
(aucune écriture CRM ici). Lancé par le dev runner (`uv run bot.py`, UI
http://localhost:7860/client).
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.gladia.config import LanguageConfig
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_flows import FlowManager, NodeConfig, flows_tool_options

load_dotenv(override=True)

# Persona FR de l'agent, posée une seule fois dans le node initial via `role_message`
# (system instruction qui persiste entre nodes). Pas de qualification ni de CRM ici.
ROLE_MESSAGE = (
    "Tu es l'agent vocal d'accueil d'une agence immobilière belge. Tu réponds en "
    "français, sur un ton chaleureux et professionnel. Tes réponses sont lues à voix "
    "haute : pas d'emojis, de listes à puces ni de mise en forme. Sois bref et naturel."
)

# En Phase 4 on ne sert que le transport SmallWebRTC en local. Les autres transports
# (Daily, téléphonie) viendront aux phases ultérieures avec leurs extras dédiés.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# --- Nodes d'intention (terminaux pour cette phase, aucune fonction / CRM) ----------


def create_vendeur_node(sous_intention: str) -> NodeConfig:
    """Crée le node « vendeur » (tunnel partagé vente directe / estimation).

    L'appelant qui veut vendre OU faire estimer atterrit ici ; le message d'accueil
    est adapté à `sous_intention` pour que la distinction soit audible, mais le node
    reste terminal pour cette phase (aucune écriture CRM).

    Args:
        sous_intention: « vente_directe » ou « estimation ».

    Returns:
        La configuration du node « vendeur ».
    """
    if sous_intention == "estimation":
        objectif = (
            "L'appelant veut faire estimer la valeur de son bien. Confirme avec "
            "enthousiasme que tu vas organiser une estimation et qu'un expert le "
            "recontactera pour convenir d'un rendez-vous."
        )
    else:
        objectif = (
            "L'appelant souhaite mettre son bien en vente. Confirme avec enthousiasme "
            "que tu vas l'accompagner pour la mise en vente et qu'un conseiller le "
            "recontactera pour la suite."
        )
    return NodeConfig(
        name="vendeur",
        task_messages=[{"role": "developer", "content": objectif}],
    )


def create_acheteur_node() -> NodeConfig:
    """Crée le node « acheteur » : l'appelant cherche à acheter.

    Returns:
        La configuration du node « acheteur ».
    """
    return NodeConfig(
        name="acheteur",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "L'appelant cherche à acheter un bien. Confirme que tu vas l'aider "
                    "et indique qu'un conseiller reviendra vers lui avec des biens "
                    "correspondants."
                ),
            }
        ],
    )


def create_location_node() -> NodeConfig:
    """Crée le node « location » : l'appelant cherche à louer.

    Returns:
        La configuration du node « location ».
    """
    return NodeConfig(
        name="location",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "L'appelant cherche à louer un bien. Confirme que tu vas l'aider et "
                    "indique qu'un conseiller le recontactera avec les offres de "
                    "location disponibles."
                ),
            }
        ],
    )


# --- Edge functions du router : le LLM en choisit une selon l'intention -------------
# Direct functions à paramètre `flow_manager` seul ; leur docstring sert de description
# routante exposée au LLM. Pas de section Args (aucun paramètre métier à documenter).


@flows_tool_options(cancel_on_interruption=False)
async def router_vendeur(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """L'appelant souhaite VENDRE / mettre en vente un bien immobilier.

    Returns:
        Un tuple (résultat, node suivant) : transition vers le tunnel « vendeur ».
    """
    flow_manager.state["intention"] = "vendeur"
    flow_manager.state["sous_intention"] = "vente_directe"
    logger.info("Intention : vendeur / vente_directe")
    return None, create_vendeur_node("vente_directe")


@flows_tool_options(cancel_on_interruption=False)
async def router_estimation(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """L'appelant souhaite faire ESTIMER la valeur d'un bien immobilier.

    Returns:
        Un tuple (résultat, node suivant) : transition vers le tunnel « vendeur ».
    """
    flow_manager.state["intention"] = "vendeur"
    flow_manager.state["sous_intention"] = "estimation"
    logger.info("Intention : vendeur / estimation")
    return None, create_vendeur_node("estimation")


@flows_tool_options(cancel_on_interruption=False)
async def router_acheteur(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """L'appelant souhaite ACHETER un bien immobilier.

    Returns:
        Un tuple (résultat, node suivant) : transition vers le node « acheteur ».
    """
    flow_manager.state["intention"] = "acheteur"
    logger.info("Intention : acheteur")
    return None, create_acheteur_node()


@flows_tool_options(cancel_on_interruption=False)
async def router_location(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """L'appelant souhaite LOUER un bien immobilier.

    Returns:
        Un tuple (résultat, node suivant) : transition vers le node « location ».
    """
    flow_manager.state["intention"] = "location"
    logger.info("Intention : location")
    return None, create_location_node()


# --- Router + accueil ----------------------------------------------------------------


def create_router_node(prenom: str) -> NodeConfig:
    """Crée le node router : demande l'objet de l'appel et branche vers l'intention.

    Expose les quatre edge functions ; le LLM appelle celle qui correspond à la
    réponse de l'appelant (vendre / estimer / acheter / louer). La persona posée
    par `greeting` persiste, donc on ne re-pose pas `role_message`.

    Args:
        prenom: Le prénom de l'appelant, pour une adresse personnalisée.

    Returns:
        La configuration du node router.
    """
    return NodeConfig(
        name="router",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Remercie {prenom} puis demande-lui en une phrase l'objet de son "
                    "appel : vendre, faire estimer, acheter ou louer un bien. Dès que "
                    "son intention est claire, appelle la fonction correspondante."
                ),
            }
        ],
        functions=[router_vendeur, router_estimation, router_acheteur, router_location],
    )


@flows_tool_options(cancel_on_interruption=False)
async def enregistrer_nom(flow_manager: FlowManager, prenom: str) -> tuple[None, NodeConfig]:  # noqa: D417
    """Enregistre le prénom de l'appelant puis passe au node router.

    `flow_manager` est le premier paramètre injecté par Flows : on ne le documente
    pas volontairement (il ne fait pas partie du schema exposé au LLM).

    Args:
        prenom: Le prénom communiqué par l'appelant.

    Returns:
        Un tuple (résultat, node suivant) : pas de résultat, transition vers le router.
    """
    flow_manager.state["prenom"] = prenom
    logger.info(f"Prénom enregistré : {prenom}")
    return None, create_router_node(prenom)


def create_greeting_node() -> NodeConfig:
    """Crée le node initial : l'agent accueille et demande le prénom.

    Inbound, donc l'agent parle en premier (`respond_immediately=True`, défaut rendu
    explicite). Expose `enregistrer_nom`, qui déclenche la transition vers le router.

    Returns:
        La configuration du node d'accueil.
    """
    return NodeConfig(
        name="greeting",
        role_message=ROLE_MESSAGE,
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Accueille chaleureusement l'appelant, présente-toi comme l'accueil "
                    "de l'agence et demande-lui son prénom."
                ),
            }
        ],
        functions=[enregistrer_nom],
        respond_immediately=True,
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Construit le pipeline voix, branche le FlowManager et fait tourner le worker.

    Args:
        transport: Transport fourni par le dev runner (SmallWebRTC en local).
        runner_args: Arguments de session passés par le runner.
    """
    logger.info("Démarrage du bot")

    region = os.getenv("GLADIA_REGION", "eu-west")
    assert region in ("us-west", "eu-west"), f"GLADIA_REGION invalide : {region}"

    stt = GladiaSTTService(
        api_key=os.environ["GLADIA_API_KEY"],
        region=region,
        settings=GladiaSTTService.Settings(
            language_config=LanguageConfig(languages=[Language.FR]),
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            model=os.getenv("CARTESIA_MODEL", "sonic-3.5"),
            voice=os.environ["CARTESIA_VOICE_ID"],
            language=Language.FR,
        ),
    )

    # Pas de system_instruction ici : c'est le `role_message` du node qui pose la
    # persona (Pipecat Flows l'envoie via LLMUpdateSettingsFrame).
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Entrée audio (utilisateur)
            stt,  # Gladia (fr)
            context_aggregator.user(),  # Tour de parole utilisateur
            llm,  # OpenAI
            tts,  # Cartesia (Sonic)
            transport.output(),  # Sortie audio (bot)
            context_aggregator.assistant(),  # Tour de parole assistant
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Inbound : démarre le flow sur le node d'accueil dès la connexion."""
        logger.info("Client connecté")
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Arrête le worker quand le client se déconnecte."""
        logger.info("Client déconnecté")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Point d'entrée appelé par le dev runner pour chaque session.

    Args:
        runner_args: Arguments de session fournis par le runner.
    """
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
