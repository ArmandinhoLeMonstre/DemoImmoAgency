"""Phase 3 — deux nodes + une transition (dynamic flow).

Même pipeline voix qu'avant (SmallWebRTC dev local : Gladia fr -> OpenAI ->
Cartesia Sonic, VAD Silero), piloté par un `FlowManager` (Pipecat Flows 1.0).
Deux nodes reliés par une seule transition linéaire :

  greeting  --(enregistrer_nom)-->  besoin

`greeting` accueille et demande le prénom ; quand l'appelant le donne, l'agent
appelle `enregistrer_nom`, qui mémorise le prénom et transitionne vers `besoin`.
`besoin` s'adresse à l'appelant par son prénom et demande l'objet de l'appel.
Pas encore de branching ni de CRM — ça arrive aux phases suivantes. Lancé par le
dev runner (`uv run bot.py`, UI http://localhost:7860/client).
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

# En Phase 3 on ne sert que le transport SmallWebRTC en local. Les autres transports
# (Daily, téléphonie) viendront aux phases ultérieures avec leurs extras dédiés.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def create_besoin_node(prenom: str) -> NodeConfig:
    """Crée le second node : l'agent demande l'objet de l'appel.

    Node terminal pour cette phase (aucune fonction, pas de branching). La persona
    posée par `greeting` persiste, donc on ne re-pose pas `role_message`.

    Args:
        prenom: Le prénom de l'appelant, pour une adresse personnalisée.

    Returns:
        La configuration du node « besoin ».
    """
    return NodeConfig(
        name="besoin",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Remercie {prenom} et demande-lui en une phrase l'objet de son "
                    "appel (vendre, acheter, louer ou faire estimer un bien)."
                ),
            }
        ],
    )


@flows_tool_options(cancel_on_interruption=False)
async def enregistrer_nom(flow_manager: FlowManager, prenom: str) -> tuple[None, NodeConfig]:  # noqa: D417
    """Enregistre le prénom de l'appelant puis passe au node « besoin ».

    `flow_manager` est le premier paramètre injecté par Flows : on ne le documente
    pas volontairement (il ne fait pas partie du schema exposé au LLM).

    Args:
        prenom: Le prénom communiqué par l'appelant.

    Returns:
        Un tuple (résultat, node suivant) : pas de résultat, transition vers « besoin ».
    """
    flow_manager.state["prenom"] = prenom
    logger.info(f"Prénom enregistré : {prenom}")
    return None, create_besoin_node(prenom)


def create_greeting_node() -> NodeConfig:
    """Crée le node initial : l'agent accueille et demande le prénom.

    Inbound, donc l'agent parle en premier (`respond_immediately=True`, défaut rendu
    explicite). Expose `enregistrer_nom`, qui déclenche la transition vers « besoin ».

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
