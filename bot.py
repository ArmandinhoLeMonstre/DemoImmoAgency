"""Phase 5 — premier tool Odoo : qualification vendeur avec écritures progressives.

Même pipeline voix qu'avant (SmallWebRTC dev local : Gladia fr -> OpenAI ->
Cartesia Sonic, VAD Silero), piloté par un `FlowManager` (Pipecat Flows 1.0).

Graphe de conversation :

    greeting --(enregistrer_nom)--> router --+--(router_vendeur)----> vendeur_contact
                                             +--(router_estimation)-> vendeur_contact
                                             +--(router_acheteur)---> acheteur
                                             +--(router_location)---> location

    vendeur_contact --(CREATE)--> vendeur_bien --(UPDATE)--> vendeur_qualif
                    --(UPDATE)--> vendeur_fin

Le tunnel vendeur écrit dans Odoo de façon PROGRESSIVE : un lead minimal est créé
dès le contact, puis enrichi à chaque étape (un handler = une écriture). Toutes les
écritures Odoo (synchrones, XML-RPC) passent par `asyncio.to_thread` bornées à
`ODOO_TIMEOUT_SECS` pour ne jamais geler la voix ; tout échec/timeout est loggé et
la conversation continue. acheteur / location restent des placeholders (Phase 6).
Lancé par le dev runner (`uv run bot.py`, UI http://localhost:7860/client).
"""

import asyncio
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

from odoo_seller_tool import SellerLead, get_client

load_dotenv(override=True)

# Chaque écriture Odoo (XML-RPC synchrone) est bornée à ~4s : au-delà on logge et on
# poursuit la conversation. Important pour la démo : un Odoo lent ne gèle pas la voix.
ODOO_TIMEOUT_SECS = 4.0

# Persona FR de l'agent, posée une seule fois dans le node initial via `role_message`
# (system instruction qui persiste entre nodes). Pas de qualification ni de CRM ici.
ROLE_MESSAGE = (
    "Tu es l'agent vocal d'accueil d'une agence immobilière belge. Tu réponds en "
    "français, sur un ton chaleureux et professionnel. Tes réponses sont lues à voix "
    "haute : pas d'emojis, de listes à puces ni de mise en forme. Sois bref et naturel."
)

# En Phase 5 on ne sert que le transport SmallWebRTC en local. Les autres transports
# (Daily, téléphonie) viendront aux phases ultérieures avec leurs extras dédiés.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# --- Tunnel vendeur : qualification + écritures Odoo progressives -------------------
# Un handler = une écriture Odoo. Les écritures (XML-RPC synchrone) tournent dans un
# thread (asyncio.to_thread) borné par ODOO_TIMEOUT_SECS ; échec/timeout -> log + on
# poursuit. `state["seller_data"]` accumule les champs ; `state["lead_id"]` mémorise le
# lead créé, partagé par les updates suivantes.


def _build_seller(seller_data: dict) -> SellerLead:
    """Construit un SellerLead depuis l'accumulateur d'état.

    `sous_intention` est porté dans l'état mais n'est pas un champ SellerLead : on
    l'exclut. Pydantic coerce les chaînes vers les enums PropertyType / Timeframe.

    Args:
        seller_data: Champs collectés au fil de l'appel.

    Returns:
        Un SellerLead validé.
    """
    fields = {k: v for k, v in seller_data.items() if k != "sous_intention"}
    return SellerLead(**fields)


def _build_call_summary(data: dict) -> str:
    """Assemble un résumé d'appel par simple concaténation des champs collectés.

    Pas d'appel LLM séparé : on agrège ce qu'on a déjà en mémoire d'état.

    Args:
        data: L'accumulateur `state["seller_data"]`.

    Returns:
        Un résumé d'une ligne.
    """
    parts = [
        f"Vendeur {data.get('first_name', '')} {data.get('last_name', '')}".strip(),
        f"intention {data.get('sous_intention', '?')}",
        f"bien {data.get('property_type', '?')} à {data.get('city', '?')}",
        f"prix {data.get('expected_price', '?')}",
        f"délai {data.get('timeframe', '?')}",
        f"raison {data.get('reason', '?')}",
        f"propriétaire unique {data.get('sole_owner', '?')}",
    ]
    return " — ".join(parts) + "."


async def _odoo_update(flow_manager: FlowManager, etape: str) -> None:
    """Met à jour (une seule écriture) le lead Odoo courant depuis l'état.

    Bornée à ODOO_TIMEOUT_SECS dans un thread ; ignorée s'il n'y a pas de lead_id
    (création précédente échouée) ; tout échec/timeout est loggé et la conversation
    continue.

    Args:
        flow_manager: Le FlowManager courant (porte l'état).
        etape: Libellé court pour les logs (ex. « bien », « qualif »).
    """
    lead_id = flow_manager.state.get("lead_id")
    if not lead_id:
        logger.warning(f"update {etape} ignoré : pas de lead_id (création Odoo échouée)")
        return
    try:
        seller = _build_seller(flow_manager.state["seller_data"])
        await asyncio.wait_for(
            asyncio.to_thread(lambda: get_client().update_lead(lead_id, seller)),
            timeout=ODOO_TIMEOUT_SECS,
        )
        logger.info(f"Lead {lead_id} mis à jour ({etape})")
    except asyncio.TimeoutError:
        logger.error(f"Odoo update_lead ({etape}) : timeout >{ODOO_TIMEOUT_SECS}s — on poursuit")
    except Exception as e:
        logger.error(f"Odoo update_lead ({etape}) a échoué — on poursuit : {e}")


def create_vendeur_contact_node() -> NodeConfig:
    """Node 1 du tunnel vendeur : collecte du contact (nom + téléphone).

    Returns:
        La configuration du node « vendeur_contact ».
    """
    return NodeConfig(
        name="vendeur_contact",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Pour enregistrer la demande, demande à l'appelant son nom de "
                    "famille et son numéro de téléphone. Dès que tu as les deux, "
                    "appelle enregistrer_contact_vendeur."
                ),
            }
        ],
        functions=[enregistrer_contact_vendeur],
    )


@flows_tool_options(cancel_on_interruption=False)
async def enregistrer_contact_vendeur(  # noqa: D417
    flow_manager: FlowManager, nom: str, telephone: str
) -> tuple[None, NodeConfig]:
    """Enregistre le contact du vendeur et CRÉE le lead Odoo (création minimale).

    Args:
        nom: Nom de famille du vendeur.
        telephone: Numéro de téléphone du vendeur (format +32… si possible).

    Returns:
        Un tuple (résultat, node suivant) : transition vers « vendeur_bien ».
    """
    data = flow_manager.state.setdefault("seller_data", {})
    data["first_name"] = flow_manager.state.get("prenom", "")
    data["last_name"] = nom
    data["phone"] = telephone
    data["sous_intention"] = flow_manager.state.get("sous_intention", "vente_directe")
    flow_manager.state["lead_id"] = None
    try:
        seller = _build_seller(data)
        result = await asyncio.wait_for(
            asyncio.to_thread(lambda: get_client().create_lead_minimal(seller)),
            timeout=ODOO_TIMEOUT_SECS,
        )
        flow_manager.state["lead_id"] = result["lead_id"]
        logger.info(f"Lead vendeur créé (id={result['lead_id']}, priorité={result['priority']})")
    except asyncio.TimeoutError:
        logger.error(f"Odoo create_lead_minimal : timeout >{ODOO_TIMEOUT_SECS}s — on poursuit")
    except Exception as e:
        logger.error(f"Odoo create_lead_minimal a échoué — on poursuit : {e}")
    return None, create_vendeur_bien_node()


def create_vendeur_bien_node() -> NodeConfig:
    """Node 2 du tunnel vendeur : caractéristiques du bien.

    Returns:
        La configuration du node « vendeur_bien ».
    """
    return NodeConfig(
        name="vendeur_bien",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Demande le type de bien (appartement, maison, terrain, commerce "
                    "ou autre), la commune et le prix de vente espéré. Dès que tu as "
                    "ces trois informations, appelle qualifier_bien_vendeur."
                ),
            }
        ],
        functions=[qualifier_bien_vendeur],
    )


@flows_tool_options(cancel_on_interruption=False)
async def qualifier_bien_vendeur(  # noqa: D417
    flow_manager: FlowManager, type_bien: str, ville: str, prix: float
) -> tuple[None, NodeConfig]:
    """Enregistre les caractéristiques du bien et MET À JOUR le lead Odoo.

    Args:
        type_bien: Type de bien. L'une des valeurs : "appartement", "maison",
            "terrain", "commerce", "autre".
        ville: Commune / ville du bien.
        prix: Prix de vente espéré, en euros.

    Returns:
        Un tuple (résultat, node suivant) : transition vers « vendeur_qualif ».
    """
    data = flow_manager.state.setdefault("seller_data", {})
    data["property_type"] = type_bien
    data["city"] = ville
    data["expected_price"] = prix
    await _odoo_update(flow_manager, "bien")
    return None, create_vendeur_qualif_node()


def create_vendeur_qualif_node() -> NodeConfig:
    """Node 3 du tunnel vendeur : qualifiers (délai, raison, propriétaire).

    Returns:
        La configuration du node « vendeur_qualif ».
    """
    return NodeConfig(
        name="vendeur_qualif",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Demande le délai de vente souhaité, la raison de la vente et si "
                    "l'appelant est le seul propriétaire / décideur. Dès que tu as ces "
                    "informations, appelle finaliser_vendeur."
                ),
            }
        ],
        functions=[finaliser_vendeur],
    )


@flows_tool_options(cancel_on_interruption=False)
async def finaliser_vendeur(  # noqa: D417
    flow_manager: FlowManager, delai: str, raison: str, proprietaire_unique: bool
) -> tuple[None, NodeConfig]:
    """Enregistre la qualification finale et MET À JOUR le lead Odoo.

    Args:
        delai: Délai de vente souhaité. L'une des valeurs : "asap", "1_month",
            "3_months", "6_months_plus", "unsure".
        raison: Raison de la vente (succession, déménagement, etc.).
        proprietaire_unique: True si l'appelant est le seul propriétaire / décideur.

    Returns:
        Un tuple (résultat, node suivant) : transition vers « vendeur_fin ».
    """
    data = flow_manager.state.setdefault("seller_data", {})
    data["timeframe"] = delai
    data["reason"] = raison
    data["sole_owner"] = proprietaire_unique
    data["call_summary"] = _build_call_summary(data)
    await _odoo_update(flow_manager, "qualif")
    return None, create_vendeur_fin_node()


def create_vendeur_fin_node() -> NodeConfig:
    """Node terminal du tunnel vendeur : remerciement + promesse de rappel.

    Returns:
        La configuration du node « vendeur_fin ».
    """
    return NodeConfig(
        name="vendeur_fin",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Remercie chaleureusement l'appelant pour ces informations et "
                    "indique qu'un conseiller le recontactera très prochainement. "
                    "Conclus l'échange."
                ),
            }
        ],
    )


# --- Nodes acheteur / location (placeholders inchangés, Phase 6) --------------------


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
    return None, create_vendeur_contact_node()


@flows_tool_options(cancel_on_interruption=False)
async def router_estimation(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """L'appelant souhaite faire ESTIMER la valeur d'un bien immobilier.

    Returns:
        Un tuple (résultat, node suivant) : transition vers le tunnel « vendeur ».
    """
    flow_manager.state["intention"] = "vendeur"
    flow_manager.state["sous_intention"] = "estimation"
    logger.info("Intention : vendeur / estimation")
    return None, create_vendeur_contact_node()


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
