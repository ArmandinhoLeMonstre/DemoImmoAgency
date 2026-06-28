"""Phase 1 — agent vocal inbound (FR) : pipeline STT -> LLM -> TTS, sans Flows.

Pipeline SmallWebRTC (dev local) : Gladia (fr) -> OpenAI -> Cartesia (Sonic),
VAD Silero. Lancé par le dev runner Pipecat (`uv run bot.py`, UI sur
http://localhost:7860/client). La logique de qualification arrive aux phases
suivantes ; ici on valide seulement que la voix fonctionne.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
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

load_dotenv(override=True)

# Persona FR de l'agent d'accueil. En Phase 1 il accueille et discute simplement :
# pas encore de qualification structurée (vendeur/acheteur/...) ni de CRM.
SYSTEM_INSTRUCTION = (
    "Tu es l'agent vocal d'accueil d'une agence immobilière belge. Tu réponds en "
    "français, sur un ton chaleureux et professionnel. Tes réponses sont lues à voix "
    "haute : pas d'emojis, de listes à puces ni de mise en forme. Sois bref et "
    "naturel. Accueille l'appelant et demande comment tu peux l'aider."
)

# En Phase 1 on ne sert que le transport SmallWebRTC en local. Les autres transports
# (Daily, téléphonie) viendront aux phases ultérieures avec leurs extras dédiés.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Construit le pipeline STT -> LLM -> TTS et le fait tourner.

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

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Entrée audio (utilisateur)
            stt,  # Gladia (fr)
            user_aggregator,  # Tour de parole utilisateur
            llm,  # OpenAI
            tts,  # Cartesia (Sonic)
            transport.output(),  # Sortie audio (bot)
            assistant_aggregator,  # Tour de parole assistant
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Inbound : l'agent parle en premier dès la connexion du client."""
        logger.info("Client connecté")
        context.add_message({"role": "developer", "content": "Accueille l'appelant en une phrase."})
        await worker.queue_frames([LLMRunFrame()])

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
