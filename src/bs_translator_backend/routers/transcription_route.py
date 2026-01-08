import io
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import Field

from bs_translator_backend.container import Container
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.services.transcription_service import TranscriptionService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService

logger = get_logger(__name__)


@inject
def create_router(
    transcription_service: TranscriptionService = Provide[Container.transcription_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create and configure the transcription API router.

    Args:
        request: The incoming request object, used to check for client disconnects.
        audio_file: Uploaded audio file to transcribe.
        x_client_id: Optional client identifier for usage tracking.
        language: Language of the audio file, or 'auto' to detect.

    Returns:
        APIRouter: Configured router with transcription endpoints
    """
    logger.info("Creating transcription router")
    router: APIRouter = APIRouter(prefix="/transcription", tags=["transcription"])

    @router.post("/audio", summary="Transcribe audio")
    async def transcribe_audio(
        request: Request,
        audio_file: UploadFile,
        x_client_id: Annotated[str | None, Header()],
        language: Annotated[
            LanguageOrAuto,
            Form(title="Language", description="Language of the audio file", example=Field),
        ] = DetectLanguage.AUTO,
    ) -> StreamingResponse:
        """
        Transcribe the provided audio file.

        Args:
            audio_file: Uploaded audio file to transcribe

        Returns:
            StreamingResponse: Streaming response with transcribed text
        """

        usage_tracking_service.log_event(
            __name__, transcribe_audio.__name__, user_id=x_client_id, file_size=audio_file.size
        )

        content = await audio_file.read()
        buffer = io.BytesIO(content)

        async def stream_response():
            async for chunk in transcription_service.transcribe(buffer, language):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping transcription stream")
                    break
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/plain")

    logger.info("Transcription router configured")
    return router
