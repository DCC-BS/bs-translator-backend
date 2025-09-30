"""
Translation API Router

This module defines the FastAPI routes for text translation services.
It provides endpoints for retrieving supported languages and translating
text with customizable parameters.
"""
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import Field

from bs_translator_backend.container import Container
from bs_translator_backend.models.langugage import DetectLanguage, LanguageOrAuto
from bs_translator_backend.services.transcription_service import TranscriptionService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService
from bs_translator_backend.utils.logger import get_logger

logger = get_logger("transcription_router")


@inject
def create_router(
    transcription_service: TranscriptionService = Provide[Container.transcription_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create and configure the transcription API router.

    Args:
        transcription_service: Injected transcription service instance

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
        language: Annotated[LanguageOrAuto, Form(title="Language", description="Language of the audio file", example=Field)] = DetectLanguage.AUTO,
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

        content = audio_file.file.read()

        async def stream_response():
            async for chunk in transcription_service.transcribe(content, language):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping transcription stream")
                    break
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/plain")

    logger.info("Transcription router configured")
    return router
