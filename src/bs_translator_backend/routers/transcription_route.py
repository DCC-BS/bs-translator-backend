import io
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from bs_translator_backend.container import Container
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.services.transcription_service import TranscriptionService
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.cancelation import ClientDisconnectedError, run_until_disconnected
from bs_translator_backend.utils.streaming import peek_first
from bs_translator_backend.utils.uploads import read_upload_limited

logger = get_logger(__name__)

# nginx closes the connection without a response; used when the client went away.
CLIENT_CLOSED_REQUEST = 499


async def _stream_transcript(
    request: Request, peeked: tuple[str, AsyncGenerator[str, None]] | None
) -> AsyncGenerator[str, None]:
    if peeked is None:
        return
    _, chunks = peeked
    async with aclosing(chunks) as chunks:
        try:
            async for chunk in chunks:
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping transcription stream")
                    break
                yield chunk
        except ApiErrorException as e:
            # Headers (status 200) are already sent: all that is left is to stop.
            logger.warning(
                "Transcription failed mid-stream", error_id=e.error_response.get("errorId")
            )


@inject
def create_router(
    transcription_service: TranscriptionService = Provide[Container.transcription_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
    app_config: AppConfig = Provide[Container.app_config],
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
    logger.debug("Creating transcription router")
    router: APIRouter = APIRouter(prefix="/transcription", tags=["transcription"])

    @router.post("/audio", summary="Transcribe audio")
    async def transcribe_audio(
        request: Request,
        audio_file: UploadFile,
        x_client_id: Annotated[str | None, Header()],
        language: Annotated[
            LanguageOrAuto,
            Form(title="Language", description="Language of the audio file", examples=["de"]),
        ] = DetectLanguage.AUTO,
    ) -> Response:
        """
        Transcribe the provided audio file.

        Args:
            audio_file: Uploaded audio file to transcribe

        Returns:
            StreamingResponse: Streaming response with transcribed text
        """

        usage_tracking_service.log_event(
            "transcription.audio",
            user_id=x_client_id,
            file_size=audio_file.size,
            language=language.value,
        )

        content = await read_upload_limited(audio_file, app_config.max_upload_bytes)

        # Wait for Whisper's first chunk before committing to a 200, so a connect error,
        # timeout or Whisper error status still becomes a proper JSON error response.
        try:
            peeked = await run_until_disconnected(
                request, peek_first(transcription_service.transcribe(io.BytesIO(content), language))
            )
        except ClientDisconnectedError:
            return Response(status_code=CLIENT_CLOSED_REQUEST)

        return StreamingResponse(
            _stream_transcript(request, peeked),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
            },
        )

    logger.debug("Transcription router configured")
    return router
