from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header, Request
from fastapi.responses import Response, StreamingResponse

from bs_translator_backend.container import Container
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.translation import (
    DetectLanguageInput,
    DetectLanguageOutput,
    TranslationInput,
)
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.cancelation import ClientDisconnectedError, run_until_disconnected
from bs_translator_backend.utils.streaming import (
    peek_first,
    stream_error_trailer,
    strip_stream_error_separator,
)

# nginx closes the connection without a response; used when the client went away.
CLIENT_CLOSED_REQUEST = 499

logger = get_logger(__name__)


async def _stream_with_error_trailer(
    request: Request, peeked: tuple[str, AsyncGenerator[str, None]] | None
) -> AsyncGenerator[str, None]:
    """Stream the translation, ending it with an error trailer if it fails mid-way."""
    if peeked is None:
        return
    _, chunks = peeked
    # aclosing: on disconnect the service generator must be closed here, inside the
    # request context, so its cleanup (llm_call usage logging) runs deterministically
    # instead of at garbage collection.
    async with aclosing(chunks) as chunks:
        try:
            async for chunk in chunks:
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping translation stream")
                    break
                yield strip_stream_error_separator(chunk)
        except ApiErrorException as e:
            # Headers (status 200) are already sent by this point: mark the stream as
            # failed so the client does not take the truncated text for a complete
            # translation.
            logger.warning(
                "Translation failed mid-stream", error_id=e.error_response.get("errorId")
            )
            yield stream_error_trailer(e.error_response)
        except Exception:
            logger.exception("Unexpected error mid-stream")
            yield stream_error_trailer()


@inject
def create_router(
    translation_service: TranslationService = Provide[Container.translation_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create and configure the translation API router.

    Args:
        translation_service: Injected translation service instance

    Returns:
        APIRouter: Configured router with translation endpoints
    """
    logger.debug("Creating translation router")
    router: APIRouter = APIRouter(prefix="/translation", tags=["translation"])

    @router.get("/languages", summary="Get supported languages")
    def get_languages() -> list[str]:
        """
        Retrieve the list of supported languages for translation.

        Returns:
            list[str]: List of supported language codes
        """
        return translation_service.get_supported_languages()

    @router.post("/text", summary="Translate text")
    async def translate_text(
        request: Request,
        translation_input: TranslationInput,
        x_client_id: Annotated[str | None, Header()],
    ) -> Response:
        """Translate the provided text using the specified configuration.

        Streams plain text. If the translation fails after the first chunk was sent, the
        stream ends with U+001E followed by `{"errorId": ..., "status": ...}`.
        """
        config = translation_input.config
        usage_tracking_service.log_event(
            "translation.text",
            user_id=x_client_id,
            text_length=len(translation_input.text),
            target_language=config.target_language.value,
            source_language=config.source_language.value
            if config.source_language is not None
            else None,
            domain=config.domain,
            tone=config.tone,
        )

        # Peeking the first chunk here, before the StreamingResponse exists, means a
        # failure that happens right away (e.g. vLLM already overloaded) raises inside
        # this handler, where it still becomes a proper ApiErrorException JSON response
        # instead of a response that's already committed to status 200.
        try:
            peeked = await run_until_disconnected(
                request,
                peek_first(
                    translation_service.translate_text(
                        translation_input.text, translation_input.config
                    )
                ),
            )
        except ClientDisconnectedError:
            return Response(status_code=CLIENT_CLOSED_REQUEST)

        return StreamingResponse(
            _stream_with_error_trailer(request, peeked),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.post("/detect-language", summary="Detect language")
    async def detect_language(
        request: Request,
        detect_language_input: DetectLanguageInput,
    ) -> DetectLanguageOutput:
        """Detect the language of the text"""
        return await translation_service.detect_language(detect_language_input)

    logger.debug("Translation router configured")
    return router
