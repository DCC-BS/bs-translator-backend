from collections.abc import AsyncGenerator
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header, Request, UploadFile
from fastapi.params import Form
from fastapi.responses import StreamingResponse

from bs_translator_backend.container import Container
from bs_translator_backend.models.translation import (
    DetectLanguageInput,
    DetectLanguageOutput,
    TranslationConfig,
    TranslationInput,
)
from bs_translator_backend.services.translation_service import TranslationService

logger = get_logger(__name__)


@inject
def create_router(  # noqa: C901
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
    ) -> StreamingResponse:
        """Translate the provided text using the specified configuration."""
        config = translation_input.config
        usage_tracking_service.log_event(
            "translation.text",
            user_id=x_client_id,
            text_length=len(translation_input.text),
            target_language=config.target_language.value,
            source_language=config.source_language.value if config.source_language is not None else None,
            domain=config.domain,
            tone=config.tone,
        )

        async def generate_stream() -> AsyncGenerator[str, None]:
            async for chunk in translation_service.translate_text(
                translation_input.text, translation_input.config
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping translation stream")
                    break
                yield chunk

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.post("/image", summary="Translate text in image")
    async def translate_image(
        request: Request,
        image_file: UploadFile,
        x_client_id: Annotated[str | None, Header()],
        translation_config: Annotated[str, Form()] = '{"target_language": "de"}',
    ) -> StreamingResponse:
        """
        Translate text within an uploaded image file.

        Args:
            image_file: Uploaded image file containing text to translate
            source_language: Source language code (default: auto-detect)
            target_language: Target language code for translation

        Returns:
            StreamingResponse: Streaming response with translated text
        """
        config = TranslationConfig.model_validate_json(translation_config)

        usage_tracking_service.log_event(
            "translation.image",
            user_id=x_client_id,
            target_language=config.target_language.value,
            source_language=config.source_language.value if config.source_language is not None else None,
        )

        # Read the file content once to avoid issues with file being closed in streaming context
        try:
            file_content = image_file.file.read()
            from io import BytesIO

            file_stream = BytesIO(file_content)
            filename = image_file.filename
            content_type = image_file.content_type
        except Exception:
            logger.exception("Failed to read uploaded file", filename=image_file.filename)
            raise

        async def generate_translation() -> AsyncGenerator[str, None]:
            async for translation in translation_service.translate_image(
                file_stream, config, filename, content_type
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping translation stream")
                    break

                yield translation.model_dump_json()

        return StreamingResponse(
            generate_translation(),
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
