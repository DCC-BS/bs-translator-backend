"""
Translation API Router

This module defines the FastAPI routes for text translation services.
It provides endpoints for retrieving supported languages and translating
text with customizable parameters.
"""

from collections.abc import Generator
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from bs_translator_backend.container import Container
from bs_translator_backend.models.translation_input import TranslationInput
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService
from bs_translator_backend.utils.logger import get_logger

logger = get_logger("translation_router")


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
    logger.info("Creating translation router")
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
    def translate_text(
        translation_input: TranslationInput, x_client_id: Annotated[str | None, Header()]
    ) -> StreamingResponse:
        """
        Translate the provided text using the specified configuration.

        Args:
            translation_input: Translation request containing text and configuration

        Returns:
            StreamingResponse: Streaming response with translated text
        """
        usage_tracking_service.log_event(
            __name__,
            translate_text.__name__,
            user_id=x_client_id,
            text_length=len(translation_input.text),
            target_language=str(translation_input.config.target_language),
            source_language=str(translation_input.config.source_language),
            domain=translation_input.config.domain,
            tone=translation_input.config.tone,
        )

        def generate_translation() -> Generator[str, None, None]:
            yield from translation_service.translate_text(
                translation_input.text, translation_input.config
            )

        return StreamingResponse(generate_translation(), media_type="text/plain")

    logger.info("Translation router configured")
    return router
