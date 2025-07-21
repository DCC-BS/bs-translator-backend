from collections.abc import Generator
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import StreamingResponse
from llama_index.core import Document

from bs_translator_backend.container import Container
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.models.translation_input import TranslationInput
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.logger import get_logger

logger = get_logger("translation_router")


@inject
def create_router(translation_service: TranslationService = Provide[Container.translation_service]) -> APIRouter:
    logger.info("Creating translation router")
    router: APIRouter = APIRouter(prefix="/translation", tags=["translation"])

    @router.get("/languages")
    def get_languages() -> list[str]:
        return translation_service.get_supported_languages()

    @router.post("/text")
    def translate_text(translation_input: TranslationInput) -> StreamingResponse:
        """
        Translate the provided text using the specified configuration.
        """
        logger.info("Translating text")

        def generate_translation() -> Generator[str, None, None]:
            yield from translation_service.translate_text(translation_input.text, translation_input.config)

        return StreamingResponse(generate_translation(), media_type="text/plain")

    @router.post("/doc")
    def convert(file: UploadFile) -> str:
        """
        Translate the content of an uploaded document.
        """

        markdown = DocumentConversionService().convert(file)

        return markdown

    logger.info("Translation router configured")
    return router
