from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter

from bs_translator_backend.container import Container
from bs_translator_backend.models.translation_input import TranslationInput
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

    @router.post("/translate")
    def translate_text(input: TranslationInput) -> str:
        """
        Translate the provided text using the specified configuration.
        """
        logger.info("Translating text")
        return translation_service.translate_text(input.text, input.config)

    logger.info("Translation router configured")
    return router
