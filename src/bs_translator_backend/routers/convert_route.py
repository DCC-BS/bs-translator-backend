from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, UploadFile

from bs_translator_backend.container import Container
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.utils.logger import get_logger

logger = get_logger("convert_router")


@inject
def create_router(
    document_conversion_service: DocumentConversionService = Provide[Container.document_conversion_service],
) -> APIRouter:
    logger.info("Creating convert router")
    router: APIRouter = APIRouter(prefix="/convert", tags=["convert"])

    @router.post("/doc")
    def convert(file: UploadFile) -> str:
        """
        Translate the content of an uploaded document.
        """

        markdown = document_conversion_service.convert(file)

        return markdown

    logger.info("Conversion router configured")
    return router
