"""
Document Conversion API Router

This module defines the FastAPI routes for document conversion services.
It provides endpoints for converting various document formats (PDF, DOCX)
to markdown with image extraction capabilities.
"""

from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, UploadFile

from bs_translator_backend.container import Container
from bs_translator_backend.models.conversion_result import ConversionOutput
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService
from bs_translator_backend.utils.logger import get_logger
from bs_translator_backend.models.langugage import LanguageOrAuto

logger = get_logger("convert_router")


@inject
def create_router(
    document_conversion_service: DocumentConversionService = Provide[
        Container.document_conversion_service
    ],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create and configure the document conversion API router.

    Args:
        document_conversion_service: Injected document conversion service instance

    Returns:
        APIRouter: Configured router with conversion endpoints
    """
    logger.info("Creating convert router")
    router: APIRouter = APIRouter(prefix="/convert", tags=["convert"])

    @router.post("/doc", summary="Convert document to markdown")
    async def convert(
        file: UploadFile,
        x_client_id: Annotated[str | None, Header()],
        source_language: LanguageOrAuto = Form(),
    ) -> ConversionOutput:
        """
        Convert the content of an uploaded document to markdown with images.

        This endpoint accepts various document formats (PDF, DOCX) and converts
        them to markdown format while extracting and encoding any embedded images.

        Args:
            file: Uploaded document file to convert

        Returns:
            ConversionOutput: Conversion result with markdown content and images
        """

        usage_tracking_service.log_event(
            __name__, convert.__name__, user_id=x_client_id, file_size=file.size
        )

        result = await document_conversion_service.convert(file, source_language)
        return ConversionOutput(markdown=result.markdown, images=result.images)

    logger.info("Conversion router configured")
    return router
