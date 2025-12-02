import asyncio
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, Request, UploadFile

from bs_translator_backend.container import Container
from bs_translator_backend.models.conversion_result import ConversionOutput
from bs_translator_backend.models.language import LanguageOrAuto
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService
from bs_translator_backend.utils.logger import get_logger

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
        request: Request,
        file: UploadFile,
        x_client_id: Annotated[str | None, Header()],
        source_language: Annotated[LanguageOrAuto, Form()],
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

        task = asyncio.create_task(document_conversion_service.convert(file, source_language))

        while task.done() is False:
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                task.cancel()
                logger.info("Conversion task cancelled due to client disconnect")
                return ConversionOutput(markdown="", images=[])

        result = task.result()
        return ConversionOutput(markdown=result.markdown, images=result.images)

    logger.info("Conversion router configured")
    return router
