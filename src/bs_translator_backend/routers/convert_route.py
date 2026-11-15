from collections.abc import Callable
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, Request, Response, UploadFile

from bs_translator_backend.container import Container
from bs_translator_backend.models.conversion_result import ConversionOutput
from bs_translator_backend.models.language import LanguageOrAuto
from bs_translator_backend.services.document_conversion_service import (
    DocumentConversionService,
)
from bs_translator_backend.utils.cancelation import ClientDisconnectedError, run_until_disconnected

# nginx closes the connection without a response; used when the client went away.
CLIENT_CLOSED_REQUEST = 499

logger = get_logger(__name__)


@inject
def create_router(
    document_conversion_service_factory: Callable[[], DocumentConversionService] = Provide[
        Container.document_conversion_service.provider
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
    logger.debug("Creating convert router")
    router: APIRouter = APIRouter(prefix="/convert", tags=["convert"])

    @router.post("/doc", summary="Convert document to markdown", response_model=ConversionOutput)
    async def convert(
        request: Request,
        file: UploadFile,
        x_client_id: Annotated[str | None, Header()],
        source_language: Annotated[LanguageOrAuto, Form()],
    ) -> ConversionOutput | Response:
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
            "document.convert", user_id=x_client_id, file_size=file.size
        )

        async with document_conversion_service_factory() as conversion_service:
            # The conversion is cancelled and awaited on disconnect, before the service
            # (and its HTTP client) is closed by the `async with`.
            try:
                result = await run_until_disconnected(
                    request, conversion_service.convert(file, source_language)
                )
            except ClientDisconnectedError:
                logger.info("Conversion task cancelled due to client disconnect")
                return Response(status_code=CLIENT_CLOSED_REQUEST)

            return ConversionOutput(markdown=result.markdown, images=result.images)

    logger.debug("Conversion router configured")
    return router
