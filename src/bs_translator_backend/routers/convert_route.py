import base64
from io import BytesIO

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, UploadFile

from bs_translator_backend.container import Container
from bs_translator_backend.models.conversion_result import ConversionOutput
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
    def convert(file: UploadFile) -> ConversionOutput:
        """
        Convert the content of an uploaded document to markdown with images.
        """

        result = document_conversion_service.convert(file)

        # Convert PIL Images to base64 strings for JSON serialization
        images: dict[int, str] = {}
        for i, img in result.images.items():
            buffer = BytesIO()
            # Save as PNG format for consistency
            img.save(buffer, format='PNG')
            _ = buffer.seek(0)
            # Encode to base64 with data URI prefix
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images[i] = f"data:image/png;base64,{img_base64}"

        return ConversionOutput(
            markdown=result.markdown,
            images=images
        )

    logger.info("Conversion router configured")
    return router
