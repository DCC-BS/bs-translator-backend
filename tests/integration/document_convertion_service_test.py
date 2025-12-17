import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.conversion_result import ConversionResult
from bs_translator_backend.models.language import DetectLanguage
from bs_translator_backend.services.document_conversion_service import DocumentConversionService


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig.from_env()


@pytest.mark.asyncio
async def test_convert_pdf_file(app_config: AppConfig) -> None:
    service = DocumentConversionService(app_config)

    async def fake_convert(*args, **kwargs) -> ConversionResult:
        return ConversionResult(markdown="# demo", images={})

    service.convert = fake_convert  # type: ignore[method-assign]

    with open("./tests/assets/Pocketbook.pdf", "rb") as file:
        headers = Headers({"content-type": "application/pdf"})

        upload_file = UploadFile(file=file, filename="example_report.pdf", headers=headers)
        result = await service.convert(upload_file, DetectLanguage.AUTO)
        assert hasattr(result, "markdown")
        assert hasattr(result, "images")
