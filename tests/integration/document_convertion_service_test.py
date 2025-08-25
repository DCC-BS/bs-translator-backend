import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.langugage import DetectLanguage
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.utils.load_env import load_env


@pytest.fixture
def app_config() -> AppConfig:
    load_env()

    # Provide a minimal AppConfig stub
    # Adjust the constructor as needed to match your AppConfig definition
    return AppConfig.from_env()


@pytest.mark.asyncio
async def test_convert_pdf_file(app_config: AppConfig) -> None:
    service = DocumentConversionService(app_config)

    with open("./tests/assets/Pocketbook.pdf", "rb") as file:
        headers = Headers({"content-type": "application/pdf"})

        upload_file = UploadFile(file=file, filename="example_report.pdf", headers=headers)
        result = service.convert(upload_file, DetectLanguage.AUTO)
        assert hasattr(result, "markdown")
        assert hasattr(result, "images")
