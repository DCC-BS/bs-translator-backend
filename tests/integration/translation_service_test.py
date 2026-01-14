from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from bs_translator_backend.models.docling_response import (
    BoundingBox,
    DoclingDocument,
    ProvenanceItem,
    TextItem,
)
from bs_translator_backend.models.language import Language
from bs_translator_backend.models.translation import TranslationConfig
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.app_config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig.from_env()


@pytest.fixture
def translation_service(app_config: AppConfig, monkeypatch) -> TranslationService:
    async def fake_convert_to_docling(*args, **kwargs) -> DoclingDocument:
        bbox = BoundingBox(l=0, t=0, r=10, b=10)
        provenance = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, 5))
        text_item = TextItem(
            self_ref="#/texts/0",
            orig="Hallo",
            text="Hallo",
            label="text",
            prov=[provenance],
        )
        return DoclingDocument(name="demo", texts=[text_item])

    def conversion_service_factory() -> DocumentConversionService:
        service = DocumentConversionService(app_config)
        service.convert_to_docling = fake_convert_to_docling  # type: ignore[method-assign]
        return service

    text_chunk_service = TextChunkService()

    service = TranslationService(app_config, text_chunk_service, conversion_service_factory)

    # Mock the translation agent to avoid requiring actual LLM
    async def mock_stream_text(delta: bool = False):
        yield "[german] Hallo"

    mock_stream = MagicMock()
    mock_stream.stream_text = mock_stream_text
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    service.translation_agent = MagicMock()
    service.translation_agent.run_stream = MagicMock(return_value=mock_stream)

    return service


@pytest.mark.asyncio
async def test_image_translate(translation_service: TranslationService) -> None:
    with open("./tests/assets/ReportView.png", "rb") as file:
        headers = Headers({"content-type": "image/png"})

        upload_file = UploadFile(file=file, filename="ReportView.png", headers=headers)
        translate_config = TranslationConfig(
            source_language=Language.DE, target_language=Language.EN
        )

        translation_entries = []

        async for entry in translation_service.translate_image(upload_file, translate_config):
            translation_entries.append(entry)

        assert len(translation_entries) > 0, "Should have translated at least one text segment"
        assert translation_entries[0].translated.startswith("[german]")
