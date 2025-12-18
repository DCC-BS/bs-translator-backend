from typing import cast

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
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.dspy_config.translation_program import TranslationModule
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.image_overlay import (
    create_side_by_side_comparison,
    overlay_translations_on_image,
)


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig.from_env()


class StubTranslationModule:
    async def stream(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        domain: str = "",
        tone: str = "",
        glossary: str = "",
        context: str = "",
    ):
        normalized_target = (
            target_language.lower() if hasattr(target_language, "lower") else str(target_language)
        )
        yield f"[{normalized_target}] {source_text.strip()}"


@pytest.fixture
def translation_service(app_config: AppConfig) -> TranslationService:
    async def fake_convert_to_docling(*args, **kwargs) -> DoclingDocument:
        bbox = BoundingBox(l=5, t=5, r=120, b=40)
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
    translation_module = cast(TranslationModule, StubTranslationModule())

    return TranslationService(translation_module, text_chunk_service, conversion_service_factory)


@pytest.mark.asyncio
async def test_image_translate_with_overlay(translation_service: TranslationService) -> None:
    """Test image translation and create visual overlay of translated text."""
    with open("./tests/assets/ReportView.png", "rb") as file:
        headers = Headers({"content-type": "image/png"})
        upload_file = UploadFile(file=file, filename="ReportView.png", headers=headers)
        translate_config = TranslationConfig(source_language=Language.DE)

        # Collect all translation entries
        translation_entries = [
            entry
            async for entry in translation_service.translate_image(upload_file, translate_config)
        ]

        # Create overlay visualization if we have translations
        if translation_entries:
            # You can provide the original image path or bytes
            # For this demo, let's assume we have an image file
            # In a real scenario, you might extract the image from the PDF first

            try:
                # Try to create overlay on a test image
                # Note: You'll need an actual image file for this to work
                # This is just demonstrating the API

                # Option 1: Simple overlay with default styling
                _ = overlay_translations_on_image(
                    "./tests/assets/ReportView.png",  # Use the existing test image
                    translation_entries,
                    output_path="./tests/output_translated.png",
                    font_size=14,
                    text_color="red",
                    background_color="white",
                    background_opacity=180,
                )
                assert _ is not None

                _ = create_side_by_side_comparison(
                    "./tests/assets/ReportView.png",
                    translation_entries,
                    output_path="./tests/output_comparison.png",
                    font_size=12,
                    text_color="blue",
                    background_color="yellow",
                    background_opacity=150,
                )
                assert _ is not None

            except Exception as e:
                raise AssertionError(f"Overlay creation failed: {e}") from e

        assert len(translation_entries) > 0, "Should have at least one translation entry"


@pytest.mark.asyncio
async def test_translation_bbox_coordinates(translation_service: TranslationService) -> None:
    """Test that bbox coordinates are properly extracted and can be used for overlay."""
    with open("./tests/assets/ReportView.png", "rb") as file:
        headers = Headers({"content-type": "image/png"})
        upload_file = UploadFile(file=file, filename="ReportView.png", headers=headers)
        translate_config = TranslationConfig(source_language=Language.DE)

        async for entry in translation_service.translate_image(upload_file, translate_config):
            # Check that bbox coordinates are valid
            assert entry.bbox is not None
            assert hasattr(entry.bbox, "left")
            assert hasattr(entry.bbox, "top")
            assert hasattr(entry.bbox, "right")
            assert hasattr(entry.bbox, "bottom")

            # Check coordinate validity
            assert entry.bbox.left >= 0
            assert entry.bbox.top >= 0
            assert entry.bbox.right > entry.bbox.left
            # Handle different coordinate origins - in BOTTOMLEFT origin, bottom can be less than top
            if hasattr(entry.bbox, "coord_origin") and entry.bbox.coord_origin == "BOTTOMLEFT":
                assert entry.bbox.bottom < entry.bbox.top  # For bottom-left origin
            else:
                assert entry.bbox.bottom > entry.bbox.top  # For top-left origin
