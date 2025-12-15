import logging

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.language import Language
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.services.dspy_config.qwen3 import DspyVllm
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.llm_facade import LLMFacade
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.load_env import load_env

logger = logging.getLogger(__name__)
logger.info("Test started")


@pytest.fixture
def app_config() -> AppConfig:
    load_env()

    # Provide a minimal AppConfig stub
    # Adjust the constructor as needed to match your AppConfig definition
    return AppConfig.from_env()


@pytest.mark.asyncio
async def test_image_translate(app_config: AppConfig) -> None:
    conversion_service = DocumentConversionService(app_config)

    llm = DspyVllm.from_config(app_config).inference_lm
    llm_facade = LLMFacade(llm)
    text_chunk_service = TextChunkService()
    translation_service = TranslationService(llm_facade, text_chunk_service, conversion_service)

    with open("./tests/assets/ReportView.png", "rb") as file:
        headers = Headers({"content-type": "image/png"})

        upload_file = UploadFile(file=file, filename="ReportView.png", headers=headers)
        translate_config = TranslationConfig(source_language=Language.DE)

        # Collect translation entries for potential image overlay
        translation_entries = []

        async for entry in translation_service.translate_image(upload_file, translate_config):
            logger.info(f"{entry.original} ==> {entry.translated} BBox: {entry.bbox}")
            translation_entries.append(entry)

        # Simple demonstration of how you could use the image overlay functionality
        logger.info("\n📊 Translation Summary:")
        logger.info(f"   Total translations: {len(translation_entries)}")

        if translation_entries:
            logger.info("\n🎨 To create image overlays, you can use:")
            logger.info(
                "   from bs_translator_backend.utils.image_overlay import overlay_translations_on_image"
            )
            logger.info(
                "   overlay_translations_on_image(image_path, translation_entries, output_path)"
            )

            # Show first few bbox coordinates as examples
            for i, entry in enumerate(translation_entries[:3]):
                bbox = entry.bbox
                logger.info(
                    f"   Entry {i + 1}: text at ({bbox.left:.1f}, {bbox.top:.1f}) -> ({bbox.right:.1f}, {bbox.bottom:.1f})"
                )

            logger.info(
                "\n💡 Tip: Run the enhanced test 'translation_service_with_overlay_test.py' to see actual image generation!"
            )

        assert len(translation_entries) > 0, "Should have translated at least one text segment"
