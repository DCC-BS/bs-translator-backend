"""
Translation Service Module

This module provides the core translation functionality for the BS Translator Backend.
It handles text translation using LLM models with customizable parameters including
tone, domain, glossary, and context settings.
"""

from collections.abc import AsyncGenerator, Callable
from io import BytesIO
from typing import final

from fastapi import UploadFile

from bs_translator_backend.models.conversion_result import (
    BBox,
    ConversionImageTextEntry,
)
from bs_translator_backend.models.language import DetectLanguage, Language, get_language_name
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.dspy_config.translation_program import TranslationModule
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.utils.language_detection import detect_language


@final
class TranslationService:
    """
    Service for handling text translation with AI models.

    This service provides high-quality text translation capabilities using
    large language models with support for:
    - Automatic language detection
    - Customizable translation parameters (tone, domain, glossary, context)
    - Text chunking for large documents
    - Streaming translation responses
    """

    def __init__(
        self,
        translation_module: TranslationModule,
        text_chunk_service: TextChunkService,
        conversion_service_factory: Callable[[], DocumentConversionService],
    ) -> None:
        self.translation_module = translation_module
        self.text_chunk_service = text_chunk_service
        self._conversion_service_factory = conversion_service_factory

    async def translate_text(
        self, text: str, config: TranslationConfig
    ) -> AsyncGenerator[str, None]:
        """Base translation method"""
        if not text.strip() or len(text.strip()) == 1:
            yield text
            return

        if not config.source_language or config.source_language == DetectLanguage.AUTO:
            config.source_language = detect_language(text).value_or(Language.DE)

        text_chunks = self.text_chunk_service.chunk_text(text)
        context = ""
        for text_chunk in text_chunks:
            translated_chunks = self.translation_module.stream(
                source_language=get_language_name(config.source_language),
                target_language=get_language_name(config.target_language),
                context=context,
                source_text=text_chunk,
                domain=config.domain or "",
                tone=config.tone or "",
                glossary=config.glossary or "",
                reasoning=config.reasoning,
            )
            chunk = ""
            async for chunk in translated_chunks:
                context += chunk
                yield chunk

    async def translate_image(
        self,
        image: UploadFile | BytesIO,
        config: TranslationConfig,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> AsyncGenerator[ConversionImageTextEntry, None]:
        """Translate text extracted from an image or document upload."""

        async with self._conversion_service_factory() as conversion_service:
            doc = await conversion_service.convert_to_docling(
                image, config.source_language or DetectLanguage.AUTO, filename, content_type
            )

        for txt in doc.texts:
            content = txt.text or ""
            if not txt.prov:
                continue
            bbox = txt.prov[0].bbox

            translated = ""
            async for chunk in self.translate_text(content, config):
                translated += chunk

            yield ConversionImageTextEntry(
                original=content, translated=translated, bbox=BBox(**bbox.model_dump())
            )

    def get_supported_languages(self) -> list[str]:
        """Returns a list of supported languages for translation"""
        return [lang.value for lang in Language]
