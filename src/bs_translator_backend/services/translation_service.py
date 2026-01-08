"""
Translation Service Module

This module provides the core translation functionality for the BS Translator Backend.
It handles text translation using LLM models with customizable parameters including
tone, domain, glossary, and context settings.
"""

from collections.abc import Callable
from io import BytesIO
from typing import final

from beartype.typing import AsyncGenerator
from fastapi import UploadFile

from bs_translator_backend.agents.translation_agent import create_translation_agent
from bs_translator_backend.models.conversion_result import BBox, ConversionImageTextEntry
from bs_translator_backend.models.language import DetectLanguage, Language, get_language_name
from bs_translator_backend.models.translation import TranslationConfig
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.utils.app_config import AppConfig


@final
class TranslationService:
    """
    Service for handling text translation with AI models.

    This service provides high-quality text translation capabilities using
    large language models with support for:
    - Automatic language detection
    - Customizable translation parameters (tone, domain, glossary, context)
    - Text chunking for large documents
    - Streaming translation responses via SSE
    """

    def __init__(
        self,
        app_config: AppConfig,
        text_chunk_service: TextChunkService,
        conversion_service_factory: Callable[[], DocumentConversionService],
    ) -> None:
        self.app_config = app_config
        self.text_chunk_service = text_chunk_service
        self._conversion_service_factory = conversion_service_factory
        self.translation_agent = create_translation_agent(app_config)

    def _create_user_message(
        self, text: str, translation_config: TranslationConfig, reasoning: bool = False
    ) -> str:
        """Create the prompt message for the translation agent."""
        target_language_name: str = get_language_name(translation_config.target_language)
        prompt = f"""Translate the following text into {target_language_name}.
Domain: {translation_config.domain}
Tone: {translation_config.tone}
Glossary: {translation_config.glossary}
Context:
{translation_config.context}

Text to translate:
{text}
"""
        if not reasoning:
            prompt += "/no_think"
        return prompt

    async def translate_text(
        self, text: str, config: TranslationConfig
    ) -> AsyncGenerator[str, None]:
        """
        Translate text with streaming response (plain text chunks).

        Handles chunking for large texts and streams translation results.

        Args:
            text: The text to translate
            config: Translation configuration parameters

        Yields:
            Translated text chunks as plain strings
        """
        if not text.strip() or len(text.strip()) == 1:
            yield text
            return

        text_chunks = self.text_chunk_service.chunk_text(text)
        accumulated_context = ""

        for text_chunk in text_chunks:
            # Update context with previous translations for consistency
            chunk_config = TranslationConfig(
                target_language=config.target_language,
                source_language=config.source_language,
                domain=config.domain,
                tone=config.tone,
                glossary=config.glossary,
                context=f"{config.context}\n\nPrevious translations:\n{accumulated_context}"
                if accumulated_context
                else config.context,
            )

            user_message = self._create_user_message(text_chunk, chunk_config)
            chunk_translation = ""

            async with self.translation_agent.run_stream(user_message) as stream:
                async for text_part in stream.stream_text(delta=True):
                    chunk_translation += text_part
                    yield text_part

            # Accumulate context for next chunk (keep last ~500 chars for context)
            accumulated_context = (accumulated_context + chunk_translation)[-500:]

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
