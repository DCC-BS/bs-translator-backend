"""
Translation Service Module

This module provides the core translation functionality for the BS Translator Backend.
It handles text translation using LLM models with customizable parameters including
tone, domain, glossary, and context settings.
"""

from collections.abc import AsyncGenerator, Generator
from io import BytesIO
from typing import final

from fastapi import UploadFile
from llama_index.core import PromptTemplate

from bs_translator_backend.models.conversion_result import (
    BBox,
    ConversionImageTextEntry,
)
from bs_translator_backend.models.langugage import DetectLanguage, Language
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.llm_facade import LLMFacade
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.utils.language_detection import detect_language

SYSTEM_MESSAGE = """You are an expert translator.

Requirements:
    1. Accuracy: The translation should be accurate and convey the same meaning as the original text.
    2. Fluency: The translated text should be natural and fluent in the target language.
    3. Style: Maintain the original style and tone of the text as much as possible.
    4. Context: Consider the context provided when translating.
    5. No Unnecessary Translations: Do not translate proper nouns like names (e.g., "Yanick Schraner"), brands (e.g., "Apple"), places (e.g., "Basel-Stadt"), addresses, URLs, email addresses, phone numbers, or any element that would lose its meaning or functionality if translated. These should remain in their original form.
    6. Idioms and Cultural References: Adapt idiomatic expressions and culturally specific references to their equivalents in the target language to maintain meaning and readability.
    7. Source Text Errors: If there are any obvious errors or typos in the source text, correct them in the translation to improve clarity.
    8. Formatting: Preserve the original markdown formatting of the text, including line breaks, bullet points, and any emphasis like bold or italics.
    9. Special characters: Use '\n' for line breaks. Preserve line breaks and paragraphs as in the source text. Keep carriage return characters ('\r') if they are used in the source text.
    10. Output Requirements: Provide only the translated text without explanations, notes, comments, or any additional text.
"""


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
        llm_facade: LLMFacade,
        text_chunk_service: TextChunkService,
        conversion_service: DocumentConversionService,
    ):
        self.llm_facade = llm_facade
        self.text_chunk_service = text_chunk_service
        self.conversion_service = conversion_service

    def _create_user_message(self, text: str, config: TranslationConfig) -> str:
        """Creates the user message for the chat API"""
        tone_prompt = config.get_tone_prompt()
        domain_prompt = config.get_domain_prompt()
        glossary_prompt = config.get_glossary_prompt()

        context_section = f"Context: {config.context}\n\n" if config.context else ""

        return PromptTemplate(
            """Translate the following text from {source_language} to {target_language}.

            {context_section}Domain-Specific Terminology: {domain_prompt}
            Tone: {tone_prompt}
            Glossary: {glossary_prompt}

            Text to translate:
            {text}"""
        ).format(
            source_language=config.source_language or "auto-detected",
            target_language=config.target_language,
            context_section=context_section,
            domain_prompt=domain_prompt,
            tone_prompt=tone_prompt,
            glossary_prompt=glossary_prompt,
            text=text,
        )

    def translate_text(self, text: str, config: TranslationConfig) -> Generator[str, None, None]:
        """Base translation method"""
        if not text.strip() or len(text.strip()) == 1:
            yield text

        if not config.source_language or config.source_language == DetectLanguage.AUTO:
            config.source_language = detect_language(text).value_or(Language.DE)

        endswith_r = text.endswith("\r")

        text_chunks = self.text_chunk_service.chunk_text(text)

        for text_chunk in text_chunks:
            translated_chunks = self.llm_facade.stream_complete(
                PromptTemplate("{system_message} {user_message}").format(
                    system_message=SYSTEM_MESSAGE,
                    user_message=self._create_user_message(text_chunk, config),
                ),
            )

            at_the_beginning = True

            r = ""

            for text_chunk in translated_chunks:
                if at_the_beginning and text_chunk.strip() == "":
                    continue

                at_the_beginning = False

                # Replace '\n' with '\r\n' to preserve line breaks
                text_chunk = text_chunk.replace("\n", "\r\n").replace("ß", "ss")

                r += text_chunk

                yield text_chunk

        # If the last chunk ends with a newline, preserve it
        if endswith_r:
            yield "\r"

    async def translate_image(
        self, image: UploadFile | BytesIO, config: TranslationConfig, filename: str | None = None, content_type: str | None = None
    ) -> AsyncGenerator[ConversionImageTextEntry, None]:
        doc = await self.conversion_service.convert_to_docling(
            image, config.source_language or DetectLanguage.AUTO, filename, content_type
        )

        for txt in doc.texts:
            content = txt.text or ""
            if not txt.prov:
                continue
            bbox = txt.prov[0].bbox

            translated = ""
            for chunk in self.translate_text(content, config):
                translated += chunk

            yield ConversionImageTextEntry(
                original=content, translated=translated, bbox=BBox(**bbox.model_dump())
            )

    def get_supported_languages(self) -> list[str]:
        """Returns a list of supported languages for translation"""
        return [lang.value for lang in Language]
