"""
Translation Service Module

This module provides the core translation functionality for the BS Translator Backend.
It handles text translation using LLM models with customizable parameters including
tone, domain, glossary, and context settings.
"""

from collections.abc import AsyncGenerator, Callable
from functools import partial
from io import BytesIO
from typing import final

from fastapi import UploadFile

from bs_translator_backend.agents.translation_agent import (
    ShortTextTranslationAgent,
    TranslationAgent,
)
from bs_translator_backend.models.conversion_result import BBox, ConversionImageTextEntry
from bs_translator_backend.models.language import DetectLanguage, Language, get_language_name
from bs_translator_backend.models.translation import (
    DetectLanguageInput,
    DetectLanguageOutput,
    TranslationConfig,
)
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.language_detection import detect_language

# Inputs at or below this many words are treated as dictionary/lexical lookups
# (e.g. "Hirsch", "Der Hirsch") and routed to the short-text translation agent,
# whose instructions counter the model's tendency to copy short, capitalized
# German nouns verbatim under a mistaken "proper noun" assumption.
SHORT_TEXT_WORD_THRESHOLD = 3

# Minimum fast-langdetect confidence required before the short-text prompt is
# allowed to assert the detected source language. Measured on this branch:
#   'Hirsch'      -> en-us  conf=0.36   <- German word, detected as English
#   'Auto'        -> en-us  conf=0.38   <- German word, detected as English
#   'Hund'        -> de     conf=0.67
#   'Wasser'      -> de     conf=0.91
#   'Der Hirsch'  -> de     conf=0.96
#   'jelen'       -> hu     conf=1.00   <- confidently wrong (Hungarian for "deer")
# 0.9 accepts the two high-confidence correct detections above and rejects the
# two confidently-wrong low-confidence ones. No threshold catches 'jelen' -
# the goal is only to stop confidently-wrong *low*-confidence claims, not to
# guarantee correctness.
SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD = 0.9


def _is_short_text(text: str) -> bool:
    """Whether text is a short, single-line phrase (1-3 words) suited to a lexical lookup."""
    stripped = text.strip()
    if "\n" in stripped:
        return False
    word_count = len(stripped.split())
    return 1 <= word_count <= SHORT_TEXT_WORD_THRESHOLD


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
        self.translation_agent = TranslationAgent(app_config)
        self.short_text_translation_agent = ShortTextTranslationAgent(app_config)

    def _create_user_message(self, text: str, translation_config: TranslationConfig) -> str:
        """Create the prompt message for the translation agent."""
        target_language_name: str = get_language_name(translation_config.target_language)
        return f"""Translate the following text into {target_language_name}.
Domain: {translation_config.domain}
Tone: {translation_config.tone}
Glossary: {translation_config.glossary}
Context:
{translation_config.context}

Text to translate:
{text}
"""

    def _create_short_text_user_message(
        self,
        text: str,
        translation_config: TranslationConfig,
        assert_source_language: bool = True,
    ) -> str:
        """Create the prompt message for the short-text (lexical lookup) translation agent.

        The "from {source}" clause is only asserted when `assert_source_language` is
        true, i.e. when the source language is trustworthy: explicitly chosen by the
        caller, or auto-detected with confidence at or above
        SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD. Asserting a wrong source
        language (e.g. claiming a German noun is English) makes the model more
        likely to copy it verbatim instead of translating it - exactly the bug this
        agent exists to fix. When the source language is not trustworthy, the
        instruction falls back to omitting it, letting the model infer it itself.
        """
        target_language_name: str = get_language_name(translation_config.target_language)
        if (
            assert_source_language
            and translation_config.source_language
            and translation_config.source_language != DetectLanguage.AUTO
        ):
            source_language_name: str = get_language_name(translation_config.source_language)
            instruction = (
                f"Translate the following text from {source_language_name} "
                f"into {target_language_name}."
            )
        else:
            instruction = f"Translate the following text into {target_language_name}."

        lines = [instruction]
        if translation_config.domain:
            lines.append(f"Domain: {translation_config.domain}")
        if translation_config.tone:
            lines.append(f"Tone: {translation_config.tone}")
        if translation_config.glossary:
            lines.append(f"Glossary: {translation_config.glossary}")
        if translation_config.context:
            lines.append(f"Context:\n{translation_config.context}")

        metadata_block = "\n".join(lines)
        return f"""{metadata_block}

Text to translate:
{text}
"""

    async def translate_text(
        self, text: str, config: TranslationConfig
    ) -> AsyncGenerator[str, None]:
        """Translate text and stream the result."""
        if not text.strip() or len(text.strip()) == 1:
            yield text
            return

        use_short_text_agent = _is_short_text(text)

        if not config.source_language or config.source_language == DetectLanguage.AUTO:
            detection_result = detect_language(text)
            detected_confidence = detection_result.map(lambda result: result.confidence).value_or(
                0.0
            )
            detected = detection_result.map(lambda result: result.language).value_or(None)

            if (
                detected is not None
                and not isinstance(detected, DetectLanguage)
                and (
                    detected_confidence >= SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD
                    if use_short_text_agent
                    else True
                )
            ):
                config.source_language = detected
                source_language_trustworthy = True
            else:
                source_language_trustworthy = False
        else:
            # The caller (i.e. the user, via the UI) explicitly chose this language.
            source_language_trustworthy = True

        if source_language_trustworthy and config.source_language == config.target_language:
            yield text
            return

        translation_agent = (
            self.short_text_translation_agent if use_short_text_agent else self.translation_agent
        )
        create_user_message = (
            partial(
                self._create_short_text_user_message,
                assert_source_language=source_language_trustworthy,
            )
            if use_short_text_agent
            else self._create_user_message
        )

        text_chunks = self.text_chunk_service.chunk_text(text)
        accumulated_context = ""

        for text_chunk in text_chunks:
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

            user_message = create_user_message(
                text=text_chunk,
                translation_config=chunk_config,
            )
            chunk_translation = ""
            async for text_part in translation_agent.run_stream_text(
                user_prompt=user_message, delta=True
            ):
                chunk_translation += text_part
                yield text_part

            # Accumulate context for next chunk (keep last ~500 chars for context)
            accumulated_context = (accumulated_context + chunk_translation)[-500:]

    async def aclose(self) -> None:
        """Close both agents' HTTP clients. Called from the FastAPI lifespan."""
        await self.translation_agent.close()
        await self.short_text_translation_agent.close()

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

    async def detect_language(
        self, detect_language_input: DetectLanguageInput
    ) -> DetectLanguageOutput:
        """Detect the language of the text"""
        if not detect_language_input.text.strip() or len(detect_language_input.text.strip()) < 10:
            return DetectLanguageOutput(language=DetectLanguage.AUTO, confidence=0.0)

        return detect_language(detect_language_input.text).value_or(
            DetectLanguageOutput(language=DetectLanguage.AUTO, confidence=0.0)
        )

    def get_supported_languages(self) -> list[str]:
        """Returns a list of supported languages for translation"""
        return [lang.value for lang in Language]
