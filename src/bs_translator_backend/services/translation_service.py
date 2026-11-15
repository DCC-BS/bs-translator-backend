"""
Translation Service Module

This module provides the core translation functionality for the BS Translator Backend.
It handles text translation using LLM models with customizable parameters including
tone, domain, glossary, and context settings.
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from functools import partial
from typing import final

from dcc_backend_common.logger import get_logger
from fastapi import status

from bs_translator_backend.agents.errors import (
    TranslationRejectedError,
    TranslationUnavailableError,
)
from bs_translator_backend.agents.translation_agent import (
    ShortTextTranslationAgent,
    TranslationAgent,
)
from bs_translator_backend.models.error_codes import (
    INPUT_TOO_LONG,
    TRANSLATION_SERVICE_UNAVAILABLE,
)
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import (
    DetectLanguage,
    Language,
    LanguageOrAuto,
    get_language_name,
)
from bs_translator_backend.models.translation import (
    DetectLanguageInput,
    DetectLanguageOutput,
    TranslationConfig,
)
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.language_detection import detect_language, sample_windows

logger = get_logger(__name__)

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

# Minimum confidence before an auto-detected language of a longer text is trusted:
# named as the source language in the prompt, or used to return the text untranslated
# because it already is in the target language.
SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD = 0.8

# Below this confidence /translation/detect-language answers "auto", so the client never
# sends a low-confidence guess back as an explicit (and therefore trusted) source.
DETECT_LANGUAGE_CONFIDENCE_THRESHOLD = 0.5

# Scripts written without spaces between words, where counting whitespace-separated
# words says nothing: a whole Chinese or Thai sentence is a single "word". Inputs in
# these scripts are short only when they have at most this many characters.
_CJK_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SOUTHEAST_ASIAN_CHARS = re.compile(r"[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff]")
SHORT_TEXT_CJK_CHAR_THRESHOLD = 4
SHORT_TEXT_SOUTHEAST_ASIAN_CHAR_THRESHOLD = 10

# A single character of these scripts can be a whole word (水, 물), so it is translated.
_SINGLE_CHAR_WORD = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)

# Detection cannot tell these regional variants apart (fastText's "zh" covers Simplified
# and Traditional script, "en" both British and American English), so a detected match
# never proves a text is already in one of them.
_UNDETECTABLE_VARIANTS = frozenset({
    Language.EN_GB,
    Language.EN_US,
    Language.ZH_CN,
    Language.ZH_HK,
    Language.ZH_TW,
})
_ENGLISH = frozenset({Language.EN, Language.EN_GB, Language.EN_US})


def _is_short_text(text: str) -> bool:
    """Whether text is a short, single-line phrase (1-3 words) suited to a lexical lookup."""
    stripped = text.strip()
    if "\n" in stripped:
        return False
    word_count = len(stripped.split())
    if not 1 <= word_count <= SHORT_TEXT_WORD_THRESHOLD:
        return False
    char_count = len("".join(stripped.split()))
    if _CJK_CHARS.search(stripped):
        return char_count <= SHORT_TEXT_CJK_CHAR_THRESHOLD
    if _SOUTHEAST_ASIAN_CHARS.search(stripped):
        return char_count <= SHORT_TEXT_SOUTHEAST_ASIAN_CHAR_THRESHOLD
    return True


def _has_nothing_to_translate(text: str) -> bool:
    """Blank text, pure punctuation/digits/symbols, or a lone letter such as "H"."""
    stripped = text.strip()
    if not any(char.isalpha() for char in stripped):
        return True
    return len(stripped) == 1 and not _SINGLE_CHAR_WORD.match(stripped)


def _detected_language_is_target(detected: Language, target: Language) -> bool:
    if target in _UNDETECTABLE_VARIANTS:
        return False
    if target == Language.EN:
        return detected in _ENGLISH
    return detected == target


def _rest_is_in_language(text: str, language: Language) -> bool:
    """Whether every sample after the first (already checked) is `language` too.

    Catches mixed-language documents whose beginning happens to be in the target
    language.
    """
    for window in sample_windows(text)[1:]:
        detection = detect_language(window)
        if not detection.map(
            lambda result: (
                isinstance(result.language, Language)
                and _detected_language_is_target(result.language, language)
                and result.confidence >= SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD
            )
        ).value_or(False):
            return False
    return True


def _trailing_whitespace(text: str) -> str:
    return text[len(text.rstrip()) :]


def _missing_trailing_whitespace(source_trailing: str, translation: str) -> str:
    """The part of the source chunk's trailing whitespace the translation did not end with.

    Re-emitted after the chunk's translation so adjacent chunks never fuse. When the
    translation ends with different whitespace (e.g. a space where the source had a
    blank line), the missing line breaks are still added.
    """
    translated_trailing = _trailing_whitespace(translation)
    if source_trailing.startswith(translated_trailing):
        return source_trailing[len(translated_trailing) :]
    missing_newlines = source_trailing.count("\n") - translated_trailing.count("\n")
    return "\n" * max(0, missing_newlines)


async def _stream_llm(
    agent: TranslationAgent | ShortTextTranslationAgent,
    user_message: str,
    config: TranslationConfig,
) -> AsyncGenerator[str, None]:
    """Stream one LLM translation, turning LLM failures into API errors."""
    try:
        async for text_part in agent.run_stream_text_safe(user_prompt=user_message, delta=True):
            yield text_part
    except (TranslationRejectedError, TranslationUnavailableError) as e:
        logger.exception(
            "Translation LLM call failed",
            target_language=config.target_language.value,
            short_text_agent=isinstance(agent, ShortTextTranslationAgent),
        )
        raise _to_api_error(e) from e


class _HeldWhitespace:
    """Whitespace held back until the LLM produced its first output.

    A failure of the first LLM call then still happens before anything was streamed,
    so the route can answer with a proper error status instead of a committed 200.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._released = False

    def add(self, whitespace: str) -> str:
        """Returns what may be streamed now (nothing until the first release)."""
        if self._released:
            return whitespace
        self._pending += whitespace
        return ""

    def release_with(self, text: str) -> str:
        """Returns `text` preceded by any held whitespace; nothing is held afterwards."""
        self._released = True
        released, self._pending = self._pending + text, ""
        return released


def _to_api_error(
    error: TranslationRejectedError | TranslationUnavailableError,
) -> ApiErrorException:
    if isinstance(error, TranslationUnavailableError):
        return ApiErrorException({
            "errorId": TRANSLATION_SERVICE_UNAVAILABLE,
            "status": status.HTTP_503_SERVICE_UNAVAILABLE,
            "debugMessage": f"Translation service is currently unavailable: {error!s}",
        })
    if error.input_too_long:
        return ApiErrorException({
            "errorId": INPUT_TOO_LONG,
            "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
            "debugMessage": str(error),
        })
    # Not transient (e.g. a misconfigured model name or API key), so not a 503.
    return ApiErrorException({
        "errorId": TRANSLATION_SERVICE_UNAVAILABLE,
        "status": status.HTTP_502_BAD_GATEWAY,
        "debugMessage": str(error),
    })


def _split_surrounding_whitespace(chunk: str) -> tuple[str, str, str]:
    """Split a chunk into leading blank lines, the text to translate, trailing whitespace.

    Only the middle part goes to the LLM; the rest is re-emitted verbatim, so line and
    paragraph breaks between chunks survive the translation. Plain string scans, no
    regex: the input is user text and may hold long whitespace runs.
    """
    start = 0
    for line in chunk.splitlines(keepends=True):
        if line.strip() or not line.endswith(("\n", "\r")):
            break
        start += len(line)
    end = max(start, len(chunk.rstrip()))
    return chunk[:start], chunk[start:end], chunk[end:]


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
    ) -> None:
        self.app_config = app_config
        self.text_chunk_service = text_chunk_service
        self.translation_agent = TranslationAgent(app_config)
        self.short_text_translation_agent = ShortTextTranslationAgent(app_config)

    @staticmethod
    def _instruction(translation_config: TranslationConfig, assert_source_language: bool) -> str:
        target_language_name: str = get_language_name(translation_config.target_language)
        if (
            assert_source_language
            and translation_config.source_language
            and translation_config.source_language != DetectLanguage.AUTO
        ):
            source_language_name: str = get_language_name(translation_config.source_language)
            return (
                f"Translate the following text from {source_language_name} "
                f"into {target_language_name}."
            )
        return f"Translate the following text into {target_language_name}."

    def _create_user_message(
        self,
        text: str,
        translation_config: TranslationConfig,
        assert_source_language: bool = True,
    ) -> str:
        """Create the prompt message for the translation agent.

        The source language is named only when it is trustworthy (see
        `_create_short_text_user_message`).
        """
        instruction = self._instruction(translation_config, assert_source_language)
        return f"""{instruction}
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
        lines = [self._instruction(translation_config, assert_source_language)]
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

    @staticmethod
    async def _resolve_source_language(
        text: str, config: TranslationConfig, use_short_text_agent: bool
    ) -> tuple[LanguageOrAuto | None, bool, bool]:
        """Returns the source language, whether it is trustworthy, and whether `text` is
        already in the target language (so it is returned untranslated).

        Resolved locally rather than written back onto `config`: the caller owns that
        object, and image translation reuses one config across every text segment -
        mutating it would leak the first segment's detected language into all the others.
        """
        source_language = config.source_language
        if source_language and source_language != DetectLanguage.AUTO:
            # The caller (i.e. the user, via the UI) explicitly chose this language.
            return source_language, True, source_language == config.target_language

        # fastText is quick but synchronous; keep it off the event loop.
        detection_result = await asyncio.to_thread(detect_language, text)
        detected_confidence = detection_result.map(lambda result: result.confidence).value_or(0.0)
        detected = detection_result.map(lambda result: result.language).value_or(None)
        threshold = (
            SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD
            if use_short_text_agent
            else SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD
        )
        if not isinstance(detected, Language) or detected_confidence < threshold:
            return source_language, False, False

        already_in_target_language = _detected_language_is_target(
            detected, config.target_language
        ) and await asyncio.to_thread(_rest_is_in_language, text, config.target_language)
        return detected, True, already_in_target_language

    async def translate_text(
        self, text: str, config: TranslationConfig
    ) -> AsyncGenerator[str, None]:
        """Translate text and stream the result."""
        if _has_nothing_to_translate(text):
            yield text
            return

        use_short_text_agent = _is_short_text(text)

        (
            source_language,
            source_language_trustworthy,
            already_in_target_language,
        ) = await self._resolve_source_language(text, config, use_short_text_agent)
        if already_in_target_language:
            yield text
            return

        translation_agent = (
            self.short_text_translation_agent if use_short_text_agent else self.translation_agent
        )
        create_user_message = partial(
            self._create_short_text_user_message
            if use_short_text_agent
            else self._create_user_message,
            assert_source_language=source_language_trustworthy,
        )

        text_chunks = self.text_chunk_service.chunk_text(text)
        accumulated_context = ""
        held = _HeldWhitespace()

        for text_chunk in text_chunks:
            leading, chunk_body, trailing = _split_surrounding_whitespace(text_chunk)
            if not chunk_body:
                if released := held.add(leading + trailing):
                    yield released
                continue
            if released := held.add(leading):
                yield released

            chunk_config = TranslationConfig(
                target_language=config.target_language,
                source_language=source_language,
                domain=config.domain,
                tone=config.tone,
                glossary=config.glossary,
                context=f"{config.context}\n\nPrevious translations:\n{accumulated_context}"
                if accumulated_context
                else config.context,
            )

            user_message = create_user_message(
                text=chunk_body,
                translation_config=chunk_config,
            )
            chunk_translation = ""
            async for text_part in _stream_llm(translation_agent, user_message, config):
                chunk_translation += text_part
                yield held.release_with(text_part)

            if released := held.add(_missing_trailing_whitespace(trailing, chunk_translation)):
                yield released

            # Accumulate context for next chunk (keep last ~500 chars for context)
            accumulated_context = (accumulated_context + chunk_translation)[-500:]

        if rest := held.release_with(""):
            yield rest

    async def aclose(self) -> None:
        """Close both agents' HTTP clients. Called from the FastAPI lifespan."""
        await self.translation_agent.close()
        await self.short_text_translation_agent.close()

    async def detect_language(
        self, detect_language_input: DetectLanguageInput
    ) -> DetectLanguageOutput:
        """Detect the language of the text"""
        if not detect_language_input.text.strip() or len(detect_language_input.text.strip()) < 10:
            return DetectLanguageOutput(language=DetectLanguage.AUTO, confidence=0.0)

        detected = (await asyncio.to_thread(detect_language, detect_language_input.text)).value_or(
            DetectLanguageOutput(language=DetectLanguage.AUTO, confidence=0.0)
        )
        if detected.confidence < DETECT_LANGUAGE_CONFIDENCE_THRESHOLD:
            return DetectLanguageOutput(
                language=DetectLanguage.AUTO, confidence=detected.confidence
            )
        return detected

    def get_supported_languages(self) -> list[str]:
        """Returns a list of supported languages for translation"""
        return [lang.value for lang in Language]
