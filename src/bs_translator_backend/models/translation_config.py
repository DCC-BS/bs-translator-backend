"""
Translation Configuration Models

This module defines the configuration classes for translation parameters,
allowing users to customize translation behavior including target language,
tone, domain specialization, custom glossaries, and additional context.
"""

from pydantic import BaseModel, Field

from bs_translator_backend.models.language import Language, LanguageOrAuto


class TranslationConfig(BaseModel):
    """
    Configuration class for translation parameters.

    This class allows fine-tuning of translation behavior through various
    parameters that control the style, domain specialization, and context
    of the translation process.

    Attributes:
        target_language: Target language for translation (default: German)
        source_language: Source language (auto-detected if None)
        domain: Domain or subject area for specialized terminology
        tone: Translation tone (formal, informal, technical, neutral)
        glossary: Custom glossary or terminology for consistent translations
        context: Additional context to guide translation decisions
    """

    target_language: Language = Field(
        default=Language.DE, description="Target language for translation"
    )
    source_language: LanguageOrAuto | None = Field(
        default=None, description="Source language (auto-detected if None)"
    )
    domain: str | None = Field(default=None, description="Domain or subject area for translation")
    tone: str | None = Field(default=None, description="Tone or style for translation")
    glossary: str | None = Field(default=None, description="Custom glossary or terminology")
    context: str | None = Field(default=None, description="Additional context for translation")
    reasoning: bool = Field(
        default=False, description="Enable LLM reasoning; when false, disable with /no_think hint"
    )
