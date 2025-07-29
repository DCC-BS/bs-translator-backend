"""
Translation Configuration Models

This module defines the configuration classes for translation parameters,
allowing users to customize translation behavior including target language,
tone, domain specialization, custom glossaries, and additional context.
"""

from pydantic import BaseModel, Field

from bs_translator_backend.models.langugage import Language, LanguageOrAuto


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

    target_language: Language = Field(default=Language.DE, description="Target language for translation")
    source_language: LanguageOrAuto | None = Field(default=None, description="Source language (auto-detected if None)")
    domain: str | None = Field(default=None, description="Domain or subject area for translation")
    tone: str | None = Field(default=None, description="Tone or style for translation")
    glossary: str | None = Field(default=None, description="Custom glossary or terminology")
    context: str | None = Field(default=None, description="Additional context for translation")

    def get_tone_prompt(self) -> str:
        """Generates the tone-specific part of the prompt"""
        if self.tone is None:
            return "Use a neutral tone that is objective, informative, and unbiased."

        tone_prompts = {
            "formal": "Use a formal and professional tone appropriate for official documents.",
            "informal": "Use an informal and conversational tone that is friendly and engaging.",
            "technical": f"Use a technical tone appropriate for {self.domain if self.domain else 'professional'} writing.",
        }
        return tone_prompts.get(self.tone.lower(), "Use a neutral tone.")

    def get_domain_prompt(self) -> str:
        """Generates the domain-specific part of the prompt"""
        return (
            f"Use terminology specific to the {self.domain} field."
            if self.domain
            else "No specific domain requirements."
        )

    def get_glossary_prompt(self) -> str:
        """Generates the glossary-specific part of the prompt"""
        if self.glossary is None:
            return "No specific glossary provided."

        glossary = "\n".join(self.glossary.replace(":", ": ").split(";"))
        return f"Use the following glossary to ensure accurate translations:\n{glossary}"
