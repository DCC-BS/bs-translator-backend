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
    domain: str = Field(default="General", description="Domain or subject area for translation")
    tone: str = Field(
        default="Keep the tone of the source text", description="Tone or style for translation"
    )
    glossary: str = Field(default="No glossary", description="Custom glossary or terminology")
    context: str = Field(default="No context", description="Additional context for translation")


class TranslationInput(BaseModel):
    """
    Input model for translation requests.

    This model combines the text to be translated with optional
    configuration parameters to customize the translation behavior.

    Attributes:
        text: The text content to be translated
        config: Translation configuration parameters (optional)
    """

    text: str = Field(..., description="Text to be translated")
    config: TranslationConfig = Field(
        default=TranslationConfig(), description="Optional translation configuration parameters"
    )
