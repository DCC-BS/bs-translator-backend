from pydantic import BaseModel, Field

from bs_translator_backend.models.language import Language, LanguageOrAuto


class TranslationConfig(BaseModel):
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
    text: str = Field(..., description="Text to be translated")
    config: TranslationConfig = Field(
        default=TranslationConfig(), description="Optional translation configuration parameters"
    )
