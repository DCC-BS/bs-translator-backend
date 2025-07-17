from pydantic import BaseModel, Field

from bs_translator_backend.models.translation_config import TranslationConfig


class TranslationInput(BaseModel):
    text: str = Field(..., description="Text to be translated")
    config: TranslationConfig = Field(
        default=TranslationConfig(), description="Optional translation configuration parameters"
    )
