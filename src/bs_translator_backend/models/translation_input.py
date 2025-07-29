"""
Translation Input Models

This module defines the input models for translation requests,
combining the text to be translated with configuration parameters.
"""

from pydantic import BaseModel, Field

from bs_translator_backend.models.translation_config import TranslationConfig


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
