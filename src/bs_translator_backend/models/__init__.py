"""Pydantic models for the bs-translator-backend application."""

from .conversion_result import ConversionOutput, ConversionResult
from .docling_response import (
    ConversionStatus,
    ConvertDocumentResponse,
    DoclingDocument,
    DoclingResponse,
    DocumentResponse,
    ErrorItem,
)
from .language import DetectLanguage, Language, LanguageOrAuto
from .translation import (
    TranslationConfig,
    TranslationInput,
)

__all__ = [
    "ConversionOutput",
    "ConversionResult",
    "ConversionStatus",
    "ConvertDocumentResponse",
    "DetectLanguage",
    "DoclingDocument",
    "DoclingResponse",
    "DocumentResponse",
    "ErrorItem",
    "Language",
    "LanguageOrAuto",
    "TranslationConfig",
    "TranslationInput",
]
