"""Pydantic models for the bs-translator-backend application."""

from .app_config import AppConfig
from .conversion_result import ConversionOutput, ConversionResult
from .docling_response import (
    ConversionStatus,
    ConvertDocumentResponse,
    DoclingDocument,
    DoclingResponse,
    DocumentResponse,
    ErrorItem,
)
from .langugage import DetectLanguage, Language, LanguageOrAuto
from .translation_config import TranslationConfig
from .translation_input import TranslationInput

__all__ = [
    "AppConfig",
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
