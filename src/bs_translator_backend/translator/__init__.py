from .base_translator import BaseTranslator
from .config import LLMConfig, TranslationConfig
from .docx_translator import DocxTranslator
from .pdf_translator import PdfTranslator
from .text_translator import TextTranslator

__all__ = ["BaseTranslator", "DocxTranslator", "LLMConfig", "PdfTranslator", "TextTranslator", "TranslationConfig"]

# Version of the translator package
__version__ = "1.0.0"
