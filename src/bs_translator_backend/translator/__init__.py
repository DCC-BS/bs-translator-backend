from .base_translator import BaseTranslator
from .text_translator import TextTranslator
from .docx_translator import DocxTranslator
from .pdf_translator import PdfTranslator
from .config import LLMConfig, TranslationConfig

__all__ = [
    'BaseTranslator',
    'TextTranslator',
    'DocxTranslator',
    'LLMConfig',
    'TranslationConfig',
    'PdfTranslator'
]

# Version of the translator package
__version__ = '1.0.0'