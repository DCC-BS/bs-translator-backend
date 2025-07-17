from bs_translator_backend.translator.config import TranslationConfig
from bs_translator_backend.translator.base_translator import BaseTranslator


class TextTranslator(BaseTranslator):
    """Translator for plain text"""

    def translate(
        self, input_path: str, output_path: str, config: TranslationConfig
    ) -> None:
        """
        This method is required by the abstract base class but not used for text translation.
        Text translation is handled by translate_text method instead.
        """
        raise NotImplementedError("Use translate_text method for text translation")
