from llama_index.core import PromptTemplate

from bs_translator_backend.models.langugage import Language
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.services.llm_facade import LLMFacade
from bs_translator_backend.utils.language_detection import detect_language

SYSTEM_MESSAGE = """You are an expert translator.

Requirements:
    1. Accuracy: The translation should be accurate and convey the same meaning as the original text.
    2. Fluency: The translated text should be natural and fluent in the target language.
    3. Style: Maintain the original style and tone of the text as much as possible.
    4. Context: Consider the context provided when translating.
    5. No Unnecessary Translations: Do not translate proper nouns like names (e.g., "Yanick Schraner"), brands (e.g., "Apple"), places (e.g., "Basel-Stadt"), addresses, URLs, email addresses, phone numbers, or any element that would lose its meaning or functionality if translated. These should remain in their original form.
    6. Idioms and Cultural References: Adapt idiomatic expressions and culturally specific references to their equivalents in the target language to maintain meaning and readability.
    7. Source Text Errors: If there are any obvious errors or typos in the source text, correct them in the translation to improve clarity.
    8. Formatting: Preserve the original markdown formatting of the text, including line breaks, bullet points, and any emphasis like bold or italics.
    9. Special characters: Use '\n' for line breaks. Preserve line breaks and paragraphs as in the source text. Keep carriage return characters ('\r') if they are used in the source text.
    10. Output Requirements: Provide only the translated text without explanations, notes, comments, or any additional text.
"""


class TranslationService:
    def __init__(self, llm_facade: LLMFacade):
        self.llm_facade = llm_facade

    def _create_user_message(self, text: str, config: TranslationConfig) -> str:
        """Creates the user message for the chat API"""
        tone_prompt = config.get_tone_prompt()
        domain_prompt = config.get_domain_prompt()
        glossary_prompt = config.get_glossary_prompt()

        context_section = f"Context: {config.context}\n\n" if config.context else ""

        return PromptTemplate(
            """Translate the following text from {source_language} to {target_language}.

            {context_section}Domain-Specific Terminology: {domain_prompt}
            Tone: {tone_prompt}
            Glossary: {glossary_prompt}

            Text to translate:
            {text}"""
        ).format(
            source_language=config.source_language or "auto-detected",
            target_language=config.target_language,
            context_section=context_section,
            domain_prompt=domain_prompt,
            tone_prompt=tone_prompt,
            glossary_prompt=glossary_prompt,
            text=text,
        )

    def translate_text(self, text: str, config: TranslationConfig) -> str:
        """Base translation method"""
        if not text.strip() or len(text.strip()) == 1:
            return text

        if not config.source_language:
            config.source_language = detect_language(text).unwrap()

        endswith_r = text.endswith("\r")

        # Call the chat API
        response = self.llm_facade.complete(
            PromptTemplate("{system_message} {user_message}").format(
                system_message=SYSTEM_MESSAGE,
                user_message=self._create_user_message(text, config),
            ),
        )

        return response + ("\r" if endswith_r else "")

    def get_supported_languages(self) -> list[str]:
        """Returns a list of supported languages for translation"""
        return [lang.value for lang in Language]
