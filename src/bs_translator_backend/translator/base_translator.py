from abc import ABC, abstractmethod

import httpx
from openai import Client
from openai.types.chat import ChatCompletionMessageParam

from bs_translator_backend.translator.config import LLMConfig, TranslationConfig
from bs_translator_backend.translator.utils import detect_language


class BaseTranslator(ABC):
    """Base class for all translators"""

    def __init__(self):
        self.llm_config = LLMConfig()
        self.translation_config = TranslationConfig()
        self.client = Client(
            base_url=self.llm_config.base_url,
            http_client=httpx.Client(verify=ssl_context),
        )
        models = self.client.models.list()
        self.model_name = models.data[0].id

    def _create_system_message(self) -> str:
        """Creates the system message for the chat API"""
        return """You are an expert translator.

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

    def _create_user_message(self, text: str, config: TranslationConfig) -> str:
        """Creates the user message for the chat API"""
        tone_prompt = self._get_tone_prompt(config.tone, config.domain)
        domain_prompt = self._get_domain_prompt(config.domain)
        glossary_prompt = self._get_glossary_prompt(config.glossary)

        context_section = f"Context: {config.context}\n\n" if config.context else ""

        return f"""Translate the following text from {config.source_language} to {config.target_language}.

{context_section}Domain-Specific Terminology: {domain_prompt}
Tone: {tone_prompt}
Glossary: {glossary_prompt}

Text to translate:
{text}"""

    def _get_tone_prompt(self, tone: str | None, domain: str | None) -> str:
        """Generates the tone-specific part of the prompt"""
        if tone is None:
            return "Use a neutral tone that is objective, informative, and unbiased."

        tone_prompts = {
            "formal": "Use a formal and professional tone appropriate for official documents.",
            "informal": "Use an informal and conversational tone that is friendly and engaging.",
            "technical": f"Use a technical tone appropriate for {domain if domain else 'professional'} writing.",
        }
        return tone_prompts.get(tone.lower(), "Use a neutral tone.")

    def _get_domain_prompt(self, domain: str | None) -> str:
        """Generates the domain-specific part of the prompt"""
        return f"Use terminology specific to the {domain} field." if domain else "No specific domain requirements."

    def _get_glossary_prompt(self, glossary: str | None) -> str:
        """Generates the glossary-specific part of the prompt"""
        if glossary is None:
            return "No specific glossary provided."

        glossary = "\n".join(glossary.replace(":", ": ").split(";"))
        return f"Use the following glossary to ensure accurate translations:\n{glossary}"

    def translate_text(self, text: str, config: TranslationConfig) -> str:
        """Base translation method"""
        if not text.strip() or len(text.strip()) == 1:
            return text

        if not config.source_language or config.source_language.lower() in [
            "auto",
            "automatisch erkennen",
        ]:
            config.source_language = detect_language(text)

        endswith_r = text.endswith("\r")

        # Create messages for the chat API
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._create_system_message()},
            {"role": "user", "content": self._create_user_message(text, config)},
        ]

        # Call the chat API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.num_ctx,
            frequency_penalty=self.llm_config.frequency_penalty,
            top_p=self.llm_config.top_p,
        )

        translation_text = self._process_response(response.choices[0].message.content)
        return translation_text + ("\r" if endswith_r else "")

    def _process_response(self, text: str) -> str:
        """Process the translation response"""
        if text is None:
            return ""

        text = text.strip().replace("ß", "ss")

        # Check if the response contains the translation_text tags
        if "<translation_text>" in text and "</translation_text>" in text:
            start_index = text.find("<translation_text>") + len("<translation_text>")
            end_index = text.find("</translation_text>")
            return text[start_index:end_index].strip()

        return text

    @abstractmethod
    def translate(self, input_path: str, output_path: str, config: TranslationConfig) -> None:
        """Abstract method to be implemented by specific translators"""
        pass
