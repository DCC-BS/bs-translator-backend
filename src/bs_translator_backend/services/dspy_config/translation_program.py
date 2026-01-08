import dspy
from dcc_backend_common.logger import get_logger


class TranslationSignature(dspy.Signature):
    """source_text, source_language, target_language, domain, tone, glossary, context -> translated_text"""

    source_text = dspy.InputField(desc="Input text to translate. May contain markdown formatting.")
    source_language = dspy.InputField(desc="Source language")
    target_language = dspy.InputField(desc="Target language")
    domain = dspy.InputField(desc="Domain or subject area for translation")
    tone = dspy.InputField(desc="Tone or style for translation")
    glossary = dspy.InputField(desc="Glossary definitions for translation")
    context = dspy.InputField(
        desc="Context containing previous translations to get consistent translations"
    )
    translated_text = dspy.OutputField(
        desc="Translated text. Contains markdown formatting if the input text contains markdown formatting."
    )


class TranslationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(TranslationSignature)
        self.logger = get_logger(__name__)

    def predict_with_context(
        self,
        **kwargs: object,
    ) -> dspy.Prediction:
        return self.predict(
            **kwargs,
        )

    async def stream_with_context(self, **kwargs: object) -> None:
        pass
