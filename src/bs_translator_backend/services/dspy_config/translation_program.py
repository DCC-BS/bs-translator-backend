import os
from collections.abc import AsyncIterator, Iterable

import dspy
from backend_common.dspy_common import (
    AbstractDspyModule,
    SwissGermanStreamListener,
    edit_distance_metric,
)
from backend_common.logger import get_logger
from dspy.streaming.messages import StreamResponse

from bs_translator_backend.utils.app_config import AppConfig


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


class TranslationModule(AbstractDspyModule):
    def __init__(
        self,
        app_config: AppConfig,
    ):
        super().__init__()
        self.predict = dspy.Predict(TranslationSignature)
        stream_listener = SwissGermanStreamListener(
            signature_field_name="translated_text", allow_reuse=True
        )
        self.stream_predict = dspy.streamify(self.predict, stream_listeners=[stream_listener])
        self.logger = get_logger(__name__)
        if os.path.exists(app_config.translation_module_path):
            self.load(app_config.translation_module_path)

    def predict_with_context(
        self,
        **kwargs: object,
    ) -> dspy.Prediction:
        return self.predict(
            **kwargs,
        )

    async def stream_with_context(self, **kwargs: object) -> AsyncIterator[StreamResponse]:
        # Runs inside the adapter-aware dspy.context created by AbstractDspyModule.stream
        output = self.stream_predict(**kwargs)
        async for chunk in output:
            self.logger.info(str(chunk))
            yield chunk  # AbstractDspyModule.stream converts StreamResponse to text


def optimize_translation_module(
    base_program: TranslationModule,
    trainset: Iterable[dspy.Example],
    valset: Iterable[dspy.Example],
    reflection_lm: dspy.LM,
    task_lm: dspy.LM,
) -> TranslationModule:
    """Optimize the translation module using MIPRO with edit distance metric."""
    optimizer = dspy.MIPROv2(
        prompt_model=reflection_lm, task_model=task_lm, metric=edit_distance_metric
    )
    optimized = optimizer.compile(base_program, trainset=list(trainset), valset=list(valset))
    return optimized
