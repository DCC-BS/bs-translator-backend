import os
from collections.abc import AsyncGenerator, Iterable
from typing import Any, cast

import dspy
import jiwer
from dspy.signatures import Signature
from dspy.streaming.messages import StreamResponse
from dspy.streaming.streaming_listener import StreamListener
from litellm.types.utils import ModelResponseStream

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.utils.logger import get_logger


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
    def __init__(
        self,
        app_config: AppConfig,
    ):
        super().__init__()
        self.predict = dspy.Predict(TranslationSignature)
        stream_listener = SwissGermanStreamListener(
            signature_field_name="translated_text", allow_reuse=True
        )
        self.stream_predict: Any = dspy.streamify(self.predict, stream_listeners=[stream_listener])
        self.logger = get_logger(__name__)
        if os.path.exists(app_config.translation_module_path):
            self.load(app_config.translation_module_path)

    def forward(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        domain: str = "",
        tone: str = "",
        glossary: str = "",
        context: str = "",
        reasoning: bool = False,
    ) -> dspy.Prediction:
        adapter = dspy.ChatAdapter() if reasoning else DisableReasoningAdapter()
        with dspy.context(adapter=adapter):
            return self.predict(
                source_text=source_text,
                source_language=source_language,
                target_language=target_language,
                domain=domain,
                tone=tone,
                glossary=glossary,
                context=context,
            )

    def stream(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        domain: str = "",
        tone: str = "",
        glossary: str = "",
        context: str = "",
        reasoning: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream translated text chunks from the DSPy model."""
        adapter = dspy.ChatAdapter() if reasoning else DisableReasoningAdapter()

        async def generate_chunks():
            with dspy.context(adapter=adapter):
                output = self.stream_predict(
                    source_text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                    domain=domain,
                    tone=tone,
                    glossary=glossary,
                    context=context,
                )
                async for chunk in output:
                    self.logger.info(str(chunk))
                    if isinstance(chunk, StreamResponse):
                        yield chunk.chunk

        return generate_chunks()


class DisableReasoningAdapter(dspy.ChatAdapter):
    """Adapter that adds a no-think hint when reasoning is disabled."""

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        custom_suffix = "\no_think"
        return super().format_user_message_content(
            signature=signature,
            inputs=inputs,
            prefix=prefix,
            suffix=suffix + custom_suffix,
            main_request=main_request,
        )


class SwissGermanStreamListener(StreamListener):
    """Stream listener that normalizes Swiss German characters in streamed chunks."""

    def __init__(
        self,
        signature_field_name: str,
        predict: Any = None,
        predict_name: str | None = None,
        allow_reuse: bool = False,
    ):
        """
        Extend StreamListener to accept DisableReasoningAdapter (a ChatAdapter subclass).
        """
        super().__init__(
            signature_field_name=signature_field_name,
            predict=predict,
            predict_name=predict_name,
            allow_reuse=allow_reuse,
        )
        self.adapter_identifiers["DisableReasoningAdapter"] = self.adapter_identifiers[
            "ChatAdapter"
        ]

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Replace German sharp S with its Swiss German equivalent."""
        return value.replace("ß", "ss")

    def _normalize_chunk_fields(self, chunk: ModelResponseStream) -> ModelResponseStream:
        """Normalize both content and reasoning_content fields on a chunk."""
        delta = chunk.choices[0].delta
        if hasattr(delta, "content"):
            content = getattr(delta, "content", None)
            if content is not None:
                delta.content = self._normalize_text(content)
        if hasattr(delta, "reasoning_content"):
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content is not None:
                delta.reasoning_content = self._normalize_text(reasoning_content)
        return chunk

    def receive(self, chunk: ModelResponseStream) -> StreamResponse | None:
        """
        Intercept streamed chunks to apply Swiss German normalization before yielding.

        Normalization is applied to both standard content and reasoning content so the
        parent StreamListener can continue handling buffering and chunk parsing without
        having to know about the transformation.
        """
        normalized_chunk = self._normalize_chunk_fields(chunk)
        delta = normalized_chunk.choices[0].delta

        if getattr(delta, "content", None) is None:
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content is not None:
                delta.content = reasoning_content

        parent_result = super().receive(normalized_chunk)
        if parent_result is None:
            return None

        if isinstance(parent_result.chunk, str):
            parent_result.chunk = self._normalize_text(parent_result.chunk)
        return parent_result


def optimize_translation_module(
    base_program: TranslationModule,
    trainset: Iterable[dspy.Example],
    valset: Iterable[dspy.Example],
    reflection_lm: dspy.LM,
    task_lm: dspy.LM,
) -> TranslationModule:
    optimizer = dspy.MIPROv2(
        prompt_model=reflection_lm, task_model=task_lm, metric=translation_metric
    )
    optimized = optimizer.compile(base_program, trainset=list(trainset), valset=list(valset))
    return cast(TranslationModule, optimized)


def translation_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    predicted = pred.translated_text
    reference = gold.translated_text

    wer_value = jiwer.wer(reference, predicted)
    cer_value = jiwer.cer(reference, predicted)

    # Combine WER and CER with equal weight
    combined_error = (wer_value + cer_value) / 2.0
    # DSPy maximizes the score, so we need to invert it
    score = max(0.0, 1.0 - combined_error)
    return score
