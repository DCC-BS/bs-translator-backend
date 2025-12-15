import os
from collections.abc import AsyncGenerator, Iterable
from typing import Any, cast

import dspy
import jiwer
from dspy.primitives.module import Module

from bs_translator_backend.models.app_config import AppConfig


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
        self.stream_predict: Any = dspy.streamify(self.predict)
        if os.path.exists(app_config.translation_module_path):
            self.load(app_config.translation_module_path)

    def __call__(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        domain: str = "",
        tone: str = "",
        glossary: str = "",
        context: str = "",
    ) -> dspy.Prediction:
        return self.predict(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            domain=domain,
            tone=tone,
            glossary=glossary,
            context=context,
        )

    def forward(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        domain: str = "",
        tone: str = "",
        glossary: str = "",
        context: str = "",
    ) -> dspy.Prediction:
        return self.predict(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            domain=domain,
            tone=tone,
            glossary=glossary,
            context=context,
        )

    async def stream(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        domain: str = "",
        tone: str = "",
        glossary: str = "",
        context: str = "",
    ) -> AsyncGenerator[str, None]:
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
            if isinstance(chunk, dspy.Prediction):
                yield chunk.translated_text


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


def _optimize_translation_module(
    base_program: TranslationModule,
    trainset: Iterable[dspy.Example],
    valset: Iterable[dspy.Example],
    reflection_lm: dspy.LM,
) -> TranslationModule:
    """
    Optimize a translation module using DSPy's GEPA optimizer.

    Args:
        base_program: The initial translation module.
        trainset: Examples used to optimize prompts.
        valset: Examples used to evaluate the optimizer.
        reflection_lm: LM for reflection.
    """

    optimizer = dspy.GEPA(
        metric=translation_metric_with_feedback,
        auto="light",
        reflection_lm=reflection_lm,
    )

    optimized: Module = optimizer.compile(
        base_program,
        trainset=list(trainset),
        valset=list(valset),
    )

    return cast(TranslationModule, optimized)


def translation_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    return translation_metric_with_feedback(gold, pred).score


def translation_metric_with_feedback(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> dspy.Prediction:
    """
    Compute a combined WER/CER score with feedback to guide GEPA.
    """
    predicted = pred.translated_text
    reference = gold.translated_text

    wer_value = jiwer.wer(reference, predicted)
    cer_value = jiwer.cer(reference, predicted)

    # Combine WER and CER with equal weight
    combined_error = (wer_value + cer_value) / 2.0
    # DSPy maximizes the score, so we need to invert it
    score = max(0.0, 1.0 - combined_error)

    feedback_lines = [
        f"WER: {wer_value:.3f}, CER: {cer_value:.3f}, combined error: {combined_error:.3f}.",
    ]

    if wer_value > 0.3:
        feedback_lines.append(
            "High word-level error; ensure terminology and phrasing match the reference more closely."
        )
    if cer_value > 0.3:
        feedback_lines.append(
            "High character-level error; watch for spelling, diacritics, and punctuation fidelity."
        )
    if reference:
        feedback_lines.append(f"Reference snippet: {reference[:200]}")
    if predicted:
        feedback_lines.append(f"Your output snippet: {predicted[:200]}")

    feedback = " ".join(feedback_lines)
    return dspy.Prediction(score=score, feedback=feedback)
