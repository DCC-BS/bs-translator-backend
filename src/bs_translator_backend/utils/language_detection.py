import math
import os
from functools import cache
from pathlib import Path

from dcc_backend_common.logger import get_logger
from fast_langdetect import LangDetectConfig, LangDetector
from fast_langdetect.infer import CACHE_DIRECTORY, FASTTEXT_LARGE_MODEL_NAME
from returns.pipeline import is_successful
from returns.result import Failure, ResultE, Success, safe

from bs_translator_backend.models.language import Language
from bs_translator_backend.models.translation import DetectLanguageOutput

logger = get_logger(__name__)

# Characters handed to fastText per detection. fast-langdetect's own default silently
# truncates to 80 characters; a longer sample is cheap and far more accurate.
DETECTION_SAMPLE_CHARS = 1000
# Upper bound on the windows `sample_windows` spreads across a long text.
MAX_DETECTION_SAMPLES = 8

_FT_LANGUAGE_MAPPING: dict[str, Language] = {
    "af": Language.AF,
    "am": Language.AM,
    "ar": Language.AR,
    "bg": Language.BG,
    "bn": Language.BN,
    "ca": Language.CA,
    "cs": Language.CS,
    "cy": Language.CY,
    "da": Language.DA,
    "de": Language.DE,
    "el": Language.EL,
    "en": Language.EN_US,
    "es": Language.ES,
    "et": Language.ET,
    "fa": Language.FA,
    "fi": Language.FI,
    "fr": Language.FR,
    "gu": Language.GU,
    "he": Language.HE,
    "hi": Language.HI,
    "hr": Language.HR,
    "hu": Language.HU,
    "id": Language.ID,
    "it": Language.IT,
    "ja": Language.JA,
    "ka": Language.KA,
    "kn": Language.KN,
    "ko": Language.KO,
    "ku": Language.KU,
    "lt": Language.LT,
    "lv": Language.LV,
    "mk": Language.MK,
    "ml": Language.ML,
    "mr": Language.MR,
    "ne": Language.NE,
    "nl": Language.NL,
    "no": Language.NO,
    "pa": Language.PA,
    "pl": Language.PL,
    "ps": Language.PS,
    "pt": Language.PT,
    "ro": Language.RO,
    "ru": Language.RU,
    "sk": Language.SK,
    "sl": Language.SL,
    "so": Language.SO,
    "sq": Language.SQ,
    "sr": Language.SR,
    "sv": Language.SV,
    "sw": Language.SW,
    "ta": Language.TA,
    "te": Language.TE,
    "th": Language.TH,
    "ti": Language.TI,
    "tl": Language.TL,
    "tr": Language.TR,
    "uk": Language.UK,
    "ur": Language.UR,
    "vi": Language.VI,
    "zh": Language.ZH_CN,
    "zh-cn": Language.ZH_CN,
    "zh-tw": Language.ZH_TW,
    "yue": Language.ZH_HK,  # Cantonese
    "pt-br": Language.PT,
    "en-us": Language.EN_US,
    "en-gb": Language.EN_GB,
}


def detect_language(text: str) -> ResultE[DetectLanguageOutput]:
    detection = detect_language_str(text)
    if not is_successful(detection):
        return Failure(detection.failure())

    code, confidence = detection.unwrap()
    lang = _FT_LANGUAGE_MAPPING.get(code.lower().strip())
    if lang and confidence > 0.1:
        return Success(DetectLanguageOutput(language=lang, confidence=confidence))
    return Failure(Exception(f"Unsupported language code: {code}"))


def sample_windows(text: str) -> list[str]:
    """Evenly spaced `DETECTION_SAMPLE_CHARS`-long windows covering a long text.

    The first window is the text's beginning, i.e. what `detect_language` looks at.
    A text that fits into one window is returned as the only sample.
    """
    if len(text) <= DETECTION_SAMPLE_CHARS:
        return [text]
    count = min(MAX_DETECTION_SAMPLES, math.ceil(len(text) / DETECTION_SAMPLE_CHARS))
    last_start = len(text) - DETECTION_SAMPLE_CHARS
    starts = sorted({round(i * last_start / (count - 1)) for i in range(count)})
    return [text[start : start + DETECTION_SAMPLE_CHARS] for start in starts]


def _full_model_path() -> Path:
    """Where the full lid.176.bin model is expected (baked into the Docker image)."""
    return Path(os.getenv("FTLANG_CACHE", CACHE_DIRECTORY)) / FASTTEXT_LARGE_MODEL_NAME


@cache
def _get_detector() -> LangDetector:
    """The detector, never downloading anything at runtime.

    fast-langdetect's default (`model="auto"`) downloads the 126 MB full model on first
    use, synchronously and on every call while offline. Instead the full model is used
    only when it is already on disk (the Docker image bakes it in at `$FTLANG_CACHE`);
    otherwise the small model bundled with the package is used.
    """
    model_path = _full_model_path()
    if model_path.exists():
        config = LangDetectConfig(custom_model_path=str(model_path), max_input_length=None)
    else:
        logger.warning(
            "Full fastText language model not found, using the bundled lite model",
            expected_path=str(model_path),
        )
        config = LangDetectConfig(model="lite", max_input_length=None)
    return LangDetector(config)


def warm_up_language_detection() -> None:
    """Load the fastText model up front so the first request does not pay for it.

    Blocking (the full model takes a moment to load): call it from a worker thread.
    """
    _get_detector().detect("warm up", k=1)


@safe
def detect_language_str(text: str) -> tuple[str, float]:
    """Detect the language of the given text.

    If no language is detected, return the default language (German).
    """
    result = _get_detector().detect(text[:DETECTION_SAMPLE_CHARS], k=1)
    if not result:
        return "de", 0.0
    return str(result[0]["lang"]), float(result[0]["score"])
