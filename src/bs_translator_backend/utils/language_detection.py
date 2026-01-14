from fast_langdetect import detect
from returns.result import Failure, ResultE, Success, safe

from bs_translator_backend.models.language import Language
from bs_translator_backend.models.translation import DetectLanguageOutput

_FT_LANGUAGE_MAPPING: dict[str, Language] = {
    "af": Language.AF,
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
    "kn": Language.KN,
    "ko": Language.KO,
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
    "pt": Language.PT,
    "ro": Language.RO,
    "ru": Language.RU,
    "sk": Language.SK,
    "sl": Language.SL,
    "so": Language.SO,
    "sq": Language.SQ,
    "sv": Language.SV,
    "sw": Language.SW,
    "ta": Language.TA,
    "te": Language.TE,
    "th": Language.TH,
    "tl": Language.TL,
    "tr": Language.TR,
    "uk": Language.UK,
    "ur": Language.UR,
    "vi": Language.VI,
    "zh": Language.ZH_CN,
    "zh-cn": Language.ZH_CN,
    "zh-tw": Language.ZH_TW,
    "pt-br": Language.PT,
    "en-us": Language.EN_US,
    "en-gb": Language.EN_GB,
}


def detect_language(text: str) -> ResultE[DetectLanguageOutput]:
    def map_to_language(result: tuple[str, float]) -> ResultE[DetectLanguageOutput]:
        code, confidence = result
        lang = _FT_LANGUAGE_MAPPING.get(code.lower().strip())
        if lang and confidence > 0.1:
            return Success(DetectLanguageOutput(language=lang, confidence=confidence))
        return Failure(Exception(f"Unsupported language code: {code}"))

    return detect_language_str(text).bind(map_to_language)


@safe
def detect_language_str(text: str) -> tuple[str, float]:
    """Detect the language of the given text.

    If no language is detected, return the default language (German).
    """
    result = detect(text[:1000], k=1)
    if not result:
        return "de", 0.0
    return str(result[0]["lang"]), float(result[0]["score"])
