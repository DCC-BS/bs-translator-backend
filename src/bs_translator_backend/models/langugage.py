"""
Language Definition Models

This module defines the supported languages for translation services
and related language detection functionality.
"""

from __future__ import annotations

from enum import Enum


class Language(Enum):
    """
    Enumeration of supported languages for translation.

    Each language is represented by its ISO 639-1 language code.
    This enum provides a comprehensive list of languages supported
    by the translation service.
    """

    AF = "af"  # Afrikaans
    AR = "ar"  # Arabic
    BG = "bg"  # Bulgarian
    BN = "bn"  # Bengali
    CA = "ca"  # Catalan
    CS = "cs"  # Czech
    CY = "cy"  # Welsh
    DA = "da"  # Danish
    DE = "de"  # German
    EL = "el"  # Greek
    EN = "en"  # English
    EN_GB = "en-gb"  # English (United Kingdom)
    EN_US = "en-us"  # English (United States)
    ES = "es"  # Spanish
    ET = "et"  # Estonian
    FA = "fa"  # Persian
    FI = "fi"  # Finnish
    FR = "fr"  # French
    GU = "gu"  # Gujarati
    HE = "he"  # Hebrew
    HI = "hi"  # Hindi
    HR = "hr"  # Croatian
    HU = "hu"  # Hungarian
    ID = "id"  # Indonesian
    IT = "it"  # Italian
    JA = "ja"  # Japanese
    KN = "kn"  # Kannada
    KO = "ko"  # Korean
    LT = "lt"  # Lithuanian
    LV = "lv"  # Latvian
    MK = "mk"  # Macedonian
    ML = "ml"  # Malayalam
    MR = "mr"  # Marathi
    NE = "ne"  # Nepali
    NL = "nl"  # Dutch
    NO = "no"  # Norwegian
    PA = "pa"  # Punjabi
    PL = "pl"  # Polish
    PT = "pt"  # Portuguese
    RO = "ro"  # Romanian
    RU = "ru"  # Russian
    SK = "sk"  # Slovak
    SL = "sl"  # Slovenian
    SO = "so"  # Somali
    SQ = "sq"  # Albanian
    SV = "sv"  # Swedish
    SW = "sw"  # Swahili
    TA = "ta"  # Tamil
    TE = "te"  # Telugu
    TH = "th"  # Thai
    TL = "tl"  # Filipino
    TR = "tr"  # Turkish
    UK = "uk"  # Ukrainian
    UR = "ur"  # Urdu
    VI = "vi"  # Vietnamese
    ZH_CN = "zh-cn"  # Chinese (Simplified)
    ZH_TW = "zh-tw"  # Chinese (Traditional)


class DetectLanguage(Enum):
    """
    Enumeration for automatic language detection.

    This enum provides the option to automatically detect
    the source language of the input text.
    """

    AUTO = "auto"


# Mapping of language codes to their full names
_LANGUAGE_NAMES = {
    Language.AF: "Afrikaans",
    Language.AR: "Arabic",
    Language.BG: "Bulgarian",
    Language.BN: "Bengali",
    Language.CA: "Catalan",
    Language.CS: "Czech",
    Language.CY: "Welsh",
    Language.DA: "Danish",
    Language.DE: "German",
    Language.EL: "Greek",
    Language.EN: "English",
    Language.EN_GB: "English (United Kingdom)",
    Language.EN_US: "English (United States)",
    Language.ES: "Spanish",
    Language.ET: "Estonian",
    Language.FA: "Persian",
    Language.FI: "Finnish",
    Language.FR: "French",
    Language.GU: "Gujarati",
    Language.HE: "Hebrew",
    Language.HI: "Hindi",
    Language.HR: "Croatian",
    Language.HU: "Hungarian",
    Language.ID: "Indonesian",
    Language.IT: "Italian",
    Language.JA: "Japanese",
    Language.KN: "Kannada",
    Language.KO: "Korean",
    Language.LT: "Lithuanian",
    Language.LV: "Latvian",
    Language.MK: "Macedonian",
    Language.ML: "Malayalam",
    Language.MR: "Marathi",
    Language.NE: "Nepali",
    Language.NL: "Dutch",
    Language.NO: "Norwegian",
    Language.PA: "Punjabi",
    Language.PL: "Polish",
    Language.PT: "Portuguese",
    Language.RO: "Romanian",
    Language.RU: "Russian",
    Language.SK: "Slovak",
    Language.SL: "Slovenian",
    Language.SO: "Somali",
    Language.SQ: "Albanian",
    Language.SV: "Swedish",
    Language.SW: "Swahili",
    Language.TA: "Tamil",
    Language.TE: "Telugu",
    Language.TH: "Thai",
    Language.TL: "Filipino",
    Language.TR: "Turkish",
    Language.UK: "Ukrainian",
    Language.UR: "Urdu",
    Language.VI: "Vietnamese",
    Language.ZH_CN: "Chinese (Simplified)",
    Language.ZH_TW: "Chinese (Traditional)",
}

# Type alias for Pydantic compatibility - accepts Language enum or DetectLanguage class
LanguageOrAuto = Language | DetectLanguage


def get_language_name(language: LanguageOrAuto | None) -> str:
    """
    Get the human-readable name for a given language code.

    Args:
        language: A Language enum member or DetectLanguage enum member

    Returns:
        The full name of the language (e.g., "English", "German")

    Raises:
        ValueError: If the language code is not recognized

    Examples:
        >>> get_language_name(Language.EN)
        'English'
        >>> get_language_name(DetectLanguage.AUTO)
        'auto-detected'
    """
    # Handle auto-detection and None cases
    if isinstance(language, DetectLanguage) or language is None:
        return "auto-detected"

    # Look up the language name (language is now guaranteed to be Language enum)
    name = _LANGUAGE_NAMES.get(language)
    if name is None:
        raise ValueError(f"Language name not found for: {language}") from None

    return name
