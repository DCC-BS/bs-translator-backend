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


# Type alias for Pydantic compatibility - accepts Language enum or DetectLanguage class
LanguageOrAuto = Language | DetectLanguage
