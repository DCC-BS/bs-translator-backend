from __future__ import annotations

from enum import Enum
from typing import override


class Language(Enum):
    AF = "af"
    AR = "ar"
    BG = "bg"
    BN = "bn"
    CA = "ca"
    CS = "cs"
    CY = "cy"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    FA = "fa"
    FI = "fi"
    FR = "fr"
    GU = "gu"
    HE = "he"
    HI = "hi"
    HR = "hr"
    HU = "hu"
    ID = "id"
    IT = "it"
    JA = "ja"
    KN = "kn"
    KO = "ko"
    LT = "lt"
    LV = "lv"
    MK = "mk"
    ML = "ml"
    MR = "mr"
    NE = "ne"
    NL = "nl"
    NO = "no"
    PA = "pa"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    RU = "ru"
    SK = "sk"
    SL = "sl"
    SO = "so"
    SQ = "sq"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TH = "th"
    TL = "tl"
    TR = "tr"
    UK = "uk"
    UR = "ur"
    VI = "vi"
    ZH_CN = "zh-cn"
    ZH_TW = "zh-tw"


class DetectLanguage:
    """Class representing automatic language detection.

    This class serves as a marker type for automatic language detection
    in translation operations. When used in Pydantic models, it indicates
    that the language should be automatically detected rather than specified.
    """

    @override
    def __str__(self) -> str:
        return "detect_language"

    @override
    def __repr__(self) -> str:
        return "DetectLanguage()"


# Type alias for Pydantic compatibility - accepts Language enum or DetectLanguage class
LanguageOrAuto = Language | DetectLanguage
