import pytest
from returns.result import Failure, Success

from bs_translator_backend.models.language import Language
from bs_translator_backend.utils.language_detection import (
    _FT_LANGUAGE_MAPPING,
    detect_language,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", Language.EN_US),
        ("en-us", Language.EN_US),
        ("en-gb", Language.EN_GB),
        ("zh", Language.ZH_CN),
        ("zh-cn", Language.ZH_CN),
        ("zh-tw", Language.ZH_TW),
        ("pt", Language.PT),
        ("pt-br", Language.PT),
    ],
)
def test_regional_aliases_map_to_a_language(code: str, expected: Language) -> None:
    assert _FT_LANGUAGE_MAPPING[code] == expected


def test_detected_codes_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """fast_langdetect codes are lowercased and stripped before lookup."""
    monkeypatch.setattr(
        "bs_translator_backend.utils.language_detection.detect_language_str",
        lambda _text: Success((" ZH-TW ", 0.9)),
    )
    result = detect_language("irrelevant")
    assert isinstance(result, Success)
    assert result.unwrap().language == Language.ZH_TW


def test_unmapped_code_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bs_translator_backend.utils.language_detection.detect_language_str",
        lambda _text: Success(("xx", 0.9)),
    )
    assert isinstance(detect_language("irrelevant"), Failure)


def test_detect_language_english():
    text = "This is a simple English sentence."
    result = detect_language(text)
    assert isinstance(result, Success)
    output = result.unwrap()
    assert output.language == Language.EN_US
    assert output.confidence > 0.1


def test_detect_language_german():
    text = "Dies ist ein einfacher deutscher Satz."
    result = detect_language(text)
    assert isinstance(result, Success)
    output = result.unwrap()
    assert output.language == Language.DE
    assert output.confidence > 0.1


def test_detect_language_french():
    text = "C'est une phrase simple en français."
    result = detect_language(text)
    assert isinstance(result, Success)
    output = result.unwrap()
    assert output.language == Language.FR
    assert output.confidence > 0.1


def test_detect_language_chinese():
    text = "这是一个简单的中文句子。"
    result = detect_language(text)
    assert isinstance(result, Success)
    output = result.unwrap()
    assert output.language == Language.ZH_CN
    assert output.confidence > 0.1
