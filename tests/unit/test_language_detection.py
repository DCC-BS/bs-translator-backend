from returns.result import Success

from bs_translator_backend.models.language import Language
from bs_translator_backend.utils.language_detection import detect_language


def test_detect_language_english():
    text = "This is a simple English sentence."
    result = detect_language(text)
    assert isinstance(result, Success)
    output = result.unwrap()
    assert output.language == Language.EN
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
