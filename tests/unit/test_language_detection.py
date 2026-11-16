import pytest
from returns.result import Failure, Success

from bs_translator_backend.models.language import Language
from bs_translator_backend.utils import language_detection
from bs_translator_backend.utils.language_detection import (
    _FT_LANGUAGE_MAPPING,
    DETECTION_SAMPLE_CHARS,
    detect_language,
    sample_windows,
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
        ("yue", Language.ZH_HK),
        ("pt", Language.PT),
        ("pt-br", Language.PT),
        ("sr", Language.SR),
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


class TestNeverDownloads:
    """fast-langdetect's default would download the 126 MB model on first use."""

    @pytest.fixture(autouse=True)
    def _fresh_detector(self):
        language_detection._get_detector.cache_clear()
        yield
        language_detection._get_detector.cache_clear()

    @pytest.fixture
    def no_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("fast-langdetect tried to download a model")

        monkeypatch.setattr("fast_langdetect.infer.ModelDownloader.download", fail)

    def test_without_full_model_uses_bundled_lite_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, no_download: None
    ) -> None:
        monkeypatch.setenv("FTLANG_CACHE", str(tmp_path))

        result = detect_language("Dies ist ein einfacher deutscher Satz.")

        assert result.unwrap().language == Language.DE

    def test_full_model_is_used_from_ftlang_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, no_download: None
    ) -> None:
        (tmp_path / "lid.176.bin").write_bytes(b"")
        monkeypatch.setenv("FTLANG_CACHE", str(tmp_path))

        detector = language_detection._get_detector()

        assert detector.config.custom_model_path == str(tmp_path / "lid.176.bin")

    def test_input_is_not_truncated_to_80_characters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, no_download: None
    ) -> None:
        monkeypatch.setenv("FTLANG_CACHE", str(tmp_path))

        assert language_detection._get_detector().config.max_input_length is None


class TestSampleWindows:
    def test_short_text_is_its_own_sample(self) -> None:
        assert sample_windows("kurz") == ["kurz"]

    def test_long_text_is_sampled_from_start_to_end(self) -> None:
        text = "a" * DETECTION_SAMPLE_CHARS + "b" * DETECTION_SAMPLE_CHARS * 3 + "c" * 10
        windows = sample_windows(text)

        assert windows[0] == text[:DETECTION_SAMPLE_CHARS]
        assert windows[-1] == text[-DETECTION_SAMPLE_CHARS:]
        assert all(len(window) == DETECTION_SAMPLE_CHARS for window in windows)

    def test_number_of_samples_is_bounded(self) -> None:
        assert len(sample_windows("x" * DETECTION_SAMPLE_CHARS * 100)) == 8
