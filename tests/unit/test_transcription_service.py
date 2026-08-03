"""
Unit tests for the transcription service language handling.
"""

from bs_translator_backend.models.language import Language
from bs_translator_backend.services.transcription_service import (
    transform_language_code_for_whisper,
)


class TestTransformLanguageCodeForWhisper:
    """Tests for mapping Language codes onto codes Whisper understands."""

    def test_english_variants_are_collapsed(self) -> None:
        assert transform_language_code_for_whisper("en-gb") == "en"
        assert transform_language_code_for_whisper("en-us") == "en"

    def test_dari_is_transcribed_as_persian(self) -> None:
        assert transform_language_code_for_whisper(Language.FA_AF.value) == "fa"

    def test_unsupported_languages_return_none(self) -> None:
        """Kurdish and Tigrinya have no Whisper model, so we fall back to auto-detection."""
        assert transform_language_code_for_whisper(Language.KU.value) is None
        assert transform_language_code_for_whisper(Language.TI.value) is None

    def test_known_codes_pass_through(self) -> None:
        assert transform_language_code_for_whisper(Language.KA.value) == "ka"
        assert transform_language_code_for_whisper(Language.PS.value) == "ps"
        assert transform_language_code_for_whisper(Language.DE.value) == "de"
