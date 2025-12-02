"""
Unit tests for the Language module.

This module tests the Language enum and related functionality,
ensuring data integrity and proper configuration.
"""

from bs_translator_backend.models.language import (
    _LANGUAGE_NAMES,
    DetectLanguage,
    Language,
    get_language_name,
)


class TestLanguageEnumCompleteness:
    """Tests to ensure the Language enum and _LANGUAGE_NAMES dictionary are in sync."""

    def test_all_languages_have_names(self) -> None:
        """
        Verify that every member of the Language enum has a corresponding entry
        in the _LANGUAGE_NAMES dictionary.

        This test ensures that the manual mapping doesn't become out-of-sync
        when new languages are added to the Language enum.
        """
        missing_languages = []

        for language in Language:
            if language not in _LANGUAGE_NAMES:
                missing_languages.append(language)

        assert not missing_languages, (
            f"The following languages are missing from _LANGUAGE_NAMES: {[lang.value for lang in missing_languages]}"
        )

    def test_no_extra_names_in_mapping(self) -> None:
        """
        Verify that _LANGUAGE_NAMES doesn't contain entries for non-existent languages.

        This test ensures we don't have stale entries in the mapping dictionary.
        """
        valid_languages = set(Language)
        mapped_languages = set(_LANGUAGE_NAMES.keys())

        extra_languages = mapped_languages - valid_languages

        assert not extra_languages, (
            f"_LANGUAGE_NAMES contains entries for non-existent languages: {[lang.value for lang in extra_languages]}"
        )

    def test_language_names_are_non_empty(self) -> None:
        """
        Verify that all language names in _LANGUAGE_NAMES are non-empty strings.
        """
        invalid_entries = []

        for language, name in _LANGUAGE_NAMES.items():
            if not isinstance(name, str) or not name.strip():
                invalid_entries.append((language, name))

        assert not invalid_entries, (
            f"The following languages have invalid names: {[(lang.value, name) for lang, name in invalid_entries]}"
        )


class TestGetLanguageName:
    """Tests for the get_language_name function."""

    def test_get_language_name_with_valid_language(self) -> None:
        """Test that get_language_name returns correct names for valid Language enums."""
        assert get_language_name(Language.EN) == "English"
        assert get_language_name(Language.DE) == "German"
        assert get_language_name(Language.FR) == "French"
        assert get_language_name(Language.ES) == "Spanish"
        assert get_language_name(Language.EN_GB) == "English (United Kingdom)"
        assert get_language_name(Language.EN_US) == "English (United States)"
        assert get_language_name(Language.ZH_CN) == "Chinese (Simplified)"
        assert get_language_name(Language.ZH_TW) == "Chinese (Traditional)"

    def test_get_language_name_with_auto_detect(self) -> None:
        """Test that get_language_name handles DetectLanguage.AUTO correctly."""
        assert get_language_name(DetectLanguage.AUTO) == "auto-detected"

    def test_get_language_name_with_none(self) -> None:
        """Test that get_language_name handles None correctly."""
        assert get_language_name(None) == "auto-detected"

    def test_get_language_name_for_all_languages(self) -> None:
        """
        Test that get_language_name works for every Language enum member
        without raising exceptions.
        """
        for language in Language:
            name = get_language_name(language)
            assert isinstance(name, str)
            assert len(name) > 0
