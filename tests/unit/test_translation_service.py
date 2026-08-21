from unittest.mock import AsyncMock, MagicMock

import pytest
from returns.result import Success

from bs_translator_backend.models.language import DetectLanguage, Language
from bs_translator_backend.models.translation import DetectLanguageOutput, TranslationConfig
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.app_config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    """Static config so unit tests do not depend on a local environment or a live LLM."""
    return AppConfig(
        llm_url="http://localhost:8001/v1",
        llm_api_key="none",
        llm_model="test/test-model",
        client_url="http://localhost:3000",
        docling_url="http://localhost:8004/v1",
        docling_api_key="none",
        hmac_secret="test-secret",  # noqa: S106
        whisper_url="http://localhost:50001/v1",
    )


def _make_mock_stream(output: str) -> MagicMock:
    async def mock_stream_text(delta: bool = False):
        yield output

    mock_stream = MagicMock()
    mock_stream.stream_text = mock_stream_text
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    return mock_stream


@pytest.fixture
def translation_service(app_config: AppConfig) -> TranslationService:
    text_chunk_service = TextChunkService()
    service = TranslationService(
        app_config, text_chunk_service, conversion_service_factory=lambda: MagicMock()
    )

    # Mock both agents to avoid requiring an actual LLM, and to let tests assert
    # which agent a given input was routed to.
    service.translation_agent = MagicMock()
    service.translation_agent.run_stream = MagicMock(
        return_value=_make_mock_stream("long-agent-output")
    )

    service.short_text_translation_agent = MagicMock()
    service.short_text_translation_agent.run_stream = MagicMock(
        return_value=_make_mock_stream("short-agent-output")
    )

    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Hirsch", "Der Hirsch", "Der grosse Hirsch"])
async def test_short_word_counts_route_to_short_text_agent(
    translation_service: TranslationService, text: str
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [c async for c in translation_service.translate_text(text, config)]

    assert "".join(chunks) == "short-agent-output"
    translation_service.short_text_translation_agent.run_stream.assert_called_once()
    translation_service.translation_agent.run_stream.assert_not_called()


@pytest.mark.asyncio
async def test_four_word_input_routes_to_normal_agent(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [
        c async for c in translation_service.translate_text("Der grosse braune Hirsch", config)
    ]

    assert "".join(chunks) == "long-agent-output"
    translation_service.translation_agent.run_stream.assert_called_once()
    translation_service.short_text_translation_agent.run_stream.assert_not_called()


@pytest.mark.asyncio
async def test_multiline_short_input_routes_to_normal_agent(
    translation_service: TranslationService,
) -> None:
    """Few words but multiple lines should not be treated as a lexical lookup."""
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [c async for c in translation_service.translate_text("Hirsch\nHirsch", config)]

    assert "".join(chunks) == "long-agent-output"
    translation_service.translation_agent.run_stream.assert_called_once()
    translation_service.short_text_translation_agent.run_stream.assert_not_called()


@pytest.mark.asyncio
async def test_single_character_early_return_short_circuits_before_any_agent(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [c async for c in translation_service.translate_text("H", config)]

    assert chunks == ["H"]
    translation_service.translation_agent.run_stream.assert_not_called()
    translation_service.short_text_translation_agent.run_stream.assert_not_called()


def test_short_text_user_message_includes_source_and_target_language(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    message = translation_service._create_short_text_user_message(
        text="Hirsch", translation_config=config, reasoning=False
    )

    assert "German" in message
    assert "French" in message


def test_short_text_user_message_omits_source_language_when_not_asserted(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    message = translation_service._create_short_text_user_message(
        text="Hirsch",
        translation_config=config,
        reasoning=False,
        assert_source_language=False,
    )

    assert "German" not in message
    assert "Translate the following text into French." in message


@pytest.mark.asyncio
async def test_explicit_source_language_is_asserted_in_short_text_prompt(
    translation_service: TranslationService,
) -> None:
    """An explicitly-chosen source language is authoritative and safe to assert."""
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    [c async for c in translation_service.translate_text("Hirsch", config)]

    prompt = translation_service.short_text_translation_agent.run_stream.call_args[0][0]
    assert "Translate the following text from German into French." in prompt


@pytest.mark.asyncio
async def test_auto_source_high_confidence_detection_is_asserted_in_short_text_prompt(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """High-confidence auto-detection is trustworthy enough to assert."""
    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Success(DetectLanguageOutput(language=Language.DE, confidence=0.96)),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.FR)

    [c async for c in translation_service.translate_text("Der Hirsch", config)]

    prompt = translation_service.short_text_translation_agent.run_stream.call_args[0][0]
    assert "Translate the following text from German into French." in prompt


@pytest.mark.asyncio
async def test_auto_source_low_confidence_detection_is_not_asserted_in_short_text_prompt(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low-confidence auto-detection is unreliable on short text and must not be asserted."""
    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Success(DetectLanguageOutput(language=Language.EN_US, confidence=0.36)),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.FR)

    [c async for c in translation_service.translate_text("Hirsch", config)]

    prompt = translation_service.short_text_translation_agent.run_stream.call_args[0][0]
    assert "from" not in prompt.split("\n", 1)[0]
    assert "Translate the following text into French." in prompt


@pytest.mark.parametrize(
    ("source_language", "detect_return", "text"),
    [
        (Language.DE, None, "Der grosse braune Hirsch"),
        (
            DetectLanguage.AUTO,
            Success(DetectLanguageOutput(language=Language.DE, confidence=0.96)),
            "Der grosse braune Hirsch",
        ),
        (
            DetectLanguage.AUTO,
            Success(DetectLanguageOutput(language=Language.EN_US, confidence=0.36)),
            "Der grosse braune Hirsch",
        ),
    ],
)
@pytest.mark.asyncio
async def test_long_path_prompt_never_mentions_source_language(
    translation_service: TranslationService,
    monkeypatch: pytest.MonkeyPatch,
    source_language: Language | DetectLanguage,
    detect_return: object,
    text: str,
) -> None:
    """The long path's prompt is unaffected by source-language trustworthiness."""
    if detect_return is not None:
        monkeypatch.setattr(
            "bs_translator_backend.services.translation_service.detect_language",
            lambda _text: detect_return,
        )
    config = TranslationConfig(source_language=source_language, target_language=Language.FR)

    [c async for c in translation_service.translate_text(text, config)]

    prompt = translation_service.translation_agent.run_stream.call_args[0][0]
    assert "Translate the following text into French." in prompt
    assert "from" not in prompt.split("\n", 1)[0]
