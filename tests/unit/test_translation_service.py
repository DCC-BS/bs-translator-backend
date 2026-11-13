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
        hmac_secret="test-secret",
        whisper_url="http://localhost:50001/v1",
    )


def _make_mock_agent(output: str) -> MagicMock:
    agent = MagicMock()

    async def mock_run_stream_text_safe(user_prompt: str, delta: bool = True):
        yield output

    agent.run_stream_text_safe = MagicMock(side_effect=mock_run_stream_text_safe)
    return agent


@pytest.fixture
def translation_service(app_config: AppConfig) -> TranslationService:
    text_chunk_service = TextChunkService()
    service = TranslationService(
        app_config, text_chunk_service, conversion_service_factory=lambda: MagicMock()
    )

    # Mock both agents to avoid requiring an actual LLM, and to let tests assert
    # which agent a given input was routed to.
    service.translation_agent = _make_mock_agent("long-agent-output")
    service.short_text_translation_agent = _make_mock_agent("short-agent-output")

    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Hirsch", "Der Hirsch", "Der grosse Hirsch"])
async def test_short_word_counts_route_to_short_text_agent(
    translation_service: TranslationService, text: str
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [c async for c in translation_service.translate_text(text, config)]

    assert "".join(chunks) == "short-agent-output"
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_called_once()
    translation_service.translation_agent.run_stream_text_safe.assert_not_called()


@pytest.mark.asyncio
async def test_four_word_input_routes_to_normal_agent(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [
        c async for c in translation_service.translate_text("Der grosse braune Hirsch", config)
    ]

    assert "".join(chunks) == "long-agent-output"
    translation_service.translation_agent.run_stream_text_safe.assert_called_once()
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_not_called()


@pytest.mark.asyncio
async def test_multiline_short_input_routes_to_normal_agent(
    translation_service: TranslationService,
) -> None:
    """Few words but multiple lines should not be treated as a lexical lookup."""
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [c async for c in translation_service.translate_text("Hirsch\nHirsch", config)]

    assert "".join(chunks) == "long-agent-output"
    translation_service.translation_agent.run_stream_text_safe.assert_called_once()
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_not_called()


@pytest.mark.asyncio
async def test_single_character_early_return_short_circuits_before_any_agent(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    chunks = [c async for c in translation_service.translate_text("H", config)]

    assert chunks == ["H"]
    translation_service.translation_agent.run_stream_text_safe.assert_not_called()
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_not_called()


def test_short_text_user_message_omits_empty_metadata_fields(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(
        source_language=Language.DE,
        target_language=Language.FR,
        domain="",
        tone="",
        glossary="",
        context="",
    )

    message = translation_service._create_short_text_user_message(
        text="Hirsch",
        translation_config=config,
        assert_source_language=False,
    )

    expected = "Translate the following text into French.\n\nText to translate:\nHirsch\n"
    assert message == expected
    assert "Domain:" not in message
    assert "Tone:" not in message
    assert "Glossary:" not in message
    assert "Context:" not in message


def test_short_text_user_message_includes_non_empty_metadata_fields(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(
        source_language=Language.DE,
        target_language=Language.FR,
        domain="Legal",
        tone="Formal",
        glossary="Hirsch -> Cerf",
        context="Forest document",
    )

    message = translation_service._create_short_text_user_message(
        text="Hirsch",
        translation_config=config,
        assert_source_language=True,
    )

    assert "Translate the following text from German into French." in message
    assert "Domain: Legal" in message
    assert "Tone: Formal" in message
    assert "Glossary: Hirsch -> Cerf" in message
    assert "Context:\nForest document" in message
    assert "Text to translate:\nHirsch\n" in message


def test_short_text_user_message_includes_source_and_target_language(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

    message = translation_service._create_short_text_user_message(
        text="Hirsch", translation_config=config
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

    prompt = translation_service.short_text_translation_agent.run_stream_text_safe.call_args.kwargs[
        "user_prompt"
    ]
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

    prompt = translation_service.short_text_translation_agent.run_stream_text_safe.call_args.kwargs[
        "user_prompt"
    ]
    assert "Translate the following text from German into French." in prompt


@pytest.mark.asyncio
async def test_auto_detection_does_not_mutate_the_callers_config(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller owns the config: `translate_image` reuses one across every text
    segment, so a detected language must not leak out of a single call."""
    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Success(DetectLanguageOutput(language=Language.DE, confidence=0.96)),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.FR)

    [c async for c in translation_service.translate_text("Der Hirsch", config)]

    assert config.source_language == DetectLanguage.AUTO


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

    prompt = translation_service.short_text_translation_agent.run_stream_text_safe.call_args.kwargs[
        "user_prompt"
    ]
    assert "from" not in prompt.split("\n", 1)[0]
    assert "Translate the following text into French." in prompt


@pytest.mark.asyncio
async def test_auto_source_low_confidence_does_not_skip_when_target_is_german(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low confidence detection must not default to German and skip when target is German."""
    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Success(DetectLanguageOutput(language=Language.DE, confidence=0.35)),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.DE)

    chunks = [c async for c in translation_service.translate_text("Hirsch", config)]

    assert "".join(chunks) == "short-agent-output"
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_called_once()
    prompt = translation_service.short_text_translation_agent.run_stream_text_safe.call_args.kwargs[
        "user_prompt"
    ]
    assert "Translate the following text into German." in prompt
    assert "from" not in prompt


@pytest.mark.asyncio
async def test_auto_source_failed_detection_routes_to_short_agent_when_target_is_german(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed language detection must route to short-text agent with minimal prompt."""
    from returns.result import Failure

    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Failure(Exception("Unsupported")),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.DE)

    chunks = [c async for c in translation_service.translate_text("hi", config)]

    assert "".join(chunks) == "short-agent-output"
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_called_once()
    prompt = translation_service.short_text_translation_agent.run_stream_text_safe.call_args.kwargs[
        "user_prompt"
    ]
    assert "Translate the following text into German." in prompt


@pytest.mark.asyncio
async def test_explicit_same_source_and_target_still_skips(
    translation_service: TranslationService,
) -> None:
    """Explicitly setting source=DE and target=DE must still short-circuit without calling agents."""
    config = TranslationConfig(source_language=Language.DE, target_language=Language.DE)

    chunks = [c async for c in translation_service.translate_text("Hirsch", config)]

    assert chunks == ["Hirsch"]
    translation_service.short_text_translation_agent.run_stream_text_safe.assert_not_called()
    translation_service.translation_agent.run_stream_text_safe.assert_not_called()


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

    prompt = translation_service.translation_agent.run_stream_text_safe.call_args.kwargs[
        "user_prompt"
    ]
    assert "Translate the following text into French." in prompt
    assert "from" not in prompt.split("\n", 1)[0]


class TestServiceUsesBaseAgents:
    def test_service_builds_base_agent_instances(self, app_config) -> None:
        from dcc_backend_common.llm_agent import BaseAgent

        service = TranslationService(app_config, TextChunkService(), lambda: MagicMock())
        assert isinstance(service.translation_agent, BaseAgent)
        assert isinstance(service.short_text_translation_agent, BaseAgent)

    @pytest.mark.asyncio
    async def test_aclose_closes_both_agents(self, app_config) -> None:
        service = TranslationService(app_config, TextChunkService(), lambda: MagicMock())
        service.translation_agent.close = AsyncMock()
        service.short_text_translation_agent.close = AsyncMock()

        await service.aclose()

        service.translation_agent.close.assert_awaited_once()
        service.short_text_translation_agent.close.assert_awaited_once()

    def test_module_does_not_import_pydantic_ai(self) -> None:
        from pathlib import Path

        import bs_translator_backend.services.translation_service as mod

        assert "pydantic_ai" not in Path(mod.__file__).read_text()

    def test_user_message_has_no_no_think_marker(self, app_config) -> None:
        service = TranslationService(app_config, TextChunkService(), lambda: MagicMock())
        config = TranslationConfig(
            target_language=Language.FR,
            source_language=Language.DE,
            domain="",
            tone="",
            glossary="",
            context="",
        )
        for build in (service._create_user_message, service._create_short_text_user_message):
            assert "/no_think" not in build(text="Hirsch", translation_config=config)


class TestLlmUnavailable:
    """TranslationUnavailableError from an agent (vLLM overloaded/down) must surface as a
    clean, typed ApiErrorException rather than an opaque or hanging failure."""

    @pytest.mark.asyncio
    async def test_failure_on_first_chunk_raises_api_error_exception(
        self, translation_service: TranslationService
    ) -> None:
        from bs_translator_backend.agents.errors import TranslationUnavailableError
        from bs_translator_backend.models.error_response import ApiErrorException

        async def failing_stream(user_prompt: str, delta: bool = True):
            raise TranslationUnavailableError("vLLM overloaded")
            yield  # pragma: no cover - makes this an async generator function

        translation_service.translation_agent.run_stream_text_safe = MagicMock(
            side_effect=failing_stream
        )
        config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

        with pytest.raises(ApiErrorException) as exc_info:
            [
                c
                async for c in translation_service.translate_text(
                    "Dies ist ein langer Satz für Tests.", config
                )
            ]

        assert exc_info.value.error_response["errorId"] == "translation_service_unavailable"
        assert exc_info.value.error_response["status"] == 503

    @pytest.mark.asyncio
    async def test_failure_on_a_later_chunk_still_raises_after_earlier_chunks_streamed(
        self, app_config: AppConfig
    ) -> None:
        """Multi-chunk input (e.g. a long document) where only a later chunk fails: the
        caller must still see the typed error, not a silently truncated translation."""
        from bs_translator_backend.agents.errors import TranslationUnavailableError
        from bs_translator_backend.models.error_response import ApiErrorException

        call_count = 0

        async def flaky_stream(user_prompt: str, delta: bool = True):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "first chunk ok"
                return
            raise TranslationUnavailableError("vLLM overloaded")
            yield  # pragma: no cover - makes this an async generator function

        service = TranslationService(
            app_config,
            # A tiny max_tokens forces the two-line input below into two chunks.
            TextChunkService(max_tokens=1),
            lambda: MagicMock(),
        )
        service.translation_agent.run_stream_text_safe = MagicMock(side_effect=flaky_stream)
        config = TranslationConfig(source_language=Language.DE, target_language=Language.FR)

        received: list[str] = []
        with pytest.raises(ApiErrorException) as exc_info:
            async for chunk in service.translate_text("first line\nsecond line", config):
                received.append(chunk)

        assert received == ["first chunk ok"]
        assert exc_info.value.error_response["errorId"] == "translation_service_unavailable"
