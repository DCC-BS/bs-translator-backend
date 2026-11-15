"""
Unit tests for the transcription service language handling.
"""

from collections.abc import AsyncGenerator
from io import BytesIO
from typing import Any, cast

import httpx
import pytest

from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, Language, LanguageOrAuto
from bs_translator_backend.services.transcription_service import (
    TranscriptionService,
    transform_language_code_for_whisper,
)
from bs_translator_backend.utils.app_config import AppConfig

_WHISPER_URL = "http://whisper"


class _FakeResponse:
    def __init__(self, chunks: tuple[str, ...]) -> None:
        self._chunks = chunks
        self.status_code = 200
        self.is_error = False

    async def aiter_text(self) -> AsyncGenerator[str, None]:
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _RecordingClient:
    """Stand-in for httpx.AsyncClient that records the streamed request."""

    def __init__(self, chunks: tuple[str, ...] = ("hello",)) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _FakeStreamContext(_FakeResponse(self.chunks))


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        llm_url="http://openai",
        llm_api_key="test-key",
        llm_model="gpt-4",
        client_url="http://client",
        docling_url="http://docling",
        docling_api_key="test-key",
        hmac_secret="secret",
        whisper_url=_WHISPER_URL,
    )


async def _transcribe(config: AppConfig, language: LanguageOrAuto) -> dict[str, Any]:
    """Run a transcription against a fake client and return the sent form data."""
    service = TranscriptionService(config)
    client = _RecordingClient()
    service.client = cast("httpx.AsyncClient", client)

    chunks = [chunk async for chunk in service.transcribe(BytesIO(b"audio"), language)]

    assert chunks == list(client.chunks)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"] == f"{_WHISPER_URL}/audio/transcriptions/stream"
    return call["data"]


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

    @pytest.mark.parametrize("language", [Language.ZH_CN, Language.ZH_TW, Language.ZH_HK])
    def test_chinese_variants_are_collapsed(self, language: Language) -> None:
        assert transform_language_code_for_whisper(language.value) == "zh"

    def test_known_codes_pass_through(self) -> None:
        assert transform_language_code_for_whisper(Language.KA.value) == "ka"
        assert transform_language_code_for_whisper(Language.PS.value) == "ps"
        assert transform_language_code_for_whisper(Language.DE.value) == "de"


class TestTranscribeRequest:
    """Tests for the language field the service sends to Whisper."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            (Language.DE, "de"),
            (Language.EN_GB, "en"),
            (Language.EN_US, "en"),
            (Language.FA_AF, "fa"),
        ],
    )
    async def test_supported_languages_are_sent(
        self, config: AppConfig, language: Language, expected: str
    ) -> None:
        data = await _transcribe(config, language)
        assert data["language"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", [Language.KU, Language.TI])
    async def test_unsupported_languages_omit_the_language_key(
        self, config: AppConfig, language: Language
    ) -> None:
        data = await _transcribe(config, language)
        assert "language" not in data

    @pytest.mark.asyncio
    async def test_auto_omits_the_language_key(self, config: AppConfig) -> None:
        data = await _transcribe(config, DetectLanguage.AUTO)
        assert "language" not in data


def _service_with_transport(config: AppConfig, handler: Any) -> TranscriptionService:
    service = TranscriptionService(config)
    service.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return service


async def _collect(service: TranscriptionService) -> list[str]:
    return [chunk async for chunk in service.transcribe(BytesIO(b"audio"), DetectLanguage.AUTO)]


class TestWhisperErrors:
    """Whisper failures must raise a typed ApiErrorException, never stream as a transcript."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("whisper_status", "error_id", "status"),
        [
            (500, "transcription_service_unavailable", 503),
            (503, "transcription_service_unavailable", 503),
            (429, "transcription_service_unavailable", 503),
            (400, "transcription_failed", 422),
            (415, "transcription_failed", 422),
            (413, "file_too_large", 413),
        ],
    )
    async def test_error_status_is_mapped(
        self, config: AppConfig, whisper_status: int, error_id: str, status: int
    ) -> None:
        service = _service_with_transport(
            config, lambda _request: httpx.Response(whisper_status, text="internal-host:1234 boom")
        )

        with pytest.raises(ApiErrorException) as exc_info:
            await _collect(service)

        error = exc_info.value.error_response
        assert error["errorId"] == error_id
        assert error["status"] == status
        assert "internal-host" not in (error.get("debugMessage") or "")

    @pytest.mark.asyncio
    async def test_connect_error_is_unavailable(self, config: AppConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(ApiErrorException) as exc_info:
            await _collect(_service_with_transport(config, handler))

        assert exc_info.value.error_response["errorId"] == "transcription_service_unavailable"
        assert exc_info.value.error_response["status"] == 503

    @pytest.mark.asyncio
    async def test_timeout_is_unavailable(self, config: AppConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(ApiErrorException) as exc_info:
            await _collect(_service_with_transport(config, handler))

        assert exc_info.value.error_response["status"] == 503

    @pytest.mark.asyncio
    async def test_success_streams_the_transcript(self, config: AppConfig) -> None:
        service = _service_with_transport(
            config, lambda _request: httpx.Response(200, text="hello world")
        )

        assert "".join(await _collect(service)) == "hello world"

    @pytest.mark.asyncio
    async def test_aclose_closes_the_client(self, config: AppConfig) -> None:
        service = TranscriptionService(config)

        await service.aclose()

        assert service.client.is_closed
