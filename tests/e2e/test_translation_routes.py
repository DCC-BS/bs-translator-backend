"""Tier 1 e2e tests: real FastAPI app, real DI container, real pydantic-ai/OpenAI
SDK code paths, LLM stubbed at the HTTP transport layer (see `conftest.py`).

These are the tests that exercise what `tests/unit/` and `tests/integration/`
cannot: the actual ASGI routes, SSE streaming, and the request bodies the
real OpenAI SDK builds and sends -- including which translation agent
(long-form vs. short-text) a given input is routed to, which is only
observable at this level because the unit tests mock the agent object
itself.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from bs_translator_backend.utils.app_config import AppConfig
from tests.e2e.conftest import CLIENT_HEADERS, FakeLLM, error_response, stream_response


class TestLanguagesRoute:
    """GET /translation/languages."""

    @pytest.mark.asyncio
    async def test_returns_supported_languages_including_serbian(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/translation/languages")

        assert response.status_code == 200
        languages = response.json()
        assert isinstance(languages, list)
        assert "sr" in languages
        assert "de" in languages
        assert "fr" in languages


class TestDetectLanguageRoute:
    """POST /translation/detect-language. No LLM call is involved -- this
    route uses fast-langdetect directly."""

    @pytest.mark.asyncio
    async def test_detects_german_text(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/translation/detect-language",
            json={
                "text": "Der Bundesrat hat heute eine wichtige Entscheidung getroffen, "
                "die viele Menschen betrifft."
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["language"] == "de"
        assert body["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_short_text_reports_auto_with_zero_confidence(
        self, client: httpx.AsyncClient
    ) -> None:
        """Below the service's 10-character floor, detection is skipped entirely."""
        response = await client.post("/translation/detect-language", json={"text": "hi"})

        assert response.status_code == 200
        body = response.json()
        assert body["language"] == "auto"
        assert body["confidence"] == 0.0


class TestTranslateTextRoute:
    """POST /translation/text -- the main streaming translation route."""

    @pytest.mark.asyncio
    async def test_streams_the_stubbed_translation(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.respond_with("Bonjour le monde")

        chunks: list[str] = []
        async with client.stream(
            "POST",
            "/translation/text",
            json={
                "text": "Hallo zusammen, wie geht es euch heute?",
                "config": {"source_language": "de", "target_language": "fr"},
            },
            headers=CLIENT_HEADERS,
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                chunks.append(chunk)

        # The stub splits its response into two SSE deltas ("Bonjour" + " le monde",
        # see `_split_for_streaming`). Reassembling to the full string proves both
        # deltas were consumed from the LLM and forwarded through the route.
        #
        # Deliberately NOT asserting len(chunks) >= 2: `aiter_text` yields per network
        # read, and httpx.ASGITransport runs in-process with no socket to fragment the
        # body, so the two deltas routinely arrive coalesced into a single read. That
        # count measures the test transport, not the application. Incremental yielding
        # is covered where it is actually observable: at the service layer in
        # tests/unit/test_translation_service.py.
        assert chunks, "expected at least one chunk from the stream"
        assert "".join(chunks) == "Bonjour le monde"

    @pytest.mark.asyncio
    async def test_response_has_streaming_headers(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.respond_with("Bonjour")

        async with client.stream(
            "POST",
            "/translation/text",
            json={
                "text": "Hallo zusammen, wie geht es euch heute?",
                "config": {"source_language": "de", "target_language": "fr"},
            },
            headers=CLIENT_HEADERS,
        ) as response:
            assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
            assert response.headers["content-type"].startswith("text/plain")
            async for _ in response.aiter_text():
                pass


class TestPromptRouting:
    """Short-vs-long prompt routing, asserted on the request actually sent to
    the (stubbed) LLM by the real pydantic-ai + OpenAI SDK code path.

    This is the regression coverage for the short-text feature: it proves
    that a 1-3 word input reaches the LLM carrying the short-text
    (dictionary-lookup) system prompt and an explicit "from X into Y"
    instruction, while a longer input carries the ordinary long-form system
    prompt.
    """

    @pytest.mark.asyncio
    async def test_short_input_uses_short_text_agent_with_asserted_source_language(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.respond_with("Cerf")

        async with client.stream(
            "POST",
            "/translation/text",
            json={
                "text": "Hirsch",
                "config": {"source_language": "de", "target_language": "fr"},
            },
            headers=CLIENT_HEADERS,
        ) as response:
            async for _ in response.aiter_text():
                pass

        assert len(fake_llm.requests) == 1
        body = fake_llm.last_body()
        system_prompt = fake_llm.system_prompt(body)
        user_message = fake_llm.user_message(body)

        assert "dictionary-style lookup" in system_prompt
        assert "Translate the following text from German into French." in user_message
        assert "Hirsch" in user_message

    @pytest.mark.asyncio
    async def test_long_input_uses_long_form_agent_with_source_language_clause(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.respond_with("Le grand cerf brun a traversé la forêt.")

        async with client.stream(
            "POST",
            "/translation/text",
            json={
                "text": "Der grosse braune Hirsch lief durch den Wald.",
                "config": {"source_language": "de", "target_language": "fr"},
            },
            headers=CLIENT_HEADERS,
        ) as response:
            async for _ in response.aiter_text():
                pass

        assert len(fake_llm.requests) == 1
        body = fake_llm.last_body()
        system_prompt = fake_llm.system_prompt(body)
        user_message = fake_llm.user_message(body)

        assert "dictionary-style lookup" not in system_prompt
        assert "senior translator and terminologist" in system_prompt
        assert "Translate the following text from German into French." in user_message

    @pytest.mark.asyncio
    async def test_short_input_auto_source_translating_into_german_uses_fallback_prompt(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.respond_with("Hallo")

        chunks: list[str] = []
        async with client.stream(
            "POST",
            "/translation/text",
            json={
                "text": "hi",
                "config": {"source_language": "auto", "target_language": "de"},
            },
            headers=CLIENT_HEADERS,
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                chunks.append(chunk)

        assert "".join(chunks) == "Hallo"
        assert len(fake_llm.requests) == 1
        body = fake_llm.last_body()
        system_prompt = fake_llm.system_prompt(body)
        user_message = fake_llm.user_message(body)

        assert "dictionary-style lookup" in system_prompt
        assert "Translate the following text into German." in user_message
        assert "from" not in user_message.split("\n", 1)[0]
        assert "hi" in user_message


class TestErrorPaths:
    """The stub LLM returns an HTTP error; the failure must not hang the
    request and must not be silently swallowed."""

    @pytest.mark.asyncio
    async def test_llm_http_error_on_first_chunk_returns_clean_503(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        """`StreamingResponse` commits its 200 status as soon as the ASGI app is
        called, before the body generator runs - so once streaming has started,
        an LLM failure cannot change the status code. The route works around
        this for the common case (the LLM is already down/overloaded before
        the first chunk) by awaiting that first chunk before constructing the
        `StreamingResponse`, so a failure there still becomes a normal,
        bounded-time JSON error response instead of a broken connection.
        """
        fake_llm.fail(status_code=500, message="upstream boom")

        async def make_request() -> httpx.Response:
            return await client.post(
                "/translation/text",
                json={
                    "text": "Der grosse braune Hirsch lief durch den Wald.",
                    "config": {"source_language": "de", "target_language": "fr"},
                },
                headers=CLIENT_HEADERS,
            )

        response = await asyncio.wait_for(make_request(), timeout=5.0)

        assert response.status_code == 503
        body = response.json()
        assert body["errorId"] == "translation_service_unavailable"
        # The upstream response body (which can name internal hosts) never leaks.
        assert "upstream boom" not in response.text

    @pytest.mark.asyncio
    async def test_context_length_error_returns_422_input_too_long(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.fail(
            status_code=400,
            message="This model's maximum context length is 8192 tokens. However, you requested 9000.",
        )

        response = await client.post(
            "/translation/text",
            json={
                "text": "Der grosse braune Hirsch lief durch den Wald.",
                "config": {"source_language": "de", "target_language": "fr"},
            },
            headers=CLIENT_HEADERS,
        )

        assert response.status_code == 422
        assert response.json()["errorId"] == "input_too_long"
        assert "maximum context length" not in response.text

    @pytest.mark.asyncio
    async def test_failure_after_first_chunk_ends_stream_with_error_trailer(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        """A long text is translated chunk by chunk; the second chunk's LLM call fails."""
        calls = 0

        def respond(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return stream_response("Premier bloc")
            return error_response(500, "upstream boom")

        fake_llm._respond = respond
        paragraph = ("Der grosse braune Hirsch lief durch den Wald. " * 70).rstrip()
        text = f"{paragraph}\n\n{paragraph}"

        response = await client.post(
            "/translation/text",
            json={"text": text, "config": {"source_language": "de", "target_language": "fr"}},
            headers=CLIENT_HEADERS,
        )

        assert response.status_code == 200
        translated, separator, trailer = response.text.partition("\x1e")
        assert translated == "Premier bloc\n\n"
        assert separator == "\x1e"
        assert json.loads(trailer) == {"errorId": "translation_service_unavailable", "status": 503}

    @pytest.mark.asyncio
    async def test_invalid_config_returns_422_invalid_request(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/translation/text",
            json={"text": "Hallo", "config": {"target_language": "xx"}},
            headers=CLIENT_HEADERS,
        )

        assert response.status_code == 422
        assert response.json()["errorId"] == "invalid_request"


class TestTranscriptionRoute:
    """POST /transcription/audio, with Whisper stubbed at the HTTP transport layer."""

    @pytest.fixture
    def app_config(self, app_config: AppConfig) -> AppConfig:
        # About 1 KB, so the size limit can be tested with a small upload.
        return app_config.model_copy(update={"max_upload_size_mb": 0.001})

    @staticmethod
    def _stub_whisper(app: FastAPI, handler) -> None:
        service = app.state.container.transcription_service()
        service.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @staticmethod
    async def _post(client: httpx.AsyncClient, content: bytes = b"audio") -> httpx.Response:
        return await client.post(
            "/transcription/audio",
            files={"audio_file": ("audio.webm", content, "audio/webm")},
            data={"language": "zh-tw"},
            headers=CLIENT_HEADERS,
        )

    @pytest.mark.asyncio
    async def test_streams_the_transcript_unbuffered(
        self, client: httpx.AsyncClient, built_app: FastAPI
    ) -> None:
        sent: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request.read())
            return httpx.Response(200, text="你好")

        self._stub_whisper(built_app, handler)

        response = await self._post(client)

        assert response.status_code == 200
        assert response.text == "你好"
        assert response.headers["x-accel-buffering"] == "no"
        assert b'name="language"\r\n\r\nzh\r\n' in sent[0]

    @pytest.mark.asyncio
    async def test_whisper_error_status_is_a_json_error(
        self, client: httpx.AsyncClient, built_app: FastAPI
    ) -> None:
        self._stub_whisper(built_app, lambda _request: httpx.Response(500, text="whisper boom"))

        response = await self._post(client)

        assert response.status_code == 503
        assert response.json()["errorId"] == "transcription_service_unavailable"
        assert "whisper boom" not in response.text

    @pytest.mark.asyncio
    async def test_whisper_unreachable_is_a_json_error(
        self, client: httpx.AsyncClient, built_app: FastAPI
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        self._stub_whisper(built_app, handler)

        response = await self._post(client)

        assert response.status_code == 503
        assert response.json()["errorId"] == "transcription_service_unavailable"

    @pytest.mark.asyncio
    async def test_upload_over_the_limit_is_413(
        self, client: httpx.AsyncClient, built_app: FastAPI
    ) -> None:
        self._stub_whisper(built_app, lambda _request: httpx.Response(200, text="never"))

        response = await self._post(client, content=b"x" * 2048)

        assert response.status_code == 413
        assert response.json()["errorId"] == "file_too_large"
