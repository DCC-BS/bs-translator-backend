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

import httpx
import pytest

from tests.e2e.conftest import CLIENT_HEADERS, FakeLLM


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
    prompt and no "from" clause.
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
    async def test_long_input_uses_long_form_agent_without_source_language_clause(
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
        assert "Translate the following text into French." in user_message
        assert "from German" not in user_message


class TestErrorPaths:
    """The stub LLM returns an HTTP error; the failure must not hang the
    request and must not be silently swallowed."""

    @pytest.mark.asyncio
    async def test_llm_http_error_does_not_hang_and_is_not_silently_swallowed(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
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

        # `StreamingResponse` commits its 200 status as soon as the ASGI app is
        # called, before the body generator runs -- so an LLM failure during
        # streaming cannot change the status code. The observable, current
        # behaviour is that the failure propagates as a raised exception
        # through the ASGI transport (matching what a real ASGI server would
        # see: the connection breaks rather than the client receiving a wrong
        # "200 OK, translation succeeded"). The important properties this
        # test protects are: (a) no hang -- bounded by the timeout below, and
        # (b) not a quiet, unnoticed failure -- the exception carries the
        # upstream status.
        with pytest.raises(Exception) as exc_info:
            await asyncio.wait_for(make_request(), timeout=5.0)

        assert "500" in str(exc_info.value)
