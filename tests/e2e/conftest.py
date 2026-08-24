"""Shared fixtures for the deterministic (Tier 1) e2e test suite.

These tests drive the *real* FastAPI app -- built via the factory functions in
``bs_translator_backend/app.py`` -- through an ASGI client (``httpx.ASGITransport``).
Nothing about the app, the dependency-injector container, the translation
service, pydantic-ai, or the OpenAI SDK is mocked: only the LLM's HTTP
transport is faked, via ``httpx.MockTransport``.

How the stub is wired in
-------------------------
``TranslationService.__init__`` builds two translation agents (``TranslationAgent``
and ``ShortTextTranslationAgent``) subclassing ``BaseAgent``, which constructs
a real ``openai.AsyncOpenAI`` client:

    client = AsyncOpenAI(max_retries=..., base_url=app_config.llm_url, api_key=app_config.llm_api_key, ...)

There is no parameter to inject a custom transport, and none was added --
instead, the ``AsyncOpenAI`` name imported into
``dcc_backend_common.llm_agent.base_agent`` is monkeypatched (see
``fake_llm`` below) to a factory that forwards to the *real* ``openai.AsyncOpenAI``
class, adding only ``http_client=httpx.AsyncClient(transport=httpx.MockTransport(...))``.
Production code is unmodified: ``BaseAgent`` runs exactly as shipped and ends up
holding a genuine ``AsyncOpenAI`` instance -- just one whose socket is replaced
by an in-process handler. Every layer above that (pydantic-ai's ``OpenAIChatModel``,
the OpenAI SDK's request building and SSE parsing, ``TranslationService``, and the
FastAPI routers) executes for real.

Env vars and import-time config
--------------------------------
Importing ``bs_translator_backend.container`` (transitively imported by
``bs_translator_backend.app``) evaluates ``AppConfig.from_env()`` and
``init_logger(...)`` at *module import time*. Dummy env vars are installed
below, before any package import, purely so collecting this test package
never depends on a real ``.env`` file or real secrets being present. The
config actually used by the built app comes from the ``app_config`` fixture,
applied via ``container.app_config.override(...)``.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable

# Must happen before any bs_translator_backend import: Container.app_config
# is `providers.Object(AppConfig.from_env())`, evaluated at class-body
# (i.e. import) time, and app.py's `init_logger` needs IS_PROD.
os.environ.setdefault("LLM_URL", "http://llm.invalid/v1")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_MODEL", "test/test-model")
os.environ.setdefault("CLIENT_URL", "http://localhost:3000")
os.environ.setdefault("DOCLING_URL", "http://docling.invalid/v1")
os.environ.setdefault("DOCLING_API_KEY", "test-docling-key")
os.environ.setdefault("HMAC_SECRET", "test-secret")
os.environ.setdefault("WHISPER_URL", "http://whisper.invalid/v1")
os.environ.setdefault("IS_PROD", "false")

import dcc_backend_common.llm_agent.base_agent as base_agent_module
import httpx
import pytest
import pytest_asyncio
from dcc_backend_common.fastapi_logging_middleware import add_logging_middleware
from dcc_backend_common.logger import get_logger
from fastapi import FastAPI
from openai import AsyncOpenAI as RealAsyncOpenAI

from bs_translator_backend import app as app_module
from bs_translator_backend.utils.app_config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    """Static test config so e2e tests do not depend on a local environment or real secrets.

    Field values mirror `tests/integration/conftest.py`.
    """
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


def _sse_event(payload: dict) -> bytes:
    """Encode one payload as an SSE `data:` frame, as the LLM endpoint would."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk(*, chunk_id: str, model: str, delta: dict, finish_reason: str | None) -> dict:
    """Build one OpenAI `chat.completion.chunk` body."""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _split_for_streaming(text: str) -> list[str]:
    """Split into >=2 pieces (when possible) so tests can prove chunks really stream."""
    if len(text) < 2:
        return [text]
    mid = len(text) // 2
    return [text[:mid], text[mid:]]


def stream_response(text: str, model: str = "test/test-model") -> httpx.Response:
    """Build an OpenAI-shaped streaming chat-completion HTTP response for `text`."""
    chunk_id = "chatcmpl-e2e-test"
    pieces = _split_for_streaming(text)
    body = b"".join([
        _sse_event(
            _chunk(
                chunk_id=chunk_id,
                model=model,
                delta={"role": "assistant", "content": ""},
                finish_reason=None,
            )
        ),
        *(
            _sse_event(
                _chunk(chunk_id=chunk_id, model=model, delta={"content": piece}, finish_reason=None)
            )
            for piece in pieces
        ),
        _sse_event(_chunk(chunk_id=chunk_id, model=model, delta={}, finish_reason="stop")),
        b"data: [DONE]\n\n",
    ])
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def error_response(status_code: int, message: str) -> httpx.Response:
    """Build an OpenAI-shaped error HTTP response."""
    return httpx.Response(
        status_code, json={"error": {"message": message, "type": "server_error", "code": None}}
    )


class FakeLLM:
    """Records outbound chat-completion requests and returns a scripted response.

    Defaults to echoing a fixed stub translation; call `respond_with` or
    `fail` to script a different response for the next (and subsequent)
    requests.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._respond: Callable[[httpx.Request], httpx.Response] = lambda _request: stream_response(
            "stub-translation"
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    def respond_with(self, text: str) -> None:
        self._respond = lambda _request: stream_response(text)

    def fail(self, status_code: int = 500, message: str = "upstream boom") -> None:
        self._respond = lambda _request: error_response(status_code, message)

    def request_bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]

    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @staticmethod
    def system_prompt(body: dict) -> str:
        return next(m["content"] for m in body["messages"] if m["role"] == "system")

    @staticmethod
    def user_message(body: dict) -> str:
        return next(m["content"] for m in body["messages"] if m["role"] == "user")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Stub the LLM at the HTTP transport layer (see module docstring)."""
    fake = FakeLLM()

    def _factory(*args: object, **kwargs: object) -> RealAsyncOpenAI:
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        # Real retries would add real backoff sleeps to the error-path test;
        # this is a test-double concern, not a change to production behaviour.
        kwargs["max_retries"] = 0
        return RealAsyncOpenAI(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(base_agent_module, "AsyncOpenAI", _factory)
    return fake


@pytest.fixture
def built_app(app_config: AppConfig, fake_llm: FakeLLM) -> FastAPI:
    """Build a real app via app.py's own factory functions.

    `fake_llm` is requested (even though unused directly) so the
    `AsyncOpenAI` monkeypatch is active *before* the DI container builds the
    translation agents below.
    """
    logger = get_logger("e2e-test")

    app = app_module._build_fastapi_app()
    app_module._register_exception_handlers(app)

    container = app_module._configure_container(app=app, logger=logger)
    container.app_config.override(app_config)
    resolved_config = container.app_config()

    app_module._register_health_routes(app=app, config=resolved_config)
    app_module._configure_cors(app=app, client_url=resolved_config.client_url, logger=logger)
    add_logging_middleware(app)
    app_module._register_routes(app=app, logger=logger)

    return app


@pytest_asyncio.fixture
async def client(built_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An ASGI-backed httpx client for the real app, real routers included.

    The app's lifespan runs around the requests so shutdown (which closes the
    agents' HTTP clients) is exercised too.
    """
    transport = httpx.ASGITransport(app=built_app)
    async with (
        built_app.router.lifespan_context(built_app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac


CLIENT_HEADERS = {"x-client-id": "e2e-test-client"}
