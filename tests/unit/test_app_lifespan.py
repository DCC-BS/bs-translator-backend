import os
from unittest.mock import AsyncMock

import pytest

# Set default env vars before imports because Container.app_config evaluates AppConfig.from_env() at import time
os.environ.setdefault("LLM_URL", "http://localhost:8001/v1")
os.environ.setdefault("LLM_API_KEY", "none")
os.environ.setdefault("LLM_MODEL", "test/test-model")
os.environ.setdefault("CLIENT_URL", "http://localhost:3000")
os.environ.setdefault("DOCLING_URL", "http://localhost:8004/v1")
os.environ.setdefault("DOCLING_API_KEY", "none")
os.environ.setdefault("HMAC_SECRET", "test-secret")
os.environ.setdefault("WHISPER_URL", "http://localhost:50001/v1")
os.environ.setdefault("IS_PROD", "false")

from bs_translator_backend.app import _build_fastapi_app, create_app


@pytest.mark.asyncio
async def test_lifespan_closes_translation_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown must release the agents' HTTP connection pools."""
    monkeypatch.setenv("LLM_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("LLM_API_KEY", "none")
    monkeypatch.setenv("LLM_MODEL", "test/test-model")
    monkeypatch.setenv("CLIENT_URL", "http://localhost:3000")
    monkeypatch.setenv("DOCLING_URL", "http://localhost:8004/v1")
    monkeypatch.setenv("DOCLING_API_KEY", "none")
    monkeypatch.setenv("HMAC_SECRET", "test-secret")
    monkeypatch.setenv("WHISPER_URL", "http://localhost:50001/v1")

    app = create_app()
    service = app.state.container.translation_service()
    service.aclose = AsyncMock()

    async with app.router.lifespan_context(app):
        pass

    service.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_without_container_does_not_raise() -> None:
    """A partially-configured app without container must not raise during lifespan shutdown."""
    app = _build_fastapi_app()
    async with app.router.lifespan_context(app):
        pass
