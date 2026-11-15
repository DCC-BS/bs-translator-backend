"""Every error the API returns is JSON `{errorId, status, debugMessage?}`."""

import os

# app.py builds the app (and its logger, which needs IS_PROD) at import time.
os.environ.setdefault("IS_PROD", "false")

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from bs_translator_backend.app import _build_fastapi_app, _register_exception_handlers
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.translation import TranslationInput


class _Body(BaseModel):
    count: int


@pytest.fixture
def app() -> FastAPI:
    app = _build_fastapi_app()
    _register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal-host:1234 detail")

    @app.get("/api-error")
    def api_error() -> None:
        raise ApiErrorException({"errorId": "no_document", "status": 400})

    @app.post("/validated")
    def validated(body: _Body) -> int:
        return body.count

    @app.post("/translation-input")
    def translation_input(body: TranslationInput) -> str:
        return body.text

    return app


async def _request(app: FastAPI, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_unhandled_exception_is_json_500_without_details(app: FastAPI) -> None:
    response = await _request(app, "GET", "/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["errorId"] == "unexpected_error"
    assert body["status"] == 500
    assert "internal-host" not in response.text


@pytest.mark.asyncio
async def test_api_error_exception_is_passed_through(app: FastAPI) -> None:
    response = await _request(app, "GET", "/api-error")

    assert response.status_code == 400
    assert response.json() == {"errorId": "no_document", "status": 400}


@pytest.mark.asyncio
async def test_validation_error_is_422_invalid_request_with_detail(app: FastAPI) -> None:
    response = await _request(app, "POST", "/validated", json={"count": "not a number"})

    assert response.status_code == 422
    body = response.json()
    assert body["errorId"] == "invalid_request"
    assert body["status"] == 422
    assert body["detail"][0]["loc"] == ["body", "count"]


@pytest.mark.asyncio
async def test_invalid_translation_config_is_422(app: FastAPI) -> None:
    response = await _request(
        app,
        "POST",
        "/translation-input",
        json={"text": "Hallo", "config": {"target_language": "xx"}},
    )

    assert response.status_code == 422
    assert response.json()["errorId"] == "invalid_request"


@pytest.mark.asyncio
async def test_unknown_route_is_json_404(app: FastAPI) -> None:
    response = await _request(app, "GET", "/does-not-exist")

    assert response.status_code == 404
    assert response.json()["errorId"] == "not_found"
