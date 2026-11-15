import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from fastapi import Request

from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.routers.translation_route import _stream_with_error_trailer
from bs_translator_backend.utils.streaming import (
    STREAM_ERROR_SEPARATOR,
    peek_first,
    stream_error_trailer,
)


class _ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


_REQUEST = cast("Request", _ConnectedRequest())


async def _stream(source: AsyncGenerator[str, None]) -> str:
    peeked = await peek_first(source)
    return "".join([chunk async for chunk in _stream_with_error_trailer(_REQUEST, peeked)])


def _trailer(body: str) -> dict[str, Any]:
    _, separator, trailer = body.partition(STREAM_ERROR_SEPARATOR)
    assert separator, "stream must end with the error trailer"
    return json.loads(trailer)


@pytest.mark.asyncio
async def test_successful_stream_has_no_trailer() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "Bonjour "
        yield "le monde"

    assert await _stream(source()) == "Bonjour le monde"


@pytest.mark.asyncio
async def test_error_after_first_chunk_ends_stream_with_trailer() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "Bonjour"
        raise ApiErrorException({
            "errorId": "translation_service_unavailable",
            "status": 503,
            "debugMessage": "http://internal-vllm:8000 body",
        })

    body = await _stream(source())

    assert body.startswith("Bonjour" + STREAM_ERROR_SEPARATOR)
    assert _trailer(body) == {"errorId": "translation_service_unavailable", "status": 503}
    assert "internal-vllm" not in body


@pytest.mark.asyncio
async def test_unexpected_error_after_first_chunk_is_unexpected_error() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "Bonjour"
        raise RuntimeError("boom")

    assert _trailer(await _stream(source())) == {"errorId": "unexpected_error", "status": 500}


@pytest.mark.asyncio
async def test_separator_in_llm_output_is_stripped() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield f"a{STREAM_ERROR_SEPARATOR}b"
        yield STREAM_ERROR_SEPARATOR

    assert await _stream(source()) == "ab"


def test_trailer_defaults_to_unexpected_error() -> None:
    assert stream_error_trailer() == STREAM_ERROR_SEPARATOR + json.dumps({
        "errorId": "unexpected_error",
        "status": 500,
    })


@pytest.mark.asyncio
async def test_transcription_error_after_first_chunk_ends_with_trailer() -> None:
    from bs_translator_backend.routers.transcription_route import _stream_transcript

    async def source() -> AsyncGenerator[str, None]:
        yield "Guten Tag"
        raise ApiErrorException({"errorId": "transcription_service_unavailable", "status": 503})

    peeked = await peek_first(source())
    body = "".join([chunk async for chunk in _stream_transcript(_REQUEST, peeked)])

    assert body.startswith("Guten Tag" + STREAM_ERROR_SEPARATOR)
    assert _trailer(body) == {"errorId": "transcription_service_unavailable", "status": 503}
