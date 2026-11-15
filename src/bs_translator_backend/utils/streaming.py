"""Helpers for plain-text streaming responses."""

import json
from collections.abc import AsyncGenerator

from bs_translator_backend.models.error_codes import UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ErrorResponse

# Marks the end of the text in a stream that failed after its first chunk was sent: the
# stream then ends with this character followed by a JSON object
# `{"errorId": ..., "status": ...}`. The text itself never contains it.
STREAM_ERROR_SEPARATOR = "\x1e"


async def peek_first[T](
    source: AsyncGenerator[T, None],
) -> tuple[T, AsyncGenerator[T, None]] | None:
    """Pull the first item out of `source`, handing back a generator that still yields it.

    Awaiting the first item here - before the route returns a `StreamingResponse` - means
    a failure that happens immediately (e.g. vLLM already overloaded) raises inside the
    route handler itself, where FastAPI's normal exception handling still applies and can
    turn it into a proper status code + JSON body. Once headers are sent, as they are for
    every later item, an error can only be handled from within the generator body.

    Returns `None` if `source` was empty.
    """
    try:
        first = await source.__anext__()
    except StopAsyncIteration:
        return None

    async def _rest() -> AsyncGenerator[T, None]:
        # `source` is only ever iterated from here on, so closing `_rest()` (the only
        # generator the caller sees from this point) must close `source` too.
        try:
            yield first
            async for item in source:
                yield item
        finally:
            await source.aclose()

    return first, _rest()


def strip_stream_error_separator(text: str) -> str:
    """Remove `STREAM_ERROR_SEPARATOR` so streamed text can never fake an error trailer."""
    return text.replace(STREAM_ERROR_SEPARATOR, "")


def stream_error_trailer(error: ErrorResponse | None = None) -> str:
    """The trailer that ends a stream which failed after its first chunk was sent.

    Only `errorId` and `status` are sent: the debug message can carry upstream details.
    """
    error = error or {}
    return STREAM_ERROR_SEPARATOR + json.dumps({
        "errorId": error.get("errorId", UNEXPECTED_ERROR),
        "status": error.get("status", 500),
    })
