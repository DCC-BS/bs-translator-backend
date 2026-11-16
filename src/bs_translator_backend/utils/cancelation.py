import asyncio
import contextlib
from collections.abc import Awaitable

from dcc_backend_common.logger import get_logger
from fastapi import Request

logger = get_logger(__name__)

DISCONNECT_POLL_INTERVAL_SECONDS = 0.25


class ClientDisconnectedError(Exception):
    """Raised when the client went away while the server was still working on its request."""


async def run_until_disconnected[T](
    request: Request,
    awaitable: Awaitable[T],
    poll_interval: float = DISCONNECT_POLL_INTERVAL_SECONDS,
) -> T:
    """Await `awaitable`, cancelling it as soon as the client disconnects.

    Uvicorn does not cancel a handler when the client goes away, so long-running work
    (docling conversions, the first LLM token) would otherwise keep running for a
    response nobody reads. The work runs as its own task; this coroutine polls
    `request.is_disconnected()` alongside it.

    Raises:
        ClientDisconnectedError: the client disconnected before the work finished.
            The work task has been cancelled and awaited by then.

    If the caller itself is cancelled, the work task is cancelled and awaited too, so it
    never outlives the request (e.g. keeps using an HTTP client that is being closed).
    """
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_interval)
            if done:
                return task.result()
            if await request.is_disconnected():
                logger.info(
                    "Client disconnected, cancelling request",
                    method=request.method,
                    path=request.url.path,
                )
                raise ClientDisconnectedError
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
