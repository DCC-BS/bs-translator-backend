from fastapi import status
from starlette.datastructures import UploadFile

from bs_translator_backend.models.error_codes import FILE_TOO_LARGE
from bs_translator_backend.models.error_response import ApiErrorException

_READ_CHUNK_BYTES = 1024 * 1024


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload into memory without blocking the event loop, enforcing a size limit.

    `UploadFile.read()` runs the (possibly on-disk) spooled file read in a thread. Reading
    in chunks stops as soon as the limit is exceeded instead of buffering the whole file.

    Raises:
        ApiErrorException: 413 `file_too_large` when the upload exceeds `max_bytes`.
    """
    if upload.size is not None and upload.size > max_bytes:
        raise _too_large(upload.size, max_bytes)

    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise _too_large(total, max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _too_large(size: int, max_bytes: int) -> ApiErrorException:
    return ApiErrorException({
        "errorId": FILE_TOO_LARGE,
        "status": status.HTTP_413_CONTENT_TOO_LARGE,
        "debugMessage": f"Upload of {size} bytes exceeds the limit of {max_bytes} bytes",
    })
