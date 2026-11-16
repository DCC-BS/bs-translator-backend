import asyncio
import re
import time
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any, Self, final

import httpx
from dcc_backend_common.logger import get_logger
from fastapi import status
from starlette.datastructures import UploadFile

from bs_translator_backend.models.conversion_result import Base64EncodedImage, ConversionResult
from bs_translator_backend.models.docling_response import (
    DoclingDocument,
    DoclingResponse,
    DocumentResponse,
)
from bs_translator_backend.models.error_codes import (
    DOCLING_TASK_FAILED,
    DOCLING_TIMEOUT,
    INVALID_MIME_TYPE,
    NO_DOCUMENT,
    UNEXPECTED_ERROR,
)
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.uploads import read_upload_limited

logger = get_logger(__name__)


def get_mimetype(path_source: Path) -> str:
    """Get MIME type based on file extension."""

    extension = path_source.suffix.lower()
    mimetypes = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".html": "text/html",
        ".htm": "text/html",
        ".xhtml": "application/xhtml+xml",
        ".adoc": "text/asciidoc",
        ".asciidoc": "text/asciidoc",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".txt": "text/plain",
    }

    resolved = mimetypes.get(extension, "invalid")
    logger.debug(
        "Determined MIME type for file",
        mimetype=resolved,
        extension=extension,
        path=str(path_source),
    )
    return resolved


def validate_mimetype(mimetype: str, logger_context: dict[str, Any]) -> None:
    if not mimetype or mimetype == "invalid":
        logger.error("Invalid MIME type", mimetype=mimetype, **logger_context)
        raise ApiErrorException({
            "errorId": INVALID_MIME_TYPE,
            "status": status.HTTP_400_BAD_REQUEST,
            "debugMessage": f"Invalid MIME type: {mimetype!r}",
        })


# Image types accepted for photo OCR. The browser re-encodes camera captures and
# gallery uploads to JPEG, the others are there for API clients.
IMAGE_MIMETYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
})


def extract_docling_document(
    response: dict[str, Any], content_field: str, logger_context: dict[str, Any]
) -> DocumentResponse:
    """Validate a docling-serve conversion result and check the requested output is present."""
    docling_response = DoclingResponse.model_validate(response)
    if getattr(docling_response.document, content_field) is None:
        logger.error(
            "Docling response does not contain the requested content",
            content_field=content_field,
            status=docling_response.status,
            errors=[error.error_message for error in docling_response.errors],
            **logger_context,
        )
        raise ApiErrorException({
            "errorId": NO_DOCUMENT,
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "debugMessage": f"Document conversion returned no {content_field}",
        })

    return docling_response.document


def _docling_auth_headers(api_key: str) -> dict[str, str]:
    """Headers authenticating against docling-serve (its own X-Api-Key) or a gateway (Bearer)."""
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-Api-Key": api_key}


# Bodies above this size are JSON-decoded in a worker thread.
_PARSE_IN_THREAD_BYTES = 512 * 1024

_EMBEDDED_IMAGE = re.compile(r"(!\[[^\]]*\]\()data:image/[^;]+;base64,([^)]+)\)")


def extract_embedded_images(markdown: str) -> tuple[str, dict[int, Base64EncodedImage]]:
    """Replace embedded base64 images by `image<n>.png` references, in one pass."""
    images: dict[int, Base64EncodedImage] = {}

    def _replace(match: re.Match[str]) -> str:
        idx = len(images)
        images[idx] = match.group(2)
        return f"{match.group(1)}image{idx}.png)"

    return _EMBEDDED_IMAGE.sub(_replace, markdown), images


@final
class DocumentConversionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers=_docling_auth_headers(config.docling_api_key),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if it is still open."""

        if self.client.is_closed:
            return

        await self.client.aclose()

    async def __aenter__(self) -> Self:
        """Return the service instance for use within an async context."""

        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Ensure the HTTP client is closed when leaving an async context."""

        await self.aclose()

    async def _make_request(
        self, method: str, url: str, context: str, **kwargs: Any
    ) -> httpx.Response:
        """Execute an HTTP request, raising ApiErrorException on network errors or non-2xx status."""
        try:
            response = await self.client.request(method, url, **kwargs)
        except httpx.TimeoutException as e:
            logger.exception("Request timed out", context=context)
            raise ApiErrorException({
                "errorId": DOCLING_TIMEOUT,
                "status": status.HTTP_504_GATEWAY_TIMEOUT,
                "debugMessage": f"{context} timed out",
            }) from e
        except httpx.RequestError as e:
            logger.exception("Request error", context=context)
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"{context} request error: {e!s}",
            }) from e

        if not (200 <= response.status_code < 300):
            logger.error(
                "Request failed",
                context=context,
                status_code=response.status_code,
                response_text=response.text,
            )
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"{context} failed with status {response.status_code}",
            })

        return response

    async def _parse_json(self, response: httpx.Response, context: str) -> dict[str, Any]:
        """Parse response JSON, raising ApiErrorException if the body is not valid JSON.

        Conversion results can be many megabytes; those are parsed in a thread to keep
        the event loop (and every other request's stream) responsive.
        """
        try:
            if len(response.content) > _PARSE_IN_THREAD_BYTES:
                return await asyncio.to_thread(response.json)
            return response.json()
        except ValueError as e:
            logger.exception("Response body is not valid JSON", context=context)
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"{context} response is not valid JSON",
            }) from e

    async def _resolve_file(
        self,
        file: UploadFile | BytesIO | bytes,
        filename: str | None,
        content_type: str | None,
    ) -> tuple[bytes, str, str]:
        """Read file content and resolve filename and MIME type.

        Returns (content, filename, content_type). Raises ApiErrorException
        for unsupported MIME types and uploads above the configured size limit.
        """
        if isinstance(file, UploadFile):
            content = await read_upload_limited(file, self.config.max_upload_bytes)
            filename = filename or file.filename or "uploaded_document"
        elif isinstance(file, bytes):
            content = file
            filename = filename or "uploaded_document"
        else:
            content = file.read()
            filename = filename or "uploaded_document"

        if content_type is None:
            content_type = get_mimetype(Path(filename))

        validate_mimetype(content_type, logger_context={"content_type": content_type})
        return content, filename, content_type

    async def _submit_async_convert(
        self,
        files: dict[str, tuple[str, BytesIO, str]],
        options: dict[str, str | list[str] | bool],
    ) -> str:
        """POST /convert/file/async and return the task_id string."""
        response = await self._make_request(
            "POST",
            self.config.docling_url + "/convert/file/async",
            "async submit",
            files=files,
            data=options,
        )
        body = await self._parse_json(response, "async submit")

        task_id = body.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            logger.error("Docling async submit response missing task_id", body=body)
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"Docling async submit response missing task_id: {body}",
            })

        logger.debug("Docling async task submitted", task_id=task_id)
        return task_id

    _POLL_VALID_STATES = frozenset({"pending", "started", "success", "failure"})

    async def _poll_task(self, task_id: str) -> None:
        """Poll /status/poll/{task_id} until the task succeeds or fails.

        Raises ApiErrorException on timeout (DOCLING_TIMEOUT), task failure
        (DOCLING_TASK_FAILED), or an unrecognised task_status value. Note:
        docling-serve has no cancel endpoint, so on timeout the remote task
        keeps running.
        """
        deadline = time.monotonic() + self.config.docling_task_timeout
        poll_url = self.config.docling_url + f"/status/poll/{task_id}"

        while True:
            if time.monotonic() >= deadline:
                logger.error(
                    "Docling task timed out; remote task cannot be cancelled "
                    "(no cancel API in docling-serve)",
                    task_id=task_id,
                    timeout_seconds=self.config.docling_task_timeout,
                )
                raise ApiErrorException({
                    "errorId": DOCLING_TIMEOUT,
                    "status": status.HTTP_504_GATEWAY_TIMEOUT,
                    "debugMessage": f"Docling task {task_id} did not complete within {self.config.docling_task_timeout}s",
                })

            poll_response = await self._make_request("GET", poll_url, f"poll task {task_id}")
            body = await self._parse_json(poll_response, f"poll task {task_id}")
            task_status: str = body.get("task_status", "")
            logger.debug("Docling task status polled", task_id=task_id, task_status=task_status)

            if task_status not in self._POLL_VALID_STATES:
                logger.error(
                    "Docling task returned unexpected status",
                    task_id=task_id,
                    task_status=task_status,
                )
                raise ApiErrorException({
                    "errorId": UNEXPECTED_ERROR,
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "debugMessage": f"Docling task {task_id} returned unexpected status: {task_status!r}",
                })

            if task_status == "success":
                return
            if task_status == "failure":
                raise ApiErrorException({
                    "errorId": DOCLING_TASK_FAILED,
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "debugMessage": f"Docling task {task_id} failed",
                })

            await asyncio.sleep(self.config.docling_poll_interval)

    async def _fetch_result(self, task_id: str) -> httpx.Response:
        """GET /result/{task_id} and return the raw response."""
        return await self._make_request(
            "GET",
            self.config.docling_url + f"/result/{task_id}",
            f"result fetch {task_id}",
        )

    async def fetch_docling_file_convert(
        self,
        files: dict[str, tuple[str, BytesIO, str]],
        options: dict[str, str | list[str] | bool],
    ) -> httpx.Response:
        task_id = await self._submit_async_convert(files, options)
        await self._poll_task(task_id)
        return await self._fetch_result(task_id)

    @staticmethod
    def _ocr_languages(source_lang: LanguageOrAuto) -> list[str]:
        if source_lang == DetectLanguage.AUTO:
            return ["de", "en", "fr", "it"]
        if source_lang.value.startswith("en"):
            return ["en"]
        return [source_lang.value.split("-")[0]]

    async def _docling_convert(
        self,
        content: bytes,
        filename: str,
        content_type: str,
        content_field: str,
        docling_options: dict[str, str | list[str] | bool],
    ) -> DocumentResponse:
        files = {"files": (filename, BytesIO(content), content_type)}
        response = await self.fetch_docling_file_convert(files, docling_options)
        return extract_docling_document(
            await self._parse_json(response, f"{content_field} result"),
            content_field=content_field,
            logger_context={"options": docling_options, "content_type": content_type},
        )

    async def ocr_image(
        self,
        content: bytes,
        source_lang: LanguageOrAuto,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> DoclingDocument:
        """OCR a photo with the configured OCR engine (GLM-OCR by default).

        The photo is OCR'd per layout region, embedded pictures are not exported (only
        text and its position are needed), tables keep their cell structure.
        """
        filename = filename or "photo.jpg"
        content_type = content_type or get_mimetype(Path(filename))
        if content_type not in IMAGE_MIMETYPES:
            logger.error("Unsupported image type for OCR", content_type=content_type)
            raise ApiErrorException({
                "errorId": INVALID_MIME_TYPE,
                "status": status.HTTP_400_BAD_REQUEST,
                "debugMessage": f"Unsupported image type: {content_type!r}",
            })

        options: dict[str, str | list[str] | bool] = {
            "to_formats": ["json"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            # No `force_ocr`: with it docling OCRs the photo as one full-page crop, and
            # GLM-OCR then returns all text as one block without positions. Without it
            # each layout region is its own crop, so every block keeps its box (an image
            # has no text layer, so nothing is skipped).
            "force_ocr": False,
            "ocr_preset": self.config.docling_image_ocr_preset,
            "ocr_lang": self._ocr_languages(source_lang),
            "do_table_structure": True,
            "table_mode": "accurate",
        }
        if self.config.docling_image_layout_preset:
            options["layout_preset"] = self.config.docling_image_layout_preset

        document = await self._docling_convert(
            content, filename, content_type, "json_content", options
        )
        return await asyncio.to_thread(DoclingDocument.model_validate, document.json_content)

    async def convert(
        self,
        file: UploadFile | BytesIO | bytes,
        source_lang: LanguageOrAuto,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> ConversionResult:
        """Convert a document to markdown; embedded images are returned separately."""
        content, filename, content_type = await self._resolve_file(file, filename, content_type)
        options: dict[str, str | list[str] | bool] = {
            "to_formats": ["md"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "images_scale": "1",
            "ocr_lang": self._ocr_languages(source_lang),
            "table_mode": "accurate",
            "pdf_backend": "docling_parse",
            "ocr_preset": "rapidocr",
            "md_compact_tables": True,
        }
        document = await self._docling_convert(
            content, filename, content_type, "md_content", options
        )
        markdown, images = await asyncio.to_thread(
            extract_embedded_images, document.md_content or ""
        )
        return ConversionResult(markdown=markdown, images=images)
