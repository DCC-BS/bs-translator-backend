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

logger = get_logger(__name__)


def get_mimetype(path_source: Path) -> str:
    """Get MIME type based on file extension."""

    extension = path_source.suffix.lower()
    mimetypes = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".html": "text/html",
        ".adoc": "text/asciidoc",
        ".md": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".txt": "text/plain",
    }

    logger.info(
        f"Determined MIME type '{mimetypes.get(extension, 'invalid')}' for extension '{extension}' and path '{path_source}'"
    )
    return mimetypes.get(extension, "invalid")


def validate_mimetype(mimetype: str, logger_context: dict[str, Any]) -> None:
    if not mimetype or mimetype == "invalid":
        logger.error(f"Invalid MIME type: {mimetype!r}", extra=logger_context)
        raise ApiErrorException({
            "errorId": INVALID_MIME_TYPE,
            "status": status.HTTP_400_BAD_REQUEST,
            "debugMessage": f"Invalid MIME type: {mimetype!r}",
        })


def extract_docling_document(
    response: dict[str, Any], logger_context: dict[str, Any]
) -> DocumentResponse:
    docling_response = DoclingResponse.model_validate(response)
    if docling_response.document.json_content is None:
        logger.error(
            "Docling response does not contain a document",
            extra=logger_context,
        )
        raise ApiErrorException({
            "errorId": NO_DOCUMENT,
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "debugMessage": "Document conversion failed the json content is None",
        })

    return docling_response.document


@final
class DocumentConversionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": config.llm_api_key},
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
            logger.exception(f"{context} timed out")
            raise ApiErrorException({
                "errorId": DOCLING_TIMEOUT,
                "status": status.HTTP_504_GATEWAY_TIMEOUT,
                "debugMessage": f"{context} timed out",
            }) from e
        except httpx.RequestError as e:
            logger.exception(f"{context} request error")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"{context} request error: {e!s}",
            }) from e

        if not (200 <= response.status_code < 300):
            logger.error(f"{context} failed ({response.status_code}): {response.text}")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"{context} failed with status {response.status_code}",
            })

        return response

    def _parse_json(self, response: httpx.Response, context: str) -> dict[str, Any]:
        """Parse response JSON, raising ApiErrorException if the body is not valid JSON."""
        try:
            return response.json()
        except ValueError as e:
            logger.exception(f"{context} returned non-JSON body")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"{context} response is not valid JSON",
            }) from e

    def _resolve_file(
        self,
        file: UploadFile | BytesIO,
        filename: str | None,
        content_type: str | None,
    ) -> tuple[bytes, str, str]:
        """Read file content and resolve filename and MIME type.

        Returns (content, filename, content_type). Raises ApiErrorException
        for unsupported MIME types.
        """
        if isinstance(file, UploadFile):
            content = file.file.read()
            filename = filename or file.filename or "uploaded_document"
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
        body = self._parse_json(response, "async submit")

        task_id = body.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            logger.error(f"Docling async submit response missing task_id: {body}")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"Docling async submit response missing task_id: {body}",
            })

        logger.info(f"Docling async task submitted: {task_id}")
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
                logger.warning(
                    f"Docling task {task_id} timed out after {self.config.docling_task_timeout}s. "
                    "Remote task cannot be cancelled (no cancel API in docling-serve)."
                )
                raise ApiErrorException({
                    "errorId": DOCLING_TIMEOUT,
                    "status": status.HTTP_504_GATEWAY_TIMEOUT,
                    "debugMessage": f"Docling task {task_id} did not complete within {self.config.docling_task_timeout}s",
                })

            poll_response = await self._make_request("GET", poll_url, f"poll task {task_id}")
            body = self._parse_json(poll_response, f"poll task {task_id}")
            task_status: str = body.get("task_status", "")
            logger.info(f"Docling task {task_id} status: {task_status}")

            if task_status not in self._POLL_VALID_STATES:
                logger.error(f"Docling task {task_id} returned unexpected status: {task_status!r}")
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

    async def convert_to_docling(
        self,
        file: UploadFile | BytesIO,
        source_lang: LanguageOrAuto,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> DoclingDocument:
        languages = [source_lang.value]

        if source_lang == DetectLanguage.AUTO:
            languages = ["de", "en", "fr", "it"]
        elif source_lang.value.startswith("en"):
            languages = ["en"]

        content, filename, content_type = self._resolve_file(file, filename, content_type)

        files = {"files": (filename, BytesIO(content), content_type)}
        options: dict[str, str | list[str] | bool] = {
            "to_formats": ["json"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "images_scale": "1",
            "ocr_engine": "easyocr",
            "ocr_lang": languages,
            "table_mode": "accurate",
            "pdf_backend": "pypdfium2",
        }

        response = await self.fetch_docling_file_convert(files, options)
        document = extract_docling_document(
            self._parse_json(response, "convert_to_docling result"),
            logger_context={"options": options, "content_type": content_type},
        )

        if isinstance(document.json_content, dict):
            return DoclingDocument.model_validate(document.json_content)
        assert document.json_content is not None  # noqa: S101  # guaranteed by extract_docling_document
        return document.json_content

    async def convert(
        self,
        file: UploadFile | BytesIO,
        source_lang: LanguageOrAuto,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> ConversionResult:
        languages = [source_lang.value]

        if source_lang == DetectLanguage.AUTO:
            languages = ["de", "en", "fr", "it"]

        logger.info(f"type of file: {type(file)}")
        content, filename, content_type = self._resolve_file(file, filename, content_type)

        files = {"files": (filename, BytesIO(content), content_type)}
        options: dict[str, str | list[str] | bool] = {
            "images_scale": "1",
            "to_formats": ["md", "json"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "ocr_engine": "easyocr",
            "ocr_lang": languages,
            "table_mode": "accurate",
            "pdf_backend": "pypdfium2",
        }

        response = await self.fetch_docling_file_convert(files, options)
        docling_response = extract_docling_document(
            self._parse_json(response, "convert result"),
            logger_context={"options": options, "content_type": content_type},
        )

        markdown = docling_response.md_content or ""
        images: dict[int, Base64EncodedImage] = {}

        base64_pattern = r"!\[.*?\]\(data:image/[^;]+;base64,([^)]+)\)"
        for idx, base64_data in enumerate(re.findall(base64_pattern, markdown)):
            try:
                images[idx] = base64_data
                old_pattern = f"data:image/[^;]+;base64,{re.escape(base64_data)}"
                markdown = re.sub(old_pattern, f"image{idx}.png", markdown)
            except Exception:
                logger.exception(f"Error decoding base64 image {idx}")

        return ConversionResult(markdown=markdown, images=images)
