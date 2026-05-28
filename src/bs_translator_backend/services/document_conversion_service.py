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
    if len(mimetype) == 0:
        logger.error("MIME type is empty", extra=logger_context)

        raise ApiErrorException({
            "errorId": INVALID_MIME_TYPE,
            "status": status.HTTP_400_BAD_REQUEST,
            "debugMessage": "MIME type is empty",
        })

    if mimetype == "invalid":
        logger.error("Invalid MIME type", extra=logger_context)
        raise ApiErrorException({
            "errorId": INVALID_MIME_TYPE,
            "status": status.HTTP_400_BAD_REQUEST,
            "debugMessage": "Invalid MIME type",
        })


def extract_docling_document(response: str, logger_context: dict[str, Any]) -> DocumentResponse:
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
        self.client = httpx.AsyncClient(timeout=60.0)

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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Ensure the HTTP client is closed when leaving an async context."""

        await self.aclose()

    async def _submit_async_convert(
        self,
        files: dict[str, tuple[str, BytesIO, str]],
        options: dict[str, str | list[str] | bool],
    ) -> str:
        """POST /convert/file/async and return the task_id string.

        Raises ApiErrorException on timeout, request error, non-2xx response,
        or a response body that is not valid JSON / missing the task_id field.
        """
        try:
            response = await self.client.post(
                self.config.docling_url + "/convert/file/async",
                files=files,
                data=options,
                headers={"Authorization": self.config.openai_api_key},
            )
        except httpx.TimeoutException as e:
            logger.exception("Docling async submit timeout")
            raise ApiErrorException({
                "errorId": DOCLING_TIMEOUT,
                "status": status.HTTP_504_GATEWAY_TIMEOUT,
                "debugMessage": "Docling async submit request timed out",
            }) from e
        except httpx.RequestError as e:
            logger.exception("Docling async submit request error")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"Docling async submit request error: {e!s}",
            }) from e

        if not (200 <= response.status_code < 300):
            logger.error(f"Docling async submit error response: {response.text}")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": "Unexpected error on async submit",
            })

        try:
            body = response.json()
        except ValueError as e:
            logger.exception("Docling async submit returned non-JSON body")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": "Docling async submit response is not valid JSON",
            }) from e

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

    _POLL_TERMINAL_STATES = frozenset({"success", "failure"})
    _POLL_VALID_STATES = frozenset({"pending", "started", "success", "failure"})

    async def _poll_task(self, task_id: str) -> None:
        """Poll /status/poll/{task_id} until the task succeeds or fails.

        Raises ApiErrorException on timeout (DOCLING_TIMEOUT), task failure
        (DOCLING_TASK_FAILED), network errors, non-JSON responses, or an
        unrecognised task_status value. Note: docling-serve has no cancel
        endpoint, so on timeout the remote task keeps running.
        """
        deadline = time.monotonic() + self.config.docling_task_timeout
        poll_url = self.config.docling_url + f"/status/poll/{task_id}"

        while True:
            if time.monotonic() >= deadline:
                # docling-serve has no cancel endpoint; remote task continues running
                logger.warning(
                    f"Docling task {task_id} timed out after {self.config.docling_task_timeout}s. "
                    "Remote task cannot be cancelled (no cancel API in docling-serve)."
                )
                raise ApiErrorException({
                    "errorId": DOCLING_TIMEOUT,
                    "status": status.HTTP_504_GATEWAY_TIMEOUT,
                    "debugMessage": f"Docling task {task_id} did not complete within {self.config.docling_task_timeout}s",
                })

            try:
                poll_response = await self.client.get(
                    poll_url,
                    headers={"Authorization": self.config.openai_api_key},
                )
            except httpx.TimeoutException as e:
                logger.exception(f"Docling poll timeout for task {task_id}")
                raise ApiErrorException({
                    "errorId": DOCLING_TIMEOUT,
                    "status": status.HTTP_504_GATEWAY_TIMEOUT,
                    "debugMessage": f"Docling poll request timed out for task {task_id}",
                }) from e
            except httpx.RequestError as e:
                logger.exception(f"Docling poll request error for task {task_id}")
                raise ApiErrorException({
                    "errorId": UNEXPECTED_ERROR,
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "debugMessage": f"Docling poll request error: {e!s}",
                }) from e

            try:
                poll_body = poll_response.json()
            except ValueError as e:
                logger.exception(f"Docling poll returned non-JSON body for task {task_id}")
                raise ApiErrorException({
                    "errorId": UNEXPECTED_ERROR,
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "debugMessage": f"Docling poll response is not valid JSON for task {task_id}",
                }) from e

            task_status: str = poll_body.get("task_status", "")
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
        """GET /result/{task_id} and return the raw response.

        Raises ApiErrorException on timeout, request error, or non-2xx status.
        """
        try:
            response = await self.client.get(
                self.config.docling_url + f"/result/{task_id}",
                headers={"Authorization": self.config.openai_api_key},
            )
        except httpx.TimeoutException as e:
            logger.exception(f"Docling result fetch timeout for task {task_id}")
            raise ApiErrorException({
                "errorId": DOCLING_TIMEOUT,
                "status": status.HTTP_504_GATEWAY_TIMEOUT,
                "debugMessage": f"Docling result fetch timed out for task {task_id}",
            }) from e
        except httpx.RequestError as e:
            logger.exception(f"Docling result fetch request error for task {task_id}")
            raise ApiErrorException({
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": f"Docling result fetch error: {e!s}",
            }) from e

        if 200 <= response.status_code < 300:
            return response

        try:
            error_text = response.text
            logger.error(f"Docling result error response: {error_text}")
        except UnicodeDecodeError:
            logger.exception(
                f"Docling result contains binary data (status: {response.status_code})"
            )

        raise ApiErrorException({
            "errorId": UNEXPECTED_ERROR,
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "debugMessage": "Unexpected error fetching docling task result",
        })

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

        # Handle both UploadFile and BytesIO cases
        if isinstance(file, UploadFile):
            content = file.file.read()
            filename = filename or file.filename or "uploaded_document"
            if content_type is None:
                content_type = get_mimetype(Path(filename))
        else:
            # It's a BytesIO object
            content = file.read()
            filename: str = filename or "uploaded_document"
            if content_type is None:
                content_type: str = get_mimetype(Path(filename))

        assert isinstance(content_type, str)  # noqa: S101
        assert isinstance(filename, str)  # noqa: S101
        validate_mimetype(content_type, logger_context={"content_type": content_type})

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

        logger_context = {"options": options, "content_type": content_type}

        response = await self.fetch_docling_file_convert(files, options)
        json_response = response.json()

        document = extract_docling_document(json_response, logger_context)

        if document.json_content is None:
            logger.error("Docling response does not contain a document", extra=logger_context)

            raise ApiErrorException({
                "errorId": NO_DOCUMENT,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": "Docling response does not contain a document",
            })

        # Ensure we return a DoclingDocument instance
        if isinstance(document.json_content, dict):
            return DoclingDocument.model_validate(document.json_content)
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

        # Handle both UploadFile and BytesIO cases
        if isinstance(file, UploadFile):
            content = file.file.read()
            filename = file.filename or "uploaded_document"
            logger.info(f"Filename from UploadFile: {filename}")
            if content_type is None:
                content_type: str = get_mimetype(Path(filename))
        else:
            logger.info("File is not an UploadFile instance")
            # It's a BytesIO object
            content = file.read()
            filename: str = filename or "uploaded_document"
            if content_type is None:
                content_type: str = get_mimetype(Path(filename))
        assert isinstance(content_type, str)  # noqa: S101
        assert isinstance(filename, str)  # noqa: S101
        validate_mimetype(content_type, logger_context={"content_type": content_type})

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
        json_response = response.json()
        docling_response = extract_docling_document(
            json_response, logger_context={"options": options, "content_type": content_type}
        )

        # Extract markdown content from the docling response
        markdown = docling_response.md_content or ""

        images: dict[int, Base64EncodedImage] = {}

        # Extract base64 images directly from markdown
        base64_pattern = r"!\[.*?\]\(data:image/[^;]+;base64,([^)]+)\)"
        matches = re.findall(base64_pattern, markdown)

        for idx, base64_data in enumerate(matches):
            try:
                images[idx] = base64_data
                # Replace base64 data in markdown with file path
                old_pattern = f"data:image/[^;]+;base64,{re.escape(base64_data)}"
                new_path = f"image{idx}.png"
                markdown = re.sub(old_pattern, new_path, markdown)
            except Exception:
                logger.exception(f"Error decoding base64 image {idx}")

        return ConversionResult(markdown=markdown, images=images)
