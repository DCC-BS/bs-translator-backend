from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.datastructures import UploadFile

from bs_translator_backend.models.error_codes import (
    DOCLING_TASK_FAILED,
    DOCLING_TIMEOUT,
    INVALID_MIME_TYPE,
    UNEXPECTED_ERROR,
)
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, Language
from bs_translator_backend.services.document_conversion_service import (
    DocumentConversionService,
    get_mimetype,
    validate_mimetype,
)
from bs_translator_backend.utils.app_config import AppConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TASK_ID = "task-abc-123"
_DOCLING_URL = "http://docling/v1"

_SUBMIT_OK = {"task_id": _TASK_ID, "task_status": "pending"}
_POLL_PENDING = {"task_status": "pending", "task_id": _TASK_ID}
_POLL_SUCCESS = {"task_status": "success", "task_id": _TASK_ID}
_POLL_FAILURE = {"task_status": "failure", "task_id": _TASK_ID}

_RESULT_BODY = {
    "document": {
        "filename": "test.pdf",
        "md_content": "# Hello world",
        "json_content": {"name": "test_doc"},
    },
    "status": "success",
    "errors": [],
    "processing_time": 1.0,
}

_DUMMY_REQUEST = httpx.Request("POST", _DOCLING_URL)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        openai_api_base_url="http://openai",
        openai_api_key="test-key",
        llm_model="gpt-4",
        client_url="http://client",
        docling_url=_DOCLING_URL,
        hmac_secret="secret",
        whisper_url="http://whisper",
        docling_poll_interval=0.001,  # minimal sleep; patched to no-op where needed
        docling_task_timeout=600.0,
    )


@pytest.fixture
def service(config: AppConfig) -> DocumentConversionService:
    return DocumentConversionService(config)


def _responses(*bodies: dict | None, status: int = 200) -> AsyncMock:
    """Build an AsyncMock for client.request returning one Response per call."""
    return AsyncMock(
        side_effect=[
            httpx.Response(status, json=b) if b is not None else httpx.Response(status)
            for b in bodies
        ]
    )


# ---------------------------------------------------------------------------
# get_mimetype
# ---------------------------------------------------------------------------


class TestGetMimetype:
    def test_known_extensions(self) -> None:
        assert get_mimetype(pytest.importorskip("pathlib").Path("doc.pdf")) == "application/pdf"
        assert get_mimetype(pytest.importorskip("pathlib").Path("doc.docx")).startswith(
            "application/vnd"
        )
        assert get_mimetype(pytest.importorskip("pathlib").Path("img.png")) == "image/png"
        assert get_mimetype(pytest.importorskip("pathlib").Path("data.csv")) == "text/csv"

    def test_unknown_extension_returns_invalid(self) -> None:
        from pathlib import Path

        assert get_mimetype(Path("file.xyz")) == "invalid"

    def test_case_insensitive(self) -> None:
        from pathlib import Path

        assert get_mimetype(Path("DOC.PDF")) == "application/pdf"


# ---------------------------------------------------------------------------
# validate_mimetype
# ---------------------------------------------------------------------------


class TestValidateMimetype:
    def test_valid_mimetype_passes(self) -> None:
        validate_mimetype("application/pdf", {})  # no exception

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ApiErrorException) as exc_info:
            validate_mimetype("", {})
        assert exc_info.value.args[0]["errorId"] == INVALID_MIME_TYPE

    def test_invalid_sentinel_raises(self) -> None:
        with pytest.raises(ApiErrorException) as exc_info:
            validate_mimetype("invalid", {})
        assert exc_info.value.args[0]["errorId"] == INVALID_MIME_TYPE


# ---------------------------------------------------------------------------
# _resolve_file
# ---------------------------------------------------------------------------


class TestResolveFile:
    def test_bytesio_uses_provided_filename_and_content_type(
        self, service: DocumentConversionService
    ) -> None:
        data = b"hello"
        content, filename, ct = service._resolve_file(BytesIO(data), "doc.pdf", "application/pdf")
        assert content == data
        assert filename == "doc.pdf"
        assert ct == "application/pdf"

    def test_bytesio_detects_mimetype_from_filename(
        self, service: DocumentConversionService
    ) -> None:
        _, _, ct = service._resolve_file(BytesIO(b"x"), "report.pdf", None)
        assert ct == "application/pdf"

    def test_bytesio_falls_back_to_uploaded_document_name(
        self, service: DocumentConversionService
    ) -> None:
        _, filename, _ = service._resolve_file(BytesIO(b"x"), None, "text/plain")
        assert filename == "uploaded_document"

    def test_upload_file_reads_content_and_filename(
        self, service: DocumentConversionService
    ) -> None:
        data = b"pdf content"
        mock_file = MagicMock()
        mock_file.file.read.return_value = data
        mock_file.filename = "upload.pdf"
        mock_file.__class__ = UploadFile

        content, filename, ct = service._resolve_file(mock_file, None, "application/pdf")
        assert content == data
        assert filename == "upload.pdf"

    def test_upload_file_param_filename_overrides_upload_filename(
        self, service: DocumentConversionService
    ) -> None:
        mock_file = MagicMock()
        mock_file.file.read.return_value = b""
        mock_file.filename = "original.pdf"
        mock_file.__class__ = UploadFile

        _, filename, _ = service._resolve_file(mock_file, "override.pdf", "application/pdf")
        assert filename == "override.pdf"

    def test_unknown_extension_raises_invalid_mime(
        self, service: DocumentConversionService
    ) -> None:
        with pytest.raises(ApiErrorException) as exc_info:
            service._resolve_file(BytesIO(b"x"), "file.xyz", None)
        assert exc_info.value.args[0]["errorId"] == INVALID_MIME_TYPE


# ---------------------------------------------------------------------------
# _make_request
# ---------------------------------------------------------------------------


class TestMakeRequest:
    @pytest.mark.asyncio
    async def test_returns_response_on_2xx(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(return_value=httpx.Response(200, json={"ok": True}))
        response = await service._make_request("GET", "http://x", "test")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_timeout_raises_docling_timeout(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(
            side_effect=httpx.ReadTimeout("timed out", request=_DUMMY_REQUEST)
        )
        with pytest.raises(ApiErrorException) as exc_info:
            await service._make_request("GET", "http://x", "ctx")
        assert exc_info.value.args[0]["errorId"] == DOCLING_TIMEOUT

    @pytest.mark.asyncio
    async def test_request_error_raises_unexpected(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = AsyncMock(
            side_effect=httpx.ConnectError("refused", request=_DUMMY_REQUEST)
        )
        with pytest.raises(ApiErrorException) as exc_info:
            await service._make_request("GET", "http://x", "ctx")
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_non_2xx_raises_unexpected(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(return_value=httpx.Response(500, text="oops"))
        with pytest.raises(ApiErrorException) as exc_info:
            await service._make_request("POST", "http://x", "ctx")
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR


# ---------------------------------------------------------------------------
# _submit_async_convert
# ---------------------------------------------------------------------------


class TestSubmitAsyncConvert:
    @pytest.mark.asyncio
    async def test_returns_task_id(self, service: DocumentConversionService) -> None:
        service.client.request = _responses(_SUBMIT_OK)
        task_id = await service._submit_async_convert({}, {})
        assert task_id == _TASK_ID

    @pytest.mark.asyncio
    async def test_missing_task_id_raises(self, service: DocumentConversionService) -> None:
        service.client.request = _responses({"status": "pending"})  # no task_id
        with pytest.raises(ApiErrorException) as exc_info:
            await service._submit_async_convert({}, {})
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_non_json_body_raises(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(return_value=httpx.Response(200, text="not-json"))
        with pytest.raises(ApiErrorException) as exc_info:
            await service._submit_async_convert({}, {})
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_non_2xx_raises(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(return_value=httpx.Response(503, text="unavailable"))
        with pytest.raises(ApiErrorException) as exc_info:
            await service._submit_async_convert({}, {})
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR


# ---------------------------------------------------------------------------
# _poll_task
# ---------------------------------------------------------------------------


class TestPollTask:
    @pytest.mark.asyncio
    async def test_immediate_success(self, service: DocumentConversionService) -> None:
        service.client.request = _responses(_POLL_SUCCESS)
        with patch(
            "bs_translator_backend.services.document_conversion_service.time.monotonic",
            side_effect=[0.0, 1.0],
        ):
            await service._poll_task(_TASK_ID)  # no exception = success

    @pytest.mark.asyncio
    async def test_polls_multiple_times_before_success(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = _responses(_POLL_PENDING, _POLL_PENDING, _POLL_SUCCESS)
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 1.0, 2.0, 3.0],
            ),
            patch(
                "bs_translator_backend.services.document_conversion_service.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await service._poll_task(_TASK_ID)
        assert service.client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_failure_status_raises_docling_task_failed(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = _responses(_POLL_FAILURE)
        with patch(
            "bs_translator_backend.services.document_conversion_service.time.monotonic",
            side_effect=[0.0, 1.0],
        ):
            with pytest.raises(ApiErrorException) as exc_info:
                await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == DOCLING_TASK_FAILED

    @pytest.mark.asyncio
    async def test_timeout_raises_docling_timeout(self, service: DocumentConversionService) -> None:
        # monotonic: deadline = 0.0 + 600 = 600.0; first check returns 700.0 → expired
        service.client.request = AsyncMock()
        with patch(
            "bs_translator_backend.services.document_conversion_service.time.monotonic",
            side_effect=[0.0, 700.0],
        ):
            with pytest.raises(ApiErrorException) as exc_info:
                await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == DOCLING_TIMEOUT
        service.client.request.assert_not_called()  # no poll issued before timeout

    @pytest.mark.asyncio
    async def test_unknown_status_raises_unexpected(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = _responses({"task_status": "UNKNOWN_STATE"})
        with patch(
            "bs_translator_backend.services.document_conversion_service.time.monotonic",
            side_effect=[0.0, 1.0],
        ):
            with pytest.raises(ApiErrorException) as exc_info:
                await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_non_json_poll_response_raises(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(return_value=httpx.Response(200, text="not-json"))
        with patch(
            "bs_translator_backend.services.document_conversion_service.time.monotonic",
            side_effect=[0.0, 1.0],
        ):
            with pytest.raises(ApiErrorException) as exc_info:
                await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_network_error_during_poll_raises(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = AsyncMock(
            side_effect=httpx.ConnectError("gone", request=_DUMMY_REQUEST)
        )
        with patch(
            "bs_translator_backend.services.document_conversion_service.time.monotonic",
            side_effect=[0.0, 1.0],
        ):
            with pytest.raises(ApiErrorException) as exc_info:
                await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR


# ---------------------------------------------------------------------------
# Full async flow: fetch_docling_file_convert
# ---------------------------------------------------------------------------


class TestFetchDoclingFileConvert:
    @pytest.mark.asyncio
    async def test_happy_path_submit_poll_fetch(self, service: DocumentConversionService) -> None:
        service.client.request = _responses(_SUBMIT_OK, _POLL_PENDING, _POLL_SUCCESS, _RESULT_BODY)
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 1.0, 2.0, 3.0],
            ),
            patch(
                "bs_translator_backend.services.document_conversion_service.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            response = await service.fetch_docling_file_convert({}, {})

        assert response.status_code == 200
        assert response.json()["document"]["md_content"] == "# Hello world"
        assert service.client.request.call_count == 4  # submit + 2 polls + result


# ---------------------------------------------------------------------------
# convert (end-to-end with mocked fetch_docling_file_convert)
# ---------------------------------------------------------------------------


class TestConvert:
    @pytest.mark.asyncio
    async def test_returns_conversion_result(self, service: DocumentConversionService) -> None:
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        result = await service.convert(BytesIO(b"data"), DetectLanguage.AUTO, filename="file.pdf")
        assert result.markdown == "# Hello world"
        assert result.images == {}

    @pytest.mark.asyncio
    async def test_extracts_base64_images_from_markdown(
        self, service: DocumentConversionService
    ) -> None:
        b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        result_with_image = {
            **_RESULT_BODY,
            "document": {
                **_RESULT_BODY["document"],
                "md_content": f"![img](data:image/png;base64,{b64})",
            },
        }
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=result_with_image)
        )
        result = await service.convert(BytesIO(b"data"), Language.DE, filename="file.pdf")
        assert 0 in result.images
        assert "image0.png" in result.markdown

    @pytest.mark.asyncio
    async def test_invalid_mimetype_raises_before_request(
        self, service: DocumentConversionService
    ) -> None:
        service.fetch_docling_file_convert = AsyncMock()
        with pytest.raises(ApiErrorException) as exc_info:
            await service.convert(BytesIO(b"data"), DetectLanguage.AUTO, filename="file.xyz")
        assert exc_info.value.args[0]["errorId"] == INVALID_MIME_TYPE
        service.fetch_docling_file_convert.assert_not_called()


# ---------------------------------------------------------------------------
# convert_to_docling (end-to-end with mocked fetch_docling_file_convert)
# ---------------------------------------------------------------------------


class TestConvertToDocling:
    @pytest.mark.asyncio
    async def test_returns_docling_document(self, service: DocumentConversionService) -> None:
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        doc = await service.convert_to_docling(
            BytesIO(b"data"), DetectLanguage.AUTO, filename="file.pdf"
        )
        assert doc.name == "test_doc"

    @pytest.mark.asyncio
    async def test_auto_language_uses_multilingual_ocr(
        self, service: DocumentConversionService
    ) -> None:
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        await service.convert_to_docling(BytesIO(b"data"), DetectLanguage.AUTO, filename="file.pdf")
        call_kwargs = service.fetch_docling_file_convert.call_args
        options = call_kwargs[0][1]  # second positional arg
        assert set(options["ocr_lang"]) == {"de", "en", "fr", "it"}

    @pytest.mark.asyncio
    async def test_english_language_uses_single_ocr_lang(
        self, service: DocumentConversionService
    ) -> None:
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        await service.convert_to_docling(BytesIO(b"data"), Language.EN, filename="file.pdf")
        options = service.fetch_docling_file_convert.call_args[0][1]
        assert options["ocr_lang"] == ["en"]
