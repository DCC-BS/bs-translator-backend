from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.datastructures import UploadFile

from bs_translator_backend.models.error_codes import (
    DOCLING_TASK_FAILED,
    DOCLING_TIMEOUT,
    FILE_TOO_LARGE,
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
        llm_url="http://openai",
        llm_api_key="test-key",
        llm_model="gpt-4",
        client_url="http://client",
        docling_url=_DOCLING_URL,
        docling_api_key="test-key",
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
    @pytest.mark.asyncio
    async def test_bytesio_uses_provided_filename_and_content_type(
        self, service: DocumentConversionService
    ) -> None:
        data = b"hello"
        content, filename, ct = await service._resolve_file(
            BytesIO(data), "doc.pdf", "application/pdf"
        )
        assert content == data
        assert filename == "doc.pdf"
        assert ct == "application/pdf"

    @pytest.mark.asyncio
    async def test_bytesio_detects_mimetype_from_filename(
        self, service: DocumentConversionService
    ) -> None:
        _, _, ct = await service._resolve_file(BytesIO(b"x"), "report.pdf", None)
        assert ct == "application/pdf"

    @pytest.mark.asyncio
    async def test_bytesio_falls_back_to_uploaded_document_name(
        self, service: DocumentConversionService
    ) -> None:
        _, filename, _ = await service._resolve_file(BytesIO(b"x"), None, "text/plain")
        assert filename == "uploaded_document"

    @pytest.mark.asyncio
    async def test_upload_file_reads_content_and_filename(
        self, service: DocumentConversionService
    ) -> None:
        data = b"pdf content"
        upload = UploadFile(file=BytesIO(data), filename="upload.pdf")

        content, filename, _ct = await service._resolve_file(upload, None, "application/pdf")
        assert content == data
        assert filename == "upload.pdf"

    @pytest.mark.asyncio
    async def test_upload_file_param_filename_overrides_upload_filename(
        self, service: DocumentConversionService
    ) -> None:
        upload = UploadFile(file=BytesIO(b""), filename="original.pdf")

        _, filename, _ = await service._resolve_file(upload, "override.pdf", "application/pdf")
        assert filename == "override.pdf"

    @pytest.mark.asyncio
    async def test_upload_above_size_limit_raises_file_too_large(self, config: AppConfig) -> None:
        service = DocumentConversionService(config.model_copy(update={"max_upload_size_mb": 0.001}))
        upload = UploadFile(file=BytesIO(b"x" * 2048), filename="big.pdf")

        with pytest.raises(ApiErrorException) as exc_info:
            await service._resolve_file(upload, None, None)
        assert exc_info.value.args[0]["errorId"] == FILE_TOO_LARGE
        assert exc_info.value.args[0]["status"] == 413

    @pytest.mark.asyncio
    async def test_unknown_extension_raises_invalid_mime(
        self, service: DocumentConversionService
    ) -> None:
        with pytest.raises(ApiErrorException) as exc_info:
            await service._resolve_file(BytesIO(b"x"), "file.xyz", None)
        assert exc_info.value.args[0]["errorId"] == INVALID_MIME_TYPE

    @pytest.mark.parametrize("name", ["scan.tif", "page.htm", "page.xhtml", "notes.markdown"])
    def test_docling_supported_extensions_are_accepted(self, name: str) -> None:
        assert get_mimetype(Path(name)) != "invalid"


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
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 1.0],
            ),
            pytest.raises(ApiErrorException) as exc_info,
        ):
            await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == DOCLING_TASK_FAILED

    @pytest.mark.asyncio
    async def test_timeout_raises_docling_timeout(self, service: DocumentConversionService) -> None:
        # monotonic: deadline = 0.0 + 600 = 600.0; first check returns 700.0 → expired
        service.client.request = AsyncMock()
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 700.0],
            ),
            pytest.raises(ApiErrorException) as exc_info,
        ):
            await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == DOCLING_TIMEOUT
        service.client.request.assert_not_called()  # no poll issued before timeout

    @pytest.mark.asyncio
    async def test_unknown_status_raises_unexpected(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = _responses({"task_status": "UNKNOWN_STATE"})
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 1.0],
            ),
            pytest.raises(ApiErrorException) as exc_info,
        ):
            await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_non_json_poll_response_raises(self, service: DocumentConversionService) -> None:
        service.client.request = AsyncMock(return_value=httpx.Response(200, text="not-json"))
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 1.0],
            ),
            pytest.raises(ApiErrorException) as exc_info,
        ):
            await service._poll_task(_TASK_ID)
        assert exc_info.value.args[0]["errorId"] == UNEXPECTED_ERROR

    @pytest.mark.asyncio
    async def test_network_error_during_poll_raises(
        self, service: DocumentConversionService
    ) -> None:
        service.client.request = AsyncMock(
            side_effect=httpx.ConnectError("gone", request=_DUMMY_REQUEST)
        )
        with (
            patch(
                "bs_translator_backend.services.document_conversion_service.time.monotonic",
                side_effect=[0.0, 1.0],
            ),
            pytest.raises(ApiErrorException) as exc_info,
        ):
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


class TestOcrImage:
    @pytest.mark.asyncio
    async def test_returns_docling_document(self, service: DocumentConversionService) -> None:
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        doc = await service.ocr_image(b"data", DetectLanguage.AUTO, "photo.jpg", "image/jpeg")
        assert doc.name == "test_doc"

    @pytest.mark.asyncio
    async def test_uses_configured_ocr_and_layout_presets(
        self, service: DocumentConversionService
    ) -> None:
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        await service.ocr_image(b"data", DetectLanguage.AUTO, "photo.jpg", "image/jpeg")
        options = service.fetch_docling_file_convert.call_args[0][1]
        assert options["ocr_preset"] == "glm-ocr-remote"
        assert options["layout_preset"] == "ppdoclayout-v3"
        assert options["force_ocr"] is False
        assert options["image_export_mode"] == "placeholder"
        assert set(options["ocr_lang"]) == {"de", "en", "fr", "it"}

    @pytest.mark.asyncio
    async def test_empty_layout_preset_is_not_sent(self, config: AppConfig) -> None:
        service = DocumentConversionService(
            config.model_copy(update={"docling_image_layout_preset": ""})
        )
        service.fetch_docling_file_convert = AsyncMock(
            return_value=httpx.Response(200, json=_RESULT_BODY)
        )
        await service.ocr_image(b"data", Language.EN, "photo.png", "image/png")
        options = service.fetch_docling_file_convert.call_args[0][1]
        assert "layout_preset" not in options
        assert options["ocr_lang"] == ["en"]

    @pytest.mark.asyncio
    async def test_rejects_non_image_types(self, service: DocumentConversionService) -> None:
        service.fetch_docling_file_convert = AsyncMock()
        with pytest.raises(ApiErrorException) as exc_info:
            await service.ocr_image(b"data", DetectLanguage.AUTO, "x.heic", "image/heic")
        assert exc_info.value.args[0]["errorId"] == INVALID_MIME_TYPE
        service.fetch_docling_file_convert.assert_not_called()

    @pytest.mark.asyncio
    async def test_tolerates_null_meta_fields_from_newer_docling(
        self, service: DocumentConversionService
    ) -> None:
        """docling-serve >= 1.36 serializes `meta: null` on every item (SVC-18)."""
        body = {
            **_RESULT_BODY,
            "document": {
                **_RESULT_BODY["document"],
                "json_content": {
                    "name": "photo",
                    "body": {"self_ref": "#/body", "name": "_root_", "meta": None},
                    "texts": [
                        {
                            "self_ref": "#/texts/0",
                            "label": "text",
                            "orig": "Hallo",
                            "text": "Hallo",
                            "meta": None,
                            "prov": [
                                {
                                    "page_no": 1,
                                    "bbox": {
                                        "l": 1,
                                        "t": 9,
                                        "r": 5,
                                        "b": 2,
                                        "coord_origin": "BOTTOMLEFT",
                                    },
                                    "charspan": [0, 5],
                                }
                            ],
                        }
                    ],
                    "pages": {"1": {"page_no": 1, "size": {"width": 10, "height": 10}}},
                },
            },
            "errors": [{"component_type": "pipeline", "module_name": "x", "error_message": "y"}],
            "status": "partial_success",
        }
        service.fetch_docling_file_convert = AsyncMock(return_value=httpx.Response(200, json=body))
        doc = await service.ocr_image(b"data", DetectLanguage.AUTO, "photo.jpg", "image/jpeg")
        assert doc.texts[0].text == "Hallo"


class TestDoclingAuth:
    def test_sends_docling_key_not_llm_key(self, config: AppConfig) -> None:
        service = DocumentConversionService(
            config.model_copy(update={"docling_api_key": "doc-key", "llm_api_key": "llm-key"})
        )
        assert service.client.headers["Authorization"] == "Bearer doc-key"
        assert service.client.headers["X-Api-Key"] == "doc-key"

    def test_no_auth_header_without_key(self, config: AppConfig) -> None:
        service = DocumentConversionService(config.model_copy(update={"docling_api_key": ""}))
        assert "Authorization" not in service.client.headers
