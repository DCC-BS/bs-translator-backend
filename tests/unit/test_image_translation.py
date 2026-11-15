import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bs_translator_backend.models.conversion_result import (
    BlockToTranslate,
    BlockTranslatedEvent,
    ImageOcrResult,
)
from bs_translator_backend.models.docling_response import BoundingBox, DoclingDocument
from bs_translator_backend.models.error_codes import TRANSLATION_SERVICE_UNAVAILABLE
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, Language
from bs_translator_backend.models.translation import DetectLanguageOutput, TranslationConfig
from bs_translator_backend.routers import image_translation_route
from bs_translator_backend.services.image_translation_service import (
    ImageTranslationService,
    extract_blocks,
    normalize_bbox,
)
from bs_translator_backend.utils.app_config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        llm_url="http://llm/v1",
        llm_api_key="k",
        llm_model="m",
        client_url="http://client",
        docling_url="http://docling/v1",
        docling_api_key="",
        hmac_secret="s",
        whisper_url="http://whisper/v1",
        image_translation_concurrency=2,
    )


def _doc(texts: list[dict], tables: list[dict] | None = None) -> DoclingDocument:
    return DoclingDocument.model_validate({
        "name": "photo",
        "texts": texts,
        "tables": tables or [],
        "pages": {"1": {"page_no": 1, "size": {"width": 200, "height": 100}}},
    })


def _text(index: int, text: str, bbox: dict, label: str = "text") -> dict:
    return {
        "self_ref": f"#/texts/{index}",
        "label": label,
        "orig": text,
        "text": text,
        "meta": None,
        "prov": [{"page_no": 1, "bbox": bbox, "charspan": [0, len(text)]}],
    }


class TestNormalizeBbox:
    def test_bottomleft_origin_is_flipped_and_normalized(self) -> None:
        box = BoundingBox(l=20, t=90, r=120, b=70, coord_origin="BOTTOMLEFT")
        normalized = normalize_bbox(box, page_width=200, page_height=100)
        assert normalized.left == pytest.approx(0.1)
        assert normalized.right == pytest.approx(0.6)
        assert normalized.top == pytest.approx(0.1)
        assert normalized.bottom == pytest.approx(0.3)

    def test_topleft_origin_is_kept(self) -> None:
        box = BoundingBox(l=0, t=10, r=200, b=30, coord_origin="TOPLEFT")
        normalized = normalize_bbox(box, page_width=200, page_height=100)
        assert (normalized.top, normalized.bottom) == (pytest.approx(0.1), pytest.approx(0.3))

    def test_boxes_slightly_outside_the_page_are_clamped(self) -> None:
        box = BoundingBox(l=-5, t=-2, r=210, b=104, coord_origin="TOPLEFT")
        normalized = normalize_bbox(box, page_width=200, page_height=100)
        assert (normalized.left, normalized.top, normalized.right, normalized.bottom) == (
            0,
            0,
            1,
            1,
        )


class TestExtractBlocks:
    def test_texts_in_reading_order_then_table_cells(self) -> None:
        doc = _doc(
            texts=[
                _text(
                    0,
                    "Speisekarte",
                    {"l": 10, "t": 95, "r": 190, "b": 80, "coord_origin": "BOTTOMLEFT"},
                    "title",
                ),
                _text(1, "   ", {"l": 10, "t": 50, "r": 20, "b": 40, "coord_origin": "BOTTOMLEFT"}),
                {**_text(2, "no prov", {"l": 0, "t": 0, "r": 1, "b": 1}), "prov": []},
            ],
            tables=[
                {
                    "self_ref": "#/tables/0",
                    "label": "table",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 0,
                                "t": 60,
                                "r": 200,
                                "b": 0,
                                "coord_origin": "BOTTOMLEFT",
                            },
                            "charspan": [0, 0],
                        }
                    ],
                    "data": {
                        "num_rows": 1,
                        "num_cols": 2,
                        "table_cells": [
                            {
                                "bbox": {
                                    "l": 0,
                                    "t": 50,
                                    "r": 100,
                                    "b": 60,
                                    "coord_origin": "TOPLEFT",
                                },
                                "start_row_offset_idx": 0,
                                "end_row_offset_idx": 1,
                                "start_col_offset_idx": 0,
                                "end_col_offset_idx": 1,
                                "text": "Suppe",
                            },
                            {
                                "bbox": None,
                                "start_row_offset_idx": 0,
                                "end_row_offset_idx": 1,
                                "start_col_offset_idx": 1,
                                "end_col_offset_idx": 2,
                                "text": "8.50",
                            },
                        ],
                    },
                }
            ],
        )

        blocks = extract_blocks(doc)

        assert [(b.id, b.text, b.kind) for b in blocks] == [
            ("t0", "Speisekarte", "title"),
            ("c0-0", "Suppe", "table_cell"),
        ]
        assert blocks[0].bbox.top == pytest.approx(0.05)
        assert blocks[1].bbox.top == pytest.approx(0.5)

    def test_document_without_pages_has_no_blocks(self) -> None:
        doc = DoclingDocument.model_validate({"name": "x"})
        assert extract_blocks(doc) == []


def _service(
    app_config: AppConfig, translate: object, detected: DetectLanguage | Language
) -> ImageTranslationService:
    translation_service = MagicMock()
    translation_service.translate_text = translate
    translation_service.detect_language = AsyncMock(
        return_value=DetectLanguageOutput(language=detected, confidence=0.9)
    )
    return ImageTranslationService(app_config, translation_service, MagicMock())


class TestTranslateBlocks:
    @pytest.mark.asyncio
    async def test_translates_each_block_with_page_language_and_context(
        self, app_config: AppConfig
    ) -> None:
        seen_configs: list[TranslationConfig] = []

        async def translate(text: str, config: TranslationConfig) -> AsyncGenerator[str, None]:
            seen_configs.append(config)
            yield f"[{text}]"

        service = _service(app_config, translate, Language.DE)
        blocks = [BlockToTranslate(id="a", text="Ausgang"), BlockToTranslate(id="b", text="12:30")]

        events = [
            e
            async for e in service.translate_blocks(
                blocks, TranslationConfig(target_language=Language.EN_US)
            )
        ]

        assert {e.id: e.translated for e in events} == {"a": "[Ausgang]", "b": "12:30"}
        # "12:30" has no letters and is not sent to the LLM.
        assert len(seen_configs) == 1
        assert seen_configs[0].source_language == Language.DE
        assert "Ausgang" in seen_configs[0].context

    @pytest.mark.asyncio
    async def test_explicit_source_language_skips_detection(self, app_config: AppConfig) -> None:
        async def translate(text: str, config: TranslationConfig) -> AsyncGenerator[str, None]:
            yield text

        service = _service(app_config, translate, Language.DE)
        config = TranslationConfig(target_language=Language.EN_US, source_language=Language.FR)
        _ = [
            e
            async for e in service.translate_blocks(
                [BlockToTranslate(id="a", text="Sortie")], config
            )
        ]
        service.translation_service.detect_language.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrency_is_limited(self, app_config: AppConfig) -> None:
        running = 0
        peak = 0

        async def translate(text: str, config: TranslationConfig) -> AsyncGenerator[str, None]:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            yield text

        service = _service(app_config, translate, DetectLanguage.AUTO)
        blocks = [BlockToTranslate(id=str(i), text=f"Wort {i}") for i in range(6)]
        events = [e async for e in service.translate_blocks(blocks, TranslationConfig())]

        assert len(events) == 6
        assert peak == app_config.image_translation_concurrency

    @pytest.mark.asyncio
    async def test_closing_the_stream_cancels_pending_translations(
        self, app_config: AppConfig
    ) -> None:
        cancelled = 0

        async def translate(text: str, config: TranslationConfig) -> AsyncGenerator[str, None]:
            nonlocal cancelled
            if text != "fast":
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    cancelled += 1
                    raise
            yield text

        service = _service(app_config, translate, DetectLanguage.AUTO)
        blocks = [BlockToTranslate(id="1", text="fast"), BlockToTranslate(id="2", text="slow")]
        stream = service.translate_blocks(blocks, TranslationConfig())
        first = await anext(stream)
        await stream.aclose()

        assert first.id == "1"
        assert cancelled == 1


def _client(service: MagicMock) -> TestClient:
    app = FastAPI()
    usage = MagicMock()
    config = MagicMock(max_upload_bytes=1024)
    app.include_router(
        image_translation_route.create_router(
            image_translation_service=service, usage_tracking_service=usage, app_config=config
        )
    )

    @app.exception_handler(ApiErrorException)
    async def _handler(_request: object, exc: ApiErrorException):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=exc.error_response["status"], content=exc.error_response)

    return TestClient(app)


class TestRoutes:
    def test_ocr_returns_blocks(self) -> None:
        service = MagicMock()
        service.ocr = AsyncMock(return_value=ImageOcrResult(blocks=[]))
        response = _client(service).post(
            "/translation/image/ocr",
            files={"image_file": ("photo.jpg", b"jpegbytes", "image/jpeg")},
            data={"source_language": "auto"},
        )
        assert response.status_code == 200
        assert response.json() == {"blocks": []}
        service.ocr.assert_awaited_once()

    def test_ocr_rejects_uploads_above_limit(self) -> None:
        service = MagicMock()
        service.ocr = AsyncMock()
        response = _client(service).post(
            "/translation/image/ocr",
            files={"image_file": ("photo.jpg", b"x" * 2048, "image/jpeg")},
        )
        assert response.status_code == 413
        assert response.json()["errorId"] == "file_too_large"

    def test_blocks_stream_ndjson_events_and_done(self) -> None:
        async def events(*_args: object) -> AsyncGenerator[BlockTranslatedEvent, None]:
            yield BlockTranslatedEvent(id="a", translated="Exit")
            yield BlockTranslatedEvent(id="b", translated="Platform 3")

        service = MagicMock()
        service.translate_blocks = events
        response = _client(service).post(
            "/translation/image/blocks",
            json={
                "blocks": [{"id": "a", "text": "Ausgang"}, {"id": "b", "text": "Gleis 3"}],
                "config": {"target_language": "en-us"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in response.text.splitlines()]
        assert lines == [
            {"type": "translation", "id": "a", "translated": "Exit"},
            {"type": "translation", "id": "b", "translated": "Platform 3"},
            {"type": "done"},
        ]

    def test_failure_before_first_block_is_an_http_error(self) -> None:
        async def events(*_args: object) -> AsyncGenerator[BlockTranslatedEvent, None]:
            raise ApiErrorException({"errorId": TRANSLATION_SERVICE_UNAVAILABLE, "status": 503})
            yield  # pragma: no cover

        service = MagicMock()
        service.translate_blocks = events
        response = _client(service).post(
            "/translation/image/blocks",
            json={"blocks": [{"id": "a", "text": "Ausgang"}], "config": {}},
        )
        assert response.status_code == 503
        assert response.json()["errorId"] == TRANSLATION_SERVICE_UNAVAILABLE

    def test_failure_mid_stream_ends_with_error_event(self) -> None:
        async def events(*_args: object) -> AsyncGenerator[BlockTranslatedEvent, None]:
            yield BlockTranslatedEvent(id="a", translated="Exit")
            raise ApiErrorException({"errorId": TRANSLATION_SERVICE_UNAVAILABLE, "status": 503})

        service = MagicMock()
        service.translate_blocks = events
        response = _client(service).post(
            "/translation/image/blocks",
            json={
                "blocks": [{"id": "a", "text": "Ausgang"}, {"id": "b", "text": "x"}],
                "config": {},
            },
        )
        lines = [json.loads(line) for line in response.text.splitlines()]
        assert lines[-1] == {
            "type": "error",
            "errorId": TRANSLATION_SERVICE_UNAVAILABLE,
            "status": 503,
        }

    def test_no_blocks_only_sends_done(self) -> None:
        async def events(*_args: object) -> AsyncGenerator[BlockTranslatedEvent, None]:
            return
            yield  # pragma: no cover

        service = MagicMock()
        service.translate_blocks = events
        response = _client(service).post(
            "/translation/image/blocks", json={"blocks": [], "config": {}}
        )
        assert [json.loads(line) for line in response.text.splitlines()] == [{"type": "done"}]
