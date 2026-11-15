from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from bs_translator_backend.container import Container
from bs_translator_backend.models.conversion_result import (
    BlockTranslatedEvent,
    BlockTranslationDoneEvent,
    BlockTranslationErrorEvent,
    BlockTranslationInput,
    ImageOcrResult,
)
from bs_translator_backend.models.error_codes import UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.services.image_translation_service import ImageTranslationService
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.cancelation import ClientDisconnectedError, run_until_disconnected
from bs_translator_backend.utils.uploads import read_upload_limited

logger = get_logger(__name__)

CLIENT_CLOSED_REQUEST = 499

_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "X-Accel-Buffering": "no",
}


def _ndjson(
    event: BlockTranslatedEvent | BlockTranslationDoneEvent | BlockTranslationErrorEvent,
) -> str:
    return event.model_dump_json() + "\n"


async def _block_translation_stream(
    request: Request,
    first: BlockTranslatedEvent | None,
    events: AsyncGenerator[BlockTranslatedEvent, None],
) -> AsyncGenerator[str, None]:
    """NDJSON body of the block translation response; `first` was already pulled from `events`."""
    async with aclosing(events):
        if first is None:
            yield _ndjson(BlockTranslationDoneEvent())
            return
        yield _ndjson(first)
        try:
            async for event in events:
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping block translation")
                    return
                yield _ndjson(event)
        except ApiErrorException as e:
            logger.warning("Block translation failed mid-stream", error=e.error_response)
            yield _ndjson(
                BlockTranslationErrorEvent(
                    errorId=e.error_response.get("errorId", UNEXPECTED_ERROR),
                    status=e.error_response.get("status", 500),
                )
            )
            return
        except Exception:
            logger.exception("Block translation failed mid-stream")
            yield _ndjson(BlockTranslationErrorEvent(errorId=UNEXPECTED_ERROR, status=500))
            return
        yield _ndjson(BlockTranslationDoneEvent())


@inject
def create_router(
    image_translation_service: ImageTranslationService = Provide[
        Container.image_translation_service
    ],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
    app_config: AppConfig = Provide[Container.app_config],
) -> APIRouter:
    """Photo translation endpoints: OCR a photo once, translate its blocks per language."""
    router: APIRouter = APIRouter(prefix="/translation/image", tags=["translation"])

    @router.post(
        "/ocr", summary="Recognise the text blocks of a photo", response_model=ImageOcrResult
    )
    async def ocr_image(
        request: Request,
        image_file: UploadFile,
        source_language: Annotated[LanguageOrAuto, Form()] = DetectLanguage.AUTO,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> ImageOcrResult | Response:
        """Run OCR (GLM-OCR via docling-serve) on a photo.

        Returns the text blocks in reading order with boxes normalized to the image
        (0..1, origin top-left). An image without text gives an empty block list.
        """
        content = await read_upload_limited(image_file, app_config.max_upload_bytes)
        usage_tracking_service.log_event(
            "translation.image.ocr",
            user_id=x_client_id,
            file_size=len(content),
            source_language=source_language.value,
        )
        try:
            return await run_until_disconnected(
                request,
                image_translation_service.ocr(
                    content, source_language, image_file.filename, image_file.content_type
                ),
            )
        except ClientDisconnectedError:
            return Response(status_code=CLIENT_CLOSED_REQUEST)

    @router.post("/blocks", summary="Translate the text blocks of a photo")
    async def translate_blocks(
        request: Request,
        body: BlockTranslationInput,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Translate OCR blocks, streamed as NDJSON.

        One JSON object per line: `{"type": "translation", "id", "translated"}` per block
        (in completion order), then `{"type": "done"}`. An error after the stream has
        started ends it with `{"type": "error", "errorId", "status"}` instead.
        """
        config = body.config
        usage_tracking_service.log_event(
            "translation.image.blocks",
            user_id=x_client_id,
            block_count=len(body.blocks),
            text_length=sum(len(block.text) for block in body.blocks),
            target_language=config.target_language.value,
            source_language=config.source_language.value if config.source_language else None,
        )

        events = image_translation_service.translate_blocks(body.blocks, config)

        # Wait for the first translation before committing to status 200, so a failure
        # that happens right away (LLM unavailable) still becomes a proper error response.
        try:
            first = await run_until_disconnected(request, anext(events, None))
        except ClientDisconnectedError:
            await events.aclose()
            return Response(status_code=CLIENT_CLOSED_REQUEST)
        except BaseException:
            await events.aclose()
            raise

        return StreamingResponse(
            _block_translation_stream(request, first, events),
            media_type="application/x-ndjson",
            headers=_STREAM_HEADERS,
        )

    return router
