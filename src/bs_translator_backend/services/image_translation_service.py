"""Photo translation: OCR a photo into positioned text blocks, then translate the blocks.

The two steps are separate on purpose. OCR (docling-serve with GLM-OCR) is the slow
part; keeping its result on the client means changing the target language only
re-runs the translation, not the OCR.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from typing import final

from dcc_backend_common.logger import get_logger

from bs_translator_backend.models.conversion_result import (
    BlockToTranslate,
    BlockTranslatedEvent,
    ImageOcrResult,
    NormalizedBox,
    OcrBlock,
    OcrBlockKind,
)
from bs_translator_backend.models.docling_response import BoundingBox, DoclingDocument
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.models.translation import DetectLanguageInput, TranslationConfig
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.app_config import AppConfig

logger = get_logger(__name__)

# The whole recognised text is given to every block's translation as context, so short
# fragments ("Ausgang", "Gleis 3") are translated in the sense of the whole sign/page.
PAGE_CONTEXT_MAX_CHARS = 2000

_KIND_BY_LABEL: dict[str, OcrBlockKind] = {
    "title": "title",
    "section_header": "heading",
    "text": "text",
    "paragraph": "text",
    "list_item": "list_item",
    "caption": "caption",
    "footnote": "text",
    "page_header": "text",
    "page_footer": "text",
}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalize_bbox(bbox: BoundingBox, page_width: float, page_height: float) -> NormalizedBox:
    """Convert a docling box (page units, either origin) to a top-left normalized box."""
    if bbox.coord_origin.upper() == "BOTTOMLEFT":
        top, bottom = page_height - bbox.top, page_height - bbox.bottom
    else:
        top, bottom = bbox.top, bbox.bottom
    left, right = bbox.left, bbox.right

    return NormalizedBox(
        left=_clamp(min(left, right) / page_width),
        right=_clamp(max(left, right) / page_width),
        top=_clamp(min(top, bottom) / page_height),
        bottom=_clamp(max(top, bottom) / page_height),
    )


def extract_blocks(doc: DoclingDocument) -> list[OcrBlock]:
    """Text blocks of the first page of `doc`, in docling's reading order, tables last."""
    if not doc.pages:
        return []
    first_page = min(doc.pages.values(), key=lambda page: page.page_no)
    width, height = first_page.size.width, first_page.size.height
    if width <= 0 or height <= 0:
        return []

    blocks: list[OcrBlock] = []

    for index, item in enumerate(doc.texts):
        text = (item.text or "").strip()
        prov = next((p for p in item.prov if p.page_no == first_page.page_no), None)
        if not text or prov is None:
            continue
        blocks.append(
            OcrBlock(
                id=f"t{index}",
                text=text,
                kind=_KIND_BY_LABEL.get(item.label, "other"),
                bbox=normalize_bbox(prov.bbox, width, height),
            )
        )

    for table_index, table in enumerate(doc.tables):
        if table.prov and table.prov[0].page_no != first_page.page_no:
            continue
        for cell_index, cell in enumerate(table.data.table_cells):
            text = cell.text.strip()
            if not text or cell.bbox is None:
                continue
            blocks.append(
                OcrBlock(
                    id=f"c{table_index}-{cell_index}",
                    text=text,
                    kind="table_cell",
                    bbox=normalize_bbox(cell.bbox, width, height),
                )
            )

    return blocks


def _needs_translation(text: str) -> bool:
    """Numbers, prices, times and punctuation are kept as they are."""
    return any(char.isalpha() for char in text)


@final
class ImageTranslationService:
    def __init__(
        self,
        app_config: AppConfig,
        translation_service: TranslationService,
        conversion_service_factory: Callable[[], DocumentConversionService],
    ) -> None:
        self.app_config = app_config
        self.translation_service = translation_service
        self._conversion_service_factory = conversion_service_factory

    async def ocr(
        self,
        content: bytes,
        source_language: LanguageOrAuto,
        filename: str | None,
        content_type: str | None,
    ) -> ImageOcrResult:
        async with self._conversion_service_factory() as conversion_service:
            doc = await conversion_service.ocr_image(
                content, source_language, filename, content_type
            )
        blocks = extract_blocks(doc)
        logger.info("Photo OCR finished", block_count=len(blocks))
        return ImageOcrResult(blocks=blocks)

    async def _resolve_source_language(
        self, blocks: list[BlockToTranslate], config: TranslationConfig
    ) -> LanguageOrAuto | None:
        """Detect the language once for the whole photo instead of per (short) block."""
        if config.source_language and config.source_language != DetectLanguage.AUTO:
            return config.source_language
        page_text = "\n".join(block.text for block in blocks)
        detected = await self.translation_service.detect_language(
            DetectLanguageInput(text=page_text)
        )
        return detected.language

    async def translate_blocks(
        self, blocks: list[BlockToTranslate], config: TranslationConfig
    ) -> AsyncGenerator[BlockTranslatedEvent, None]:
        """Translate each block, yielding results as they complete (not in input order).

        Up to `image_translation_concurrency` blocks are translated at once. Closing the
        generator (client disconnect) cancels the translations still running.
        """
        if not blocks:
            return

        source_language = await self._resolve_source_language(blocks, config)
        page_text = "\n".join(block.text for block in blocks)[:PAGE_CONTEXT_MAX_CHARS]
        block_config = config.model_copy(
            update={
                "source_language": source_language,
                "context": (
                    f"{config.context}\n\n"
                    "The text to translate is one block of text recognised in a photo "
                    "(a sign, a letter, a menu, a form, ...). Translate only that block. "
                    f"All text recognised in the photo, for context:\n{page_text}"
                ),
            }
        )
        semaphore = asyncio.Semaphore(self.app_config.image_translation_concurrency)

        async def translate_one(block: BlockToTranslate) -> BlockTranslatedEvent:
            if not _needs_translation(block.text):
                return BlockTranslatedEvent(id=block.id, translated=block.text)
            async with semaphore:
                parts = [
                    part
                    async for part in self.translation_service.translate_text(
                        block.text, block_config
                    )
                ]
            return BlockTranslatedEvent(id=block.id, translated="".join(parts).strip())

        tasks = [asyncio.create_task(translate_one(block)) for block in blocks]
        try:
            for next_done in asyncio.as_completed(tasks):
                yield await next_done
        finally:
            for task in tasks:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)
