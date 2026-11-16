from typing import Literal

from pydantic import BaseModel, Field

from bs_translator_backend.models.translation import TranslationConfig

type Base64EncodedImage = str


class ConversionResult:
    markdown: str
    images: dict[int, Base64EncodedImage]

    def __init__(self, markdown: str, images: dict[int, Base64EncodedImage]):
        self.markdown = markdown
        self.images = images


class ConversionOutput(BaseModel):
    markdown: str
    images: dict[int, str] = {}


class NormalizedBox(BaseModel):
    """A rectangle relative to the image: (0, 0) is the top-left corner, (1, 1) the bottom-right.

    Normalized coordinates make the box independent of the image resolution and of the
    unit docling measures in (points, i.e. pixels * 72 / dpi), so the client only has to
    multiply by the size it draws the image at.
    """

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)


type OcrBlockKind = Literal[
    "title", "heading", "text", "list_item", "caption", "table_cell", "other"
]


class OcrBlock(BaseModel):
    """One piece of text recognised in a photo, with where it stands in the image."""

    id: str
    text: str
    kind: OcrBlockKind
    bbox: NormalizedBox


class ImageOcrResult(BaseModel):
    """Text blocks of a photo in reading order."""

    blocks: list[OcrBlock]


class BlockToTranslate(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=10_000)


class BlockTranslationInput(BaseModel):
    """Request body of the block translation endpoint."""

    blocks: list[BlockToTranslate] = Field(max_length=1000)
    config: TranslationConfig


class BlockTranslatedEvent(BaseModel):
    type: Literal["translation"] = "translation"
    id: str
    translated: str


class BlockTranslationDoneEvent(BaseModel):
    type: Literal["done"] = "done"


class BlockTranslationErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    errorId: str
    status: int
