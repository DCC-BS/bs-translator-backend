from pydantic import BaseModel

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


class BBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: str = "TOPLEFT"


class ConversionImageTextEntry(BaseModel):
    original: str
    translated: str
    bbox: BBox


class ConversionImageOutput(BaseModel):
    items: list[ConversionImageTextEntry]
