from PIL.Image import Image
from pydantic import BaseModel


class ConversionResult:
    markdown: str
    images: dict[int, Image]

    def __init__(self, markdown: str, images: dict[int, Image]):
        self.markdown = markdown
        self.images = images


class ConversionOutput(BaseModel):
    markdown: str
    images: dict[int, str] = {}
