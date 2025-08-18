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
