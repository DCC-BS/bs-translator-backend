import re
from pathlib import Path
from typing import BinaryIO, final

import httpx
from fastapi import HTTPException, UploadFile

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.conversion_result import Base64EncodedImage, ConversionResult
from bs_translator_backend.models.docling_response import (
    DoclingDocument,
    DoclingResponse,
    DocumentResponse,
)
from bs_translator_backend.models.langugage import DetectLanguage, LanguageOrAuto
from bs_translator_backend.services.image_reader_serivice import ImageReaderService
from bs_translator_backend.utils.logger import get_logger

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
    }
    return mimetypes.get(extension, "invalid")


def validate_mimetype(mimetype: str, logger_context: dict[str, any]) -> None:
    if len(mimetype) == 0:
        logger.error("MIME type is empty", extra=logger_context)
        raise HTTPException(status_code=400, detail="Invalid MIME type")
    if mimetype == "invalid":
        logger.error("Invalid MIME type", extra=logger_context)
        raise HTTPException(status_code=400, detail="Invalid document type")


def extract_docling_document(response: str, logger_context: dict[str, any]) -> DocumentResponse:
    docling_response = DoclingResponse.model_validate(response)
    if docling_response.document.json_content is None:
        logger.error(
            "Docling response does not contain a document",
            extra=logger_context,
        )
        raise HTTPException(status_code=500, detail="Document conversion failed")

    return docling_response.document


@final
class DocumentConversionService:
    def __init__(self, config: AppConfig) -> None:
        self.image_reader = ImageReaderService()
        self.config = config

    def fetch_docling_file_convert(
        self,
        files: dict[str, tuple[str, BinaryIO, str]],
        options: dict[str, str | list[str] | bool],
    ) -> httpx.Response:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                self.config.docling_url + "/convert/file",
                files=files,
                data=options,
            )

            if response.status_code <= 200:
                return response
            else:
                # For error responses, safely handle potential binary content
                try:
                    error_text = response.text
                    logger.error(f"Error response: {error_text}")

                    raise HTTPException(status_code=response.status_code, detail=error_text)
                except UnicodeDecodeError as e:
                    logger.exception(
                        f"Error response contains binary data (status: {response.status_code})"
                    )
                    raise HTTPException(
                        status_code=response.status_code, detail="Binary data in error response"
                    ) from e

    def convert_to_docling(self, file: UploadFile, source_lang: LanguageOrAuto) -> DoclingDocument:
        languages = [source_lang.value]

        if source_lang == DetectLanguage.AUTO:
            languages = ["de", "en", "fr", "it"]

        content = file.file
        filename = file.filename or "uploaded_document"
        content_type = get_mimetype(Path(filename))
        validate_mimetype(content_type, logger_context={"content_type": content_type})

        files = {"files": (filename, content, content_type)}
        options: dict[str, str | list[str] | bool] = {
            "to_formats": ["json"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "ocr_engine": "easyocr",
            "ocr_lang": languages,
            "table_mode": "accurate",
            "pdf_backend": "pypdfium2",
        }

        logger_context = {"options": options, "content_type": content_type}

        response = self.fetch_docling_file_convert(files, options)
        json_response = response.json()

        document = extract_docling_document(json_response, logger_context)

        if document.json_content is None:
            logger.error("Docling response does not contain a document", extra=logger_context)
            raise HTTPException(status_code=500, detail="Document conversion failed")

        return document.json_content

    def convert(self, file: UploadFile, source_lang: LanguageOrAuto) -> ConversionResult:
        languages = [source_lang.value]

        if source_lang == DetectLanguage.AUTO:
            languages = ["de", "en", "fr", "it"]

        content = file.file
        filename = file.filename or "uploaded_document"
        content_type = get_mimetype(Path(filename))
        validate_mimetype(content_type, logger_context={"content_type": content_type})

        files = {"files": (filename, content, content_type)}
        options: dict[str, str | list[str] | bool] = {
            "to_formats": ["md", "json"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "ocr_engine": "easyocr",
            "ocr_lang": languages,
            "table_mode": "accurate",
            "pdf_backend": "pypdfium2",
        }

        response = self.fetch_docling_file_convert(files, options)
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
