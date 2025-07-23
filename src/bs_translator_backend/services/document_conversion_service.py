from io import BytesIO
from typing import final
from uuid import RESERVED_FUTURE

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ExcelFormatOption,
    PdfFormatOption,
    WordFormatOption,
)
from docling_core.types.io import DocumentStream
from fastapi import UploadFile
from PIL import Image
from PIL.Image import Image

from bs_translator_backend.models.conversion_result import ConversionResult


def create_converter() -> DocumentConverter:
    """Create a document converter with default settings."""
    # accelerator_device: AcceleratorDevice = (
    #     AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    # )
    accelerator_device: AcceleratorDevice = AcceleratorDevice.CPU

    accelerator_options: AcceleratorOptions = AcceleratorOptions(num_threads=4, device=accelerator_device)
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.MD,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.CSV,
            InputFormat.IMAGE,
            InputFormat.XLSX,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_table_structure=True,
                    table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
                    accelerator_options=accelerator_options,
                    generate_picture_images=True,
                )
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_options=PipelineOptions(
                    accelerator_options=accelerator_options,
                )
            ),
            InputFormat.XLSX: ExcelFormatOption(
                pipeline_options=PipelineOptions(
                    accelerator_options=accelerator_options,
                )
            ),
        },
    )


@final
class DocumentConversionService:
    def __init__(self):
        self.converter = create_converter()

    def convert(self, file: UploadFile) -> ConversionResult:
        document_stream = DocumentStream(
            name=file.filename if file.filename else "uploaded_document",
            stream=BytesIO(file.file.read()),
        )

        result = self.converter.convert(document_stream)

        image_tag = "<<IMG>>"

        images: dict[int, Image] = {} 

        markdown = result.document.export_to_markdown(
            image_placeholder=image_tag,
        ).strip()

        for i, picture in enumerate(result.document.pictures):
            img_name = f"img{i + 1}.png"

            markdown = markdown.replace(image_tag, f"![{img_name}]({img_name})", 1)

            img = picture.get_image(result.document)
            if img is not None:
                images[i + 1] = img

        return ConversionResult(
            markdown=markdown,
            images=images
        )
