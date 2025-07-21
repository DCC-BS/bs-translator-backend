from io import BytesIO
from typing import final
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    MarkdownFormatOption,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.io import DocumentStream
from transformers import AutoTokenizer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def create_converter() -> DocumentConverter:
    """Create a document converter with default settings."""
    # accelerator_device: AcceleratorDevice = (
    #     AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    # )
    accelerator_device: AcceleratorDevice = AcceleratorDevice.CPU

    accelerator_options: AcceleratorOptions = AcceleratorOptions(num_threads=4, device=accelerator_device)
    return DocumentConverter(
        allowed_formats=[
            InputFormat.MD,
        ],
        format_options={
            InputFormat.MD: MarkdownFormatOption(
                pipeline_options=PipelineOptions(
                    accelerator_options=accelerator_options,
                )
            )
        },
    )


@final
class TextChunkService:
    def __init__(self, max_tokens: int = 6_000):
        self.max_tokens = max_tokens

        self.converter = create_converter()

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
            max_tokens=self.max_tokens,
        )

        self.chunker = HybridChunker(tokenizer=tokenizer)

    def chunk_text(self, text: str):
        """Splits the input text into chunks of specified size."""

        stream = DocumentStream(
            name="input.md",
            stream=BytesIO(text.encode("utf-8")),
        )

        document = self.converter.convert(stream).document
        chunks = self.chunker.chunk(document)

        for chunk in chunks:
            yield chunk.text
