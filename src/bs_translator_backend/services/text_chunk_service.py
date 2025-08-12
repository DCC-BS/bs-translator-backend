from typing import final


@final
class TextChunkService:
    def __init__(self, max_tokens: int = 6_000):
        self.max_tokens = max_tokens

    def chunk_text(self, text: str) -> list[str]:
        """Splits the input text into chunks of specified size."""

        segments: list[str] = text.split("\n")

        chunks: list[str] = []
        current_chunk = ""

        for segment in segments:
            if (len(current_chunk) + len(segment) + 1) > self.max_tokens:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk += segment + "\n"

        chunks.append(current_chunk)

        return chunks
