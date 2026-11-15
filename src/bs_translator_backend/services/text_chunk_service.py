import re
from collections.abc import Iterator
from typing import final

# Opening line of a fenced code block: up to three spaces, then ``` or ~~~ (or longer).
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# A markdown table row starts with a pipe.
_TABLE_ROW = re.compile(r"^\s*\|")
# A sentence plus its trailing whitespace (Latin and CJK sentence punctuation).
_SENTENCE = re.compile(r".+?(?:[.!?\u3002\uff01\uff1f]+\s*|\Z)", re.DOTALL)
# A word plus its trailing whitespace, or a run of leading whitespace.
_WORD = re.compile(r"\S+\s*|\s+")


def _lines(text: str) -> list[str]:
    """Split on "\\n" only, each line keeping its newline (str.splitlines splits on more)."""
    lines = [line + "\n" for line in text.split("\n")]
    lines[-1] = lines[-1][:-1]
    return [line for line in lines if line]


def _is_fence_close(line: str, marker: str) -> bool:
    return re.match(rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*$", line) is not None


def _blocks(text: str) -> Iterator[str]:
    """Lines of `text`, with fenced code blocks and markdown tables kept as one block each."""
    lines = _lines(text)
    i = 0
    while i < len(lines):
        if fence := _FENCE_OPEN.match(lines[i]):
            end = i + 1
            while end < len(lines) and not _is_fence_close(lines[end], fence.group(1)):
                end += 1
            end = min(end + 1, len(lines))
        elif _TABLE_ROW.match(lines[i]):
            end = i
            while end < len(lines) and _TABLE_ROW.match(lines[end]):
                end += 1
        else:
            end = i + 1
        yield "".join(lines[i:end])
        i = end


@final
class TextChunkService:
    """Splits text into chunks of at most `max_tokens` characters (characters, not tokens).

    Concatenating the chunks gives back the input exactly: every chunk keeps its own
    trailing newline(s), so a caller translating chunk by chunk can re-emit them between
    the translations. Chunks are cut at line boundaries, never inside a fenced code
    block or a markdown table unless that block alone exceeds the limit. A line that
    alone exceeds the limit is split at sentence, then word boundaries.
    """

    def __init__(self, max_tokens: int = 6_000):
        self.max_tokens = max_tokens

    def chunk_text(self, text: str) -> list[str]:
        """Splits the input text into chunks of specified size. No chunk is empty."""
        chunks: list[str] = []
        current = ""

        for block in _blocks(text):
            for piece in self._fit(block):
                # Blank lines never start a chunk: they stay with the text they follow.
                is_blank = not piece.strip()
                if current and not is_blank and len(current) + len(piece) > self.max_tokens:
                    chunks.append(current)
                    current = ""
                current += piece

        if current:
            chunks.append(current)
        return chunks

    def _fit(self, block: str) -> list[str]:
        """`block` itself if it fits, otherwise its lines, each split to fit."""
        if len(block) <= self.max_tokens:
            return [block]
        return [piece for line in _lines(block) for piece in self._split_overlong(line)]

    def _split_overlong(self, text: str) -> list[str]:
        """Split `text` into pieces that fit: at sentences, then words, then hard cuts."""
        if len(text) <= self.max_tokens:
            return [text]
        for pattern in (_SENTENCE, _WORD):
            units = pattern.findall(text)
            if len(units) > 1:
                return [piece for unit in units for piece in self._split_overlong(unit)]
        return [text[i : i + self.max_tokens] for i in range(0, len(text), self.max_tokens)]
