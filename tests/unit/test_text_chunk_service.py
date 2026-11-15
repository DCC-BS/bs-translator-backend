import pytest

from bs_translator_backend.services.text_chunk_service import TextChunkService


def _chunks(text: str, max_tokens: int) -> list[str]:
    chunks = TextChunkService(max_tokens=max_tokens).chunk_text(text)
    # Chunking must be lossless: the chunks put back together are the input.
    assert "".join(chunks) == text
    assert all(chunk.strip() for chunk in chunks), "no empty or whitespace-only chunks"
    return chunks


def test_empty_text_has_no_chunks() -> None:
    assert TextChunkService().chunk_text("") == []


@pytest.mark.parametrize("text", ["Hallo", "Hallo\n", "line one\nline two"])
def test_short_text_is_one_unmodified_chunk(text: str) -> None:
    """No artificial trailing newline is added."""
    assert _chunks(text, max_tokens=100) == [text]


def test_chunks_end_with_their_original_newlines() -> None:
    text = "first paragraph\n\nsecond paragraph\nthird line"
    chunks = _chunks(text, max_tokens=20)
    assert chunks == ["first paragraph\n\n", "second paragraph\n", "third line"]


def test_overlong_first_line_does_not_yield_an_empty_chunk() -> None:
    text = "a" * 30 + "\nshort"
    chunks = _chunks(text, max_tokens=10)
    assert chunks[0].startswith("a")


def test_overlong_line_is_split_at_sentences() -> None:
    text = "One sentence here. Another sentence here. A third one."
    chunks = _chunks(text, max_tokens=25)
    assert chunks == ["One sentence here. ", "Another sentence here. ", "A third one."]
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_overlong_sentence_is_split_at_words() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    chunks = _chunks(text, max_tokens=12)
    assert all(len(chunk) <= 12 for chunk in chunks)
    assert all(not chunk.startswith(" ") for chunk in chunks)


def test_word_longer_than_max_is_hard_cut() -> None:
    chunks = _chunks("x" * 25, max_tokens=10)
    assert [len(chunk) for chunk in chunks] == [10, 10, 5]


def test_cjk_sentences_are_split_at_ideographic_full_stop() -> None:
    text = "这是第一句话。这是第二句话。"
    assert _chunks(text, max_tokens=8) == ["这是第一句话。", "这是第二句话。"]


def test_markdown_table_is_not_split() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    text = "Intro line\n" + table + "Outro"
    chunks = _chunks(text, max_tokens=len(table) + 5)
    assert table in chunks[1]


def test_fenced_code_block_is_not_split() -> None:
    code = "```python\nprint('a')\nprint('b')\n```\n"
    text = "Intro line\n" + code + "Outro"
    chunks = _chunks(text, max_tokens=len(code) + 2)
    assert any(code in chunk for chunk in chunks)


def test_block_larger_than_max_falls_back_to_lines() -> None:
    table = "".join(f"| row {i} |\n" for i in range(10))
    chunks = _chunks(table, max_tokens=30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
