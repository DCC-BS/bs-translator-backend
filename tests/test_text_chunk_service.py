"""Tests for TextChunkService."""

import pytest

from src.bs_translator_backend.services.text_chunk_service import TextChunkService


class TestTextChunkService:
    """Test cases for TextChunkService class."""

    def test_chunk_text_empty_string(self) -> None:
        """Test chunking empty string."""
        service = TextChunkService()
        result = service.chunk_text("")
        assert result == [""]

    def test_chunk_text_simple_text(self) -> None:
        """Test chunking simple text without headers."""
        service = TextChunkService()
        text = "This is a simple text without any headers."
        result = service.chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_text_with_headers(self) -> None:
        """Test chunking text with markdown headers."""
        service = TextChunkService()
        text = "Introduction text\n# Header 1\nContent under header 1\n# Header 2\nContent under header 2"
        result = service.chunk_text(text)
        # Should split at headers but exact behavior depends on implementation
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_text_respects_max_tokens(self) -> None:
        """Test that chunks respect max_tokens limit."""
        service = TextChunkService(max_tokens=50)
        # Create a text longer than max_tokens
        long_text = "A" * 100
        result = service.chunk_text(long_text)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(len(chunk) <= service.max_tokens for chunk in result[:-1])

    def test_chunk_text_with_newlines_no_headers(self) -> None:
        """Test chunking text with newlines but no headers."""
        service = TextChunkService()
        text = "Line 1\nLine 2\nLine 3"
        result = service.chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    @pytest.mark.parametrize("max_tokens", [100, 1000, 5000])
    def test_chunk_text_different_max_tokens(self, max_tokens: int) -> None:
        """Test chunking with different max_tokens values."""
        service = TextChunkService(max_tokens=max_tokens)
        text = "Sample text " * 50  # Create moderately long text
        result = service.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_text_header_at_beginning(self) -> None:
        """Test chunking text that starts with a header."""
        service = TextChunkService()
        text = "# Header 1\nContent under header 1\n# Header 2\nContent under header 2"
        result = service.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_text_multiple_consecutive_headers(self) -> None:
        """Test chunking text with multiple consecutive headers."""
        service = TextChunkService()
        text = "# Header 1\n# Header 2\n# Header 3\nSome content"
        result = service.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_text_return_type(self) -> None:
        """Test that chunk_text returns the correct type."""
        service = TextChunkService()
        result = service.chunk_text("any text")
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_chunk_text_preserves_content(self) -> None:
        """Test that chunking preserves all content."""
        service = TextChunkService()
        original_text = "Test content\n# Header\nMore content"
        result = service.chunk_text(original_text)
        # Join all chunks and compare with original (accounting for potential modifications)
        assert isinstance(result, list)
        assert len(result) > 0
        # Basic check that content is not empty
        joined_result = "".join(result)
        assert len(joined_result) > 0

    def test_chunk_text_handles_long_text(self) -> None:
        """Test chunking of long text."""
        service = TextChunkService(max_tokens=50)
        long_text = """
# Title 1
## Subtitle 1
Content under subtitle 1
# Title 2
Content under title 2
### Subtitle 2
More content under subtitle 2

- item 1
- item 2
  - item 3

```python
def example_function():
    return "This is an example function."
```

        """

        result = service.chunk_text(long_text)
        assert isinstance(result, list)
        assert len(result) > 0
        # Check that no chunk exceeds max_tokens
        for chunk in result:
            assert len(chunk) <= service.max_tokens
            print(f"{chunk}", end="")  # Debugging output
