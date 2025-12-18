from typing import cast

from dspy.streaming.messages import StreamResponse
from dspy.streaming.streaming_listener import StreamListener
from litellm.types.utils import ModelResponseStream

from bs_translator_backend.services.dspy_config.translation_program import (
    SwissGermanStreamListener,
)


class _FakeDelta:
    def __init__(self, content: str | None, reasoning_content: str | None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta: _FakeDelta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, content: str | None, reasoning_content: str | None = None):
        self.choices = [_FakeChoice(_FakeDelta(content, reasoning_content))]


def test_receive_normalizes_content(monkeypatch):
    listener = SwissGermanStreamListener("translated_text")
    listener.predict_name = "predict"

    captured: dict[str, str | None] = {}

    def fake_parent_receive(self, chunk):
        captured["content"] = chunk.choices[0].delta.content
        return StreamResponse(
            self.predict_name,
            self.signature_field_name,
            chunk.choices[0].delta.content,
            is_last_chunk=False,
        )

    monkeypatch.setattr(StreamListener, "receive", fake_parent_receive)

    chunk = _FakeChunk(content="große Straße")
    response = listener.receive(cast(ModelResponseStream, chunk))  # type: ignore[arg-type]

    assert captured["content"] == "grosse Strasse"
    assert isinstance(response, StreamResponse)
    assert response.chunk == "grosse Strasse"


def test_receive_uses_reasoning_when_content_missing(monkeypatch):
    listener = SwissGermanStreamListener("translated_text")
    listener.predict_name = "predict"

    def fake_parent_receive(self, chunk):
        return StreamResponse(
            self.predict_name,
            self.signature_field_name,
            chunk.choices[0].delta.content,
            is_last_chunk=False,
        )

    monkeypatch.setattr(StreamListener, "receive", fake_parent_receive)

    chunk = _FakeChunk(content=None, reasoning_content="heiße Größe")
    response = listener.receive(cast(ModelResponseStream, chunk))  # type: ignore[arg-type]

    assert isinstance(response, StreamResponse)
    assert response.chunk == "heisse Grösse"


def test_disable_reasoning_adapter_is_registered():
    listener = SwissGermanStreamListener("translated_text")
    assert "DisableReasoningAdapter" in listener.adapter_identifiers
    assert (
        listener.adapter_identifiers["DisableReasoningAdapter"]
        == listener.adapter_identifiers["ChatAdapter"]
    )
