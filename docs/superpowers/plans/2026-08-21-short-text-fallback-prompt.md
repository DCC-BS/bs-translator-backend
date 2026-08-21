# Short Text Minimal Translation Prompt Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a minimal translation prompt fallback for short text lookups when language auto-detection is unreliable, preventing false-positive skip conditions when target language is German.

**Architecture:** Update `TranslationService._create_short_text_user_message` to omit empty metadata headers, and update `TranslationService.translate_text` to only skip same-language translations when the source language is explicitly provided or detected with high confidence.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic, Returns library (`ResultE`, `Success`, `Failure`), Pytest, HTTPX (e2e stubbed transport).

## Global Constraints

- Do not import `pydantic_ai` in `translation_service.py` (architectural boundary).
- Keep single-character early return (`len == 1`) unchanged.
- Preserve exact whitespace handling without trimming.
- Adhere to `SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD = 0.9`.

---

### Task 1: Clean Minimal Prompt Generation in Short Text User Message

**Files:**
- Modify: `src/bs_translator_backend/services/translation_service.py:101-137`
- Test: `tests/unit/test_translation_service.py:109-136`

**Interfaces:**
- Consumes: `TranslationConfig` (model with `target_language`, `source_language`, `domain`, `tone`, `glossary`, `context`)
- Produces: `TranslationService._create_short_text_user_message(text: str, translation_config: TranslationConfig, assert_source_language: bool = True) -> str`

- [ ] **Step 1: Write unit tests for minimal and populated short text user messages**

Add tests to `tests/unit/test_translation_service.py`:
```python
def test_short_text_user_message_omits_empty_metadata_fields(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(
        source_language=Language.DE,
        target_language=Language.FR,
        domain="",
        tone="",
        glossary="",
        context="",
    )

    message = translation_service._create_short_text_user_message(
        text="Hirsch",
        translation_config=config,
        assert_source_language=False,
    )

    expected = "Translate the following text into French.\n\nText to translate:\nHirsch\n"
    assert message == expected
    assert "Domain:" not in message
    assert "Tone:" not in message
    assert "Glossary:" not in message
    assert "Context:" not in message


def test_short_text_user_message_includes_non_empty_metadata_fields(
    translation_service: TranslationService,
) -> None:
    config = TranslationConfig(
        source_language=Language.DE,
        target_language=Language.FR,
        domain="Legal",
        tone="Formal",
        glossary="Hirsch -> Cerf",
        context="Forest document",
    )

    message = translation_service._create_short_text_user_message(
        text="Hirsch",
        translation_config=config,
        assert_source_language=True,
    )

    assert "Translate the following text from German into French." in message
    assert "Domain: Legal" in message
    assert "Tone: Formal" in message
    assert "Glossary: Hirsch -> Cerf" in message
    assert "Context:\nForest document" in message
    assert "Text to translate:\nHirsch\n" in message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_translation_service.py::test_short_text_user_message_omits_empty_metadata_fields -v`
Expected: FAIL due to existing empty metadata headers.

- [ ] **Step 3: Implement minimal prompt formatting in `_create_short_text_user_message`**

Update `src/bs_translator_backend/services/translation_service.py`:
```python
    def _create_short_text_user_message(
        self,
        text: str,
        translation_config: TranslationConfig,
        assert_source_language: bool = True,
    ) -> str:
        """Create the prompt message for the short-text (lexical lookup) translation agent."""
        target_language_name: str = get_language_name(translation_config.target_language)
        if assert_source_language and translation_config.source_language:
            source_language_name: str = get_language_name(translation_config.source_language)
            instruction = (
                f"Translate the following text from {source_language_name} "
                f"into {target_language_name}."
            )
        else:
            instruction = f"Translate the following text into {target_language_name}."

        lines = [instruction]
        if translation_config.domain:
            lines.append(f"Domain: {translation_config.domain}")
        if translation_config.tone:
            lines.append(f"Tone: {translation_config.tone}")
        if translation_config.glossary:
            lines.append(f"Glossary: {translation_config.glossary}")
        if translation_config.context:
            lines.append(f"Context:\n{translation_config.context}")

        metadata_block = "\n".join(lines)
        return f"""{metadata_block}

Text to translate:
{text}
"""
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `uv run pytest tests/unit/test_translation_service.py -k "test_short_text_user_message" -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1 changes**

```bash
git add src/bs_translator_backend/services/translation_service.py tests/unit/test_translation_service.py
git commit -m "feat: strip empty metadata lines in short text translation prompt"
```

---

### Task 2: Untrusted Auto-Detection Fallback and Skip Logic

**Files:**
- Modify: `src/bs_translator_backend/services/translation_service.py:146-175`
- Test: `tests/unit/test_translation_service.py`

**Interfaces:**
- Consumes: `detect_language(text: str) -> ResultE[DetectLanguageOutput]`
- Produces: `TranslationService.translate_text(text: str, config: TranslationConfig) -> AsyncGenerator[str, None]`

- [ ] **Step 1: Write failing unit tests for short text fallback when target is German**

Add tests to `tests/unit/test_translation_service.py`:
```python
@pytest.mark.asyncio
async def test_auto_source_low_confidence_does_not_skip_when_target_is_german(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low confidence detection must not default to German and skip when target is German."""
    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Success(DetectLanguageOutput(language=Language.DE, confidence=0.35)),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.DE)

    chunks = [c async for c in translation_service.translate_text("Hirsch", config)]

    assert "".join(chunks) == "short-agent-output"
    translation_service.short_text_translation_agent.run_stream_text.assert_called_once()
    prompt = translation_service.short_text_translation_agent.run_stream_text.call_args.kwargs[
        "user_prompt"
    ]
    assert "Translate the following text into German." in prompt
    assert "from" not in prompt


@pytest.mark.asyncio
async def test_auto_source_failed_detection_routes_to_short_agent_when_target_is_german(
    translation_service: TranslationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed language detection must route to short-text agent with minimal prompt."""
    from returns.result import Failure

    monkeypatch.setattr(
        "bs_translator_backend.services.translation_service.detect_language",
        lambda _text: Failure(Exception("Unsupported")),
    )
    config = TranslationConfig(source_language=DetectLanguage.AUTO, target_language=Language.DE)

    chunks = [c async for c in translation_service.translate_text("hi", config)]

    assert "".join(chunks) == "short-agent-output"
    translation_service.short_text_translation_agent.run_stream_text.assert_called_once()
    prompt = translation_service.short_text_translation_agent.run_stream_text.call_args.kwargs[
        "user_prompt"
    ]
    assert "Translate the following text into German." in prompt


@pytest.mark.asyncio
async def test_explicit_same_source_and_target_still_skips(
    translation_service: TranslationService,
) -> None:
    """Explicitly setting source=DE and target=DE must still short-circuit without calling agents."""
    config = TranslationConfig(source_language=Language.DE, target_language=Language.DE)

    chunks = [c async for c in translation_service.translate_text("Hirsch", config)]

    assert chunks == ["Hirsch"]
    translation_service.short_text_translation_agent.run_stream_text.assert_not_called()
    translation_service.translation_agent.run_stream_text.assert_not_called()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_translation_service.py -k "test_auto_source_low_confidence_does_not_skip_when_target_is_german" -v`
Expected: FAIL (currently skips and returns `["Hirsch"]` because `source_language` defaults to `Language.DE`).

- [ ] **Step 3: Implement trustworthy detection and skip logic in `translate_text`**

Update `src/bs_translator_backend/services/translation_service.py`:
```python
use_short_text_agent = _is_short_text(text)

if not config.source_language or config.source_language == DetectLanguage.AUTO:
    detection_result = detect_language(text)
    detected_confidence = detection_result.map(lambda result: result.confidence).value_or(0.0)
    detected = detection_result.map(lambda result: result.language).value_or(None)

    if (
        detected is not None
        and not isinstance(detected, DetectLanguage)
        and (
            detected_confidence >= SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD
            if use_short_text_agent
            else True
        )
    ):
        config.source_language = detected
        source_language_trustworthy = True
    else:
        source_language_trustworthy = False
else:
    # The caller (i.e. the user, via the UI) explicitly chose this language.
    source_language_trustworthy = True

if source_language_trustworthy and config.source_language == config.target_language:
    yield text
    return
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `uv run pytest tests/unit/test_translation_service.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit Task 2 changes**

```bash
git add src/bs_translator_backend/services/translation_service.py tests/unit/test_translation_service.py
git commit -m "feat: fallback to minimal translation prompt when short text language is untrusted"
```

---

### Task 3: End-to-End Test and Full Suite Verification

**Files:**
- Modify: `tests/e2e/test_translation_routes.py`

- [ ] **Step 1: Add e2e test for short text fallback with target=DE and source=auto**

Add to `tests/e2e/test_translation_routes.py` under `TestPromptRouting`:
```python
    @pytest.mark.asyncio
    async def test_short_input_auto_source_translating_into_german_uses_fallback_prompt(
        self, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:
        fake_llm.respond_with("Hallo")

        chunks: list[str] = []
        async with client.stream(
            "POST",
            "/translation/text",
            json={
                "text": "hi",
                "config": {"source_language": "auto", "target_language": "de"},
            },
            headers=CLIENT_HEADERS,
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                chunks.append(chunk)

        assert "".join(chunks) == "Hallo"
        assert len(fake_llm.requests) == 1
        body = fake_llm.last_body()
        system_prompt = fake_llm.system_prompt(body)
        user_message = fake_llm.user_message(body)

        assert "dictionary-style lookup" in system_prompt
        assert "Translate the following text into German." in user_message
        assert "from" not in user_message
        assert "Domain:" not in user_message
        assert "hi" in user_message
```

- [ ] **Step 2: Run e2e tests**

Run: `uv run pytest tests/e2e/test_translation_routes.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite and linters**

Run:
```bash
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy
```
Expected: All tests pass, 0 lint/format/type errors.

- [ ] **Step 4: Commit Task 3 changes**

```bash
git add tests/e2e/test_translation_routes.py
git commit -m "test: add e2e test for short text fallback prompt routing"
```
