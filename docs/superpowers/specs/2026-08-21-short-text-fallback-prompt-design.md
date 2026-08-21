# Design Specification: Short Text Minimal Translation Prompt Fallback

**Date:** 2026-08-21
**Status:** Approved

## Overview
When translating short text inputs (1-3 words) where the source language is set to `auto` (or `None`), language detection is frequently unreliable or fails. Previously, the backend defaulted `source_language` to German (`Language.DE`). When the target language was also German (`Language.DE`), this caused a false-positive match on the `source_language == target_language` check, causing the backend to return the original text without translating it.

This feature adds a fallback path for short text: when auto-detection is untrustworthy, we do not assume German as the source language, we bypass the same-language skip check, and we construct a minimal, clean translation prompt without blank metadata headers.

---

## 1. Language Resolution and Skip Logic

### Current Behavior
In `TranslationService.translate_text`:
1. If `source_language` is `AUTO` / `None`, run `detect_language(text)`.
2. Extract language or fallback to `Language.DE` via `.value_or(Language.DE)`.
3. Set `config.source_language` to `Language.DE`.
4. If `config.source_language == config.target_language`, return `text` unchanged.

### Proposed Behavior
1. **Single-character early return:**
   Keep `if not text.strip() or len(text.strip()) == 1: yield text; return` intact.
2. **Determine source language trustworthiness:**
   * If `config.source_language` is explicitly specified (e.g., `Language.FR` or `Language.DE`), set `source_language_trustworthy = True`.
   * If `config.source_language` is `DetectLanguage.AUTO` or `None`:
     * Execute `detection_result = detect_language(text)`.
     * If detection is successful and confidence is at or above `SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD` (0.9 for short text) or standard threshold for long text:
       * Set `config.source_language = detected_language`.
       * Set `source_language_trustworthy = True`.
     * Otherwise (low confidence or unsupported/failed detection):
       * Do not assign `config.source_language` to `Language.DE`. Leave it unasserted (`None` or `DetectLanguage.AUTO`).
       * Set `source_language_trustworthy = False`.
3. **Same-language skip condition:**
   * Only skip translation (`yield text; return`) when `source_language_trustworthy` is `True` AND `config.source_language == config.target_language`.
   * If `source_language_trustworthy` is `False`, never skip translation.

---

## 2. Minimal Prompt Formatting

### `_create_short_text_user_message`
Dynamically build the user message, omitting lines for metadata fields that are empty:

1. **Instruction:**
   * If `assert_source_language` is `True`:
     `Translate the following text from {source_language_name} into {target_language_name}.`
   * If `assert_source_language` is `False`:
     `Translate the following text into {target_language_name}.`
2. **Metadata fields:**
   * `Domain: {domain}` (omitted if empty)
   * `Tone: {tone}` (omitted if empty)
   * `Glossary: {glossary}` (omitted if empty)
   * `Context:\n{context}` (omitted if empty)
3. **Payload:**
   ```text
   Text to translate:
   {text}
   ```

When metadata fields are empty and source language is untrusted, the message is minimal:
```text
Translate the following text into German.

Text to translate:
hi
```

---

## 3. Testing and Validation Plan

### Unit Tests (`tests/unit/test_translation_service.py`)
1. **Fallback when target is German:** Auto-detected short text with low confidence does not skip when target is `Language.DE`, and routes to `short_text_translation_agent`.
2. **Clean minimal prompt output:** Assert that `_create_short_text_user_message` contains no blank `Domain:`, `Tone:`, `Glossary:`, or `Context:` lines when config fields are empty.
3. **Populated metadata prompt output:** Assert that non-empty metadata fields are included in the prompt.
4. **Explicit same-language skip:** Explicit `source_language=Language.DE` and `target_language=Language.DE` still returns early without calling LLM agents.
5. **Single-character early return:** Single character inputs (`len == 1`) return immediately without LLM calls.

### E2E Tests (`tests/e2e/test_translation_routes.py`)
1. **E2E short text fallback translation:** Post `/translation/text` with `source_language="auto"`, `target_language="de"`, and `text="hi"`, verifying the streaming response and inspecting the LLM request body.

### Linter and Type Checks
* `uv run pytest`
* `uv run ruff check`
* `uv run ruff format --check`
* `uv run mypy`
