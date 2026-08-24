"""Tier 2 e2e tests: real translations against a real LLM endpoint.

These are the only tests in the suite that prove the service actually
translates. Everything else stubs the model, so everything else would keep
passing if the prompts stopped working entirely.

They are opt-in and never run in CI: set ``E2E_LIVE_LLM=1`` together with real
``LLM_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` values, e.g.

    E2E_LIVE_LLM=1 uv run --env-file .env python -m pytest tests/e2e -m live -v

Assertions are deliberately loose. Model output varies between runs and between
models, so these check the properties that must hold -- "it did not hand the
input back unchanged", "it is not empty" -- rather than exact strings. A brittle
exact-match assertion here would be worse than no test, because it would be
disabled the first time it flaked.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from bs_translator_backend.models.language import Language
from bs_translator_backend.models.translation import TranslationConfig
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.app_config import AppConfig

# tests/e2e/conftest.py sets placeholder LLM_* values via os.environ.setdefault at
# import time so the stubbed tier can build an app without real credentials.
# setdefault never overwrites, so a real environment still wins -- but a run that
# forgot to supply one would otherwise point at these placeholders and fail with a
# confusing connection error instead of a clear skip.
_PLACEHOLDER_HOST = ".invalid"
_PLACEHOLDER_API_KEY = "test-llm-key"
_PLACEHOLDER_MODEL = "test/test-model"

pytestmark = pytest.mark.live


def _live_config() -> AppConfig:
    """Build config from the real environment, skipping if it is not configured."""
    if os.getenv("E2E_LIVE_LLM") != "1":
        pytest.skip("live LLM tests are opt-in; set E2E_LIVE_LLM=1 to run them")

    llm_url = os.getenv("LLM_URL", "")
    if not llm_url or _PLACEHOLDER_HOST in llm_url:
        pytest.skip(
            f"LLM_URL is unset or still the test placeholder ({llm_url!r}); "
            "supply a real endpoint, e.g. via --env-file .env"
        )
    if os.getenv("LLM_API_KEY", "") in ("", _PLACEHOLDER_API_KEY):
        pytest.skip("LLM_API_KEY is unset or still the test placeholder; supply a real key")
    if os.getenv("LLM_MODEL", "") in ("", _PLACEHOLDER_MODEL):
        pytest.skip("LLM_MODEL is unset or still the test placeholder; supply a real model")

    return AppConfig.from_env()


@pytest_asyncio.fixture
async def translation_service() -> AsyncIterator[TranslationService]:
    """A TranslationService wired to the real LLM. Docling is never reached here."""
    service = TranslationService(_live_config(), TextChunkService(), lambda: MagicMock())
    yield service
    await service.aclose()


async def _translate(service: TranslationService, text: str, config: TranslationConfig) -> str:
    return "".join([chunk async for chunk in service.translate_text(text, config)])


def _de_to_fr() -> TranslationConfig:
    return TranslationConfig(
        target_language=Language.FR,
        source_language=Language.DE,
        domain="General",
        tone="Keep the tone of the source text",
        glossary="",
        context="",
    )


class TestShortTextTranslation:
    """The reported bug: single words came back untranslated because the model
    read a capitalized German noun as a proper name."""

    @pytest.mark.asyncio
    async def test_single_word_is_actually_translated(
        self, translation_service: TranslationService
    ) -> None:
        result = await _translate(translation_service, "Hirsch", _de_to_fr())

        assert result.strip(), "translation must not be empty"
        assert result.strip().lower() != "hirsch", (
            f"'Hirsch' came back untranslated ({result!r}) -- the short-text prompt "
            "is not overriding the proper-noun assumption"
        )

    @pytest.mark.asyncio
    async def test_short_noun_phrase_is_translated(
        self, translation_service: TranslationService
    ) -> None:
        result = await _translate(translation_service, "Der Hirsch", _de_to_fr())

        assert result.strip()
        assert "hirsch" not in result.strip().lower()


class TestLongTextTranslation:
    @pytest.mark.asyncio
    async def test_sentence_is_translated(self, translation_service: TranslationService) -> None:
        source = "Der grosse braune Hirsch lief schnell durch den dunklen Wald."
        result = await _translate(translation_service, source, _de_to_fr())

        assert result.strip()
        assert result.strip() != source

    @pytest.mark.asyncio
    async def test_markdown_structure_is_preserved(
        self, translation_service: TranslationService
    ) -> None:
        source = "# Titel\n\n- Erster Punkt\n- Zweiter Punkt\n"
        result = await _translate(translation_service, source, _de_to_fr())

        assert result.startswith("#"), f"heading marker was dropped: {result!r}"
        assert result.count("\n- ") == 2, f"list markers were not preserved: {result!r}"


class TestSwissGermanOrthography:
    @pytest.mark.asyncio
    async def test_output_never_contains_eszett(
        self, translation_service: TranslationService
    ) -> None:
        """Basel-Stadt writes 'ss', never 'ß' -- enforced by the output transform,
        not by the prompt, so this must hold for any model."""
        config = TranslationConfig(
            target_language=Language.DE,
            source_language=Language.EN_US,
            domain="General",
            tone="Keep the tone of the source text",
            glossary="",
            context="",
        )
        source = "The street was very large and the measure was excessive."
        result = await _translate(translation_service, source, config)

        assert result.strip()
        assert "ß" not in result, f"eszett leaked into the output: {result!r}"
