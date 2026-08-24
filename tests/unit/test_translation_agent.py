from pathlib import Path

import pytest
from dcc_backend_common.llm_agent import BaseAgent

import bs_translator_backend.agents.translation_agent as mod
from bs_translator_backend.agents.translation_agent import (
    ShortTextTranslationAgent,
    TranslationAgent,
)
from bs_translator_backend.utils.app_config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        llm_url="http://localhost:8001/v1",
        llm_api_key="none",
        llm_model="test/test-model",
        client_url="http://localhost:3000",
        docling_url="http://localhost:8004/v1",
        docling_api_key="none",
        hmac_secret="test-secret",  # noqa: S106
        whisper_url="http://localhost:50001/v1",
    )


class TestAgentConstruction:
    def test_translation_agent_is_a_base_agent(self, app_config: AppConfig) -> None:
        assert issubclass(TranslationAgent, BaseAgent)

    def test_short_text_agent_is_a_base_agent(self, app_config: AppConfig) -> None:
        assert issubclass(ShortTextTranslationAgent, BaseAgent)

    def test_agents_construct_from_app_config(self, app_config: AppConfig) -> None:
        assert TranslationAgent(app_config) is not None
        assert ShortTextTranslationAgent(app_config) is not None

    def test_agents_carry_distinct_instructions(self) -> None:
        """The whole point of the short agent is a different prompt."""
        from bs_translator_backend.agents.translation_agent import (
            SHORT_TEXT_TRANSLATION_INSTRUCTION,
            TRANSLATION_INSTRUCTION,
        )

        assert TRANSLATION_INSTRUCTION != SHORT_TEXT_TRANSLATION_INSTRUCTION
        assert "lexical" in SHORT_TEXT_TRANSLATION_INSTRUCTION.lower()

    def test_agents_use_eszett_postprocessor_without_trim(self, app_config: AppConfig) -> None:
        from dcc_backend_common.llm_agent.postprocessing import replace_eszett

        for agent_cls in (TranslationAgent, ShortTextTranslationAgent):
            agent = agent_cls(app_config)
            assert agent._get_postprocessors() == [replace_eszett]

    @pytest.mark.asyncio
    async def test_agents_expose_close(self, app_config: AppConfig) -> None:
        agent = TranslationAgent(app_config)
        await agent.close()


class TestNoDirectSdkUsage:
    def test_module_does_not_import_openai(self) -> None:
        """The openai SDK is not a declared dependency; it must not be imported."""
        source = Path(mod.__file__).read_text()
        assert "import openai" not in source
        assert "from openai" not in source
