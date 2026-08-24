import pytest
from dcc_backend_common.config.app_config import LlmConfig

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


class TestAppConfigIsLlmConfig:
    def test_app_config_is_an_llm_config(self) -> None:
        """AppConfig must be usable anywhere BaseAgent expects an LlmConfig."""
        assert issubclass(AppConfig, LlmConfig)

    def test_llm_timeout_and_retries_have_defaults(self, app_config: AppConfig) -> None:
        assert app_config.llm_timeout == 300
        assert app_config.llm_max_retries == 2
