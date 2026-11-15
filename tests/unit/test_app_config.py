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
        hmac_secret="test-secret",
        whisper_url="http://localhost:50001/v1",
    )


class TestAppConfigIsLlmConfig:
    def test_app_config_is_an_llm_config(self) -> None:
        """AppConfig must be usable anywhere BaseAgent expects an LlmConfig."""
        assert issubclass(AppConfig, LlmConfig)

    def test_llm_timeout_and_retries_have_defaults(self, app_config: AppConfig) -> None:
        assert app_config.llm_timeout == 300
        assert app_config.llm_max_retries == 2


_REQUIRED_ENV = {
    "LLM_URL": "http://localhost:8001/v1",
    "LLM_API_KEY": "none",
    "LLM_MODEL": "test/test-model",
    "CLIENT_URL": "http://localhost:3000",
    "DOCLING_URL": "http://localhost:8004/v1",
    "DOCLING_API_KEY": "none",
    "HMAC_SECRET": "test-secret",
    "WHISPER_URL": "http://localhost:50001/v1",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


class TestFromEnv:
    def test_trailing_slash_is_stripped_from_llm_url_not_api_key(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("LLM_URL", "http://localhost:8001/v1/")
        env.setenv("LLM_API_KEY", "key-ending-in-slash/")

        config = AppConfig.from_env()

        assert config.llm_url == "http://localhost:8001/v1"
        assert config.llm_api_key == "key-ending-in-slash/"

    @pytest.mark.parametrize(("raw", "expected"), [("30", 30), ("12.5", 13), ("0.5", 1)])
    def test_fractional_llm_timeout_is_rounded_up(
        self, env: pytest.MonkeyPatch, raw: str, expected: int
    ) -> None:
        env.setenv("LLM_TIMEOUT", raw)

        assert AppConfig.from_env().llm_timeout == expected
