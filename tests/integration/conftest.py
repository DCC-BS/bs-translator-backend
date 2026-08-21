import pytest

from bs_translator_backend.utils.app_config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    """Static config so integration tests do not depend on a local environment."""
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
