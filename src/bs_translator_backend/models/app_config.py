import os

from pydantic import BaseModel, Field


class AppConfigError(ValueError):
    """Exception raised for errors in the application configuration."""

    def __init__(self, variable_name: str) -> None:
        super().__init__(f"Configuration variable '{variable_name}' is not set or invalid.")


def get_env_or_throw(env_name: str) -> str:
    value = os.getenv(env_name)
    if value is None:
        raise AppConfigError(env_name)
    return value


def log_secret(secret: str | None) -> str:
    return "****" if secret is not None and len(secret) > 0 else "None"


class AppConfig(BaseModel):
    openai_api_base_url: str = Field(description="The base URL for the OpenAI API")
    openai_api_key: str = Field(description="The API key for authenticating with OpenAI")
    llm_model: str = Field(description="The language model to use for text generation")
    optimizer_api_key: str = Field(
        default="", description="API key for optimizer/proposal LM (defaults to OPENAI_API_KEY)"
    )
    optimizer_model: str = Field(
        default="", description="Model to use for optimization (default: gpt-4o-mini)"
    )
    optimizer_api_base_url: str = Field(
        default="", description="Optional base URL for optimizer LM (default OpenAI endpoint)"
    )
    reasoning: bool = Field(
        default=False,
        description="Enable LLM reasoning; when false, disable with /no_think hint",
    )
    translation_module_path: str = Field(
        default="src/bs_translator_backend/services/dspy_config/translation_module.pkl",
        description="The path to the translation module",
    )
    client_url: str = Field(description="The URL for the client application")
    docling_url: str = Field(description="The URL for the Docling service")
    hmac_secret: str = Field(description="The secret key for HMAC authentication")

    whisper_url: str = Field(description="The URL for the Whisper API")

    @classmethod
    def from_env(cls) -> "AppConfig":
        openai_api_base_url: str = get_env_or_throw("OPENAI_API_BASE_URL")
        openai_api_key: str = get_env_or_throw("OPENAI_API_KEY")
        llm_model: str = get_env_or_throw("LLM_MODEL")
        optimizer_api_key: str = os.getenv("OPTIMIZER_API_KEY", "")
        optimizer_model: str = os.getenv("OPTIMIZER_MODEL", "")
        optimizer_api_base_url: str = os.getenv("OPTIMIZER_API_BASE_URL", "")
        reasoning_raw = os.getenv("LLM_REASONING", "false").lower()
        reasoning = reasoning_raw in {"1", "true", "yes", "on"}
        client_url: str = get_env_or_throw("CLIENT_URL")
        docling_url: str = get_env_or_throw("DOCLING_URL")
        hmac_secret: str = get_env_or_throw("HMAC_SECRET")
        whisper_url: str = get_env_or_throw("WHISPER_URL")

        return cls(
            openai_api_base_url=openai_api_base_url,
            openai_api_key=openai_api_key,
            llm_model="openai/" + llm_model,  # Use openai prefix for DSPy
            optimizer_api_key=optimizer_api_key,
            optimizer_model="openai/" + optimizer_model,  # Use openai prefix for DSPy
            optimizer_api_base_url=optimizer_api_base_url,
            reasoning=reasoning,
            client_url=client_url,
            docling_url=docling_url,
            hmac_secret=hmac_secret,
            whisper_url=whisper_url,
        )

    def __str__(self) -> str:
        return f"""
        AppConfig(
            client_url={self.client_url},
            openai_api_base_url={self.openai_api_base_url},
            openai_api_key={log_secret(self.openai_api_key)},
            llm_model={self.llm_model},
            hmac_secret={log_secret(self.hmac_secret)},
            docling_url={self.docling_url}
            whisper_url={self.whisper_url}
            translation_module_path={self.translation_module_path}
            optimizer_api_key={log_secret(self.optimizer_api_key)}
            optimizer_model={self.optimizer_model}
            optimizer_api_base_url={self.optimizer_api_base_url}
            reasoning={self.reasoning}
        )
        """
