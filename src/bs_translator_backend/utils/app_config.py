import os

from dcc_backend_common.config import AbstractAppConfig, get_env_or_throw, log_secret
from pydantic import Field, field_validator


class AppConfig(AbstractAppConfig):
    llm_url: str = Field(description="The base URL for the LLM API")
    llm_api_key: str = Field(description="The API key for authenticating with the LLM API")
    llm_model: str = Field(description="The language model to use for text generation")
    reasoning: bool = Field(
        default=False,
        description="Enable LLM reasoning; when false, disable with /no_think hint",
    )
    client_url: str = Field(description="The URL for the client application")
    docling_url: str = Field(description="The URL for the Docling service")
    docling_api_key: str = Field(description="The API key for docling")

    docling_poll_interval: float = Field(
        default=2.0, description="Seconds between async task status polls"
    )
    docling_task_timeout: float = Field(
        default=600.0, description="Max seconds to wait for an async docling task"
    )
    hmac_secret: str = Field(description="The secret key for HMAC authentication")

    whisper_url: str = Field(description="The URL for the Whisper API")

    @field_validator(
        "llm_api_key", "client_url", "docling_url", "whisper_url", mode="after"
    )
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("docling_poll_interval", "docling_task_timeout", mode="after")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @classmethod
    def from_env(cls) -> "AppConfig":
        llm_url: str = get_env_or_throw("LLM_URL")
        llm_api_key: str = get_env_or_throw("LLM_API_KEY")
        llm_model: str = get_env_or_throw("LLM_MODEL")
        reasoning_raw = os.getenv("LLM_REASONING", "false").lower()
        reasoning = reasoning_raw in {"1", "true", "yes", "on"}
        client_url: str = get_env_or_throw("CLIENT_URL")
        docling_url: str = get_env_or_throw("DOCLING_URL")
        docling_api_key: str = get_env_or_throw("DOCLING_API_KEY")
        docling_poll_interval = float(os.getenv("DOCLING_POLL_INTERVAL", "2.0"))
        docling_task_timeout = float(os.getenv("DOCLING_TASK_TIMEOUT", "600.0"))
        hmac_secret: str = get_env_or_throw("HMAC_SECRET")
        whisper_url: str = get_env_or_throw("WHISPER_URL")

        return cls(
            llm_url=llm_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            reasoning=reasoning,
            client_url=client_url,
            docling_url=docling_url,
            docling_api_key=docling_api_key,
            docling_poll_interval=docling_poll_interval,
            docling_task_timeout=docling_task_timeout,
            hmac_secret=hmac_secret,
            whisper_url=whisper_url,
        )

    def __str__(self) -> str:
        return f"""
        AppConfig(
            client_url={self.client_url},
            llm_url={self.llm_url},
            llm_api_key={log_secret(self.llm_api_key)},
            llm_model={self.llm_model},
            hmac_secret={log_secret(self.hmac_secret)},
            docling_url={self.docling_url},
            whisper_url={self.whisper_url},
            reasoning={self.reasoning},
            docling_poll_interval={self.docling_poll_interval},
            docling_task_timeout={self.docling_task_timeout}
        )
        """
