import os


class HMACSecretNotSetError(ValueError):
    """Exception raised when HMAC_SECRET environment variable is not set."""

    def __init__(self) -> None:
        super().__init__(
            "HMAC_SECRET environment variable is not set. Please set it to a valid value."
        )


def log_secret(secret: str | None) -> str:
    return "****" if secret is not None and len(secret) > 0 else "None"


class AppConfig:
    def __init__(self) -> None:
        self.openai_api_base_url: str = os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "none")
        self.llm_model: str = os.getenv("LLM_MODEL", "ollama_chat/llama3.2")
        self.language_tool_api_url: str = os.getenv(
            "LANGUAGE_TOOL_API_URL", "http://localhost:8010/"
        )
        self.client_url: str = os.getenv("CLIENT_URL", "http://localhost:3000")

        hmac_secret = os.getenv("HMAC_SECRET")

        if hmac_secret is None:
            raise HMACSecretNotSetError()
        else:
            self.hmac_secret: str = hmac_secret

    def __str__(self) -> str:
        return f"""
        AppConfig(
            client_url={self.client_url},
            openai_api_base_url={self.openai_api_base_url},
            openai_api_key={log_secret(self.openai_api_key)},
            llm_model={self.llm_model},
            language_tool_api_url={self.language_tool_api_url},
            hmac_secret={log_secret(self.hmac_secret)}
        )
        """
