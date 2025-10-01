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
    client_url: str = Field(description="The URL for the client application")
    docling_url: str = Field(description="The URL for the Docling service")
    hmac_secret: str = Field(description="The secret key for HMAC authentication")

    whisper_url: str = Field(description="The URL for the Whisper API")

    @classmethod
    def from_env(cls) -> "AppConfig":
        openai_api_base_url: str = get_env_or_throw("OPENAI_API_BASE_URL")
        openai_api_key: str = get_env_or_throw("OPENAI_API_KEY")
        llm_model: str = get_env_or_throw("LLM_MODEL")
        client_url: str = get_env_or_throw("CLIENT_URL")
        docling_url: str = get_env_or_throw("DOCLING_URL")
        hmac_secret: str = get_env_or_throw("HMAC_SECRET")
        whisper_url: str = get_env_or_throw("WHISPER_URL")
        return cls(
            openai_api_base_url=openai_api_base_url,
            openai_api_key=openai_api_key,
            llm_model=llm_model,
            client_url=client_url,
            docling_url=docling_url,
            hmac_secret=hmac_secret,
            whisper_url=whisper_url,
        )

    # def __str__(self) -> str:
    #     return f"""
    #     AppConfig(
    #         client_url={self.client_url},
    #         openai_api_base_url={self.openai_api_base_url},
    #         openai_api_key={log_secret(self.openai_api_key)},
    #         llm_model={self.llm_model},
    #         hmac_secret={log_secret(self.hmac_secret)},
    #         docling_url={self.docling_url}
    #     )
    #     """
