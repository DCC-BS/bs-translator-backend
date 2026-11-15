import math
import os

from dcc_backend_common.config import get_env_or_throw, log_secret
from dcc_backend_common.config.app_config import LlmConfig
from pydantic import Field, field_validator


class AppConfig(LlmConfig):
    reasoning: bool = Field(
        default=False,
        description="Enable LLM reasoning; forwarded to the LLM agent enable_thinking setting",
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

    max_upload_size_mb: float = Field(
        default=50.0, description="Maximum accepted upload size in megabytes (413 above it)"
    )
    docling_image_ocr_preset: str = Field(
        default="glm-ocr-remote",
        description="docling-serve OCR preset used for photo translation (GLM-OCR by default)",
    )
    docling_image_layout_preset: str = Field(
        default="ppdoclayout-v3",
        description="docling-serve layout preset for photo translation; empty uses docling's default",
    )
    image_translation_concurrency: int = Field(
        default=4, description="Text blocks of one photo translated in parallel"
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @field_validator("llm_url", "client_url", "docling_url", "whisper_url", mode="after")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator(
        "docling_poll_interval",
        "docling_task_timeout",
        "max_upload_size_mb",
        "image_translation_concurrency",
        mode="after",
    )
    @classmethod
    def must_be_positive[N: (int, float)](cls, v: N) -> N:
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
        max_upload_size_mb = float(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
        docling_image_ocr_preset = os.getenv("DOCLING_IMAGE_OCR_PRESET", "glm-ocr-remote")
        docling_image_layout_preset = os.getenv("DOCLING_IMAGE_LAYOUT_PRESET", "ppdoclayout-v3")
        image_translation_concurrency = int(os.getenv("IMAGE_TRANSLATION_CONCURRENCY", "4"))
        # Tunable without a redeploy: when another app saturates the shared vLLM
        # instance, lowering LLM_TIMEOUT lets a stuck request fail (and get reported
        # to the user) sooner instead of hanging for the full default.
        # LlmConfig.llm_timeout is an int: round a fractional value up, so e.g. 0.5 still
        # means "a short timeout" rather than 0.
        llm_timeout = math.ceil(float(os.getenv("LLM_TIMEOUT", str(60 * 5))))
        llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

        return cls(
            llm_url=llm_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_timeout=llm_timeout,
            llm_max_retries=llm_max_retries,
            reasoning=reasoning,
            client_url=client_url,
            docling_url=docling_url,
            docling_api_key=docling_api_key,
            docling_poll_interval=docling_poll_interval,
            docling_task_timeout=docling_task_timeout,
            hmac_secret=hmac_secret,
            whisper_url=whisper_url,
            max_upload_size_mb=max_upload_size_mb,
            docling_image_ocr_preset=docling_image_ocr_preset,
            docling_image_layout_preset=docling_image_layout_preset,
            image_translation_concurrency=image_translation_concurrency,
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
            docling_task_timeout={self.docling_task_timeout},
            max_upload_size_mb={self.max_upload_size_mb},
            docling_image_ocr_preset={self.docling_image_ocr_preset},
            docling_image_layout_preset={self.docling_image_layout_preset},
            image_translation_concurrency={self.image_translation_concurrency}
        )
        """
