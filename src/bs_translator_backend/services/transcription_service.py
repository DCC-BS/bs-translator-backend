from collections.abc import AsyncGenerator
from typing import IO

import httpx
from dcc_backend_common.logger import get_logger

from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.utils.app_config import AppConfig

logger = get_logger(__name__)


def transform_language_code_for_whisper(lang_code: str) -> str:
    if lang_code == "en-gb" or lang_code == "en-us":
        return "en"
    else:
        return lang_code


class TranscriptionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient()

    async def transcribe(
        self, audio_file: "IO[bytes]", language: LanguageOrAuto
    ) -> AsyncGenerator[str, None]:
        data = {"response_format": "text"}

        if language != DetectLanguage.AUTO:
            data["language"] = transform_language_code_for_whisper(language.value.strip())
        else:
            logger.info("Language is set to auto")

        async with self.client.stream(
            "POST",
            f"{self.config.whisper_url}/audio/transcriptions/stream",
            files={"file": audio_file},
            data=data,
            headers={"Authorization": f"Bearer {self.config.llm_api_key}"},
            timeout=300,
        ) as response:
            async for chunk in response.aiter_text():
                yield chunk
