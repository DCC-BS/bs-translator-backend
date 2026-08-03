from collections.abc import AsyncGenerator
from typing import IO

import httpx
from dcc_backend_common.logger import get_logger

from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.utils.app_config import AppConfig

logger = get_logger(__name__)


# Codes Whisper does not know, mapped to the closest code it does know.
_WHISPER_LANGUAGE_ALIASES: dict[str, str] = {
    "en-gb": "en",
    "en-us": "en",
    "fa-af": "fa",  # Dari is transcribed as Persian
}

# Languages Whisper has no model support for. Transcription falls back to
# auto-detection instead of sending an unknown code.
_WHISPER_UNSUPPORTED_LANGUAGES: frozenset[str] = frozenset({"ku", "ti"})


def transform_language_code_for_whisper(lang_code: str) -> str | None:
    if lang_code in _WHISPER_UNSUPPORTED_LANGUAGES:
        return None
    return _WHISPER_LANGUAGE_ALIASES.get(lang_code, lang_code)


class TranscriptionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient()

    async def transcribe(
        self, audio_file: "IO[bytes]", language: LanguageOrAuto
    ) -> AsyncGenerator[str, None]:
        data = {"response_format": "text"}

        if language != DetectLanguage.AUTO:
            whisper_code = transform_language_code_for_whisper(language.value.strip())
            if whisper_code is None:
                logger.debug("Language %s is not supported by Whisper, using auto", language.value)
            else:
                data["language"] = whisper_code
        else:
            logger.debug("Language is set to auto")

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
