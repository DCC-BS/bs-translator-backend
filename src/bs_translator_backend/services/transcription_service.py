from collections.abc import AsyncGenerator
from io import BytesIO

import httpx

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.langugage import DetectLanguage, LanguageOrAuto


class TranscriptionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient()

    async def transcribe(self, audio_file: bytes, language: LanguageOrAuto) -> AsyncGenerator[str, None]:
        lang = None if language == DetectLanguage.AUTO else language.value

        async with self.client.stream(
            "POST",
            f"{self.config.whisper_url}/audio/transcriptions/stream",
            files={"file": BytesIO(audio_file)},
            data={"response_format": "text", "language": lang},
            timeout=None,
        ) as response:
            async for chunk in response.aiter_text():
                yield chunk[6:]
