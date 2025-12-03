from collections.abc import AsyncGenerator
from typing import IO

import httpx

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto


class TranscriptionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient()

    async def transcribe(
        self, audio_file: "IO[bytes]", language: LanguageOrAuto
    ) -> AsyncGenerator[str, None]:
        lang = None if language == DetectLanguage.AUTO else language.value

        async with self.client.stream(
            "POST",
            f"{self.config.whisper_url}/audio/transcriptions/stream",
            files={"file": audio_file},
            data={"response_format": "text", "language": lang},
            timeout=300,
        ) as response:
            async for chunk in response.aiter_text():
                yield chunk[6:]

    async def is_ready(self) -> bool:
        response = await self.client.get(f"{self.config.whisper_url}/healthz")
        if response.status_code != 200:
            raise Exception(f"Transcription service is not ready: {response.status_code}")
        return True
