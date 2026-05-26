from collections.abc import AsyncGenerator
from typing import IO

import httpx

from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.utils.app_config import AppConfig


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
        lang = None if language == DetectLanguage.AUTO else language.value

        data = {"response_format": "text"}

        if lang is not None:
            data["language"] = transform_language_code_for_whisper(lang.strip())

        async with self.client.stream(
            "POST",
            f"{self.config.whisper_url}/audio/transcriptions/stream",
            files={"file": audio_file},
            data=data,
            timeout=300,
        ) as response:
            async for chunk in response.aiter_text():
                yield chunk[6:]
