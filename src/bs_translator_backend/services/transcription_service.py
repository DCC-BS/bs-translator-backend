from collections.abc import AsyncGenerator
from typing import IO

import httpx
from dcc_backend_common.logger import get_logger
from fastapi import status

from bs_translator_backend.models.error_codes import (
    FILE_TOO_LARGE,
    TRANSCRIPTION_FAILED,
    TRANSCRIPTION_SERVICE_UNAVAILABLE,
)
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.models.language import DetectLanguage, LanguageOrAuto
from bs_translator_backend.utils.app_config import AppConfig

logger = get_logger(__name__)


# Codes Whisper does not know, mapped to the closest code it does know.
_WHISPER_LANGUAGE_ALIASES: dict[str, str] = {
    "en-gb": "en",
    "en-us": "en",
    "fa-af": "fa",  # Dari is transcribed as Persian
    "zh-cn": "zh",
    "zh-hk": "zh",
    "zh-tw": "zh",
}

# Languages Whisper has no model support for. Transcription falls back to
# auto-detection instead of sending an unknown code.
_WHISPER_UNSUPPORTED_LANGUAGES: frozenset[str] = frozenset({"ku", "ti"})


def transform_language_code_for_whisper(lang_code: str) -> str | None:
    if lang_code in _WHISPER_UNSUPPORTED_LANGUAGES:
        return None
    return _WHISPER_LANGUAGE_ALIASES.get(lang_code, lang_code)


def _unavailable(debug_message: str) -> ApiErrorException:
    return ApiErrorException({
        "errorId": TRANSCRIPTION_SERVICE_UNAVAILABLE,
        "status": status.HTTP_503_SERVICE_UNAVAILABLE,
        "debugMessage": debug_message,
    })


def _error_for_status(status_code: int) -> ApiErrorException:
    """Map a Whisper error status onto an API error (without Whisper's response body)."""
    if status_code == status.HTTP_413_CONTENT_TOO_LARGE:
        return ApiErrorException({
            "errorId": FILE_TOO_LARGE,
            "status": status.HTTP_413_CONTENT_TOO_LARGE,
            "debugMessage": "Whisper rejected the audio file as too large",
        })
    if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR or status_code in {
        status.HTTP_408_REQUEST_TIMEOUT,
        status.HTTP_429_TOO_MANY_REQUESTS,
    }:
        return _unavailable(f"Whisper returned HTTP {status_code}")
    # Any other 4xx: Whisper could not process this audio (e.g. unsupported format).
    return ApiErrorException({
        "errorId": TRANSCRIPTION_FAILED,
        "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
        "debugMessage": f"Whisper rejected the audio with HTTP {status_code}",
    })


class TranscriptionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient()

    async def aclose(self) -> None:
        """Close the HTTP client. Called from the FastAPI lifespan."""
        await self.client.aclose()

    async def transcribe(
        self, audio_file: "IO[bytes]", language: LanguageOrAuto
    ) -> AsyncGenerator[str, None]:
        """Stream the transcript from Whisper.

        Raises:
            ApiErrorException: Whisper is unreachable, timed out, or answered with an
                error status (checked before the first chunk is yielded).
        """
        data = {"response_format": "text"}

        if language != DetectLanguage.AUTO:
            whisper_code = transform_language_code_for_whisper(language.value.strip())
            if whisper_code is None:
                logger.debug(
                    "Language is not supported by Whisper, using auto", language=language.value
                )
            else:
                data["language"] = whisper_code
        else:
            logger.debug("Language is set to auto")

        try:
            async with self.client.stream(
                "POST",
                f"{self.config.whisper_url}/audio/transcriptions/stream",
                files={"file": audio_file},
                data=data,
                headers={"Authorization": f"Bearer {self.config.llm_api_key}"},
                timeout=300,
            ) as response:
                if response.is_error:
                    body = await response.aread()
                    logger.error(
                        "Whisper returned an error",
                        status_code=response.status_code,
                        body=body[:1000].decode(errors="replace"),
                    )
                    raise _error_for_status(response.status_code)
                async for chunk in response.aiter_text():
                    yield chunk
        except httpx.TimeoutException as e:
            logger.exception("Whisper request timed out")
            raise _unavailable("Whisper request timed out") from e
        except httpx.TransportError as e:
            logger.exception("Whisper request failed")
            raise _unavailable(f"Whisper request failed: {type(e).__name__}") from e
