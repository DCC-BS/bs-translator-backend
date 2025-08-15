from langdetect import detect  # type: ignore[import-untyped]
from returns.result import Failure, ResultE, Success, safe

from bs_translator_backend.models.langugage import Language


def detect_language(text: str) -> ResultE[Language]:
    def map_to_language(code: str) -> ResultE[Language]:
        try:
            return Success(Language[code.upper().strip().replace("-", "_")])
        except KeyError:
            return Failure(Exception(f"Unknown language code: {code}"))

    return detect_language_str(text).bind(map_to_language)


@safe
def detect_language_str(text: str) -> str:
    """Detect the language of the given text.

    Args:
        text: The text to analyze for language detection

    Returns:
        The detected language code as a string
    """
    return str(detect(text))
