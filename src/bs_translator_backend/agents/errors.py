class TranslationUnavailableError(Exception):
    """The LLM backing translation is unreachable, overloaded, or failed transiently.

    Raised by the translation agents in place of the underlying httpx/pydantic_ai
    exception, so callers (translation_service, routers) don't need to depend on
    those SDKs - see translation_agent.py's module docstring. The message never carries
    the upstream response body or internal hostnames: it may reach the client.
    """


class TranslationRejectedError(Exception):
    """The LLM rejected the request with a non-retryable 4xx; retrying cannot help.

    `input_too_long` is set when the rejection is about the prompt exceeding the
    model's context window, i.e. the user's input is too long. The message never
    carries the upstream response body.
    """

    def __init__(self, message: str, *, status_code: int, input_too_long: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.input_too_long = input_too_long
