class TranslationUnavailableError(Exception):
    """The LLM backing translation is unreachable, overloaded, or returned an error.

    Raised by the translation agents in place of the underlying httpx/pydantic_ai
    exception, so callers (translation_service, routers) don't need to depend on
    those SDKs - see translation_agent.py's module docstring.
    """
