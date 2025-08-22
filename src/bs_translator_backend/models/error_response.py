from typing import TypedDict


class ErrorResponse(TypedDict):
    errorId: str
    status: int
    debugMessage: str | None


class ApiErrorException(Exception):
    def __init__(self, error_response: ErrorResponse):
        self.error_response = error_response
