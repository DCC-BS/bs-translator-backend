from typing import TypedDict

from fastapi import status


class ErrorResponse(TypedDict):
    errorId: str
    status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    debugMessage: str | None


class ApiErrorException(Exception):
    def __init__(self, error_response: ErrorResponse):
        self.error_response = error_response
