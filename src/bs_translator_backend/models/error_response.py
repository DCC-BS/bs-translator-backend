from typing import TypedDict

from fastapi import status


class ErrorResponse(TypedDict, total=False):
    errorId: str
    status: int
    debugMessage: str | None


class ApiErrorException(Exception):
    def __init__(self, error_response: ErrorResponse):
        if "status" not in error_response:
            error_response["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
        self.error_response = error_response
