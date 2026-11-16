import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dcc_backend_common.fastapi_health_probes import health_probe_router
from dcc_backend_common.fastapi_health_probes.router import ServiceDependency
from dcc_backend_common.fastapi_logging_middleware import add_logging_middleware
from dcc_backend_common.logger import get_logger, init_logger
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from structlog.stdlib import BoundLogger

from bs_translator_backend.container import Container
from bs_translator_backend.models.error_codes import INVALID_REQUEST, NOT_FOUND, UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.routers import (
    convert_route,
    image_translation_route,
    transcription_route,
    translation_route,
)
from bs_translator_backend.utils.app_config import AppConfig
from bs_translator_backend.utils.language_detection import warm_up_language_detection

# Every router module: wired into the DI container and registered on the app.
_ROUTER_MODULES = (
    translation_route,
    image_translation_route,
    convert_route,
    transcription_route,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the language model before serving; release HTTP clients on shutdown."""
    # Loading the fastText model blocks for a moment; do it once, off the event loop,
    # instead of inside the first request that detects a language.
    await asyncio.to_thread(warm_up_language_detection)
    yield
    container: Container | None = getattr(app.state, "container", None)
    if container is not None:
        await container.translation_service().aclose()
        await container.transcription_service().aclose()


def _build_fastapi_app() -> FastAPI:
    """
    Instantiate the FastAPI application with metadata and lifespan.
    """
    app = FastAPI(
        title="BS Translator API",
        description="API for text translation and document conversion services with AI-powered language processing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    return app


def _register_health_routes(app: FastAPI, config: AppConfig) -> None:
    """
    Register health routes for the application.
    """
    whisper_base_url = config.whisper_url.removesuffix("/v1")
    llm_base_url = config.llm_url.removesuffix("/v1")
    docling_base_url = config.docling_url.removesuffix("/v1")
    service_dependencies: list[ServiceDependency] = [
        ServiceDependency(
            name="whisper",
            health_check_url=f"{whisper_base_url}/readyz",
            api_key=config.llm_api_key,
        ),
        ServiceDependency(
            name="llm",
            health_check_url=f"{llm_base_url}/health",
            api_key=config.llm_api_key,
        ),
        ServiceDependency(
            name="docling",
            health_check_url=f"{docling_base_url}/health",
            api_key=config.docling_api_key,
        ),
    ]
    app.include_router(health_probe_router(service_dependencies=service_dependencies))


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers so every error is JSON `{errorId, status, debugMessage?}`.

    debugMessage never carries raw exception text: it can reach the browser, and
    upstream errors may contain internal hostnames or response bodies.
    """
    logger = get_logger("app")

    def api_error_handler(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, ApiErrorException):
            return unexpected_error_handler(request, exc)
        return JSONResponse(
            status_code=exc.error_response["status"],
            media_type="application/json",
            content=exc.error_response,
        )

    def validation_error_handler(request: Request, exc: Exception) -> Response:
        detail = jsonable_encoder(exc.errors()) if isinstance(exc, RequestValidationError) else []
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "errorId": INVALID_REQUEST,
                "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
                "debugMessage": "Request validation failed",
                "detail": detail,
            },
        )

    def http_error_handler(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, StarletteHTTPException):
            return unexpected_error_handler(request, exc)
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            error_id = NOT_FOUND
        elif exc.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            error_id = INVALID_REQUEST
        else:
            error_id = UNEXPECTED_ERROR
        return JSONResponse(
            status_code=exc.status_code,
            content={"errorId": error_id, "status": exc.status_code, "debugMessage": exc.detail},
            headers=exc.headers,
        )

    def unexpected_error_handler(request: Request, exc: Exception) -> Response:
        logger.error(
            "Unhandled exception",
            method=request.method,
            path=request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "errorId": UNEXPECTED_ERROR,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "debugMessage": type(exc).__name__,
            },
        )

    app.add_exception_handler(ApiErrorException, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)


def _configure_container(app: FastAPI, logger: BoundLogger) -> Container:
    """
    Configure the dependency injection container and attach it to app state.
    """
    logger.debug("Configuring dependency injection container")
    container = Container()
    container.wire(modules=list(_ROUTER_MODULES))
    container.check_dependencies()
    logger.debug("Dependency injection configured")
    app.state.container = container
    return container


def _configure_cors(app: FastAPI, client_url: str, logger: BoundLogger) -> None:
    """
    Apply CORS middleware configuration.
    """
    logger.debug("Setting up CORS middleware")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[client_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.debug("CORS configured", origin=client_url)


def _register_routes(app: FastAPI, logger: BoundLogger) -> None:
    """
    Register API routers.
    """
    logger.debug("Registering API routers")
    for router_module in _ROUTER_MODULES:
        app.include_router(router_module.create_router())
    logger.debug("All routers registered")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This function initializes the FastAPI application with:
    - Environment variables loading
    - Logging configuration
    - Dependency injection container setup
    - CORS middleware configuration
    - API route registration

    Returns:
        FastAPI: Configured FastAPI application instance
    """

    init_logger(app_name="bs-translator")

    logger = get_logger("app")
    logger.info("Starting BS Translator API application")

    app = _build_fastapi_app()

    _register_exception_handlers(app)

    container = _configure_container(app=app, logger=logger)
    config = container.app_config()
    logger.info(f"AppConfig loaded: {config}")

    _register_health_routes(app=app, config=config)

    _configure_cors(app=app, client_url=config.client_url, logger=logger)
    add_logging_middleware(app)
    _register_routes(app=app, logger=logger)

    logger.info("API setup complete")
    return app


app = create_app()
