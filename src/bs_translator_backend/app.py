import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import dspy
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from structlog.stdlib import BoundLogger

from bs_translator_backend.container import Container
from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.error_codes import UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.routers import convert_route, transcription_route, translation_route
from bs_translator_backend.utils.load_env import load_env
from bs_translator_backend.utils.logger import get_logger, init_logger

CACHE_DURATION = 30  # seconds


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Track application lifecycle state for probes.
    """
    app.state.startup_complete = True
    app.state.startup_timestamp = datetime.now(tz=UTC).isoformat()
    try:
        yield
    finally:  # pragma: no cover - defensive cleanup
        app.state.startup_complete = False


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
        lifespan=_app_lifespan,
    )
    app.state.startup_complete = False
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers for API errors.
    """

    def api_error_handler(request: Request, exc: Exception) -> Response:
        if isinstance(exc, ApiErrorException):
            return JSONResponse(
                status_code=exc.error_response["status"],
                media_type="application/json",
                content=exc.error_response,
            )

        return JSONResponse(
            status_code=500,
            media_type="application/json",
            content={"errorId": UNEXPECTED_ERROR, "status": 500, "debugMessage": str(exc)},
        )

    app.add_exception_handler(ApiErrorException, api_error_handler)


def _configure_container(app: FastAPI, logger: BoundLogger) -> Container:
    """
    Configure the dependency injection container and attach it to app state.
    """
    logger.debug("Configuring dependency injection container")
    container = Container(app_config=AppConfig.from_env())
    container.wire(modules=[translation_route, convert_route, transcription_route])
    container.check_dependencies()
    logger.info("Dependency injection configured")
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
    logger.info(f"CORS configured with origin: {client_url}")


def _register_routes(app: FastAPI, logger: BoundLogger) -> None:
    """
    Register API routers.
    """
    logger.debug("Registering API routers")
    app.include_router(translation_route.create_router())
    app.include_router(convert_route.create_router())
    app.include_router(transcription_route.create_router())
    logger.info("All routers registered")


def _register_health_routes(app: FastAPI) -> None:
    """
    Register health probe endpoints.
    """

    @app.get("/health/liveness", tags=["Health"])
    async def liveness_probe(request: Request) -> dict[str, float | str]:
        """
        Check if the application is alive. Used for Kubernetes liveness probe.

        Returns:
            dict[str, float | str]: Liveness status and uptime in seconds
        """
        return {
            "status": "up",
            "uptime_seconds": time.time()
            - datetime.fromisoformat(request.app.state.startup_timestamp).timestamp(),
        }

    @app.get("/health/readiness", tags=["Health"])
    async def readiness_probe(request: Request, response: Response) -> dict[str, object]:
        """
        Confirm that dependencies are ready for traffic.
        """
        current_time = time.time()
        cached_result = request.app.state.last_check_result
        if (
            current_time - request.app.state.last_check_time < CACHE_DURATION
            and cached_result is not None
        ):
            return cached_result

        checks: dict[str, str] = {"dependencies": "unknown"}
        try:
            request.app.state.container.check_dependencies()
        except Exception:  # pragma: no cover - defensive path
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            result: dict[str, object] = {"status": "unhealthy", "error": "Dependency check failed"}
            request.app.state.last_check_time = current_time
            request.app.state.last_check_result = result
            return result

        checks["dependencies"] = "connected"
        result: dict[str, object] = {"status": "ready", "checks": checks}
        request.app.state.last_check_time = current_time
        request.app.state.last_check_result = result
        return result

    @app.get("/health/startup", tags=["Health"])
    async def startup_probe(request: Request, response: Response) -> dict[str, str]:
        """
        Check if the application has completed startup. Used for Kubernetes startup probe.

        Returns:
            dict[str, str]: Startup status and timestamp
        """
        if request.app.state.startup_complete:
            return {
                "status": "started",
                "timestamp": request.app.state.startup_timestamp,
            }

        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "starting"}


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

    init_logger()
    load_env()

    logger = get_logger("app")
    logger.info("Starting Text Mate API application")

    app = _build_fastapi_app()
    app.state.last_check_time = 0
    app.state.last_check_result = None

    _register_exception_handlers(app)

    container = _configure_container(app=app, logger=logger)
    config = container.app_config()
    logger.info(f"AppConfig loaded: {config}")

    dspy.configure(
        lm=dspy.LM(
            model=config.llm_model,
            api_key=config.openai_api_key,
            api_base=config.openai_api_base_url,
        )
    )

    _configure_cors(app=app, client_url=config.client_url, logger=logger)
    _register_routes(app=app, logger=logger)
    _register_health_routes(app=app)

    logger.info("API setup complete")
    return app


app = create_app()
