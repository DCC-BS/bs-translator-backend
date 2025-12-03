import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from bs_translator_backend.container import Container
from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.error_codes import UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.routers import convert_route, transcription_route, translation_route
from bs_translator_backend.utils.load_env import load_env
from bs_translator_backend.utils.logger import get_logger, init_logger

START_TIME = time.time()


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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_complete = True
        try:
            yield
        finally:  # pragma: no cover - defensive cleanup
            app.state.startup_complete = False

    app = FastAPI(
        title="BS Translator API",
        description="API for text translation and document conversion services with AI-powered language processing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.startup_complete = False
    app.state.start_time = START_TIME

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

    logger = get_logger("app")
    logger.info("Starting Text Mate API application")

    # Set up dependency injection container
    logger.debug("Configuring dependency injection container")
    container = Container(app_config=AppConfig.from_env())
    container.wire(modules=[translation_route, convert_route, transcription_route])
    container.check_dependencies()
    logger.info("Dependency injection configured")
    app.state.container = container

    config = container.app_config()
    logger.info(f"AppConfig loaded: {config}")

    # Configure CORS
    logger.debug("Setting up CORS middleware")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.client_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info(f"CORS configured with origin: {config.client_url}")

    # Include routers
    logger.debug("Registering API routers")
    app.include_router(translation_route.create_router())
    app.include_router(convert_route.create_router())
    app.include_router(transcription_route.create_router())
    logger.info("All routers registered")

    @app.get("/health/liveness", tags=["Health"])
    async def liveness_probe(request: Request) -> dict[str, float | str]:
        """
        Check if the application is alive. Used for Kubernetes liveness probe.

        Returns:
            dict[str, float | str]: Liveness status and uptime in seconds
        """
        return {"status": "up", "uptime_seconds": time.time() - request.app.state.start_time}

    @app.get("/health/readiness", tags=["Health"])
    async def readiness_probe(request: Request, response: Response) -> dict[str, object]:
        """
        Check if the application is ready to serve requests. Used for Kubernetes readiness probe.

        Returns:
            dict[str, object]: Readiness status and dependencies status
        """
        checks: dict[str, str] = {"dependencies": "unknown"}
        try:
            request.app.state.container.check_dependencies()
            await request.app.state.container.llm.is_ready()
            await request.app.state.container.transcription_service.is_ready()
        except Exception as exc:  # pragma: no cover - defensive Path
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "unhealthy", "error": str(exc)}
        else:
            checks["dependencies"] = "connected"
            return {"status": "ready", "checks": checks}

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
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }

        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "starting"}

    logger.info("API setup complete")
    return app


app = create_app()
