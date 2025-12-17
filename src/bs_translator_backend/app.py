import dspy
from backend_common.fastapi_health_probes import health_probe_router
from backend_common.fastapi_health_probes.router import ServiceDependency
from backend_common.logger import get_logger, init_logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from structlog.stdlib import BoundLogger

from bs_translator_backend.container import Container
from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.error_codes import UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.routers import convert_route, transcription_route, translation_route


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
    )

    return app


def _register_health_routes(app: FastAPI, config: AppConfig) -> None:
    """
    Register health routes for the application.
    """
    whisper_base_url = config.whisper_url.rstrip("/v1")
    llm_base_url = config.openai_api_base_url.rstrip("/v1")
    service_dependencies: list[ServiceDependency] = [
        ServiceDependency(
            name="whisper",
            health_check_url=f"{whisper_base_url}/readyz",
            api_key=config.openai_api_key,
        ),
        ServiceDependency(
            name="llm",
            health_check_url=f"{llm_base_url}/health",
            api_key=config.openai_api_key,
        ),
        ServiceDependency(
            name="docling",
            health_check_url=f"{config.docling_url}/health",
            api_key=config.openai_api_key,
        ),
    ]
    app.include_router(health_probe_router(service_dependencies=service_dependencies))


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
    container = Container()
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

    logger = get_logger("app")
    logger.info("Starting Text Mate API application")

    app = _build_fastapi_app()

    _register_exception_handlers(app)

    container = _configure_container(app=app, logger=logger)
    config = container.app_config()
    logger.info(f"AppConfig loaded: {config}")

    _register_health_routes(app=app, config=config)

    dspy.configure(
        lm=dspy.LM(
            model=config.llm_model,
            api_key=config.openai_api_key,
            api_base=config.openai_api_base_url,
        ),
    )
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    _configure_cors(app=app, client_url=config.client_url, logger=logger)
    _register_routes(app=app, logger=logger)

    logger.info("API setup complete")
    return app


app = create_app()
