from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from bs_translator_backend.container import Container
from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.error_codes import UNEXPECTED_ERROR
from bs_translator_backend.models.error_response import ApiErrorException
from bs_translator_backend.routers import convert_route, translation_route
from bs_translator_backend.utils.load_env import load_env
from bs_translator_backend.utils.logger import get_logger, init_logger
from bs_translator_backend.middlewares.request_cancel_middleware import RequestCancelledMiddleware


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

    app = FastAPI(
        title="BS Translator API",
        description="API for text translation and document conversion services with AI-powered language processing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

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
    container.wire(modules=[translation_route, convert_route])
    container.check_dependencies()
    logger.info("Dependency injection configured")

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
    app.add_middleware(RequestCancelledMiddleware)

    logger.info(f"CORS configured with origin: {config.client_url}")

    # Include routers
    logger.debug("Registering API routers")
    app.include_router(translation_route.create_router())
    app.include_router(convert_route.create_router())
    logger.info("All routers registered")

    logger.info("API setup complete")
    return app


app = create_app()
