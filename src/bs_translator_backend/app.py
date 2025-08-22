from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from bs_translator_backend.container import Container
from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.routers import convert_route, translation_route
from bs_translator_backend.utils.load_env import load_env
from bs_translator_backend.utils.logger import get_logger, init_logger
from bs_translator_backend.models.error_response import ApiErrorException, ErrorResponse


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

    @app.exception_handler(ApiErrorException)
    async def my_custom_exception_handler(
        request: Request, exception: ApiErrorException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exception.error_response["status"],
            content=exception.error_response,
        )

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
    logger.info(f"CORS configured with origin: {config.client_url}")

    # Include routers
    logger.debug("Registering API routers")
    app.include_router(translation_route.create_router())
    app.include_router(convert_route.create_router())
    logger.info("All routers registered")

    logger.info("API setup complete")
    return app


app = create_app()
