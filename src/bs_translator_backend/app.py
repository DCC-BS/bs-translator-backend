from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bs_translator_backend.container import Container
from bs_translator_backend.routers import translation_route
from bs_translator_backend.utils.load_env import load_env
from bs_translator_backend.utils.logger import get_logger, init_logger


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    load_env()
    init_logger()

    app = FastAPI(
        title="Text Mate API",
        description="API for text correction, rewriting, and other text-related services",
        version="0.1.0",
    )

    logger = get_logger("app")
    logger.info("Starting Text Mate API application")

    # Set up dependency injection container
    logger.debug("Configuring dependency injection container")
    container = Container()
    container.wire(modules=[translation_route])
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
    logger.info("All routers registered")

    logger.info("API setup complete")
    return app


app = create_app()
