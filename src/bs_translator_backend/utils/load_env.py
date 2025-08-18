from pathlib import Path

from dotenv import load_dotenv
from bs_translator_backend.utils.logger import get_logger

logger = get_logger(__name__)


def load_env() -> None:
    """
    Load environment variables from a .env file.
    This function is called at the start of the application to ensure
    that environment variables are loaded before any other modules are imported.
    """

    # Get the root directory (where .env file is located)
    root_dir: Path = Path(__file__).parent.parent.parent.parent
    logger.info(f"Root directory: {root_dir}")
    env_path: Path = root_dir / ".env"

    # Load environment variables from .env file if not in production mode
    if env_path.exists():
        _ = load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"Warning: .env file not found at {env_path}")
