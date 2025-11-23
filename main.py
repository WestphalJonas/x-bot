"""Main entry point for autonomous Twitter/X posting bot."""

import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import BotConfig
from src.scheduler import BotScheduler
from src.scheduler.jobs import (
    check_notifications,
    post_autonomous_tweet,
    post_autonomous_tweet,
    process_inspiration_queue,
    read_frontpage_posts,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_env_settings() -> dict[str, str | None]:
    """Load environment variables into a dictionary.

    Returns:
        Dictionary with environment variable values
    """
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "TWITTER_USERNAME": os.getenv("TWITTER_USERNAME"),
        "TWITTER_PASSWORD": os.getenv("TWITTER_PASSWORD"),
    }


def validate_env_settings(env_settings: dict[str, str | None]) -> None:
    """Validate that required environment variables are set.

    Args:
        env_settings: Dictionary with environment variables

    Raises:
        ValueError: If required environment variables are missing
    """
    openai_api_key = env_settings.get("OPENAI_API_KEY")
    openrouter_api_key = env_settings.get("OPENROUTER_API_KEY")
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    if not openai_api_key and not openrouter_api_key:
        raise ValueError(
            "At least one LLM provider API key is required. "
            "Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
        )
    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")


def create_job_wrapper(job_func, config: BotConfig, env_settings: dict):
    """Create a wrapper function for a job that passes config and env_settings.

    Args:
        job_func: Job function to wrap
        config: Bot configuration
        env_settings: Environment settings dictionary

    Returns:
        Wrapped function that can be called without arguments
    """
    return lambda: job_func(config, env_settings)


def main():
    """Main entry point for the bot scheduler."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config_path = Path("config/config.yaml")
    config = BotConfig.load(config_path)

    logger.info("bot_starting", extra={"config_path": str(config_path)})

    # Load and validate environment settings
    env_settings = load_env_settings()
    try:
        validate_env_settings(env_settings)
    except ValueError as e:
        logger.error("configuration_error", extra={"error": str(e)})
        sys.exit(1)

    # Initialize scheduler
    scheduler = BotScheduler(config)

    # Setup jobs with wrappers that pass config and env_settings
    scheduler.setup_posting_job(
        create_job_wrapper(post_autonomous_tweet, config, env_settings)
    )
    scheduler.setup_reading_job(
        create_job_wrapper(read_frontpage_posts, config, env_settings)
    )
    scheduler.setup_notifications_job(
        create_job_wrapper(check_notifications, config, env_settings)
    )
    scheduler.setup_inspiration_job(
        create_job_wrapper(process_inspiration_queue, config, env_settings)
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info("shutdown_signal_received", extra={"signal": signum})
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start scheduler
    try:
        scheduler.start()

        # Keep main process alive
        logger.info("scheduler_running", extra={"info": "Press Ctrl+C to stop"})
        while scheduler.is_running:
            try:
                import time

                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("keyboard_interrupt_received")
                break

    except Exception as e:
        logger.error("scheduler_error", extra={"error": str(e)}, exc_info=True)
        raise

    finally:
        scheduler.shutdown()
        logger.info("bot_shutdown_complete")


if __name__ == "__main__":
    main()
