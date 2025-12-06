"""Main entry point for autonomous Twitter/X posting bot."""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import BotConfig, EnvSettings
from src.scheduler import BotScheduler
from src.scheduler.bot_scheduler import get_job_lock, get_job_queue, set_scheduler
from src.scheduler.jobs import (
    check_notifications,
    post_autonomous_tweet,
    process_inspiration_queue,
    read_frontpage_posts,
)
from src.monitoring.logging_config import setup_logging
from src.state.manager import load_state, save_state

# Configure logging (before other imports that might log)
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir="logs",
    log_file="bot.log",
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5,
    console_output=True,  # Keep console output for development
)
logger = logging.getLogger(__name__)


def load_env_settings() -> EnvSettings:
    """Load environment variables into a typed dictionary.

    Returns:
        EnvSettings dictionary with environment variable values
    """
    return EnvSettings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
        TWITTER_USERNAME=os.getenv("TWITTER_USERNAME"),
        TWITTER_PASSWORD=os.getenv("TWITTER_PASSWORD"),
        LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY"),
        LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT"),
        LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2"),
    )


def validate_env_settings(env_settings: EnvSettings) -> None:
    """Validate that required environment variables are set.

    Args:
        env_settings: Typed dictionary with environment variables

    Raises:
        ValueError: If required environment variables are missing
    """
    openai_api_key = env_settings.get("OPENAI_API_KEY")
    openrouter_api_key = env_settings.get("OPENROUTER_API_KEY")
    google_api_key = env_settings.get("GOOGLE_API_KEY")
    anthropic_api_key = env_settings.get("ANTHROPIC_API_KEY")
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    if not any([openai_api_key, openrouter_api_key, google_api_key, anthropic_api_key]):
        raise ValueError(
            "At least one LLM provider API key is required. "
            "Please set OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY."
        )
    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")


async def set_bot_started() -> None:
    """Set the bot_started_at timestamp in state."""
    state = await load_state()
    state.bot_started_at = datetime.now(timezone.utc)
    state.last_action = "Bot started"
    state.last_action_time = datetime.now(timezone.utc)
    await save_state(state)
    logger.info(
        "bot_started_at_set", extra={"timestamp": state.bot_started_at.isoformat()}
    )


def create_job_wrapper(job_func, config: BotConfig, env_settings: EnvSettings):
    """Create a wrapper function for a job that passes config and env_settings.

    Uses a global lock to prevent parallel job execution. When the lock is busy,
    jobs are queued instead of skipped and processed after the current job completes.

    Args:
        job_func: Job function to wrap
        config: Bot configuration
        env_settings: Typed environment settings dictionary

    Returns:
        Wrapped function that can be called without arguments
    """
    job_id = job_func.__name__

    def run_job():
        """Execute the actual job."""
        job_func(config, env_settings)

    def process_queue():
        """Process any pending jobs in the queue."""
        job_queue = get_job_queue()
        while not job_queue.is_empty():
            next_job = job_queue.get_next()
            if next_job:
                next_job_id, next_func = next_job
                logger.info(
                    "processing_queued_job",
                    extra={"job_id": next_job_id, "remaining": job_queue.size()},
                )
                try:
                    next_func()
                except Exception as e:
                    logger.error(
                        "queued_job_failed",
                        extra={"job_id": next_job_id, "error": str(e)},
                        exc_info=True,
                    )

    def wrapper():
        lock = get_job_lock()
        job_queue = get_job_queue()

        if not lock.acquire(blocking=False):
            # Lock is held - queue this job instead of skipping
            job_queue.add_job(job_id, run_job)
            return

        try:
            run_job()
            # After completing, process any queued jobs
            process_queue()
        finally:
            lock.release()

    return wrapper


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

    # Register scheduler globally for API access (config reload)
    set_scheduler(scheduler)

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

    # Set bot started timestamp
    asyncio.run(set_bot_started())

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
