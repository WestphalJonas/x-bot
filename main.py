"""Main entry point for autonomous Twitter/X posting bot."""

import asyncio
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import BotConfig, EnvSettings
from src.scheduler import BotScheduler
from src.scheduler.bot_scheduler import get_job_lock, get_job_queue, set_scheduler
from src.scheduler.control_server import start_control_server
from src.scheduler.jobs import (
    check_notifications,
    post_autonomous_tweet,
    process_replies,
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
    """Set startup lifecycle state in persisted state."""
    state = await load_state()
    now = datetime.now(timezone.utc)
    state.bot_started_at = now
    state.bot_stopped_at = None
    state.running = True
    state.last_action = "Bot started"
    state.last_action_time = now
    await save_state(state)
    logger.info(
        "bot_started_at_set", extra={"timestamp": state.bot_started_at.isoformat()}
    )


def _control_server_responding(host: str, port: int, timeout: float = 0.75) -> bool:
    """Check whether the scheduler control server health endpoint is reachable."""
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 0) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


async def reconcile_stale_running_state(control_host: str, control_port: int) -> None:
    """Mark a previously running state as stopped if no control server is reachable."""
    state = await load_state()
    if not state.running:
        return

    control_alive = await asyncio.to_thread(
        _control_server_responding, control_host, control_port
    )
    if control_alive:
        logger.warning(
            "startup_reconcile_skipped_control_server_reachable",
            extra={"host": control_host, "port": control_port},
        )
        return

    now = datetime.now(timezone.utc)
    state.running = False
    state.bot_stopped_at = now
    state.last_action = "Recovered stale running state"
    state.last_action_time = now
    await save_state(state)
    logger.info(
        "stale_running_state_reconciled",
        extra={
            "host": control_host,
            "port": control_port,
            "timestamp": now.isoformat(),
        },
    )


async def set_bot_stopped() -> None:
    """Set shutdown lifecycle state in persisted state."""
    state = await load_state()
    now = datetime.now(timezone.utc)
    state.running = False
    state.bot_stopped_at = now
    state.last_action = "Bot stopped"
    state.last_action_time = now
    await save_state(state)
    logger.info("bot_stopped_at_set", extra={"timestamp": now.isoformat()})


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
        scheduler = None
        try:
            from src.scheduler.bot_scheduler import get_scheduler

            scheduler = get_scheduler()
        except Exception:
            scheduler = None

        active_config = scheduler.config if scheduler is not None else config
        # Reload env each run so rotated credentials/keys are picked up without restart.
        active_env_settings = load_env_settings()
        job_func(active_config, active_env_settings or env_settings)

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
                    started = time.perf_counter()
                    next_func()
                    logger.info(
                        "job_wrapper_completed",
                        extra={
                            "job_id": next_job_id,
                            "source": "queue",
                            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        },
                    )
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
            started = time.perf_counter()
            run_job()
            # After completing, process any queued jobs
            process_queue()
            logger.info(
                "job_wrapper_completed",
                extra={
                    "job_id": job_id,
                    "source": "scheduler_wrapper",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        finally:
            lock.release()

    return wrapper


def main():
    """Main entry point for the bot scheduler."""
    # Load environment variables
    env_path = Path("config/.env")
    load_dotenv(dotenv_path=env_path)

    # Load configuration
    config_path = Path("config/config.yaml")
    config = BotConfig.load(config_path)

    logger.info(
        "bot_starting",
        extra={"config_path": str(config_path), "env_path": str(env_path)},
    )

    # Load and validate environment settings
    env_settings = load_env_settings()
    try:
        validate_env_settings(env_settings)
    except ValueError as e:
        logger.error(
            "configuration_error: %s",
            str(e),
            extra={"error": str(e)},
        )
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
    scheduler.setup_replies_job(
        create_job_wrapper(process_replies, config, env_settings)
    )
    scheduler.setup_inspiration_job(
        create_job_wrapper(process_inspiration_queue, config, env_settings)
    )

    control_host = os.getenv("SCHEDULER_CONTROL_HOST", "127.0.0.1")
    control_port = int(os.getenv("SCHEDULER_CONTROL_PORT", "8790"))

    # Reconcile stale persisted running state after crashes/unclean exits.
    asyncio.run(reconcile_stale_running_state(control_host, control_port))

    # Start control server for cross-process reload handling
    start_control_server(scheduler, host=control_host, port=control_port)

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
        try:
            asyncio.run(set_bot_stopped())
        except Exception as e:
            logger.warning(
                "bot_stop_state_update_failed",
                extra={"error": str(e)},
                exc_info=True,
            )
        logger.info("bot_shutdown_complete")


if __name__ == "__main__":
    main()
