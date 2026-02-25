"""Main entry point for autonomous Twitter/X posting bot."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import BotConfig
from src.core.env import load_env_settings, validate_env_settings
from src.scheduler import BotScheduler
from src.scheduler.bootstrap import register_scheduler_jobs
from src.scheduler.bot_scheduler import set_scheduler
from src.scheduler.control_server import start_control_server
from src.scheduler.runtime import (
    install_signal_handlers,
    shutdown_scheduler_with_state_update,
    wait_for_scheduler_shutdown,
)
from src.monitoring.logging_config import setup_logging
from src.state.lifecycle import (
    reconcile_stale_running_state,
    set_bot_started,
    set_bot_stopped,
)

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

    register_scheduler_jobs(scheduler, config, env_settings)

    control_host = os.getenv("SCHEDULER_CONTROL_HOST", "127.0.0.1")
    control_port = int(os.getenv("SCHEDULER_CONTROL_PORT", "8790"))

    # Reconcile stale persisted running state after crashes/unclean exits.
    asyncio.run(reconcile_stale_running_state(control_host, control_port))

    # Start control server for cross-process reload handling
    start_control_server(scheduler, host=control_host, port=control_port)

    install_signal_handlers(scheduler)

    # Set bot started timestamp
    asyncio.run(set_bot_started())

    # Start scheduler
    try:
        scheduler.start()

        wait_for_scheduler_shutdown(scheduler)

    except Exception as e:
        logger.error("scheduler_error", extra={"error": str(e)}, exc_info=True)
        raise

    finally:
        shutdown_scheduler_with_state_update(
            scheduler,
            lambda: asyncio.run(set_bot_stopped()),
        )
        logger.info("bot_shutdown_complete")


if __name__ == "__main__":
    main()
