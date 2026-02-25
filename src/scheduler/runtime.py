"""Runtime helpers for signals and scheduler keepalive loop."""

from __future__ import annotations

import logging
import signal
import sys
import time
from collections.abc import Callable
from types import FrameType

from src.scheduler.bot_scheduler import BotScheduler

logger = logging.getLogger(__name__)


def install_signal_handlers(scheduler: BotScheduler) -> None:
    """Install SIGINT/SIGTERM handlers for graceful scheduler shutdown."""

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals."""
        del frame
        logger.info("shutdown_signal_received", extra={"signal": signum})
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def wait_for_scheduler_shutdown(scheduler: BotScheduler) -> None:
    """Keep the main process alive while scheduler is running."""
    logger.info("scheduler_running", extra={"info": "Press Ctrl+C to stop"})
    while scheduler.is_running:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt_received")
            break


def shutdown_scheduler_with_state_update(
    scheduler: BotScheduler,
    set_bot_stopped_func: Callable[[], None],
) -> None:
    """Shutdown scheduler and persist stop state, logging failures."""
    scheduler.shutdown()
    try:
        set_bot_stopped_func()
    except Exception as exc:
        logger.warning(
            "bot_stop_state_update_failed",
            extra={"error": str(exc)},
            exc_info=True,
        )
