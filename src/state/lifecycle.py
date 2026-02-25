"""Persisted lifecycle state helpers for bot start/stop transitions."""

from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone

from src.state.manager import load_state, save_state

logger = logging.getLogger(__name__)


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
    logger.info("bot_started_at_set", extra={"timestamp": state.bot_started_at.isoformat()})


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
        extra={"host": control_host, "port": control_port, "timestamp": now.isoformat()},
    )
