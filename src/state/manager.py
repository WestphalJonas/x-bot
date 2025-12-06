"""State management with async load/save operations."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from pydantic import ValidationError

from src.state.models import AgentState

logger = logging.getLogger(__name__)

# Global lock for state file operations to prevent race conditions
_state_lock = asyncio.Lock()


def should_reset_counters(state: AgentState, reset_time_utc: str = "00:00") -> bool:
    """Check if rate limit counters should be reset based on UTC date.

    Counters are reset once per day at reset_time_utc. The reset happens
    on the first state load after the reset time if last_counter_reset
    was before today's reset time.

    Args:
        state: Current agent state
        reset_time_utc: Time to reset counters in HH:MM format (default: "00:00")

    Returns:
        True if counters should be reset, False otherwise
    """
    now = datetime.now(timezone.utc)

    try:
        reset_hour, reset_minute = map(int, reset_time_utc.split(":"))
    except (ValueError, AttributeError):
        reset_hour, reset_minute = 0, 0

    today_reset = now.replace(
        hour=reset_hour, minute=reset_minute, second=0, microsecond=0
    )

    # Before the daily reset window, only reset if we missed yesterday's reset entirely
    if now < today_reset:
        if state.last_counter_reset is None:
            return True

        last_reset = state.last_counter_reset
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)

        hours_since_reset = (now - last_reset).total_seconds() / 3600
        return hours_since_reset >= 24

    # After the reset window, require that we have recorded a reset for today
    if state.last_counter_reset is None:
        return True

    last_reset = state.last_counter_reset
    if last_reset.tzinfo is None:
        last_reset = last_reset.replace(tzinfo=timezone.utc)

    return last_reset < today_reset


def reset_counters(state: AgentState) -> AgentState:
    """Reset rate limit counters to zero.

    Args:
        state: Current agent state

    Returns:
        Updated state with reset counters
    """
    state.counters = {"posts_today": 0, "replies_today": 0}
    state.last_counter_reset = datetime.now(timezone.utc)
    logger.info(
        "counters_reset",
        extra={
            "reset_time": state.last_counter_reset.isoformat(),
            "counters": state.counters,
        },
    )
    return state


async def load_state(
    state_path: str | Path = "data/state.json",
    reset_time_utc: str = "00:00",
) -> AgentState:
    """Load agent state from JSON file.

    Also checks if rate limit counters need to be reset based on the
    reset_time_utc and resets them if necessary.

    Uses a global lock to prevent race conditions with concurrent access.

    Args:
        state_path: Path to state JSON file
        reset_time_utc: Time to reset counters in HH:MM format (default: "00:00")

    Returns:
        AgentState instance, or default state if file doesn't exist
    """
    async with _state_lock:
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: AgentState
        if not path.exists():
            state = AgentState()
        else:
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    state = AgentState(**data)
            except (
                json.JSONDecodeError,
                ValidationError,
                OSError,
                UnicodeDecodeError,
            ) as e:
                logger.warning(
                    "state_load_error",
                    extra={"error": str(e), "path": str(path)},
                )
                state = AgentState()

        if should_reset_counters(state, reset_time_utc):
            state = reset_counters(state)
            await _save_state_internal(state, state_path)

        return state


async def _save_state_internal(
    state: AgentState, state_path: str | Path = "data/state.json"
) -> None:
    """Internal save function that doesn't acquire lock (caller must hold lock).

    Uses temp file → rename pattern for atomic writes.

    Args:
        state: AgentState instance to save
        state_path: Path to state JSON file
    """
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    try:
        async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
            await f.write(state.model_dump_json(indent=2))

        # Atomic rename keeps readers from seeing partial writes
        temp_path.replace(path)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise


async def save_state(
    state: AgentState, state_path: str | Path = "data/state.json"
) -> None:
    """Save agent state to JSON file atomically.

    Uses temp file → rename pattern for atomic writes.
    Uses a global lock to prevent race conditions with concurrent access.

    Args:
        state: AgentState instance to save
        state_path: Path to state JSON file
    """
    async with _state_lock:
        await _save_state_internal(state, state_path)
