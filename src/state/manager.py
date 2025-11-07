"""State management with async load/save operations."""

import json
from pathlib import Path

import aiofiles
from pydantic import ValidationError

from src.state.models import AgentState


async def load_state(state_path: str | Path = "data/state.json") -> AgentState:
    """Load agent state from JSON file.

    Args:
        state_path: Path to state JSON file

    Returns:
        AgentState instance, or default state if file doesn't exist
    """
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        # Return default state if file doesn't exist
        return AgentState()

    try:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
            data = json.loads(content)
            return AgentState(**data)
    except (json.JSONDecodeError, ValidationError, OSError) as e:
        # If file is corrupted, return default state
        # Log error would be added here with structured logging
        return AgentState()


async def save_state(
    state: AgentState, state_path: str | Path = "data/state.json"
) -> None:
    """Save agent state to JSON file atomically.

    Uses temp file → rename pattern for atomic writes.

    Args:
        state: AgentState instance to save
        state_path: Path to state JSON file
    """
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    temp_path = path.with_suffix(".tmp")

    try:
        async with aiofiles.open(temp_path, "w") as f:
            await f.write(state.model_dump_json(indent=2))

        # Atomic rename
        temp_path.replace(path)
    except OSError as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise

