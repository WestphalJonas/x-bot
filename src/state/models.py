"""State models for agent state management."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Agent state model for persistence."""

    personality: dict[str, Any] = Field(
        default_factory=lambda: {"tone": "professional", "topics": ["AI", "technology"]},
        description="Bot personality configuration",
    )
    counters: dict[str, int] = Field(
        default_factory=lambda: {"posts_today": 0, "replies_today": 0},
        description="Rate limit counters",
    )
    last_post_time: datetime | None = Field(
        default=None, description="Timestamp of last post"
    )
    mood: str = Field(
        default="neutral", description="Current mood state (for future use)"
    )

