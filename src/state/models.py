"""State models for agent state management."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Agent state model for persistence."""

    personality: dict[str, Any] = Field(
        default_factory=lambda: {
            "tone": "professional",
            "topics": ["AI", "technology"],
        },
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


class Post(BaseModel):
    """Post model for Twitter/X posts."""

    text: str = Field(..., description="Post content/text")
    username: str = Field(
        ..., description="Author username/handle (e.g., '@testuser' or 'testuser')"
    )
    display_name: str | None = Field(
        default=None, description="Author display name/account name"
    )
    post_id: str | None = Field(default=None, description="Post ID if extractable")
    post_type: str = Field(
        default="text_only",
        description="Type of post: text_only, text_with_media, media_only, retweet, quoted, unknown",
    )
    likes: int = Field(default=0, ge=0, description="Like count")
    retweets: int = Field(default=0, ge=0, description="Retweet count")
    replies: int = Field(default=0, ge=0, description="Reply count")
    timestamp: datetime | None = Field(default=None, description="Post timestamp")
    url: str | None = Field(default=None, description="Post URL")
