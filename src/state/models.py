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
    interesting_posts_queue: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Queue of interesting posts for later reaction processing (max 50 posts)",
    )
    notifications_queue: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Queue of notifications (replies and mentions) for later processing (max 50 notifications)",
    )
    last_notification_check_time: datetime | None = Field(
        default=None, description="Timestamp of last notification check"
    )
    processed_notification_ids: list[str] = Field(
        default_factory=list,
        description="List of processed notification IDs to avoid duplicates (max 100 IDs)",
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
    is_interesting: bool | None = Field(
        default=None,
        description="Whether post matches bot's interests (None = not evaluated yet)",
    )


class Notification(BaseModel):
    """Notification model for Twitter/X notifications."""

    notification_id: str | None = Field(
        default=None, description="Unique notification identifier"
    )
    type: str = Field(
        ...,
        description="Notification type: reply, mention, like, retweet, follow, etc.",
    )
    text: str = Field(..., description="Notification/reply content/text")
    from_username: str = Field(
        ...,
        description="Username of the user who sent the notification (e.g., '@testuser' or 'testuser')",
    )
    from_display_name: str | None = Field(
        default=None, description="Display name of the user who sent the notification"
    )
    original_post_id: str | None = Field(
        default=None, description="ID of the original post being replied to (if applicable)"
    )
    original_post_text: str | None = Field(
        default=None, description="Text of the original post being replied to (if applicable)"
    )
    timestamp: datetime | None = Field(
        default=None, description="Notification timestamp"
    )
    url: str | None = Field(default=None, description="Notification URL")
    is_reply: bool = Field(
        default=False, description="True if this is a reply notification"
    )
    is_mention: bool = Field(
        default=False, description="True if this is a mention notification"
    )