"""Data tracking helpers for the web dashboard."""

from datetime import datetime, timezone
from typing import Any

from src.state.database import get_database
from src.state.manager import load_state, save_state
from src.monitoring.token_logging import log_token_usage as core_log_token_usage


async def log_token_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    operation: str,
) -> None:
    """Log token usage for analytics (delegates to core helper)."""
    await core_log_token_usage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        operation=operation,
    )


async def log_written_tweet(
    text: str,
    tweet_type: str = "autonomous",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a successfully written tweet.

    Args:
        text: Tweet text
        tweet_type: Type of tweet (autonomous, inspiration, reply)
        metadata: Optional additional metadata
    """
    db = await get_database()
    await db.store_written_tweet(
        text=text,
        tweet_type=tweet_type,
        metadata=metadata,
    )


async def log_rejected_tweet(
    text: str,
    reason: str,
    operation: str = "autonomous",
) -> None:
    """Log a rejected tweet.

    Args:
        text: Tweet text that was rejected
        reason: Reason for rejection
        operation: Operation type (autonomous, inspiration, reply)
    """
    db = await get_database()
    await db.store_rejected_tweet(
        text=text,
        reason=reason,
        operation=operation,
    )


async def log_action(action: str) -> None:
    """Log the last action performed by the bot.

    Args:
        action: Description of the action (e.g., "Posted tweet", "Read posts")
    """
    state = await load_state()
    state.last_action = action
    state.last_action_time = datetime.now(timezone.utc)
    await save_state(state)
