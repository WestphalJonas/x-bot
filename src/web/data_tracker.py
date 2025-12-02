"""Data tracking helpers for the web dashboard."""

from datetime import datetime, timezone
from typing import Any

from src.constants import QueueLimits
from src.state.manager import load_state, save_state


async def log_token_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    operation: str,
) -> None:
    """Log token usage for analytics.

    Args:
        provider: LLM provider name
        model: Model name used
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        operation: Operation type (generate, validate, interest_check, etc.)
    """
    state = await load_state()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "operation": operation,
    }

    state.token_usage_log.append(entry)

    # Keep only recent entries
    if len(state.token_usage_log) > QueueLimits.TOKEN_USAGE_LOG:
        state.token_usage_log = state.token_usage_log[-QueueLimits.TOKEN_USAGE_LOG:]

    await save_state(state)


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
    state = await load_state()

    entry = {
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tweet_type": tweet_type,
    }

    if metadata:
        entry.update(metadata)

    state.written_tweets.append(entry)

    # Keep only recent entries
    if len(state.written_tweets) > QueueLimits.WRITTEN_TWEETS:
        state.written_tweets = state.written_tweets[-QueueLimits.WRITTEN_TWEETS:]

    await save_state(state)


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
    state = await load_state()

    entry = {
        "text": text,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
    }

    state.rejected_tweets.append(entry)

    # Keep only recent entries
    if len(state.rejected_tweets) > QueueLimits.REJECTED_TWEETS:
        state.rejected_tweets = state.rejected_tweets[-QueueLimits.REJECTED_TWEETS:]

    await save_state(state)
