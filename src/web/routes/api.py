"""API routes for the X bot dashboard."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.state.manager import load_state
from src.web.app import get_chroma_memory, get_config

router = APIRouter()


class PostResponse(BaseModel):
    """Response model for a post."""

    id: str
    text: str
    metadata: dict[str, Any]


class PostsListResponse(BaseModel):
    """Response model for list of posts."""

    posts: list[PostResponse]
    total: int


class WrittenTweetResponse(BaseModel):
    """Response model for a written tweet."""

    text: str
    timestamp: str | None
    tweet_type: str


class WrittenTweetsListResponse(BaseModel):
    """Response model for list of written tweets."""

    tweets: list[WrittenTweetResponse]
    total: int


class RejectedTweetResponse(BaseModel):
    """Response model for a rejected tweet."""

    text: str
    reason: str
    timestamp: str
    operation: str


class RejectedTweetsListResponse(BaseModel):
    """Response model for list of rejected tweets."""

    tweets: list[RejectedTweetResponse]
    total: int


class TokenUsageResponse(BaseModel):
    """Response model for token usage entry."""

    timestamp: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    operation: str


class TokenAnalyticsResponse(BaseModel):
    """Response model for token analytics."""

    entries: list[TokenUsageResponse]
    total_entries: int
    total_tokens_used: int
    tokens_by_provider: dict[str, int]
    tokens_by_operation: dict[str, int]


class StateResponse(BaseModel):
    """Response model for bot state."""

    counters: dict[str, int]
    last_post_time: str | None
    mood: str
    interesting_posts_queue_size: int
    notifications_queue_size: int
    last_notification_check_time: str | None
    memory_stats: dict[str, int] | None


@router.get("/posts/read", response_model=PostsListResponse)
async def get_read_posts(limit: int = 50) -> PostsListResponse:
    """Get last read posts from ChromaDB."""
    memory = get_chroma_memory()

    if memory is None:
        return PostsListResponse(posts=[], total=0)

    try:
        # Get all posts from the read_posts collection
        collection = memory.posts_collection
        count = collection.count()

        if count == 0:
            return PostsListResponse(posts=[], total=0)

        # Get the posts (ChromaDB returns all if limit > count)
        results = collection.get(
            limit=min(limit, count),
            include=["documents", "metadatas"],
        )

        posts = []
        for i, doc_id in enumerate(results["ids"]):
            posts.append(
                PostResponse(
                    id=doc_id,
                    text=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
            )

        # Sort by timestamp descending (most recent first)
        posts.sort(
            key=lambda p: p.metadata.get("timestamp", ""),
            reverse=True,
        )

        return PostsListResponse(posts=posts[:limit], total=count)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching read posts: {e}")


@router.get("/posts/written", response_model=WrittenTweetsListResponse)
async def get_written_tweets(limit: int = 50) -> WrittenTweetsListResponse:
    """Get written/posted tweets from state and ChromaDB."""
    # First try to get from state (more recent with metadata)
    state = await load_state()
    tweets_from_state = []

    for tweet_data in state.written_tweets[-limit:]:
        timestamp = tweet_data.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        elif timestamp is None:
            timestamp = None

        tweets_from_state.append(
            WrittenTweetResponse(
                text=tweet_data.get("text", ""),
                timestamp=timestamp,
                tweet_type=tweet_data.get("tweet_type", "autonomous"),
            )
        )

    # Also try to get from ChromaDB for historical data
    memory = get_chroma_memory()
    tweets_from_chroma = []

    if memory is not None:
        try:
            collection = memory.tweets_collection
            count = collection.count()

            if count > 0:
                results = collection.get(
                    limit=min(limit, count),
                    include=["documents", "metadatas"],
                )

                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    tweets_from_chroma.append(
                        WrittenTweetResponse(
                            text=results["documents"][i]
                            if results["documents"]
                            else "",
                            timestamp=metadata.get("timestamp"),
                            tweet_type=metadata.get("tweet_type", "unknown"),
                        )
                    )
        except Exception:
            pass  # Ignore ChromaDB errors, use state data

    # Combine and deduplicate (prefer state data as it's more recent)
    all_tweets = tweets_from_state + tweets_from_chroma
    seen_texts = set()
    unique_tweets = []

    for tweet in all_tweets:
        if tweet.text not in seen_texts:
            seen_texts.add(tweet.text)
            unique_tweets.append(tweet)

    # Sort by timestamp descending
    unique_tweets.sort(
        key=lambda t: t.timestamp or "",
        reverse=True,
    )

    return WrittenTweetsListResponse(
        tweets=unique_tweets[:limit],
        total=len(unique_tweets),
    )


@router.get("/posts/rejected", response_model=RejectedTweetsListResponse)
async def get_rejected_tweets(limit: int = 50) -> RejectedTweetsListResponse:
    """Get rejected tweets from state."""
    state = await load_state()

    tweets = []
    for tweet_data in state.rejected_tweets[-limit:]:
        timestamp = tweet_data.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        else:
            timestamp = str(timestamp) if timestamp else ""

        tweets.append(
            RejectedTweetResponse(
                text=tweet_data.get("text", ""),
                reason=tweet_data.get("reason", "Unknown"),
                timestamp=timestamp,
                operation=tweet_data.get("operation", "unknown"),
            )
        )

    # Sort by timestamp descending
    tweets.sort(key=lambda t: t.timestamp, reverse=True)

    return RejectedTweetsListResponse(
        tweets=tweets[:limit],
        total=len(state.rejected_tweets),
    )


@router.get("/analytics/tokens", response_model=TokenAnalyticsResponse)
async def get_token_analytics() -> TokenAnalyticsResponse:
    """Get token usage analytics."""
    state = await load_state()

    entries = []
    total_tokens = 0
    tokens_by_provider: dict[str, int] = {}
    tokens_by_operation: dict[str, int] = {}

    for entry_data in state.token_usage_log:
        timestamp = entry_data.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        else:
            timestamp = str(timestamp) if timestamp else ""

        provider = entry_data.get("provider", "unknown")
        operation = entry_data.get("operation", "unknown")
        entry_total = entry_data.get("total_tokens", 0)

        entries.append(
            TokenUsageResponse(
                timestamp=timestamp,
                provider=provider,
                model=entry_data.get("model", "unknown"),
                prompt_tokens=entry_data.get("prompt_tokens", 0),
                completion_tokens=entry_data.get("completion_tokens", 0),
                total_tokens=entry_total,
                operation=operation,
            )
        )

        total_tokens += entry_total
        tokens_by_provider[provider] = tokens_by_provider.get(provider, 0) + entry_total
        tokens_by_operation[operation] = (
            tokens_by_operation.get(operation, 0) + entry_total
        )

    # Sort entries by timestamp descending
    entries.sort(key=lambda e: e.timestamp, reverse=True)

    return TokenAnalyticsResponse(
        entries=entries,
        total_entries=len(entries),
        total_tokens_used=total_tokens,
        tokens_by_provider=tokens_by_provider,
        tokens_by_operation=tokens_by_operation,
    )


@router.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """Get current bot state."""
    state = await load_state()
    memory = get_chroma_memory()

    memory_stats = None
    if memory is not None:
        try:
            memory_stats = memory.get_stats()
        except Exception:
            pass

    last_post_time = None
    if state.last_post_time:
        last_post_time = state.last_post_time.isoformat()

    last_notification_check = None
    if state.last_notification_check_time:
        last_notification_check = state.last_notification_check_time.isoformat()

    return StateResponse(
        counters=state.counters,
        last_post_time=last_post_time,
        mood=state.mood,
        interesting_posts_queue_size=len(state.interesting_posts_queue),
        notifications_queue_size=len(state.notifications_queue),
        last_notification_check_time=last_notification_check,
        memory_stats=memory_stats,
    )
