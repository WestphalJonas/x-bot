"""API routes for the X bot dashboard."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.scheduler.bot_scheduler import get_job_queue
from src.state.database import get_database
from src.state.manager import load_state
from src.web.app import ChromaMemoryDep

if TYPE_CHECKING:
    from src.memory.chroma_client import ChromaMemory

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


class JobQueueResponse(BaseModel):
    """Response model for job queue status."""

    size: int
    pending_jobs: list[str]
    is_empty: bool


@router.get("/posts/read", response_model=PostsListResponse)
async def get_read_posts(
    limit: int = 50,
) -> PostsListResponse:
    """Get last read posts from SQLite."""
    try:
        db = await get_database()
        posts_data = await db.get_read_posts(limit=limit)
        total = await db.get_read_posts_count()

        posts = []
        for post in posts_data:
            posts.append(
                PostResponse(
                    id=post.get("post_id", ""),
                    text=post.get("text", ""),
                    metadata={
                        "username": post.get("username"),
                        "display_name": post.get("display_name"),
                        "post_type": post.get("post_type"),
                        "url": post.get("url"),
                        "is_interesting": post.get("is_interesting"),
                        "read_at": post.get("read_at"),
                        "post_timestamp": post.get("post_timestamp"),
                    },
                )
            )

        return PostsListResponse(posts=posts, total=total)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching read posts: {e}")


@router.get("/posts/written", response_model=WrittenTweetsListResponse)
async def get_written_tweets(
    limit: int = 50,
) -> WrittenTweetsListResponse:
    """Get written/posted tweets from SQLite."""
    try:
        db = await get_database()
        tweets_data = await db.get_written_tweets(limit=limit)
        total = await db.get_written_tweets_count()

        tweets = []
        for tweet in tweets_data:
            timestamp = tweet.get("created_at")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()

            tweets.append(
                WrittenTweetResponse(
                    text=tweet.get("text", ""),
                    timestamp=timestamp,
                    tweet_type=tweet.get("tweet_type", "autonomous"),
                )
            )

        return WrittenTweetsListResponse(tweets=tweets, total=total)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching written tweets: {e}"
        )


@router.get("/posts/rejected", response_model=RejectedTweetsListResponse)
async def get_rejected_tweets(limit: int = 50) -> RejectedTweetsListResponse:
    """Get rejected tweets from SQLite."""
    try:
        db = await get_database()
        tweets_data = await db.get_rejected_tweets(limit=limit)
        total = await db.get_rejected_tweets_count()

        tweets = []
        for tweet in tweets_data:
            timestamp = tweet.get("rejected_at")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            else:
                timestamp = str(timestamp) if timestamp else ""

            tweets.append(
                RejectedTweetResponse(
                    text=tweet.get("text", ""),
                    reason=tweet.get("reason", "Unknown"),
                    timestamp=timestamp,
                    operation=tweet.get("operation", "unknown"),
                )
            )

        return RejectedTweetsListResponse(tweets=tweets, total=total)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching rejected tweets: {e}"
        )


@router.get("/analytics/tokens", response_model=TokenAnalyticsResponse)
async def get_token_analytics(limit: int = 100) -> TokenAnalyticsResponse:
    """Get token usage analytics from SQLite."""
    try:
        db = await get_database()
        entries_data = await db.get_token_usage(limit=limit)
        stats = await db.get_token_usage_stats()

        entries = []
        for entry in entries_data:
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            else:
                timestamp = str(timestamp) if timestamp else ""

            entries.append(
                TokenUsageResponse(
                    timestamp=timestamp,
                    provider=entry.get("provider", "unknown"),
                    model=entry.get("model", "unknown"),
                    prompt_tokens=entry.get("prompt_tokens", 0),
                    completion_tokens=entry.get("completion_tokens", 0),
                    total_tokens=entry.get("total_tokens", 0),
                    operation=entry.get("operation", "unknown"),
                )
            )

        return TokenAnalyticsResponse(
            entries=entries,
            total_entries=stats.get("total_entries", 0),
            total_tokens_used=stats.get("total_tokens", 0),
            tokens_by_provider=stats.get("tokens_by_provider", {}),
            tokens_by_operation=stats.get("tokens_by_operation", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching token analytics: {e}"
        )


@router.get("/state", response_model=StateResponse)
async def get_state(memory: ChromaMemoryDep = None) -> StateResponse:
    """Get current bot state."""
    state = await load_state()

    # Get SQLite database stats
    memory_stats = None
    try:
        db = await get_database()
        memory_stats = await db.get_stats()
    except Exception:
        pass

    # Merge with ChromaDB stats if available
    if memory is not None:
        try:
            chroma_stats = memory.get_stats()
            if memory_stats:
                memory_stats.update(chroma_stats)
            else:
                memory_stats = chroma_stats
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


@router.get("/queue", response_model=JobQueueResponse)
async def get_job_queue_status() -> JobQueueResponse:
    """Get current job queue status."""
    job_queue = get_job_queue()

    # Get pending job IDs from the queue
    pending_jobs = list(job_queue._pending_job_ids)

    return JobQueueResponse(
        size=job_queue.size(),
        pending_jobs=pending_jobs,
        is_empty=job_queue.is_empty(),
    )
