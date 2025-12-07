"""API routes for the X bot dashboard."""

import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig, EnvSettings
from src.core.llm import LLMClient
from src.scheduler.bot_scheduler import get_job_queue, get_scheduler
from src.state.database import get_database
from src.state.manager import load_state
from src.web.app import ChromaMemoryDep, ConfigDep, get_config

router = APIRouter()

CONTROL_URL = os.getenv("SCHEDULER_CONTROL_URL")
CONTROL_HOST = os.getenv("SCHEDULER_CONTROL_HOST", "127.0.0.1")
CONTROL_PORT = os.getenv("SCHEDULER_CONTROL_PORT", "8790")


async def _post_control(path: str, local_method: str | None = None) -> dict[str, Any]:
    """Send a POST to the scheduler control server with optional local fallback."""
    base = CONTROL_URL or f"http://{CONTROL_HOST}:{CONTROL_PORT}"
    url = f"{base}{path}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        if local_method:
            scheduler = get_scheduler()
            if scheduler is not None:
                try:
                    method = getattr(scheduler, local_method)
                    return await asyncio.to_thread(method)
                except Exception as inner_exc:
                    return {
                        "status": "error",
                        "reason": f"local_{local_method}_failed: {inner_exc}",
                    }
        return {"status": "error", "reason": f"control_request_failed: {exc}"}


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


class ChatMessageRequest(BaseModel):
    """Request model for manual chat input."""

    message: str


class ChatMessageResponse(BaseModel):
    """Response model for manual chat output."""

    reply: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: str


def _load_env_settings() -> EnvSettings:
    """Load environment variables needed for LLM access."""
    return EnvSettings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
        TWITTER_USERNAME=os.getenv("TWITTER_USERNAME"),
        TWITTER_PASSWORD=os.getenv("TWITTER_PASSWORD"),
        LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY"),
        LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT"),
        LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2"),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _run_chat(message: str, config: BotConfig) -> ChatMessageResponse:
    """Call the LLM pipeline with retry/backoff."""
    env_settings = _load_env_settings()
    llm_client = LLMClient(config=config, env_settings=env_settings)

    system_prompt = config.get_system_prompt()
    result = await llm_client.chat(
        user_prompt=message,
        system_prompt=system_prompt,
        operation="manual_chat",
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
    )

    usage = result.usage
    return ChatMessageResponse(
        reply=result.content,
        provider=result.provider,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


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


@router.post("/chat", response_model=ChatMessageResponse)
async def chat_with_agent(
    payload: ChatMessageRequest, config: ConfigDep
) -> ChatMessageResponse:
    """Chat with the agent using the configured LLM pipeline."""
    try:
        return await _run_chat(payload.message, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


class SettingsUpdateRequest(BaseModel):
    """Request model for settings update."""

    personality: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    rate_limits: dict[str, Any] | None = None
    selenium: dict[str, Any] | None = None
    queue_limits: dict[str, Any] | None = None


class SettingsResponse(BaseModel):
    """Response model for settings."""

    personality: dict[str, Any]
    scheduler: dict[str, Any]
    llm: dict[str, Any]
    rate_limits: dict[str, Any]
    selenium: dict[str, Any]
    queue_limits: dict[str, Any]


@router.get("/settings", response_model=SettingsResponse)
async def get_settings(config: ConfigDep) -> SettingsResponse:
    """Get current bot configuration settings."""
    config_dict = config.model_dump()
    return SettingsResponse(
        personality=config_dict.get("personality", {}),
        scheduler=config_dict.get("scheduler", {}),
        llm=config_dict.get("llm", {}),
        rate_limits=config_dict.get("rate_limits", {}),
        selenium=config_dict.get("selenium", {}),
        queue_limits=config_dict.get("queue_limits", {}),
    )


@router.post("/settings")
async def update_settings(updates: SettingsUpdateRequest) -> dict[str, Any]:
    """Update bot configuration settings.

    Creates a backup of the current config before saving.
    Clears the config cache and triggers scheduler reload.
    """
    config_path = Path("config/config.yaml")
    backup_path = Path("config/config.yaml.bak")

    try:
        # Load current config
        current_config = get_config()
        current_dict = current_config.model_dump()

        # Create backup
        if config_path.exists():
            shutil.copy(config_path, backup_path)

        # Merge updates (only provided sections)
        updates_dict = updates.model_dump(exclude_none=True)
        for section, values in updates_dict.items():
            if section in current_dict and isinstance(values, dict):
                # Deep merge: update only provided fields
                current_dict[section].update(values)

        # Validate and save
        new_config = BotConfig(**current_dict)
        new_config.save(str(config_path))

        # Clear the config cache so next request gets fresh config
        get_config.cache_clear()

        # Reload scheduler with new config
        reload_result = await _reload_scheduler_config()

        return {
            "status": "ok",
            "message": "Settings saved successfully",
            "scheduler_reload": reload_result,
        }

    except Exception as e:
        # Restore backup if save failed
        if backup_path.exists():
            shutil.copy(backup_path, config_path)
        raise HTTPException(status_code=400, detail=f"Failed to save settings: {e}")


async def _reload_scheduler_config() -> dict[str, Any]:
    """Helper to reload scheduler config via control server with local fallback."""
    return await _post_control("/reload", local_method="reload_config")


async def _pause_scheduler() -> dict[str, Any]:
    """Pause scheduler via control server."""
    return await _post_control("/pause", local_method="pause_all")


async def _resume_scheduler() -> dict[str, Any]:
    """Resume scheduler via control server."""
    return await _post_control("/resume", local_method="resume_all")


@router.post("/scheduler/pause")
async def pause_scheduler() -> dict[str, Any]:
    """Pause all scheduler jobs and capture next runs."""
    result = await _pause_scheduler()
    if result.get("status") == "error":
        raise HTTPException(
            status_code=500, detail=result.get("reason", "Pause failed")
        )
    return {"status": "ok", **result}


@router.post("/scheduler/resume")
async def resume_scheduler() -> dict[str, Any]:
    """Resume scheduler jobs using captured next runs."""
    result = await _resume_scheduler()
    if result.get("status") == "error":
        raise HTTPException(
            status_code=500, detail=result.get("reason", "Resume failed")
        )
    return {"status": "ok", **result}


@router.post("/config/reload")
async def reload_config() -> dict[str, Any]:
    """Manually reload configuration and reschedule jobs.

    This endpoint allows manually triggering a config reload without
    saving new settings (useful if config.yaml was edited externally).
    """
    # Clear the config cache first
    get_config.cache_clear()

    # Reload scheduler
    result = await _reload_scheduler_config()

    if result.get("status") == "error":
        raise HTTPException(
            status_code=500, detail=result.get("reason", "Reload failed")
        )

    return {
        "status": "ok",
        "message": "Configuration reloaded",
        **result,
    }
