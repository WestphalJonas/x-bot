"""API routes for the X bot dashboard."""

import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig, EnvSettings
from src.core.llm import LLMClient
from src.scheduler.bot_scheduler import get_job_queue, get_scheduler
from src.state.database import get_database
from src.state.manager import load_state, save_state
from src.web.deps import ChromaMemoryDep, ConfigDep, get_config
from src.monitoring.data_tracker import log_action, log_written_tweet
from src.x.posting import post_tweet
from src.x.session import AsyncTwitterSession

router = APIRouter()
logger = logging.getLogger(__name__)

_DASHBOARD_LEVEL_EMOJI = {
    "DEBUG": "🔍",
    "INFO": "ℹ️",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "🔥",
}

_DASHBOARD_TASK_EMOJI = {
    "scheduler": "⏰",
    "posting": "📝",
    "reading": "📖",
    "notifications": "🔔",
    "replies": "💬",
    "llm": "🧠",
    "auth": "🔐",
    "web": "🌐",
    "db": "💾",
    "state": "💾",
    "browser": "🧭",
    "driver": "🧭",
    "x": "🧭",
    "memory": "🧠",
    "system": "🚀",
}

CONTROL_URL = (os.getenv("SCHEDULER_CONTROL_URL") or "").strip() or None
CONTROL_HOST = (os.getenv("SCHEDULER_CONTROL_HOST", "127.0.0.1") or "127.0.0.1").strip()
CONTROL_PORT = (os.getenv("SCHEDULER_CONTROL_PORT", "8790") or "8790").strip()


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


async def _get_control(path: str) -> dict[str, Any]:
    """Send a GET to the scheduler control server."""
    base = CONTROL_URL or f"http://{CONTROL_HOST}:{CONTROL_PORT}"
    url = f"{base}{path}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            return {"status": "error", "reason": "invalid_control_response"}
    except Exception as exc:
        return {"status": "error", "reason": f"control_request_failed: {exc}"}


def _tail_lines(path: Path, limit: int = 20) -> list[str]:
    """Read the last N lines from a text file."""
    if not path.exists():
        return []
    limit = max(1, min(limit, 200))
    try:
        from collections import deque

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return [line.rstrip("\n") for line in deque(f, maxlen=limit)]
    except Exception:
        return []


def _humanize_log_line(line: str) -> str:
    """Render a JSONL log line into the app's human-readable console style."""
    raw = (line or "").strip()
    if not raw:
        return ""

    try:
        payload = json.loads(raw)
    except Exception:
        return raw

    if not isinstance(payload, dict):
        return raw

    timestamp = str(payload.get("timestamp") or "")
    time_part = ""
    if "T" in timestamp:
        try:
            time_part = timestamp.split("T", 1)[1][:8]
        except Exception:
            time_part = ""

    level = str(payload.get("level") or "INFO").upper()
    task = str(payload.get("task") or "system")
    message = str(
        payload.get("message_human")
        or payload.get("message")
        or payload.get("event")
        or "log"
    )

    context = payload.get("context")
    context_text = ""
    if isinstance(context, dict) and context:
        preferred_keys = [
            "job_id",
            "operation",
            "provider",
            "model",
            "post_id",
            "notification_id",
            "username",
            "from_username",
            "duration_ms",
            "status",
            "error",
        ]
        parts: list[str] = []
        seen: set[str] = set()
        for key in preferred_keys:
            if key in context:
                parts.append(f"{key}={context.get(key)}")
                seen.add(key)
            if len(parts) >= 4:
                break
        if len(parts) < 4:
            for key in sorted(context.keys()):
                if key in seen:
                    continue
                parts.append(f"{key}={context.get(key)}")
                if len(parts) >= 4:
                    break
        if parts:
            context_text = " | " + " ".join(parts)

    level_emoji = _DASHBOARD_LEVEL_EMOJI.get(level, "")
    task_emoji = _DASHBOARD_TASK_EMOJI.get(task, "")
    icons = " ".join(part for part in [level_emoji, task_emoji] if part).strip()
    icons = f"{icons} " if icons else ""
    task_label = f"[{task}]"

    prefix = " ".join(part for part in [time_part, f"{icons}{task_label}".strip()] if part)
    if prefix:
        return f"{prefix} {message}{context_text}".strip()
    return f"{task_label} {message}{context_text}".strip()


def _summarize_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    """Create a compact queue item summary for dashboard previews."""
    text = str(
        item.get("text")
        or item.get("original_post_text")
        or item.get("post_text")
        or item.get("content")
        or ""
    ).strip()
    if len(text) > 120:
        text = text[:117].rstrip() + "..."

    timestamp = item.get("timestamp")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()

    return {
        "id": item.get("notification_id") or item.get("post_id") or item.get("id"),
        "type": item.get("type") or item.get("post_type"),
        "username": item.get("from_username") or item.get("username"),
        "display_name": item.get("from_display_name") or item.get("display_name"),
        "text": text,
        "timestamp": timestamp,
        "url": item.get("url"),
    }


async def _log_stream(
    log_path: Path, level: str | None, tail_bytes: int, human: bool = False
) -> AsyncIterator[str]:
    """Async generator that streams log lines as SSE data."""
    if not log_path.exists():
        yield "data: log file not found\n\n"
        return

    level_filter = level.upper() if level else None

    def _should_emit(line: str) -> bool:
        if not level_filter:
            return True
        return level_filter in line.upper()

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        if tail_bytes > 0:
            try:
                size = log_path.stat().st_size
                f.seek(max(size - tail_bytes, 0))
                f.readline()  # discard partial
            except OSError:
                pass

        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue

            if not _should_emit(line):
                continue

            payload = line.rstrip("\n")
            if human:
                payload = _humanize_log_line(payload)
                if not payload:
                    continue
            yield f"data: {payload}\n\n"


@router.get("/logs/stream")
async def stream_logs(
    level: str | None = None, tail_bytes: int = 8000, human: bool = False
):
    """Stream live logs from bot.log as Server-Sent Events."""
    tail_bytes = max(tail_bytes, 0)
    log_path = Path("logs/bot.log")
    generator = _log_stream(log_path, level, tail_bytes, human=human)
    return StreamingResponse(generator, media_type="text/event-stream")


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
    last_reply_time: str | None
    last_reply_status: str | None
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


class ManualPostRequest(BaseModel):
    """Request model for posting an assistant reply."""

    text: str
    metadata: dict[str, Any] | None = None


class ManualPostResponse(BaseModel):
    """Response model for manual post action."""

    status: str
    message: str
    text: str
    posted_at: str


def _chunk_text_for_stream(text: str, max_chunk_len: int = 24) -> list[str]:
    """Split text into small chunks for incremental UI streaming."""
    if not text:
        return []
    parts = re.findall(r"\S+\s*|\s+", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > max_chunk_len:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _run_chat(message: str, config: BotConfig) -> ChatMessageResponse:
    """Call the LLM pipeline with retry/backoff."""
    env_settings = _load_env_settings()
    llm_client = LLMClient(config=config, env_settings=env_settings)
    try:
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
    finally:
        await llm_client.close()


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

    last_reply_time = None
    if state.last_reply_time:
        last_reply_time = state.last_reply_time.isoformat()

    last_notification_check = None
    if state.last_notification_check_time:
        last_notification_check = state.last_notification_check_time.isoformat()

    return StateResponse(
        counters=state.counters,
        last_post_time=last_post_time,
        last_reply_time=last_reply_time,
        last_reply_status=state.last_reply_status,
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
    pending_jobs = job_queue.pending_job_ids()

    return JobQueueResponse(
        size=job_queue.size(),
        pending_jobs=pending_jobs,
        is_empty=job_queue.is_empty(),
    )


@router.get("/scheduler/jobs")
async def get_scheduler_jobs() -> dict[str, Any]:
    """Get scheduler jobs and next-run timing snapshot."""
    result = await _get_scheduler_jobs()
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("reason", "Failed"))
    return result


@router.post("/scheduler/run/{job_id}")
async def run_scheduler_job(job_id: str) -> dict[str, Any]:
    """Trigger a scheduler job to run immediately."""
    result = await _run_scheduler_job_now(job_id)
    if result.get("status") == "error":
        raise HTTPException(
            status_code=500, detail=result.get("reason", "Run job failed")
        )
    return result


@router.get("/logs/tail")
async def get_logs_tail(lines: int = 20, human: bool = True) -> dict[str, Any]:
    """Return tail of bot.log for dashboard preview."""
    tail = _tail_lines(Path("logs/bot.log"), limit=lines)
    if human:
        tail = [rendered for rendered in (_humanize_log_line(line) for line in tail) if rendered]
    return {"status": "ok", "lines": tail}


@router.get("/dashboard/overview")
async def get_dashboard_overview(config: ConfigDep) -> dict[str, Any]:
    """Aggregated dashboard snapshot for home view polling."""
    state = await load_state()
    job_queue = get_job_queue()
    now = datetime.now(timezone.utc)

    db_stats: dict[str, Any] = {}
    hourly_tokens: list[dict[str, Any]] = []
    try:
        db = await get_database()
        db_stats = await db.get_stats()
        hourly_tokens = await db.get_hourly_token_usage(hours=12)
    except Exception:
        db_stats = {}
        hourly_tokens = []

    scheduler_result = await _get_scheduler_jobs()
    jobs = scheduler_result.get("jobs", []) if scheduler_result.get("status") == "ok" else []

    health_result = await _get_control("/health")
    bot_active = health_result.get("status") == "ok"

    timeline: list[dict[str, Any]] = []
    events = [
        ("Last action", "✅", state.last_action_time, state.last_action),
        ("Last post", "📝", state.last_post_time, None),
        ("Notification check", "🔔", state.last_notification_check_time, None),
        ("Last reply", "💬", state.last_reply_time, state.last_reply_status),
        ("Bot started", "⏱️", state.bot_started_at, None),
    ]
    for label, icon, timestamp, detail in events:
        if timestamp is None:
            continue
        timeline.append(
            {
                "label": label,
                "icon": icon,
                "timestamp": timestamp.isoformat(),
                "detail": detail,
            }
        )
    timeline.sort(key=lambda item: item["timestamp"], reverse=True)

    cookie_path = Path("config/cookie.json")
    cookie_present = cookie_path.exists() and cookie_path.stat().st_size > 0

    max_posts = max(config.rate_limits.max_posts_per_day, 1)
    max_replies = max(config.rate_limits.max_replies_per_day, 1)
    posts_today = state.counters.get("posts_today", 0)
    replies_today = state.counters.get("replies_today", 0)

    return {
        "status": "ok",
        "generated_at": now.isoformat(),
        "health": {
            "active": bot_active,
            "scheduler_running": scheduler_result.get("scheduler_running"),
            "scheduler_paused": state.paused,
            "state_running": state.running,
            "cookie_present": cookie_present,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "bot_started_at": state.bot_started_at.isoformat() if state.bot_started_at else None,
            "bot_stopped_at": state.bot_stopped_at.isoformat() if state.bot_stopped_at else None,
            "last_action": state.last_action,
            "last_action_time": state.last_action_time.isoformat()
            if state.last_action_time
            else None,
            "last_reply_status": state.last_reply_status,
        },
        "scheduler": {"jobs": jobs},
        "queues": {
            "pending_jobs": job_queue.pending_job_ids(),
            "interesting_preview": [
                _summarize_queue_item(item) for item in state.interesting_posts_queue[:5]
            ],
            "notifications_preview": [
                _summarize_queue_item(item) for item in state.notifications_queue[:5]
            ],
        },
        "today": {
            "posts_today": posts_today,
            "replies_today": replies_today,
            "posts_remaining": max(max_posts - posts_today, 0),
            "replies_remaining": max(max_replies - replies_today, 0),
            "posts_pct": min(int((posts_today / max_posts) * 100), 100),
            "replies_pct": min(int((replies_today / max_replies) * 100), 100),
            "read_posts_total": db_stats.get("read_posts_count", 0),
            "written_tweets_total": db_stats.get("written_tweets_count", 0),
            "rejected_tweets_total": db_stats.get("rejected_tweets_count", 0),
            "token_usage_entries": db_stats.get("token_usage_entries", 0),
            "hourly_tokens": hourly_tokens,
        },
        "pipeline": {
            "read_posts_total": db_stats.get("read_posts_count", 0),
            "interesting_queue": len(state.interesting_posts_queue),
            "notifications_queue": len(state.notifications_queue),
            "pending_jobs": job_queue.size(),
            "written_tweets_total": db_stats.get("written_tweets_count", 0),
            "rejected_tweets_total": db_stats.get("rejected_tweets_count", 0),
            "posts_today": posts_today,
            "replies_today": replies_today,
        },
        "timeline": timeline[:8],
        "logs": {
            "tail": [
                rendered
                for rendered in (
                    _humanize_log_line(line)
                    for line in _tail_lines(Path("logs/bot.log"), limit=16)
                )
                if rendered
            ]
        },
    }


@router.post("/chat", response_model=ChatMessageResponse)
async def chat_with_agent(
    payload: ChatMessageRequest, config: ConfigDep
) -> ChatMessageResponse:
    """Chat with the agent using the configured LLM pipeline."""
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    env_settings = _load_env_settings()
    has_provider = any(
        [
            env_settings.get("OPENAI_API_KEY"),
            env_settings.get("OPENROUTER_API_KEY"),
            env_settings.get("GOOGLE_API_KEY"),
            env_settings.get("ANTHROPIC_API_KEY"),
        ]
    )
    if not has_provider:
        raise HTTPException(
            status_code=400,
            detail="No LLM API key configured (.env). Set at least one of OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY.",
        )

    try:
        return await _run_chat(message, config)
    except RetryError as exc:
        root_exc = exc.last_attempt.exception() if exc.last_attempt else exc
        raise HTTPException(status_code=500, detail=f"Chat failed: {root_exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@router.post("/chat/stream")
async def chat_with_agent_stream(
    payload: ChatMessageRequest, config: ConfigDep
) -> StreamingResponse:
    """Stream a chat response as SSE events for the dashboard chat UI."""
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    env_settings = _load_env_settings()
    has_provider = any(
        [
            env_settings.get("OPENAI_API_KEY"),
            env_settings.get("OPENROUTER_API_KEY"),
            env_settings.get("GOOGLE_API_KEY"),
            env_settings.get("ANTHROPIC_API_KEY"),
        ]
    )
    if not has_provider:
        raise HTTPException(
            status_code=400,
            detail="No LLM API key configured (.env). Set at least one of OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY.",
        )

    async def _stream() -> AsyncIterator[str]:
        try:
            yield "event: start\ndata: {}\n\n"
            result = await _run_chat(message, config)
            chunks = _chunk_text_for_stream(result.reply)
            for chunk in chunks:
                payload_json = json.dumps({"delta": chunk}, ensure_ascii=False)
                yield f"event: chunk\ndata: {payload_json}\n\n"
                await asyncio.sleep(0.015)

            meta_json = json.dumps(
                {
                    "provider": result.provider,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "timestamp": result.timestamp,
                },
                ensure_ascii=False,
            )
            yield f"event: meta\ndata: {meta_json}\n\n"
            yield "event: done\ndata: {}\n\n"
        except RetryError as exc:
            root_exc = exc.last_attempt.exception() if exc.last_attempt else exc
            err = json.dumps({"error": str(root_exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"
        except Exception as exc:
            err = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/posts/manual", response_model=ManualPostResponse)
async def post_manual_tweet(
    payload: ManualPostRequest, config: ConfigDep
) -> ManualPostResponse:
    """Post a tweet manually from a generated assistant response."""
    text = (payload.text or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Tweet text is required")

    state = await load_state(reset_time_utc=config.rate_limits.reset_time_utc)

    if state.counters.get("posts_today", 0) >= config.rate_limits.max_posts_per_day:
        raise HTTPException(
            status_code=429, detail="Daily post limit reached. Try again tomorrow."
        )

    env_settings = _load_env_settings()
    username = env_settings.get("TWITTER_USERNAME")
    password = env_settings.get("TWITTER_PASSWORD")

    if not username or not password:
        raise HTTPException(
            status_code=400, detail="Twitter credentials are not configured"
        )

    has_provider = any(
        [
            env_settings.get("OPENAI_API_KEY"),
            env_settings.get("OPENROUTER_API_KEY"),
            env_settings.get("GOOGLE_API_KEY"),
            env_settings.get("ANTHROPIC_API_KEY"),
        ]
    )

    if not has_provider:
        raise HTTPException(
            status_code=400, detail="At least one LLM provider API key is required"
        )

    llm_client = LLMClient(config=config, env_settings=env_settings)

    try:
        is_valid, validation_error = await llm_client.validate_tweet(text)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_error)

        try:
            is_aligned = await llm_client.check_brand_alignment(text)
            if not is_aligned:
                raise HTTPException(
                    status_code=400,
                    detail="Tweet not aligned with configured tone/style/topics",
                )
        except HTTPException:
            raise
        except Exception as alignment_exc:
            logger.warning(
                "brand_alignment_check_failed",
                extra={"error": str(alignment_exc)},
            )

        try:
            async with AsyncTwitterSession(config, username, password) as driver:
                posted = post_tweet(driver, text, config)
                if not posted:
                    raise RuntimeError("Tweet posting failed")
        except HTTPException:
            raise
        except Exception as post_exc:
            logger.error(
                "manual_post_failed",
                extra={"error": str(post_exc), "tweet_length": len(text)},
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"Posting failed: {post_exc}")

        state.counters["posts_today"] = state.counters.get("posts_today", 0) + 1
        state.last_post_time = datetime.now(timezone.utc)
        await save_state(state)

        metadata = payload.metadata or {"source": "chat"}
        await log_written_tweet(text=text, tweet_type="manual", metadata=metadata)
        await log_action("Posted manual tweet")

        posted_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        return ManualPostResponse(
            status="ok",
            message="Tweet posted successfully",
            text=text,
            posted_at=posted_at,
        )
    finally:
        await llm_client.close()


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


@router.get("/bot/health")
async def bot_health() -> dict[str, Any]:
    """Proxy bot control server health for the dashboard UI."""
    result = await _get_control("/health")
    is_active = result.get("status") == "ok"
    return {
        "status": "ok" if is_active else "error",
        "active": is_active,
        "control_url": CONTROL_URL or f"http://{CONTROL_HOST}:{CONTROL_PORT}",
        "reason": None if is_active else result.get("reason", "health_check_failed"),
    }


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


async def _get_scheduler_jobs() -> dict[str, Any]:
    """Get scheduler jobs snapshot via control server with local fallback."""
    result = await _get_control("/jobs")
    if result.get("status") == "ok":
        return result

    scheduler = get_scheduler()
    if scheduler is None:
        return result

    try:
        jobs = await asyncio.to_thread(scheduler.get_jobs_snapshot)
        return {"status": "ok", "jobs": jobs, "scheduler_running": scheduler.is_running}
    except Exception as exc:
        return {"status": "error", "reason": f"local_jobs_snapshot_failed: {exc}"}


async def _run_scheduler_job_now(job_id: str) -> dict[str, Any]:
    """Trigger a scheduler job immediately via control server with local fallback."""
    result = await _post_control(f"/run/{job_id}")
    if result.get("status") != "error":
        return result

    scheduler = get_scheduler()
    if scheduler is None:
        return result

    try:
        return await asyncio.to_thread(scheduler.run_job_now, job_id)
    except Exception as exc:
        return {"status": "error", "reason": f"local_run_job_failed: {exc}"}


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
