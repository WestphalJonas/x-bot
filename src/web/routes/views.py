"""HTML view routes for the X bot dashboard."""

from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.scheduler.bot_scheduler import get_job_queue
from src.state.database import get_database
from src.state.manager import load_state
from src.web.app import ChromaMemoryDep, ConfigDep, get_chroma_memory, get_config

if TYPE_CHECKING:
    from src.memory.chroma_client import ChromaMemory

router = APIRouter()


def to_iso_string(timestamp: datetime | str | None) -> str:
    """Convert datetime object or string to ISO format string.

    Args:
        timestamp: datetime object, ISO string, or None

    Returns:
        ISO format string or empty string if None
    """
    if timestamp is None:
        return ""
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    if isinstance(timestamp, str):
        # If already a string, try to parse and return ISO format
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.isoformat()
        except (ValueError, AttributeError):
            return timestamp
    return str(timestamp)


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    config: ConfigDep,
    memory: ChromaMemoryDep = None,
) -> HTMLResponse:
    """Main dashboard overview page."""
    templates = request.app.state.templates

    # Get state for overview stats
    state = await load_state()

    # Get SQLite stats
    db_stats = None
    try:
        db = await get_database()
        db_stats = await db.get_stats()
    except Exception:
        pass

    # Merge with ChromaDB stats if available
    if memory is not None:
        try:
            chroma_stats = memory.get_stats()
            if db_stats:
                db_stats.update(chroma_stats)
            else:
                db_stats = chroma_stats
        except Exception:
            pass

    # Get job queue status
    job_queue = get_job_queue()
    job_queue_info = {
        "size": job_queue.size(),
        "pending_jobs": list(job_queue._pending_job_ids),
        "is_empty": job_queue.is_empty(),
    }

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "state": state,
            "config": config,
            "stats": db_stats,
            "job_queue": job_queue_info,
        },
    )


@router.get("/posts", response_class=HTMLResponse)
async def posts_page(request: Request) -> HTMLResponse:
    """Posts listing page with tabs for read/written/rejected."""
    templates = request.app.state.templates

    return templates.TemplateResponse(
        "posts.html",
        {
            "request": request,
            "active_tab": request.query_params.get("tab", "read"),
        },
    )


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request) -> HTMLResponse:
    """Token usage analytics page."""
    templates = request.app.state.templates

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
        },
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, config: ConfigDep) -> HTMLResponse:
    """Settings page for editing bot configuration."""
    templates = request.app.state.templates

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "config": config,
        },
    )


# HTMX partial routes for dynamic content loading


@router.get("/partials/posts/read", response_class=HTMLResponse)
async def posts_read_partial(request: Request) -> HTMLResponse:
    """Partial for read posts list (HTMX)."""
    templates = request.app.state.templates

    posts = []
    total = 0

    try:
        db = await get_database()
        posts_data = await db.get_read_posts(limit=50)
        total = await db.get_read_posts_count()

        for post in posts_data:
            posts.append(
                {
                    "id": post.get("post_id", ""),
                    "text": post.get("text", ""),
                    "metadata": {
                        "username": post.get("username"),
                        "display_name": post.get("display_name"),
                        "post_type": post.get("post_type"),
                        "url": post.get("url"),
                        "is_interesting": post.get("is_interesting"),
                        "timestamp": to_iso_string(post.get("read_at")),
                        "post_timestamp": to_iso_string(post.get("post_timestamp")),
                    },
                }
            )
    except Exception:
        pass

    return templates.TemplateResponse(
        "partials/posts_list.html",
        {
            "request": request,
            "posts": posts,
            "total": total,
            "post_type": "read",
        },
    )


@router.get("/partials/posts/written", response_class=HTMLResponse)
async def posts_written_partial(request: Request) -> HTMLResponse:
    """Partial for written tweets list (HTMX)."""
    templates = request.app.state.templates

    tweets = []
    total = 0

    try:
        db = await get_database()
        tweets_data = await db.get_written_tweets(limit=50)
        total = await db.get_written_tweets_count()

        for tweet in tweets_data:
            tweets.append(
                {
                    "text": tweet.get("text", ""),
                    "timestamp": to_iso_string(tweet.get("created_at")),
                    "tweet_type": tweet.get("tweet_type", "autonomous"),
                }
            )
    except Exception:
        pass

    return templates.TemplateResponse(
        "partials/tweets_list.html",
        {
            "request": request,
            "tweets": tweets,
            "total": total,
            "tweet_type": "written",
        },
    )


@router.get("/partials/posts/rejected", response_class=HTMLResponse)
async def posts_rejected_partial(request: Request) -> HTMLResponse:
    """Partial for rejected tweets list (HTMX)."""
    templates = request.app.state.templates

    tweets = []
    total = 0

    try:
        db = await get_database()
        tweets_data = await db.get_rejected_tweets(limit=50)
        total = await db.get_rejected_tweets_count()

        for tweet in tweets_data:
            tweets.append(
                {
                    "text": tweet.get("text", ""),
                    "reason": tweet.get("reason", "Unknown"),
                    "timestamp": to_iso_string(tweet.get("rejected_at")),
                    "operation": tweet.get("operation", "unknown"),
                }
            )
    except Exception:
        pass

    return templates.TemplateResponse(
        "partials/rejected_list.html",
        {
            "request": request,
            "tweets": tweets,
            "total": total,
        },
    )


@router.get("/partials/posts/interested", response_class=HTMLResponse)
async def posts_interested_partial(request: Request) -> HTMLResponse:
    """Partial for interested posts list (HTMX)."""
    templates = request.app.state.templates

    posts = []
    total = 0

    try:
        state = await load_state()
        queue = state.interesting_posts_queue

        total = len(queue)
        for post_data in queue:
            posts.append(
                {
                    "id": post_data.get("post_id", ""),
                    "text": post_data.get("text", ""),
                    "metadata": {
                        "username": post_data.get("username"),
                        "display_name": post_data.get("display_name"),
                        "post_type": post_data.get("post_type"),
                        "url": post_data.get("url"),
                        "likes": post_data.get("likes", 0),
                        "retweets": post_data.get("retweets", 0),
                        "replies": post_data.get("replies", 0),
                        "timestamp": to_iso_string(post_data.get("timestamp")),
                    },
                }
            )
    except Exception:
        pass

    return templates.TemplateResponse(
        "partials/interested_list.html",
        {
            "request": request,
            "posts": posts,
            "total": total,
        },
    )


@router.get("/partials/analytics/tokens", response_class=HTMLResponse)
async def analytics_tokens_partial(request: Request) -> HTMLResponse:
    """Partial for token usage analytics (HTMX)."""
    templates = request.app.state.templates

    entries = []
    total_tokens = 0
    tokens_by_provider: dict[str, int] = {}
    tokens_by_operation: dict[str, int] = {}
    total_entries = 0

    try:
        db = await get_database()
        entries_data = await db.get_token_usage(limit=100)
        stats = await db.get_token_usage_stats()

        total_tokens = stats.get("total_tokens", 0)
        total_entries = stats.get("total_entries", 0)
        tokens_by_provider = stats.get("tokens_by_provider", {})
        tokens_by_operation = stats.get("tokens_by_operation", {})

        for entry in entries_data:
            entries.append(
                {
                    "timestamp": to_iso_string(entry.get("timestamp")),
                    "provider": entry.get("provider", "unknown"),
                    "model": entry.get("model", "unknown"),
                    "prompt_tokens": entry.get("prompt_tokens", 0),
                    "completion_tokens": entry.get("completion_tokens", 0),
                    "total_tokens": entry.get("total_tokens", 0),
                    "operation": entry.get("operation", "unknown"),
                }
            )
    except Exception:
        pass

    return templates.TemplateResponse(
        "partials/token_analytics.html",
        {
            "request": request,
            "entries": entries,
            "total_entries": total_entries,
            "total_tokens": total_tokens,
            "tokens_by_provider": tokens_by_provider,
            "tokens_by_operation": tokens_by_operation,
        },
    )
