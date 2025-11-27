"""HTML view routes for the X bot dashboard."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.state.manager import load_state
from src.web.app import get_chroma_memory, get_config

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard overview page."""
    templates = request.app.state.templates

    # Get state for overview stats
    state = await load_state()
    config = get_config()
    memory = get_chroma_memory()

    memory_stats = None
    if memory is not None:
        try:
            memory_stats = memory.get_stats()
        except Exception:
            pass

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "state": state,
            "config": config,
            "memory_stats": memory_stats,
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


# HTMX partial routes for dynamic content loading


@router.get("/partials/posts/read", response_class=HTMLResponse)
async def posts_read_partial(request: Request) -> HTMLResponse:
    """Partial for read posts list (HTMX)."""
    templates = request.app.state.templates
    memory = get_chroma_memory()

    posts = []
    total = 0

    if memory is not None:
        try:
            collection = memory.posts_collection
            count = collection.count()
            total = count

            if count > 0:
                results = collection.get(
                    limit=min(50, count),
                    include=["documents", "metadatas"],
                )

                for i, doc_id in enumerate(results["ids"]):
                    posts.append(
                        {
                            "id": doc_id,
                            "text": results["documents"][i]
                            if results["documents"]
                            else "",
                            "metadata": results["metadatas"][i]
                            if results["metadatas"]
                            else {},
                        }
                    )

                # Sort by timestamp descending
                posts.sort(
                    key=lambda p: p["metadata"].get("timestamp", ""),
                    reverse=True,
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

    state = await load_state()
    memory = get_chroma_memory()

    tweets = []

    # Get from state first
    for tweet_data in state.written_tweets:
        tweets.append(
            {
                "text": tweet_data.get("text", ""),
                "timestamp": tweet_data.get("timestamp"),
                "tweet_type": tweet_data.get("tweet_type", "autonomous"),
            }
        )

    # Also get from ChromaDB
    if memory is not None:
        try:
            collection = memory.tweets_collection
            count = collection.count()

            if count > 0:
                results = collection.get(
                    limit=min(50, count),
                    include=["documents", "metadatas"],
                )

                seen_texts = {t["text"] for t in tweets}

                for i, doc_id in enumerate(results["ids"]):
                    text = results["documents"][i] if results["documents"] else ""
                    if text not in seen_texts:
                        metadata = (
                            results["metadatas"][i] if results["metadatas"] else {}
                        )
                        tweets.append(
                            {
                                "text": text,
                                "timestamp": metadata.get("timestamp"),
                                "tweet_type": metadata.get("tweet_type", "unknown"),
                            }
                        )
        except Exception:
            pass

    # Sort by timestamp descending
    tweets.sort(
        key=lambda t: str(t.get("timestamp") or ""),
        reverse=True,
    )

    return templates.TemplateResponse(
        "partials/tweets_list.html",
        {
            "request": request,
            "tweets": tweets[:50],
            "total": len(tweets),
            "tweet_type": "written",
        },
    )


@router.get("/partials/posts/rejected", response_class=HTMLResponse)
async def posts_rejected_partial(request: Request) -> HTMLResponse:
    """Partial for rejected tweets list (HTMX)."""
    templates = request.app.state.templates

    state = await load_state()

    tweets = []
    for tweet_data in state.rejected_tweets:
        tweets.append(
            {
                "text": tweet_data.get("text", ""),
                "reason": tweet_data.get("reason", "Unknown"),
                "timestamp": tweet_data.get("timestamp"),
                "operation": tweet_data.get("operation", "unknown"),
            }
        )

    # Sort by timestamp descending
    tweets.sort(
        key=lambda t: str(t.get("timestamp") or ""),
        reverse=True,
    )

    return templates.TemplateResponse(
        "partials/rejected_list.html",
        {
            "request": request,
            "tweets": tweets[:50],
            "total": len(tweets),
        },
    )


@router.get("/partials/analytics/tokens", response_class=HTMLResponse)
async def analytics_tokens_partial(request: Request) -> HTMLResponse:
    """Partial for token usage analytics (HTMX)."""
    templates = request.app.state.templates

    state = await load_state()

    entries = []
    total_tokens = 0
    tokens_by_provider: dict[str, int] = {}
    tokens_by_operation: dict[str, int] = {}

    for entry_data in state.token_usage_log:
        provider = entry_data.get("provider", "unknown")
        operation = entry_data.get("operation", "unknown")
        entry_total = entry_data.get("total_tokens", 0)

        entries.append(
            {
                "timestamp": entry_data.get("timestamp"),
                "provider": provider,
                "model": entry_data.get("model", "unknown"),
                "prompt_tokens": entry_data.get("prompt_tokens", 0),
                "completion_tokens": entry_data.get("completion_tokens", 0),
                "total_tokens": entry_total,
                "operation": operation,
            }
        )

        total_tokens += entry_total
        tokens_by_provider[provider] = tokens_by_provider.get(provider, 0) + entry_total
        tokens_by_operation[operation] = (
            tokens_by_operation.get(operation, 0) + entry_total
        )

    # Sort by timestamp descending
    entries.sort(
        key=lambda e: str(e.get("timestamp") or ""),
        reverse=True,
    )

    return templates.TemplateResponse(
        "partials/token_analytics.html",
        {
            "request": request,
            "entries": entries[:100],
            "total_entries": len(entries),
            "total_tokens": total_tokens,
            "tokens_by_provider": tokens_by_provider,
            "tokens_by_operation": tokens_by_operation,
        },
    )
