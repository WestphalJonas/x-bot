"""LangGraph pipeline for reading timeline posts and classifying interest."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.constants import QueueLimits
from src.core.config import BotConfig, EnvSettings
from src.core.interest import check_interest
from src.core.llm import LLMClient
from src.state.database import get_database
from src.state.manager import load_state, save_state
from src.state.models import Post
from src.web.data_tracker import log_action
from src.x.reading import read_frontpage_posts
from src.x.session import AsyncTwitterSession

logger = logging.getLogger(__name__)


class ReadingState(BaseModel):
    """State for reading/interest graph."""

    posts: list[Post] = Field(default_factory=list)
    interesting_posts: list[Post] = Field(default_factory=list)


@dataclass
class ReadingDependencies:
    config: BotConfig
    env_settings: EnvSettings
    llm_client: LLMClient | None = None


async def _fetch_posts_node(state: ReadingState, deps: ReadingDependencies) -> dict:
    username = deps.env_settings.get("TWITTER_USERNAME")
    password = deps.env_settings.get("TWITTER_PASSWORD")
    if not username or not password:
        raise ValueError("TWITTER_USERNAME and TWITTER_PASSWORD are required")

    async with AsyncTwitterSession(deps.config, username, password) as driver:
        posts = read_frontpage_posts(driver, deps.config, count=10)

    logger.info("graph_read_posts", extra={"count": len(posts)})
    return {"posts": posts}


async def _evaluate_posts_node(state: ReadingState, deps: ReadingDependencies) -> dict:
    db = await get_database()
    llm_client = deps.llm_client

    interesting: list[Post] = []
    queued_post_ids: set[str] = set()

    bot_state = await load_state()
    queued_post_ids.update(
        {
            p.get("post_id")
            for p in bot_state.interesting_posts_queue
            if p.get("post_id")
        }
    )

    for post in state.posts:
        if post.post_id and await db.has_seen_post(post.post_id):
            continue
        if post.post_id and post.post_id in queued_post_ids:
            continue

        if llm_client:
            try:
                is_interesting = await check_interest(post, deps.config, llm_client)
                post.is_interesting = is_interesting
            except Exception as exc:
                logger.warning(
                    "interest_check_failed",
                    extra={"post_id": post.post_id, "error": str(exc)},
                )
                post.is_interesting = False
        else:
            post.is_interesting = None

        try:
            await db.store_read_post(post)
        except Exception as exc:
            logger.warning(
                "store_read_post_failed",
                extra={"post_id": post.post_id, "error": str(exc)},
            )

        if post.is_interesting:
            interesting.append(post)

    if interesting:
        bot_state.interesting_posts_queue.extend([p.model_dump() for p in interesting])
        if len(bot_state.interesting_posts_queue) > QueueLimits.INTERESTING_POSTS:
            bot_state.interesting_posts_queue = bot_state.interesting_posts_queue[
                -QueueLimits.INTERESTING_POSTS :
            ]
        await save_state(bot_state)

    await log_action(f"Read {len(state.posts)} posts from timeline")

    return {"interesting_posts": interesting}


def build_reading_graph(
    deps: ReadingDependencies,
) -> Callable[[ReadingState], Any]:
    """Compile reading/interest graph."""
    graph = StateGraph(ReadingState)

    async def fetch_posts_node(state: ReadingState) -> dict:
        return await _fetch_posts_node(state, deps)

    async def evaluate_posts_node(state: ReadingState) -> dict:
        return await _evaluate_posts_node(state, deps)

    graph.add_node("fetch_posts", fetch_posts_node)
    graph.add_node("evaluate_posts", evaluate_posts_node)

    graph.set_entry_point("fetch_posts")
    graph.add_edge("fetch_posts", "evaluate_posts")
    graph.add_edge("evaluate_posts", END)

    return graph.compile()
