"""LangGraph pipeline for processing notification replies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.core.config import BotConfig, EnvSettings
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.state.models import Notification
from src.web.data_tracker import log_action
from src.x.replies import post_reply
from src.x.session import AsyncTwitterSession

logger = logging.getLogger(__name__)


class ReplyState(BaseModel):
    """State for reply processing graph."""

    notification: Notification | None = Field(default=None)
    intent_positive: bool = Field(default=False)
    intent_reason: str | None = Field(default=None)
    reply_text: str | None = Field(default=None)


@dataclass
class ReplyDependencies:
    config: BotConfig
    env_settings: EnvSettings


async def _pop_next_notification(state: ReplyState, deps: ReplyDependencies) -> dict:
    bot_state = await load_state()
    if not bot_state.notifications_queue:
        await log_action("No notifications to process")
        return {"notification": None}

    notification_data = bot_state.notifications_queue.pop(0)
    await save_state(bot_state)
    notification = Notification(**notification_data)
    logger.info(
        "reply_notification_popped",
        extra={"notification_id": notification.notification_id, "from": notification.from_username},
    )
    return {"notification": notification}


async def _check_rate_limit(state: ReplyState, deps: ReplyDependencies) -> dict:
    if state.notification is None:
        return {}

    bot_state = await load_state()
    replies_today = bot_state.counters.get("replies_today", 0)
    max_replies = deps.config.rate_limits.max_replies_per_day

    if replies_today >= max_replies:
        # Requeue the notification so it can be processed later
        bot_state.notifications_queue.insert(0, state.notification.model_dump())
        await save_state(bot_state)
        await log_action("Reply skipped: rate limit reached")
        logger.warning(
            "reply_rate_limit_reached",
            extra={
                "replies_today": replies_today,
                "max_replies": max_replies,
                "notification_id": state.notification.notification_id,
            },
        )
        return {"notification": None}

    return {}


async def _classify_and_generate(state: ReplyState, deps: ReplyDependencies) -> dict:
    if state.notification is None:
        return {}

    llm = LLMClient(deps.config, deps.env_settings)
    try:
        positive, reason = await llm.classify_notification_intent(state.notification)
        if not positive:
            await log_action(f"Notification skipped (intent): {reason}")
            return {"intent_positive": False, "intent_reason": reason}

        reply_text = await llm.generate_reply(state.notification)
        return {
            "intent_positive": True,
            "intent_reason": reason,
            "reply_text": reply_text,
        }
    finally:
        try:
            await llm.close()
        except Exception:
            pass


async def _post_reply(state: ReplyState, deps: ReplyDependencies) -> dict:
    if state.notification is None or not state.intent_positive or not state.reply_text:
        return {}

    username = deps.env_settings.get("TWITTER_USERNAME")
    password = deps.env_settings.get("TWITTER_PASSWORD")
    if not username or not password:
        raise ValueError("TWITTER_USERNAME and TWITTER_PASSWORD are required for replies")

    async with AsyncTwitterSession(deps.config, username, password) as driver:
        success = post_reply(
            driver=driver,
            reply_text=state.reply_text,
            notification_url=state.notification.url or "https://x.com/notifications",
            config=deps.config,
        )

    if success:
        bot_state = await load_state()
        bot_state.counters["replies_today"] = bot_state.counters.get("replies_today", 0) + 1
        bot_state.last_reply_time = datetime.now(timezone.utc)
        bot_state.last_reply_status = "ok"
        bot_state.last_action = f"Replied to {state.notification.from_username}"
        bot_state.last_action_time = bot_state.last_reply_time
        await save_state(bot_state)
        await log_action(
            f"Replied to {state.notification.from_username}: {state.reply_text[:80]}"
        )

    return {}


def build_replies_graph(deps: ReplyDependencies) -> Callable[[ReplyState], Any]:
    """Compile the replies processing graph."""
    graph = StateGraph(ReplyState)

    async def fetch_node(state: ReplyState) -> dict:
        return await _pop_next_notification(state, deps)

    async def rate_limit_node(state: ReplyState) -> dict:
        return await _check_rate_limit(state, deps)

    async def llm_node(state: ReplyState) -> dict:
        return await _classify_and_generate(state, deps)

    async def reply_node(state: ReplyState) -> dict:
        return await _post_reply(state, deps)

    graph.add_node("fetch", fetch_node)
    graph.add_node("rate_limit", rate_limit_node)
    graph.add_node("llm", llm_node)
    graph.add_node("reply", reply_node)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "rate_limit")
    graph.add_edge("rate_limit", "llm")
    graph.add_edge("llm", "reply")
    graph.add_edge("reply", END)

    return graph.compile()

