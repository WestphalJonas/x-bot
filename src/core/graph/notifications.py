"""LangGraph pipeline for checking notifications and queueing replies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.constants import QueueLimits
from src.core.config import BotConfig, EnvSettings
from src.state.manager import load_state, save_state
from src.state.models import Notification
from src.web.data_tracker import log_action
from src.x.notifications import check_notifications as check_notifications_func
from src.x.session import AsyncTwitterSession

logger = logging.getLogger(__name__)


class NotificationsState(BaseModel):
    """State for notifications graph."""

    notifications: list[Notification] = Field(default_factory=list)
    new_notifications: list[Notification] = Field(default_factory=list)


@dataclass
class NotificationDependencies:
    config: BotConfig
    env_settings: EnvSettings


async def _fetch_notifications(
    state: NotificationsState, deps: NotificationDependencies
) -> dict:
    username = deps.env_settings.get("TWITTER_USERNAME")
    password = deps.env_settings.get("TWITTER_PASSWORD")
    if not username or not password:
        raise ValueError("TWITTER_USERNAME and TWITTER_PASSWORD are required")

    async with AsyncTwitterSession(deps.config, username, password) as driver:
        notifications = check_notifications_func(driver, deps.config, count=20)

    return {"notifications": notifications}


async def _filter_and_queue(
    state: NotificationsState, deps: NotificationDependencies
) -> dict:
    bot_state = await load_state()
    processed_ids = set(bot_state.processed_notification_ids)

    fresh = [n for n in state.notifications if n.notification_id not in processed_ids]

    if fresh:
        bot_state.notifications_queue.extend([n.model_dump() for n in fresh])
        if len(bot_state.notifications_queue) > QueueLimits.NOTIFICATIONS:
            bot_state.notifications_queue = bot_state.notifications_queue[
                -QueueLimits.NOTIFICATIONS :
            ]

        new_ids = [n.notification_id for n in fresh if n.notification_id]
        bot_state.processed_notification_ids.extend(new_ids)
        if (
            len(bot_state.processed_notification_ids)
            > QueueLimits.PROCESSED_NOTIFICATION_IDS
        ):
            bot_state.processed_notification_ids = bot_state.processed_notification_ids[
                -QueueLimits.PROCESSED_NOTIFICATION_IDS :
            ]

        await save_state(bot_state)

    await log_action(f"Checked notifications ({len(fresh)} new)")
    return {"new_notifications": fresh}


def build_notifications_graph(
    deps: NotificationDependencies,
) -> Callable[[NotificationsState], Any]:
    graph = StateGraph(NotificationsState)

    async def fetch_node(state: NotificationsState) -> dict:
        return await _fetch_notifications(state, deps)

    async def queue_node(state: NotificationsState) -> dict:
        return await _filter_and_queue(state, deps)

    graph.add_node("fetch", fetch_node)
    graph.add_node("queue", queue_node)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "queue")
    graph.add_edge("queue", END)

    return graph.compile()
