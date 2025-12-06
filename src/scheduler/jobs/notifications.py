"""Notification checking job leveraging LangGraph."""

import asyncio
import logging

from src.core.config import BotConfig, EnvSettings
from src.core.graph.notifications import (
    NotificationDependencies,
    NotificationsState,
    build_notifications_graph,
)

logger = logging.getLogger(__name__)


def check_notifications(config: BotConfig, env_settings: EnvSettings) -> None:
    """Check notifications (scheduled job)."""
    job_id = "check_notifications"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        asyncio.run(_check_notifications_async(config, env_settings))
        logger.info("job_completed", extra={"job_id": job_id})
    except Exception as e:
        logger.error(
            "job_failed",
            extra={
                "job_id": job_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )


async def _check_notifications_async(
    config: BotConfig, env_settings: EnvSettings
) -> None:
    graph = build_notifications_graph(
        NotificationDependencies(config=config, env_settings=env_settings)
    )
    await graph.ainvoke(NotificationsState())

