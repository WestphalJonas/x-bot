"""Reply processing job leveraging LangGraph."""

from __future__ import annotations

import asyncio
import logging

from src.core.config import BotConfig, EnvSettings
from src.core.graph.replies import ReplyDependencies, ReplyState, build_replies_graph

logger = logging.getLogger(__name__)


def process_replies(config: BotConfig, env_settings: EnvSettings) -> None:
    """Process queued notifications and send replies."""
    job_id = "process_replies"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        asyncio.run(_process_replies_async(config, env_settings))
        logger.info("job_completed", extra={"job_id": job_id})
    except Exception as exc:
        logger.error(
            "job_failed",
            extra={
                "job_id": job_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            exc_info=True,
        )


async def _process_replies_async(config: BotConfig, env_settings: EnvSettings) -> None:
    graph = build_replies_graph(ReplyDependencies(config=config, env_settings=env_settings))
    await graph.ainvoke(ReplyState())

