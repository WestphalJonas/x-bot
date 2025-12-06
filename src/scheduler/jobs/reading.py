"""Frontpage post reading job leveraging LangGraph."""

import asyncio
import logging

from src.core.config import BotConfig, EnvSettings
from src.core.graph.reading import (
    ReadingDependencies,
    ReadingState,
    build_reading_graph,
)
from src.core.llm import LLMClient

logger = logging.getLogger(__name__)


def read_frontpage_posts(config: BotConfig, env_settings: EnvSettings) -> None:
    """Read frontpage posts (scheduled job)."""
    job_id = "read_posts"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        asyncio.run(_read_frontpage_posts_async(config, env_settings))
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


async def _read_frontpage_posts_async(
    config: BotConfig, env_settings: EnvSettings
) -> None:
    has_llm_provider = any(
        [
            env_settings.get("OPENAI_API_KEY"),
            env_settings.get("OPENROUTER_API_KEY"),
            env_settings.get("GOOGLE_API_KEY"),
            env_settings.get("ANTHROPIC_API_KEY"),
        ]
    )
    llm_client = LLMClient(config=config, env_settings=env_settings) if has_llm_provider else None

    graph = build_reading_graph(
        ReadingDependencies(
            config=config,
            env_settings=env_settings,
            llm_client=llm_client,
        )
    )

    await graph.ainvoke(ReadingState())
    if llm_client:
        await llm_client.close()

