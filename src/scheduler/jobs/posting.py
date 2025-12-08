"""Autonomous tweet posting job."""

import asyncio
import logging
from datetime import datetime, timezone

from src.core.config import BotConfig, EnvSettings
from src.core.graph.tweet_generation import (
    GenerationDependencies,
    TweetGenerationState,
    build_tweet_generation_graph,
)
from src.core.llm import LLMClient
from src.memory.chroma_client import ChromaMemory
from src.state.manager import load_state, save_state
from src.state.database import get_database
from src.web.data_tracker import log_action, log_rejected_tweet, log_written_tweet
from src.x.posting import post_tweet
from src.x.session import AsyncTwitterSession

logger = logging.getLogger(__name__)


def post_autonomous_tweet(config: BotConfig, env_settings: EnvSettings) -> None:
    """Post an autonomous tweet (scheduled job)."""
    job_id = "post_tweet"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        asyncio.run(_post_autonomous_tweet_async(config, env_settings))
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


async def _post_autonomous_tweet_async(
    config: BotConfig, env_settings: EnvSettings
) -> None:
    """Async implementation of autonomous tweet posting."""
    state = await load_state(reset_time_utc=config.rate_limits.reset_time_utc)

    if state.counters["posts_today"] >= config.rate_limits.max_posts_per_day:
        logger.warning(
            "rate_limit_exceeded",
            extra={
                "posts_today": state.counters["posts_today"],
                "max_posts": config.rate_limits.max_posts_per_day,
            },
        )
        return

    # Validate credentials
    if not env_settings.get("TWITTER_USERNAME") or not env_settings.get("TWITTER_PASSWORD"):
        raise ValueError("TWITTER_USERNAME and TWITTER_PASSWORD are required")

    # Ensure at least one LLM provider
    has_provider = any(
        [
            env_settings.get("OPENAI_API_KEY"),
            env_settings.get("OPENROUTER_API_KEY"),
            env_settings.get("GOOGLE_API_KEY"),
            env_settings.get("ANTHROPIC_API_KEY"),
        ]
    )
    if not has_provider:
        raise ValueError("At least one LLM provider API key is required.")

    llm_client = LLMClient(config=config, env_settings=env_settings)
    recent_tweets: list[str] | None = None

    # Fetch recent written tweets to use as style/context references
    try:
        if config.llm.recent_tweet_context_limit > 0:
            db = await get_database()
            tweets_data = await db.get_written_tweets(
                limit=config.llm.recent_tweet_context_limit
            )
            recent_tweets = [
                tweet.get("text", "")
                for tweet in tweets_data
                if tweet.get("text")
            ]
    except Exception as exc:
        logger.warning(
            "recent_tweets_fetch_failed",
            extra={"error": str(exc)},
        )

    chroma_memory: ChromaMemory | None = None
    try:
        chroma_memory = ChromaMemory(config=config, llm_client=llm_client._client)
    except Exception as exc:
        logger.warning("chroma_unavailable", extra={"error": str(exc)})

    system_prompt = config.get_system_prompt()
    graph = build_tweet_generation_graph(
        GenerationDependencies(
            llm_client=llm_client, memory=chroma_memory, recent_tweets=recent_tweets
        )
    )

    result_state = await graph.ainvoke(TweetGenerationState(system_prompt=system_prompt))

    def _field(name: str):
        if hasattr(result_state, name):
            return getattr(result_state, name)
        if isinstance(result_state, dict):
            return result_state.get(name)
        return None

    tweet_text = _field("tweet_text")
    approved = _field("approved")
    reason = _field("reason")

    if not tweet_text:
        raise ValueError("Generation graph returned no tweet text")

    is_valid, error_message = await llm_client.validate_tweet(tweet_text)
    if not is_valid:
        await log_rejected_tweet(text=tweet_text, reason=error_message, operation="autonomous")
        raise ValueError(f"Tweet validation failed: {error_message}")

    if approved is False:
        await log_rejected_tweet(text=tweet_text, reason=reason or "Not approved", operation="autonomous")
        raise ValueError(f"Tweet re-evaluation failed: {reason}")

    username = env_settings.get("TWITTER_USERNAME")
    password = env_settings.get("TWITTER_PASSWORD")

    async with AsyncTwitterSession(config, username, password) as driver:
        logger.info("posting_tweet")
        post_success = post_tweet(driver, tweet_text, config)
        if not post_success:
            raise RuntimeError("Tweet posting failed")

        await log_written_tweet(text=tweet_text, tweet_type="autonomous")

        if chroma_memory:
            try:
                await chroma_memory.store_tweet(
                    text=tweet_text,
                    metadata={"type": "autonomous"},
                )
            except Exception as e:
                logger.warning("failed_to_store_tweet_in_memory", extra={"error": str(e)})

        state.counters["posts_today"] += 1
        state.last_post_time = datetime.now(timezone.utc)
        await save_state(state)
        await log_action("Posted autonomous tweet")

    await llm_client.close()
