"""Inspiration queue processing job."""

import asyncio
import logging
from datetime import datetime, timezone

from src.core.config import BotConfig, EnvSettings
from src.core.evaluation import re_evaluate_tweet
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.state.database import get_database
from src.state.models import Post
from src.web.data_tracker import log_action, log_rejected_tweet, log_written_tweet
from src.x.posting import post_tweet
from src.x.session import AsyncTwitterSession

from src.memory.chroma_client import ChromaMemory

logger = logging.getLogger(__name__)


def process_inspiration_queue(config: BotConfig, env_settings: EnvSettings) -> None:
    """Process inspiration queue (scheduled job).

    This function wraps async operations for APScheduler compatibility.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables (API keys, credentials)
    """
    job_id = "process_inspiration_queue"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        asyncio.run(_process_inspiration_queue_async(config, env_settings))
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


async def _process_inspiration_queue_async(
    config: BotConfig, env_settings: EnvSettings
) -> None:
    """Async implementation of inspiration queue processing.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    # Refresh state so we respect the daily limit before spending tokens
    state = await load_state(
        reset_time_utc=config.rate_limits.reset_time_utc,
    )

    if state.counters["posts_today"] >= config.rate_limits.max_posts_per_day:
        logger.warning(
            "rate_limit_exceeded",
            extra={
                "posts_today": state.counters["posts_today"],
                "max_posts": config.rate_limits.max_posts_per_day,
            },
        )
        return

    threshold = config.scheduler.inspiration_batch_size

    queue_size = len(state.interesting_posts_queue)

    if queue_size < threshold:
        logger.info(
            "inspiration_queue_below_threshold",
            extra={
                "current_size": queue_size,
                "threshold": threshold,
            },
        )
        return

    logger.info(
        "processing_inspiration_queue",
        extra={"queue_size": queue_size},
    )

    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    has_llm_provider = any(
        [
            env_settings.get("OPENAI_API_KEY"),
            env_settings.get("OPENROUTER_API_KEY"),
            env_settings.get("GOOGLE_API_KEY"),
            env_settings.get("ANTHROPIC_API_KEY"),
        ]
    )
    if not has_llm_provider:
        logger.warning(
            "no_llm_provider_configured",
            extra={"detail": "Inspiration queue processing will be skipped"},
        )
        return

    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")

    llm_client = LLMClient(config=config, env_settings=env_settings)
    recent_tweets: list[str] | None = None

    # Pull recent written tweets to guide inspiration generations
    try:
        if config.llm.recent_tweet_context_limit > 0:
            db = await get_database()
            tweets_data = await db.get_written_tweets(
                limit=config.llm.recent_tweet_context_limit
            )
            recent_tweets = [
                tweet.get("text", "") for tweet in tweets_data if tweet.get("text")
            ]
    except Exception as exc:
        logger.warning(
            "recent_tweets_fetch_failed",
            extra={"error": str(exc)},
        )

    posts_batch_dicts = state.interesting_posts_queue[:threshold]
    posts_batch = [Post(**p) for p in posts_batch_dicts]

    # Duplicate checks only run when embeddings are available
    chroma_memory = None
    try:
        chroma_memory = ChromaMemory(config=config, llm_client=llm_client._client)
    except Exception as e:
        logger.warning(
            "chroma_init_failed",
            extra={"error": str(e)},
        )

    max_attempts = 3
    tweet_text = None

    try:
        for attempt in range(1, max_attempts + 1):
            tweet_text = await llm_client.generate_inspiration_tweet(
                posts_batch, recent_tweets=recent_tweets
            )
            logger.info(
                "inspiration_tweet_generated",
                extra={
                    "tweet": tweet_text,
                    "attempt": attempt,
                    "length": len(tweet_text),
                },
            )

            # Check for duplicate using ChromaDB
            if chroma_memory:
                try:
                    is_duplicate, similarity = await chroma_memory.check_duplicate(
                        tweet_text
                    )
                    if is_duplicate:
                        logger.warning(
                            "inspiration_duplicate_detected",
                            extra={
                                "similarity": similarity,
                                "attempt": attempt,
                            },
                        )
                        continue  # Try again with new generation
                except Exception as e:
                    logger.warning(
                        "chroma_duplicate_check_failed",
                        extra={"error": str(e)},
                    )

            # Local validation before the gatekeeper LLM
            is_valid, error_message = await llm_client.validate_tweet(tweet_text)
            if is_valid:
                logger.info(
                    "inspiration_tweet_validated", extra={"length": len(tweet_text)}
                )
                break

            logger.warning(
                "inspiration_tweet_validation_retry",
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error": error_message,
                    "length": len(tweet_text),
                },
            )

            if attempt == max_attempts:
                logger.error(
                    "inspiration_tweet_validation_failed",
                    extra={"error": error_message, "attempts": max_attempts},
                )
                # Log rejected tweet for dashboard
                try:
                    await log_rejected_tweet(
                        text=tweet_text,
                        reason=error_message,
                        operation="inspiration",
                    )
                    logger.info("rejected_tweet_saved", extra={"reason": error_message})
                except Exception as save_error:
                    logger.error(
                        "failed_to_save_rejected_tweet",
                        extra={"error": str(save_error)},
                        exc_info=True,
                    )
                raise ValueError(
                    f"Inspiration tweet validation failed after {max_attempts} attempts: {error_message}"
                )

        approved, evaluation_reason = await re_evaluate_tweet(
            tweet_text=tweet_text,
            config=config,
            llm_client=llm_client,
            operation="inspiration",
        )
        if not approved:
            logger.error(
                "inspiration_re_evaluation_failed",
                extra={"reason": evaluation_reason},
            )
            try:
                await log_rejected_tweet(
                    text=tweet_text,
                    reason=evaluation_reason,
                    operation="inspiration",
                )
            except Exception as save_error:
                logger.error(
                    "failed_to_save_rejected_tweet",
                    extra={"error": str(save_error)},
                    exc_info=True,
                )
            raise ValueError(
                f"Inspiration tweet re-evaluation failed: {evaluation_reason}"
            )

        async with AsyncTwitterSession(
            config, twitter_username, twitter_password
        ) as driver:
            success = post_tweet(driver, tweet_text, config)

            if success:
                logger.info("inspiration_tweet_posted")

                await log_written_tweet(
                    text=tweet_text,
                    tweet_type="inspiration",
                )

                if chroma_memory:
                    try:
                        await chroma_memory.store_tweet(
                            text=tweet_text,
                            metadata={"type": "inspiration"},
                        )
                        logger.info("inspiration_tweet_stored_in_memory")
                    except Exception as e:
                        logger.warning(
                            "failed_to_store_inspiration_tweet_in_memory",
                            extra={"error": str(e)},
                        )

                # Reload state to avoid clobbering concurrent updates, then drop processed posts
                current_state = await load_state()

                if len(current_state.interesting_posts_queue) >= threshold:
                    current_state.interesting_posts_queue = (
                        current_state.interesting_posts_queue[threshold:]
                    )

                current_state.counters["posts_today"] += 1
                current_state.last_post_time = datetime.now(timezone.utc)

                await save_state(current_state)

                await log_action("Posted inspiration tweet")
            else:
                logger.error("failed_to_post_inspiration_tweet")

    except Exception as e:
        logger.error(
            "inspiration_processing_error",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise

    finally:
        await llm_client.close()
