"""Inspiration queue processing job."""

import asyncio
import logging
from datetime import datetime, timezone

from src.core.config import BotConfig, EnvSettings
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
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
        # Run async operations in event loop
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
    # Load state (with automatic counter reset check)
    state = await load_state(
        reset_time_utc=config.rate_limits.reset_time_utc,
    )

    # Check rate limits before processing
    if state.counters["posts_today"] >= config.rate_limits.max_posts_per_day:
        logger.warning(
            "rate_limit_exceeded",
            extra={
                "posts_today": state.counters["posts_today"],
                "max_posts": config.rate_limits.max_posts_per_day,
            },
        )
        return

    # Check if we have enough posts in the queue
    # Default threshold is 10 if not specified in config
    threshold = config.scheduler.inspiration_batch_size

    # Queue is stored as list of dicts in state.interesting_posts_queue
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

    # Get environment variables
    openai_api_key = env_settings.get("OPENAI_API_KEY")
    openrouter_api_key = env_settings.get("OPENROUTER_API_KEY")
    google_api_key = env_settings.get("GOOGLE_API_KEY")
    anthropic_api_key = env_settings.get("ANTHROPIC_API_KEY")
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    # Validate that at least one LLM provider is configured
    has_llm_provider = any(
        [openai_api_key, openrouter_api_key, google_api_key, anthropic_api_key]
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

    # Initialize LLM client
    llm_client = LLMClient(
        config=config,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        google_api_key=google_api_key,
        anthropic_api_key=anthropic_api_key,
    )

    # Take the first batch of posts
    # We need to convert dicts back to Post objects for the LLM client
    posts_batch_dicts = state.interesting_posts_queue[:threshold]
    posts_batch = [Post(**p) for p in posts_batch_dicts]

    # Initialize ChromaDB for duplicate detection
    chroma_memory = None
    if openai_api_key:
        try:
            chroma_memory = ChromaMemory(
                config=config,
                openai_api_key=openai_api_key,
            )
        except Exception as e:
            logger.warning(
                "chroma_init_failed",
                extra={"error": str(e)},
            )

    # Generate inspired tweet with retry on validation failure
    max_attempts = 3
    tweet_text = None

    try:
        for attempt in range(1, max_attempts + 1):
            tweet_text = await llm_client.generate_inspiration_tweet(posts_batch)
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

            # Validate tweet
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

        # Use session manager for browser operations
        async with AsyncTwitterSession(
            config, twitter_username, twitter_password
        ) as driver:
            success = post_tweet(driver, tweet_text, config)

            if success:
                logger.info("inspiration_tweet_posted")

                # Log written tweet for dashboard
                await log_written_tweet(
                    text=tweet_text,
                    tweet_type="inspiration",
                )

                # Store tweet in ChromaDB for future duplicate detection
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

                # Update state: remove processed posts
                # We reload state to avoid race conditions
                current_state = await load_state()

                # Remove the first 'threshold' items
                # We assume the queue hasn't changed significantly (FIFO)
                if len(current_state.interesting_posts_queue) >= threshold:
                    current_state.interesting_posts_queue = (
                        current_state.interesting_posts_queue[threshold:]
                    )

                # Increment post count
                current_state.counters["posts_today"] += 1
                current_state.last_post_time = datetime.now(timezone.utc)

                await save_state(current_state)

                # Log action
                await log_action("Posted inspiration tweet")
            else:
                logger.error("failed_to_post_inspiration_tweet")

    except Exception as e:
        logger.error(
            "inspiration_processing_error",
            extra={"error": str(e)},
            exc_info=True,
        )
        # Don't remove posts from queue if generation/posting failed
        raise

    finally:
        # Close LLM client to prevent event loop errors
        await llm_client.close()
