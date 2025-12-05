"""Autonomous tweet posting job."""

import asyncio
import logging
from datetime import datetime, timezone

from openai import AuthenticationError, RateLimitError

from src.core.config import BotConfig, EnvSettings
from src.core.evaluation import re_evaluate_tweet
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.web.data_tracker import log_action, log_rejected_tweet, log_written_tweet
from src.x.posting import post_tweet
from src.x.session import AsyncTwitterSession

from src.memory.chroma_client import ChromaMemory

logger = logging.getLogger(__name__)


def post_autonomous_tweet(config: BotConfig, env_settings: EnvSettings) -> None:
    """Post an autonomous tweet (scheduled job).

    This function wraps async operations for APScheduler compatibility.
    Extracted from main.py posting logic.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables (API keys, credentials)
    """
    job_id = "post_tweet"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        # Run async operations in event loop
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
    """Async implementation of autonomous tweet posting.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    # Load state (with automatic counter reset check)
    state = await load_state(
        reset_time_utc=config.rate_limits.reset_time_utc,
    )

    # Check rate limits
    if state.counters["posts_today"] >= config.rate_limits.max_posts_per_day:
        logger.warning(
            "rate_limit_exceeded",
            extra={
                "posts_today": state.counters["posts_today"],
                "max_posts": config.rate_limits.max_posts_per_day,
            },
        )
        return

    # Get environment variables
    openai_api_key = env_settings.get("OPENAI_API_KEY")
    openrouter_api_key = env_settings.get("OPENROUTER_API_KEY")
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    # Validate that at least one LLM provider is configured
    if not openai_api_key and not openrouter_api_key:
        raise ValueError(
            "At least one LLM provider API key is required. "
            "Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
        )
    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")

    # Get all API keys
    google_api_key = env_settings.get("GOOGLE_API_KEY")
    anthropic_api_key = env_settings.get("ANTHROPIC_API_KEY")

    # Initialize LLM client
    llm_client = LLMClient(
        config=config,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        google_api_key=google_api_key,
        anthropic_api_key=anthropic_api_key,
    )

    # Get system prompt
    system_prompt = config.get_system_prompt()

    # Generate tweet
    logger.info("generating_tweet")
    try:
        tweet_text = await llm_client.generate_tweet(system_prompt)
        logger.info("tweet_generated", extra={"length": len(tweet_text)})
    except AuthenticationError as e:
        logger.error(
            "tweet_generation_failed_auth",
            extra={
                "error": str(e),
                "detail": "Authentication failed. Please check your API keys in .env file.",
            },
        )
        raise
    except RateLimitError as e:
        logger.error(
            "tweet_generation_failed_rate_limit",
            extra={
                "error": str(e),
                "detail": "Rate limit or quota exceeded. Please check your account billing or wait before retrying.",
            },
        )
        raise
    except Exception as e:
        logger.error("tweet_generation_failed", extra={"error": str(e)}, exc_info=True)
        raise

    # Check for duplicate using ChromaDB
    chroma_memory = None
    if openai_api_key:
        try:
            chroma_memory = ChromaMemory(
                config=config,
                openai_api_key=openai_api_key,
            )
            is_duplicate, similarity = await chroma_memory.check_duplicate(tweet_text)
            if is_duplicate:
                logger.warning(
                    "duplicate_tweet_detected",
                    extra={
                        "similarity": similarity,
                        "threshold": config.llm.similarity_threshold,
                    },
                )
                # Regenerate tweet
                logger.info("regenerating_tweet_due_to_duplicate")
                tweet_text = await llm_client.generate_tweet(system_prompt)
                # Check again (simple retry, not infinite loop)
                is_duplicate, similarity = await chroma_memory.check_duplicate(
                    tweet_text
                )
                if is_duplicate:
                    raise ValueError(
                        f"Generated tweet is too similar to existing content (similarity: {similarity:.2f})"
                    )
        except ValueError:
            raise  # Re-raise duplicate error
        except Exception as e:
            logger.warning(
                "chroma_duplicate_check_failed",
                extra={"error": str(e)},
            )
            # Continue without duplicate check if ChromaDB fails

    # Validate tweet (basic)
    is_valid, error_message = await llm_client.validate_tweet(tweet_text)
    if not is_valid:
        logger.error("tweet_validation_failed", extra={"error": error_message})
        # Log rejected tweet for dashboard
        try:
            await log_rejected_tweet(
                text=tweet_text,
                reason=error_message,
                operation="autonomous",
            )
            logger.info("rejected_tweet_saved", extra={"reason": error_message})
        except Exception as save_error:
            logger.error(
                "failed_to_save_rejected_tweet",
                extra={"error": str(save_error)},
                exc_info=True,
            )
        raise ValueError(f"Tweet validation failed: {error_message}")

    # Final LLM gatekeeper check
    approved, evaluation_reason = await re_evaluate_tweet(
        tweet_text=tweet_text,
        config=config,
        llm_client=llm_client,
        operation="autonomous",
    )
    if not approved:
        logger.error(
            "tweet_re_evaluation_failed", extra={"reason": evaluation_reason}
        )
        try:
            await log_rejected_tweet(
                text=tweet_text,
                reason=evaluation_reason,
                operation="autonomous",
            )
        except Exception as save_error:
            logger.error(
                "failed_to_save_rejected_tweet",
                extra={"error": str(save_error)},
                exc_info=True,
            )
        raise ValueError(f"Tweet re-evaluation failed: {evaluation_reason}")

    logger.info(
        "tweet_validated",
        extra={"length": len(tweet_text), "approved": approved},
    )

    # Use session manager for browser operations
    try:
        async with AsyncTwitterSession(
            config, twitter_username, twitter_password
        ) as driver:
            # Post tweet
            logger.info("posting_tweet")
            post_success = post_tweet(driver, tweet_text, config)
            if not post_success:
                logger.error("tweet_post_failed")
                raise RuntimeError("Tweet posting failed")

            # Log written tweet for dashboard
            await log_written_tweet(
                text=tweet_text,
                tweet_type="autonomous",
            )

            # Store tweet in ChromaDB for future duplicate detection
            if chroma_memory:
                try:
                    await chroma_memory.store_tweet(
                        text=tweet_text,
                        metadata={"type": "autonomous"},
                    )
                    logger.info("tweet_stored_in_memory")
                except Exception as e:
                    logger.warning(
                        "failed_to_store_tweet_in_memory",
                        extra={"error": str(e)},
                    )

            # Update state
            state.counters["posts_today"] += 1
            state.last_post_time = datetime.now(timezone.utc)

            # Save state
            await save_state(state)
            logger.info(
                "state_updated", extra={"posts_today": state.counters["posts_today"]}
            )

            # Log action
            await log_action("Posted autonomous tweet")

            logger.info("bot_completed_successfully")

    except Exception as e:
        logger.error("bot_error", extra={"error": str(e)}, exc_info=True)
        raise

    finally:
        # Close LLM client to prevent event loop errors
        await llm_client.close()

