"""Scheduled job functions for bot tasks."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from openai import AuthenticationError, RateLimitError

from src.core.config import BotConfig
from src.core.interest import check_interest
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.x.auth import load_cookies, login, save_cookies
from src.x.driver import create_driver
from src.x.posting import post_tweet
from src.x.reading import read_frontpage_posts as read_posts_from_frontpage

logger = logging.getLogger(__name__)


def post_autonomous_tweet(config: BotConfig, env_settings: dict[str, Any]) -> None:
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
    config: BotConfig, env_settings: dict[str, Any]
) -> None:
    """Async implementation of autonomous tweet posting.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    # Load state
    state = await load_state()

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

    # Initialize LLM client
    llm_client = LLMClient(
        config=config,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
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

    # Validate tweet
    is_valid, error_message = await llm_client.validate_tweet(tweet_text)
    if not is_valid:
        logger.error("tweet_validation_failed", extra={"error": error_message})
        raise ValueError(f"Tweet validation failed: {error_message}")

    logger.info("tweet_validated", extra={"length": len(tweet_text)})

    # Initialize browser driver
    logger.info("initializing_browser")
    driver = None
    try:
        driver = create_driver(config)

        # Try to load cookies first
        cookies_loaded = load_cookies(driver, config)
        if not cookies_loaded:
            # Login if cookies not available
            logger.info("logging_in")
            login_success = login(driver, twitter_username, twitter_password, config)
            if not login_success:
                logger.error("login_failed")
                raise RuntimeError("Login failed")

            # Save cookies after successful login
            save_cookies(driver)

        # Post tweet
        logger.info("posting_tweet")
        post_success = post_tweet(driver, tweet_text, config)
        if not post_success:
            logger.error("tweet_post_failed")
            raise RuntimeError("Tweet posting failed")

        # Update state
        state.counters["posts_today"] += 1
        state.last_post_time = datetime.now(timezone.utc)

        # Save state
        await save_state(state)
        logger.info(
            "state_updated", extra={"posts_today": state.counters["posts_today"]}
        )

        logger.info("bot_completed_successfully")

    except Exception as e:
        logger.error("bot_error", extra={"error": str(e)}, exc_info=True)
        raise

    finally:
        if driver:
            driver.quit()
            logger.info("browser_closed")


def read_frontpage_posts(config: BotConfig, env_settings: dict[str, Any]) -> None:
    """Read frontpage posts (scheduled job).

    This function wraps async operations for APScheduler compatibility.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables (API keys, credentials)
    """
    job_id = "read_posts"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        # Run async operations in event loop
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
    config: BotConfig, env_settings: dict[str, Any]
) -> None:
    """Async implementation of frontpage post reading with interest detection.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    # Get environment variables
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")
    openai_api_key = env_settings.get("OPENAI_API_KEY")
    openrouter_api_key = env_settings.get("OPENROUTER_API_KEY")

    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")

    # Validate that at least one LLM provider is configured
    if not openai_api_key and not openrouter_api_key:
        logger.warning(
            "no_llm_provider_configured",
            extra={"detail": "Interest detection will be skipped"},
        )

    # Initialize LLM client for interest detection
    llm_client = None
    if openai_api_key or openrouter_api_key:
        llm_client = LLMClient(
            config=config,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
        )

    # Initialize browser driver
    logger.info("initializing_browser")
    driver = None
    try:
        driver = create_driver(config)

        # Try to load cookies first
        cookies_loaded = load_cookies(driver, config)
        if not cookies_loaded:
            # Login if cookies not available
            logger.info("logging_in")
            login_success = login(driver, twitter_username, twitter_password, config)
            if not login_success:
                logger.error("login_failed")
                raise RuntimeError("Login failed")

            # Save cookies after successful login
            save_cookies(driver)

        # Read posts
        logger.info("reading_frontpage_posts")
        posts = read_posts_from_frontpage(driver, config, count=10)

        logger.info(
            "posts_read",
            extra={
                "count": len(posts),
                "posts": [
                    {
                        "post_id": post.post_id,
                        "username": post.username,
                        "display_name": post.display_name,
                        "text_length": len(post.text),
                        "likes": post.likes,
                        "retweets": post.retweets,
                        "replies": post.replies,
                    }
                    for post in posts
                ],
            },
        )

        # Evaluate posts for interest if LLM client is available
        interesting_posts = []
        if llm_client and posts:
            logger.info(
                "evaluating_posts_for_interest", extra={"total_posts": len(posts)}
            )

            for post in posts:
                try:
                    # Check if post is interesting
                    is_interesting = await check_interest(post, config, llm_client)
                    post.is_interesting = is_interesting

                    # Log each post evaluation for dashboard
                    logger.info(
                        "post_evaluated",
                        extra={
                            "post_id": post.post_id,
                            "username": post.username,
                            "display_name": post.display_name,
                            "is_interesting": is_interesting,
                            "text_length": len(post.text),
                            "likes": post.likes,
                            "retweets": post.retweets,
                            "replies": post.replies,
                            "post_type": post.post_type,
                            "timestamp": post.timestamp.isoformat()
                            if post.timestamp
                            else None,
                        },
                    )

                    # Collect interesting posts
                    if is_interesting:
                        interesting_posts.append(post)

                except Exception as e:
                    # Handle LLM failures gracefully
                    logger.warning(
                        "interest_check_failed_for_post",
                        extra={
                            "post_id": post.post_id,
                            "username": post.username,
                            "error": str(e),
                        },
                        exc_info=True,
                    )
                    # Mark as not interesting on error (conservative approach)
                    post.is_interesting = False

            # Log summary statistics
            logger.info(
                "interest_evaluation_summary",
                extra={
                    "posts_evaluated": len(posts),
                    "interesting_count": len(interesting_posts),
                    "not_interesting_count": len(posts) - len(interesting_posts),
                },
            )

            # Load state and update queue
            if interesting_posts:
                state = await load_state()

                # Convert posts to dict for storage (Pydantic models need to be serialized)
                post_dicts = [post.model_dump() for post in interesting_posts]

                # Append to queue with size limit (max 50 posts)
                max_queue_size = 50
                state.interesting_posts_queue.extend(post_dicts)

                # Trim queue if it exceeds max size (keep most recent)
                if len(state.interesting_posts_queue) > max_queue_size:
                    state.interesting_posts_queue = state.interesting_posts_queue[
                        -max_queue_size:
                    ]
                    logger.info(
                        "queue_trimmed",
                        extra={
                            "max_size": max_queue_size,
                            "trimmed_to": max_queue_size,
                        },
                    )

                # Save state with updated queue
                await save_state(state)

                logger.info(
                    "interesting_posts_queued",
                    extra={
                        "new_posts_added": len(interesting_posts),
                        "queue_size": len(state.interesting_posts_queue),
                    },
                )
        elif not llm_client:
            logger.warning(
                "interest_detection_skipped",
                extra={"reason": "No LLM provider configured"},
            )
            # Mark all posts as not evaluated
            for post in posts:
                post.is_interesting = None

        logger.info("reading_completed_successfully")

    except Exception as e:
        logger.error("reading_error", extra={"error": str(e)}, exc_info=True)
        raise

    finally:
        if driver:
            driver.quit()
            logger.info("browser_closed")


def check_notifications(config: BotConfig, env_settings: dict[str, Any]) -> None:
    """Check notifications (scheduled job - stub implementation).

    This will be implemented in Phase 3 (Notifications).

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    job_id = "check_notifications"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        logger.info(
            "feature_not_implemented",
            extra={
                "job_id": job_id,
                "feature": "check_notifications",
                "phase": "Phase 3 - Notifications",
            },
        )
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
