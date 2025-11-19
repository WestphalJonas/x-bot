"""Scheduled job functions for bot tasks."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from openai import AuthenticationError, RateLimitError

from src.core.config import BotConfig
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.x.auth import load_cookies, login, save_cookies
from src.x.driver import create_driver
from src.x.posting import post_tweet

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
                "message": "Authentication failed. Please check your API keys in .env file.",
            },
        )
        raise
    except RateLimitError as e:
        logger.error(
            "tweet_generation_failed_rate_limit",
            extra={
                "error": str(e),
                "message": "Rate limit or quota exceeded. Please check your account billing or wait before retrying.",
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
    """Read frontpage posts (scheduled job - stub implementation).

    This will be implemented in Phase 1 (Frontpage Reading).

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    job_id = "read_posts"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        logger.info(
            "feature_not_implemented",
            extra={
                "job_id": job_id,
                "feature": "read_frontpage_posts",
                "phase": "Phase 1 - Frontpage Reading",
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
