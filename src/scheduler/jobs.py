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
        # Close LLM client to prevent event loop errors
        await llm_client.close()
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
            # Load state to check for already-queued posts
            state = await load_state()
            queued_post_ids = {p["post_id"] for p in state.interesting_posts_queue}

            logger.info(
                "evaluating_posts_for_interest", extra={"total_posts": len(posts)}
            )

            for post in posts:
                # Skip posts already in queue
                if post.post_id in queued_post_ids:
                    logger.info(
                        "post_skipped_already_in_queue",
                        extra={"post_id": post.post_id, "username": post.username},
                    )
                    continue

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

                # Log each post being added to queue
                for post in interesting_posts:
                    logger.info(
                        "interesting_post_added_to_queue",
                        extra={
                            "post_id": post.post_id,
                            "username": post.username,
                            "text_preview": post.text[:80] + "..."
                            if len(post.text) > 80
                            else post.text,
                            "likes": post.likes,
                        },
                    )

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
        # Close LLM client to prevent event loop errors
        if llm_client:
            await llm_client.close()
        if driver:
            driver.quit()
            logger.info("browser_closed")


def check_notifications(config: BotConfig, env_settings: dict[str, Any]) -> None:
    """Check notifications (scheduled job).

    This function wraps async operations for APScheduler compatibility.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables (API keys, credentials)
    """
    job_id = "check_notifications"
    logger.info("job_started", extra={"job_id": job_id})

    try:
        # Run async operations in event loop
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
    config: BotConfig, env_settings: dict[str, Any]
) -> None:
    """Async implementation of notification checking.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    # Get environment variables
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")

    # Load state to check for processed notifications
    state = await load_state()
    processed_ids = set(state.processed_notification_ids)

    # Initialize browser driver
    logger.info("initializing_browser_for_notifications")
    driver = None
    try:
        driver = create_driver(config)

        # Try to load cookies first
        cookies_loaded = load_cookies(driver, config)
        if not cookies_loaded:
            # Login if cookies not available
            logger.info("logging_in_for_notifications")
            login_success = login(driver, twitter_username, twitter_password, config)
            if not login_success:
                logger.error("login_failed_for_notifications")
                raise RuntimeError("Login failed for notification checking")

            # Save cookies after successful login
            save_cookies(driver)

        # Check notifications
        logger.info("checking_notifications")
        from src.x.notifications import check_notifications as check_notifications_func

        notifications = check_notifications_func(driver, config, count=20)

        logger.info(
            "notifications_checked",
            extra={
                "count": len(notifications),
                "notifications": [
                    {
                        "notification_id": notif.notification_id,
                        "type": notif.type,
                        "from_username": notif.from_username,
                        "text_length": len(notif.text),
                        "is_reply": notif.is_reply,
                        "is_mention": notif.is_mention,
                    }
                    for notif in notifications
                ],
            },
        )

        # Filter out already processed notifications
        new_notifications = [
            n for n in notifications if n.notification_id not in processed_ids
        ]

        logger.info(
            "notifications_filtered",
            extra={
                "total": len(notifications),
                "new": len(new_notifications),
                "already_processed": len(notifications) - len(new_notifications),
            },
        )

        # Store new notifications in queue
        if new_notifications:
            # Convert notifications to dict for storage
            notification_dicts = [n.model_dump() for n in new_notifications]

            # Append to queue with size limit (max 50 notifications)
            max_queue_size = 50
            state.notifications_queue.extend(notification_dicts)

            # Trim queue if it exceeds max size (keep most recent)
            if len(state.notifications_queue) > max_queue_size:
                state.notifications_queue = state.notifications_queue[-max_queue_size:]
                logger.info(
                    "notifications_queue_trimmed",
                    extra={
                        "max_size": max_queue_size,
                        "trimmed_to": max_queue_size,
                    },
                )

            # Update processed notification IDs (keep last 100)
            new_ids = [
                n.notification_id for n in new_notifications if n.notification_id
            ]
            state.processed_notification_ids.extend(new_ids)
            if len(state.processed_notification_ids) > 100:
                state.processed_notification_ids = state.processed_notification_ids[
                    -100:
                ]

            # Update last check time
            state.last_notification_check_time = datetime.now(timezone.utc)

            # Save state
            await save_state(state)

            logger.info(
                "notifications_queued",
                extra={
                    "new_notifications_added": len(new_notifications),
                    "queue_size": len(state.notifications_queue),
                },
            )
        else:
            # Still update last check time even if no new notifications
            state.last_notification_check_time = datetime.now(timezone.utc)
            await save_state(state)
            logger.info("no_new_notifications")

        logger.info("notification_checking_completed_successfully")

    except Exception as e:
        logger.error(
            "notification_checking_error", extra={"error": str(e)}, exc_info=True
        )
        raise

    finally:
        if driver:
            driver.quit()
            logger.info("browser_closed")


def process_inspiration_queue(config: BotConfig, env_settings: dict[str, Any]) -> None:
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
    config: BotConfig, env_settings: dict[str, Any]
) -> None:
    """Async implementation of inspiration queue processing.

    Args:
        config: Bot configuration
        env_settings: Dictionary with environment variables
    """
    # Load state
    state = await load_state()

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
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    # Validate that at least one LLM provider is configured
    if not openai_api_key and not openrouter_api_key:
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
    )

    # Take the first batch of posts
    # We need to convert dicts back to Post objects for the LLM client
    from src.state.models import Post

    posts_batch_dicts = state.interesting_posts_queue[:threshold]
    posts_batch = [Post(**p) for p in posts_batch_dicts]

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
                raise ValueError(
                    f"Inspiration tweet validation failed after {max_attempts} attempts: {error_message}"
                )

        # Initialize browser driver
        logger.info("initializing_browser_for_inspiration")
        driver = None
        try:
            driver = create_driver(config)

            # Try to load cookies first
            cookies_loaded = load_cookies(driver, config)
            if not cookies_loaded:
                # Login if cookies not available
                logger.info("logging_in_for_inspiration")
                login_success = login(
                    driver, twitter_username, twitter_password, config
                )
                if not login_success:
                    logger.error("login_failed_for_inspiration")
                    raise RuntimeError("Login failed for inspiration queue processing")

                # Save cookies after successful login
                save_cookies(driver)

            from src.x.posting import post_tweet

            success = post_tweet(driver, tweet_text, config)

            if success:
                logger.info("inspiration_tweet_posted")

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
            else:
                logger.error("failed_to_post_inspiration_tweet")

        finally:
            if driver:
                driver.quit()
                logger.info("browser_closed")

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
