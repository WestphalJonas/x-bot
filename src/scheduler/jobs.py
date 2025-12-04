"""Scheduled job functions for bot tasks."""

import asyncio
import logging
from datetime import datetime, timezone
from openai import AuthenticationError, RateLimitError

from src.constants import QueueLimits
from src.core.config import BotConfig, EnvSettings
from src.core.interest import check_interest
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.state.models import Post

# Optional ChromaDB import for duplicate detection
try:
    from src.memory.chroma_client import ChromaMemory

    CHROMA_AVAILABLE = True
except ImportError:
    ChromaMemory = None  # type: ignore
    CHROMA_AVAILABLE = False
from src.x.notifications import check_notifications as check_notifications_func
from src.x.posting import post_tweet
from src.x.reading import read_frontpage_posts as read_posts_from_frontpage
from src.x.session import AsyncTwitterSession
from src.state.database import get_database
from src.web.data_tracker import log_action, log_rejected_tweet, log_written_tweet

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

    # Check for duplicate using ChromaDB if available
    chroma_memory = None
    if CHROMA_AVAILABLE and openai_api_key and ChromaMemory is not None:
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

    # Validate tweet
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

    logger.info("tweet_validated", extra={"length": len(tweet_text)})

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


def read_frontpage_posts(config: BotConfig, env_settings: EnvSettings) -> None:
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
    config: BotConfig, env_settings: EnvSettings
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
    google_api_key = env_settings.get("GOOGLE_API_KEY")
    anthropic_api_key = env_settings.get("ANTHROPIC_API_KEY")

    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")

    # Validate that at least one LLM provider is configured
    has_llm_provider = any(
        [openai_api_key, openrouter_api_key, google_api_key, anthropic_api_key]
    )
    if not has_llm_provider:
        logger.warning(
            "no_llm_provider_configured",
            extra={"detail": "Interest detection will be skipped"},
        )

    # Initialize LLM client for interest detection
    llm_client = None
    if has_llm_provider:
        llm_client = LLMClient(
            config=config,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            google_api_key=google_api_key,
            anthropic_api_key=anthropic_api_key,
        )

    # Use session manager for browser operations
    try:
        async with AsyncTwitterSession(
            config, twitter_username, twitter_password
        ) as driver:
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

        # Get database for tracking read posts
        db = await get_database()

        # Evaluate posts for interest if LLM client is available (outside browser session)
        interesting_posts = []
        if llm_client and posts:
            # Load state to check for already-queued posts
            state = await load_state()
            queued_post_ids = {p["post_id"] for p in state.interesting_posts_queue}

            logger.info(
                "evaluating_posts_for_interest", extra={"total_posts": len(posts)}
            )

            for post in posts:
                # Skip posts already seen (stored in SQLite)
                if post.post_id and await db.has_seen_post(post.post_id):
                    logger.info(
                        "post_skipped_already_seen",
                        extra={"post_id": post.post_id, "username": post.username},
                    )
                    continue
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

                    # Store post in SQLite (regardless of interest)
                    if post.post_id:
                        await db.store_read_post(post)

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

                # Append to queue with size limit
                state.interesting_posts_queue.extend(post_dicts)

                # Trim queue if it exceeds max size (keep most recent)
                if len(state.interesting_posts_queue) > QueueLimits.INTERESTING_POSTS:
                    state.interesting_posts_queue = state.interesting_posts_queue[
                        -QueueLimits.INTERESTING_POSTS :
                    ]
                    logger.info(
                        "queue_trimmed",
                        extra={
                            "max_size": QueueLimits.INTERESTING_POSTS,
                            "trimmed_to": QueueLimits.INTERESTING_POSTS,
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

        # Log action
        await log_action(f"Read {len(posts)} posts from timeline")

        logger.info("reading_completed_successfully")

    except Exception as e:
        logger.error("reading_error", extra={"error": str(e)}, exc_info=True)
        raise

    finally:
        # Close LLM client to prevent event loop errors
        if llm_client:
            await llm_client.close()


def check_notifications(config: BotConfig, env_settings: EnvSettings) -> None:
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
    config: BotConfig, env_settings: EnvSettings
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

    # Use session manager for browser operations
    try:
        async with AsyncTwitterSession(
            config, twitter_username, twitter_password
        ) as driver:
            # Check notifications
            logger.info("checking_notifications")
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

        # Filter out already processed notifications (outside browser session)
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

            # Append to queue with size limit
            state.notifications_queue.extend(notification_dicts)

            # Trim queue if it exceeds max size (keep most recent)
            if len(state.notifications_queue) > QueueLimits.NOTIFICATIONS:
                state.notifications_queue = state.notifications_queue[
                    -QueueLimits.NOTIFICATIONS :
                ]
                logger.info(
                    "notifications_queue_trimmed",
                    extra={
                        "max_size": QueueLimits.NOTIFICATIONS,
                        "trimmed_to": QueueLimits.NOTIFICATIONS,
                    },
                )

            # Update processed notification IDs (keep recent ones)
            new_ids = [
                n.notification_id for n in new_notifications if n.notification_id
            ]
            state.processed_notification_ids.extend(new_ids)
            if (
                len(state.processed_notification_ids)
                > QueueLimits.PROCESSED_NOTIFICATION_IDS
            ):
                state.processed_notification_ids = state.processed_notification_ids[
                    -QueueLimits.PROCESSED_NOTIFICATION_IDS :
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

        # Log action
        await log_action(f"Checked notifications ({len(new_notifications)} new)")

        logger.info("notification_checking_completed_successfully")

    except Exception as e:
        logger.error(
            "notification_checking_error", extra={"error": str(e)}, exc_info=True
        )
        raise


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

    # Initialize ChromaDB for duplicate detection if available
    chroma_memory = None
    if CHROMA_AVAILABLE and openai_api_key and ChromaMemory is not None:
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
