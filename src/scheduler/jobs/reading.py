"""Frontpage post reading job."""

import asyncio
import logging

from src.constants import QueueLimits
from src.core.config import BotConfig, EnvSettings
from src.core.interest import check_interest
from src.core.llm import LLMClient
from src.state.database import get_database
from src.state.manager import load_state, save_state
from src.web.data_tracker import log_action
from src.x.reading import read_frontpage_posts as read_posts_from_frontpage
from src.x.session import AsyncTwitterSession

logger = logging.getLogger(__name__)


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

