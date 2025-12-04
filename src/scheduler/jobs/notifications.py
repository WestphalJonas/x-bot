"""Notification checking job."""

import asyncio
import logging
from datetime import datetime, timezone

from src.constants import QueueLimits
from src.core.config import BotConfig, EnvSettings
from src.state.manager import load_state, save_state
from src.web.data_tracker import log_action
from src.x.notifications import check_notifications as check_notifications_func
from src.x.session import AsyncTwitterSession

logger = logging.getLogger(__name__)


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

