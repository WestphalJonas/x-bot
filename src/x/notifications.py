"""Twitter/X notifications checking automation."""

import logging
import time
from datetime import datetime

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.state.models import Notification
from src.x.driver import human_delay
from src.x.parser import NotificationParser

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def check_notifications(driver, config: BotConfig, count: int = 20) -> list[Notification]:
    """Check notifications from Twitter/X notifications page.

    Args:
        driver: Selenium driver instance
        config: Bot configuration
        count: Maximum number of notifications to check (default: 20)

    Returns:
        List of Notification objects (filtered for replies and mentions)

    Raises:
        Exception: If checking fails after retries
    """
    try:
        # Navigate to notifications page
        logger.info("navigating_to_notifications", extra={"url": "https://x.com/notifications"})
        driver.get("https://x.com/notifications")
        human_delay(config)

        # Wait for notifications to load
        wait = WebDriverWait(driver, 20)
        try:
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'article[data-testid="notification"]')
                )
            )
            logger.info("notifications_loaded")
        except TimeoutException:
            logger.warning("notifications_not_found_initial_load")
            # Try alternative selector
            try:
                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'div[data-testid="notification"]')
                    )
                )
                logger.info("notifications_loaded_alternative_selector")
            except TimeoutException:
                logger.warning("no_notifications_found")
                return []

        notifications_found: list[Notification] = []
        seen_notification_ids: set[str] = set()
        max_scroll_attempts = 5
        scroll_attempts = 0

        while len(notifications_found) < count and scroll_attempts < max_scroll_attempts:
            # Find all notification elements
            notification_selectors = [
                'article[data-testid="notification"]',
                'div[data-testid="notification"]',
                'div[role="article"]',  # Fallback selector
            ]

            notification_elements = []
            for selector in notification_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        notification_elements = elements
                        logger.info(
                            "notifications_found",
                            extra={"count": len(elements), "selector": selector},
                        )
                        break
                except Exception as e:
                    logger.warning("notification_extraction_failed", extra={"error": str(e)})
                    continue

            # Extract data from each notification
            for notification_element in notification_elements:
                try:
                    # Extract notification type
                    notification_type = NotificationParser.extract_notification_type(
                        notification_element
                    )

                    # Filter for replies and mentions only (skip likes, retweets, follows)
                    if notification_type not in ["reply", "mention"]:
                        logger.debug(
                            "skipping_notification_type",
                            extra={"type": notification_type},
                        )
                        continue

                    # Extract notification ID to avoid duplicates
                    notification_id = NotificationParser.extract_notification_id(
                        notification_element
                    )
                    if notification_id and notification_id in seen_notification_ids:
                        continue
                    if notification_id:
                        seen_notification_ids.add(notification_id)

                    # Extract notification text
                    text = NotificationParser.extract_notification_text(notification_element)
                    if not text or len(text) < 5:  # Skip very short/invalid notifications
                        logger.debug(
                            "skipping_short_notification",
                            extra={
                                "notification_id": notification_id,
                                "text_length": len(text) if text else 0,
                            },
                        )
                        continue

                    # Extract author information (similar to PostParser)
                    from src.x.parser import PostParser

                    username, display_name = PostParser.extract_author_info(
                        notification_element
                    )
                    if username == "unknown":
                        logger.warning(
                            "notification_author_extraction_failed",
                            extra={"notification_id": notification_id},
                        )
                        # Still include the notification, but with unknown username

                    # Extract original post context (for replies)
                    original_post_id, original_post_text = (
                        NotificationParser.extract_original_post_context(
                            notification_element
                        )
                    )

                    # Extract timestamp and URL
                    timestamp = NotificationParser.extract_notification_timestamp(
                        notification_element
                    )
                    url = NotificationParser.extract_notification_url(notification_element)

                    # Determine if it's a reply or mention
                    is_reply = notification_type == "reply"
                    is_mention = notification_type == "mention"

                    notification = Notification(
                        notification_id=notification_id,
                        type=notification_type,
                        text=text,
                        from_username=username,
                        from_display_name=display_name,
                        original_post_id=original_post_id,
                        original_post_text=original_post_text,
                        timestamp=timestamp,
                        url=url,
                        is_reply=is_reply,
                        is_mention=is_mention,
                    )

                    notifications_found.append(notification)
                    logger.info(
                        "notification_extracted",
                        extra={
                            "notification_id": notification_id,
                            "type": notification_type,
                            "from_username": username,
                            "text_length": len(text),
                            "is_reply": is_reply,
                            "is_mention": is_mention,
                        },
                    )

                    # Stop if we have enough notifications
                    if len(notifications_found) >= count:
                        break

                except Exception as e:
                    logger.warning(
                        "notification_extraction_error",
                        extra={"error": str(e)},
                        exc_info=True,
                    )
                    continue

            # Scroll to load more notifications if needed
            if len(notifications_found) < count and scroll_attempts < max_scroll_attempts:
                logger.info(
                    "scrolling_for_more_notifications",
                    extra={
                        "current_count": len(notifications_found),
                        "target_count": count,
                        "scroll_attempt": scroll_attempts + 1,
                    },
                )
                driver.execute_script("window.scrollBy(0, 500)")
                human_delay(config)
                # Wait a bit for new content to load
                time.sleep(1)
                scroll_attempts += 1

        logger.info(
            "notification_checking_complete",
            extra={
                "notifications_found": len(notifications_found),
                "target_count": count,
                "scroll_attempts": scroll_attempts,
            },
        )

        return notifications_found

    except Exception as e:
        logger.error(
            "notification_checking_failed",
            extra={"error": str(e), "count": count},
            exc_info=True,
        )
        raise

