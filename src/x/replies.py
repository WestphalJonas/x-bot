"""Twitter/X reply automation."""

from __future__ import annotations

import logging
import random
import time
from typing import Iterable

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.x.driver import human_delay

logger = logging.getLogger(__name__)


def _find_first(driver, selectors: Iterable[str]):
    """Return the first element found across multiple CSS selectors."""
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                return elements[0], selector
        except Exception:
            continue
    return None, None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def post_reply(
    driver, reply_text: str, notification_url: str, config: BotConfig
) -> bool:
    """Post a reply to a notification thread.

    Args:
        driver: Selenium driver instance
        reply_text: Text content of the reply
        notification_url: URL of the notification/post to reply to
        config: Bot configuration

    Returns:
        True if reply succeeds, False otherwise
    """
    try:
        target_url = notification_url or "https://x.com/notifications"
        logger.info(
            "reply_navigating",
            extra={"url": target_url, "text_length": len(reply_text)},
        )
        driver.get(target_url)
        human_delay(config)

        wait = WebDriverWait(driver, 25)

        # Click reply button if available to ensure editor is visible
        try:
            reply_btn = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='reply']"))
            )
            reply_btn.click()
            human_delay(config)
        except Exception:
            logger.debug("reply_button_not_clicked")

        # Focus the reply editor (broader selectors)
        editor_selectors = [
            'div[role="textbox"][data-testid="tweetTextarea_0"]',
            'div[role="textbox"][data-testid="tweetTextarea_1"]',
            'div[contenteditable="true"][data-testid="tweetTextarea_0"]',
            'div[contenteditable="true"][data-testid="tweetTextarea_1"]',
            'div[role="textbox"]',
            '[data-testid="tweetTextarea_0"]',
            '[data-testid="tweetTextarea_1"]',
            'div[data-testid="tweetTextarea_0"] div[contenteditable="true"]',
            'div[data-testid="tweetTextarea_1"] div[contenteditable="true"]',
        ]

        compose_editor, used_selector = _find_first(driver, editor_selectors)
        if compose_editor:
            logger.info("reply_editor_found", extra={"selector": used_selector})
        else:
            try:
                compose_editor = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            'div[role="textbox"][data-testid="tweetTextarea_0"]',
                        )
                    )
                )
                logger.info("reply_editor_found_waited")
            except Exception as exc:
                logger.error(
                    "reply_editor_not_found",
                    extra={"selectors": editor_selectors, "error": str(exc)},
                    exc_info=True,
                )
                raise

        compose_editor.click()
        human_delay(config)

        # Clear any existing text
        try:
            compose_editor.clear()
        except Exception:
            pass

        # Type reply with small random delays
        for char in reply_text:
            compose_editor.send_keys(char)
            time.sleep(random.uniform(0.05, 0.12))

        human_delay(config)

        # Click the reply button (use multiple possible selectors)
        reply_button, button_selector = _find_first(
            driver, ["[data-testid='tweetButton']", "[data-testid='tweetButtonInline']"]
        )
        if not reply_button:
            reply_button = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "[data-testid='tweetButton']")
                )
            )
            button_selector = "[data-testid='tweetButton']"

        logger.info("reply_button_clicked", extra={"selector": button_selector})
        reply_button.click()

        # Wait for confirmation (button disable or editor cleared)
        try:
            wait.until(
                EC.invisibility_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='tweetTextarea_0']")
                )
            )
        except Exception:
            # Fallback: small delay to allow post to complete
            time.sleep(1.5)

        logger.info("reply_posted_successfully", extra={"length": len(reply_text)})
        return True
    except Exception as exc:
        logger.error(
            "reply_post_failed",
            extra={"error": str(exc), "url": notification_url},
            exc_info=True,
        )
        raise
