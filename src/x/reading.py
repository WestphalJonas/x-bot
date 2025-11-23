"""Twitter/X frontpage reading automation."""

import logging
import re
import time
from datetime import datetime

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.state.models import Post
from src.x.driver import human_delay


logger = logging.getLogger(__name__)


from src.x.parser import PostParser


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def read_frontpage_posts(driver, config: BotConfig, count: int = 10) -> list[Post]:
    """Read posts from Twitter/X home feed.

    Args:
        driver: Selenium driver instance
        config: Bot configuration
        count: Number of posts to read (default: 10)

    Returns:
        List of Post objects

    Raises:
        Exception: If reading fails after retries
    """
    try:
        # Navigate to X.com home
        logger.info("navigating_to_home", extra={"url": "https://x.com/home"})
        driver.get("https://x.com/home")
        human_delay(config)

        # Wait for posts to load
        wait = WebDriverWait(driver, 20)
        try:
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                )
            )
            logger.info("posts_loaded")
        except TimeoutException:
            logger.warning("posts_not_found_initial_load")
            # Try alternative selector
            try:
                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'div[data-testid="tweet"]')
                    )
                )
                logger.info("posts_loaded_alternative_selector")
            except TimeoutException:
                logger.error("no_posts_found")
                return []

        posts_found: list[Post] = []
        seen_post_ids: set[str] = set()
        max_scroll_attempts = 10
        scroll_attempts = 0

        while len(posts_found) < count and scroll_attempts < max_scroll_attempts:
            # Find all post elements
            post_selectors = [
                'article[data-testid="tweet"]',
                'div[data-testid="tweet"]',
            ]

            post_elements = []
            for selector in post_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        post_elements = elements
                        logger.info(
                            "posts_found",
                            extra={"count": len(elements), "selector": selector},
                        )
                        break
                except Exception as e:
                    logger.warning("post_extraction_failed", extra={"error": str(e)})
                    continue

            # Extract data from each post
            for post_element in post_elements:
                try:
                    # Extract post ID to avoid duplicates
                    post_id = PostParser.extract_post_id(post_element)
                    if post_id and post_id in seen_post_ids:
                        continue
                    if post_id:
                        seen_post_ids.add(post_id)

                    # Detect post type and skip non-text posts
                    post_type = PostParser.detect_post_type(post_element)
                    if post_type in ["media_only", "retweet", "unknown"]:
                        logger.info(
                            "skipping_non_text_post",
                            extra={
                                "post_id": post_id,
                                "post_type": post_type,
                                "reason": "Only text posts are processed",
                            },
                        )
                        continue

                    # Extract post data
                    text = PostParser.extract_post_text(post_element)
                    if not text or len(text) < 10:  # Skip very short/invalid posts
                        logger.info(
                            "skipping_short_post",
                            extra={
                                "post_id": post_id,
                                "text_length": len(text) if text else 0,
                            },
                        )
                        continue

                    username, display_name = PostParser.extract_author_info(post_element)
                    if username == "unknown":
                        logger.warning(
                            "author_extraction_failed", extra={"post_id": post_id}
                        )
                        # Still include the post, but with unknown username

                    likes, retweets, replies = PostParser.extract_engagement_metrics(post_element)
                    url = PostParser.extract_post_url(post_element)
                    timestamp = PostParser.extract_timestamp(post_element)

                    post = Post(
                        text=text,
                        username=username,
                        display_name=display_name,
                        post_id=post_id,
                        post_type=post_type,
                        likes=likes,
                        retweets=retweets,
                        replies=replies,
                        timestamp=timestamp,
                        url=url,
                    )

                    posts_found.append(post)
                    logger.info(
                        "post_extracted",
                        extra={
                            "post_id": post_id,
                            "username": username,
                            "display_name": display_name,
                            "text_length": len(text),
                            "likes": likes,
                            "retweets": retweets,
                            "replies": replies,
                        },
                    )

                    # Stop if we have enough posts
                    if len(posts_found) >= count:
                        break

                except Exception as e:
                    logger.warning(
                        "post_extraction_error",
                        extra={"error": str(e)},
                        exc_info=True,
                    )
                    continue

            # Scroll to load more posts if needed
            if len(posts_found) < count and scroll_attempts < max_scroll_attempts:
                logger.info(
                    "scrolling_for_more_posts",
                    extra={
                        "current_count": len(posts_found),
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
            "reading_complete",
            extra={
                "posts_found": len(posts_found),
                "target_count": count,
                "scroll_attempts": scroll_attempts,
            },
        )

        return posts_found

    except Exception as e:
        logger.error(
            "reading_failed",
            extra={"error": str(e), "count": count},
            exc_info=True,
        )
        raise
