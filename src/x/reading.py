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


def _extract_number_from_text(text: str | None) -> int:
    """Extract numeric value from text (handles K, M suffixes).

    Args:
        text: Text containing number (e.g., "1.2K", "500", "2M")

    Returns:
        Extracted number as integer, 0 if parsing fails
    """
    if not text:
        return 0

    # Remove whitespace
    text = text.strip()

    # Handle empty string
    if not text:
        return 0

    # Try to extract number with K/M suffix
    match = re.search(r"([\d.]+)\s*([KMkm]?)", text)
    if match:
        number = float(match.group(1))
        suffix = match.group(2).upper()

        if suffix == "K":
            return int(number * 1000)
        elif suffix == "M":
            return int(number * 1000000)
        else:
            return int(number)

    # Try to extract just numbers
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[0])

    return 0


def _detect_post_type(post_element) -> str:
    """Detect the type of post (text-only, media-only, retweet, quoted).

    Based on actual Twitter/X HTML structure:
    - Quoted tweets: nested article[data-testid="tweet"] AND quote labels ("Zitat", "Quote")
    - Text with media: data-testid="tweetPhoto" or data-testid="videoComponent" in main post
    - Text-only: text content without media elements

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        String indicating post type: "text_only", "text_with_media", "media_only", "retweet", "quoted", "unknown"
    """
    # Check for retweet/repost indicator
    retweet_selectors = [
        'div[data-testid="socialContext"]',  # Contains "X reposted"
        'span:contains("Reposted")',
        'span:contains("Repost")',
    ]
    for selector in retweet_selectors:
        try:
            if ":contains(" in selector:
                # XPath fallback
                elements = post_element.find_elements(
                    By.XPATH, './/span[contains(text(), "Repost")]'
                )
            else:
                elements = post_element.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                return "retweet"
        except Exception:
            continue

    # Check for quoted/embedded tweet
    # Look for nested article[data-testid="tweet"] AND quote labels ("Zitat", "Quote")
    has_nested_article = False
    has_quote_label = False

    try:
        # Check for nested article structure
        nested_articles = post_element.find_elements(
            By.CSS_SELECTOR, 'article[data-testid="tweet"] article[data-testid="tweet"]'
        )
        if nested_articles:
            has_nested_article = True

        # Also check for simpler nested structure
        if not has_nested_article:
            nested_articles = post_element.find_elements(
                By.CSS_SELECTOR, 'article[data-testid="tweet"] article'
            )
            if nested_articles:
                has_nested_article = True
    except Exception:
        pass

    # Check for quote labels ("Zitat" in German, "Quote" in English)
    try:
        # Check for quote label text in the post
        quote_text_elements = post_element.find_elements(
            By.XPATH, './/span[contains(text(), "Zitat") or contains(text(), "Quote")]'
        )
        if quote_text_elements:
            has_quote_label = True

        # Also check aria-labelledby for quote indicators
        if not has_quote_label:
            aria_labels = post_element.find_elements(By.XPATH, ".//*[@aria-labelledby]")
            for elem in aria_labels:
                aria_labelledby = elem.get_attribute("aria-labelledby") or ""
                if (
                    "zitat" in aria_labelledby.lower()
                    or "quote" in aria_labelledby.lower()
                ):
                    has_quote_label = True
                    break
    except Exception:
        pass

    # If we have both nested article and quote label, it's a quoted tweet
    if has_nested_article and has_quote_label:
        return "quoted"

    # Check for media in the main post (exclude quoted tweet area)
    # We need to check media that's NOT inside a nested quoted tweet
    has_media = False

    try:
        # First, try to find the main post article (not nested)
        main_article = post_element
        try:
            # If post_element is not an article, find the article ancestor
            if post_element.tag_name != "article":
                main_article = post_element.find_element(
                    By.XPATH, "./ancestor::article[@data-testid='tweet']"
                )
        except Exception:
            pass

        # Check for media in main post using primary selectors
        media_selectors = [
            'div[data-testid="tweetPhoto"]',
            'div[data-testid="videoComponent"]',
        ]

        for selector in media_selectors:
            try:
                media_elements = main_article.find_elements(By.CSS_SELECTOR, selector)
                # Filter out media that's inside a quoted tweet
                for media_elem in media_elements:
                    try:
                        # Check if this media is inside a nested article (quoted tweet)
                        nested_parent = media_elem.find_element(
                            By.XPATH, "./ancestor::article[@data-testid='tweet']"
                        )
                        # If we find a nested article parent, check if it's different from main article
                        if nested_parent != main_article:
                            # This media is in a quoted tweet, skip it
                            continue
                    except Exception:
                        # No nested article found, this is main post media
                        pass

                    # This is main post media
                    has_media = True
                    break

                if has_media:
                    break
            except Exception:
                continue

        # Fallback: check for images/videos if primary selectors didn't find anything
        if not has_media:
            try:
                # Check for images (but exclude profile images)
                images = main_article.find_elements(
                    By.CSS_SELECTOR, 'img[src*="pbs.twimg.com"]'
                )
                for img in images:
                    src = img.get_attribute("src") or ""
                    # Skip profile images (usually contain "profile_images" or "normal.jpg")
                    if "profile_images" not in src and "normal.jpg" not in src:
                        # Check if it's not in a quoted tweet
                        try:
                            nested_parent = img.find_element(
                                By.XPATH, "./ancestor::article[@data-testid='tweet']"
                            )
                            if nested_parent == main_article:
                                has_media = True
                                break
                        except Exception:
                            # No nested article, assume main post
                            has_media = True
                            break
            except Exception:
                pass

            # Check for videos
            if not has_media:
                try:
                    videos = main_article.find_elements(By.CSS_SELECTOR, "video")
                    for video in videos:
                        # Check if it's not in a quoted tweet
                        try:
                            nested_parent = video.find_element(
                                By.XPATH, "./ancestor::article[@data-testid='tweet']"
                            )
                            if nested_parent == main_article:
                                has_media = True
                                break
                        except Exception:
                            # No nested article, assume main post
                            has_media = True
                            break
                except Exception:
                    pass
    except Exception:
        pass

    # Check if text exists
    text = _extract_post_text(post_element)
    has_text = bool(text and len(text.strip()) > 10)

    # Determine type
    if has_media and has_text:
        return "text_with_media"
    elif has_media and not has_text:
        return "media_only"
    elif has_text:
        return "text_only"
    else:
        return "unknown"


def _extract_post_text(post_element) -> str:
    """Extract post text from a post element.

    Based on actual Twitter/X HTML structure:
    - Main post text: div[data-testid="tweetText"] (first occurrence, not in quoted tweet)
    - Quoted tweets also have tweetText, but they're nested deeper

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        Post text content
    """
    text_selectors = [
        'div[data-testid="tweetText"]',
        'div[data-testid="tweet"] span[lang]',
        "div[lang]",
        "span[lang]",
    ]

    for selector in text_selectors:
        try:
            text_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
            if text_elements:
                # Get text from first matching element (should be main post, not quoted)
                # Try to avoid quoted tweet text by checking if it's nested too deep
                for elem in text_elements:
                    text = elem.text.strip()
                    if text:
                        # Check if this is likely a quoted tweet (has ancestor with quote indicator)
                        try:
                            # Quoted tweets are typically in a container with specific classes
                            # or have a "Zitat" (Quote) label nearby
                            elem.find_element(
                                By.XPATH,
                                "./ancestor::div[contains(@class, 'r-9aw3ui')]",
                            )
                            # If we find a quote container, skip this element
                            continue
                        except Exception:
                            # No quote container found, this is likely the main post text
                            return text
        except Exception:
            continue

    # Fallback: get all text from post element
    try:
        text = post_element.text.strip()
        # Remove author and engagement metrics if present
        lines = text.split("\n")
        # Filter out lines that look like engagement metrics or author info
        filtered_lines = [
            line
            for line in lines
            if not re.match(r"^\d+[KMkm]?$", line.strip())
            and not line.strip().startswith("@")
            and "·" not in line
            and line.strip().lower() not in ["zitat", "quote"]  # Skip quote labels
        ]
        return " ".join(filtered_lines).strip()
    except Exception:
        return ""


def _extract_author_info(post_element) -> tuple[str, str | None]:
    """Extract author username and display name from a post element.

    Based on actual Twitter/X HTML structure:
    - User-Name container: div[data-testid="User-Name"]
    - Display name: First link's span text (e.g., "Will Duncan")
    - Username: Second link's span text starting with @ (e.g., "@wheelieduncan")

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        Tuple of (username, display_name). Username is always returned (defaults to "unknown"),
        display_name may be None if not found.
    """
    username = None
    display_name = None

    # Try to find User-Name container which contains both display name and username
    user_name_selectors = [
        'div[data-testid="User-Name"]',
        'div[data-testid="User-Names"]',
    ]

    for selector in user_name_selectors:
        try:
            user_name_containers = post_element.find_elements(By.CSS_SELECTOR, selector)
            if user_name_containers:
                container = user_name_containers[0]

                # Find all links in the User-Name container
                links = container.find_elements(By.CSS_SELECTOR, "a[href]")

                # First link typically contains display name
                # Look for span with display name (not starting with @, not containing ·)
                for link in links:
                    href = link.get_attribute("href")
                    if not href or "/status/" in href:
                        continue  # Skip status links

                    # Get all spans in this link
                    spans = link.find_elements(By.CSS_SELECTOR, "span")
                    for span in spans:
                        text = span.text.strip()
                        if text and not text.startswith("@") and "·" not in text:
                            # This is likely the display name
                            if not display_name:
                                display_name = text
                            break

                    # Also extract username from href if it's a profile link
                    if href.startswith("/") and "/" in href[1:]:
                        parts = href.split("/")
                        if len(parts) >= 2 and parts[1] and parts[1] != "status":
                            potential_username = parts[1]
                            if not username:
                                username = f"@{potential_username}"

                # Look for username span (starts with @)
                # Username is typically in a separate link or span
                username_spans = container.find_elements(
                    By.XPATH, './/span[starts-with(text(), "@")]'
                )
                for span in username_spans:
                    text = span.text.strip()
                    if text.startswith("@"):
                        username = text
                        break

                # Also check links for username
                if not username:
                    for link in links:
                        href = link.get_attribute("href")
                        if href and href.startswith("/") and "/status/" not in href:
                            # Extract from href like "/wheelieduncan"
                            parts = href.split("/")
                            if len(parts) >= 2 and parts[1]:
                                potential_username = parts[1]
                                if potential_username != "status":
                                    username = f"@{potential_username}"
                                    break

                if username:
                    break
        except Exception:
            continue

    # Fallback: try to find username from various selectors
    if not username:
        author_selectors = [
            'a[href*="/"] span',
            'a[role="link"] span',
        ]

        for selector in author_selectors:
            try:
                author_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                for elem in author_elements:
                    text = elem.text.strip()
                    if text.startswith("@"):
                        username = text
                        break
                    # Check if parent link contains username
                    try:
                        parent_link = elem.find_element(
                            By.XPATH, "./ancestor::a[@href]"
                        )
                        href = parent_link.get_attribute("href")
                        if href and href.startswith("/") and "/status/" not in href:
                            parts = href.split("/")
                            if len(parts) >= 2 and parts[1] and parts[1] != "status":
                                username = f"@{parts[1]}"
                                break
                    except Exception:
                        pass

                if username:
                    break
            except Exception:
                continue

    # Final fallback: try to find any @ mention in the post
    if not username:
        try:
            all_text = post_element.text
            match = re.search(r"@(\w+)", all_text)
            if match:
                username = f"@{match.group(1)}"
        except Exception:
            pass

    # Ensure we always return a username (even if unknown)
    if not username:
        username = "unknown"

    return (username, display_name)


def _extract_engagement_metrics(post_element) -> tuple[int, int, int]:
    """Extract engagement metrics (likes, retweets, replies) from a post element.

    Based on actual Twitter/X HTML structure:
    - Reply: button[data-testid="reply"] with aria-label and span containing count
    - Retweet: button[data-testid="retweet"] with aria-label and span containing count
    - Like: button[data-testid="like"] with aria-label and span containing count

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        Tuple of (likes, retweets, replies)
    """
    likes = 0
    retweets = 0
    replies = 0

    # Try to find engagement buttons by data-testid (most reliable)
    button_mappings = [
        ("reply", "replies"),
        ("retweet", "retweets"),
        ("like", "likes"),
    ]

    for testid, metric_name in button_mappings:
        try:
            buttons = post_element.find_elements(
                By.CSS_SELECTOR, f'button[data-testid="{testid}"]'
            )
            for button in buttons:
                # Try to get count from aria-label first
                aria_label = button.get_attribute("aria-label") or ""
                number = _extract_number_from_text(aria_label)

                # Also check for span with number inside button
                if number == 0:
                    spans = button.find_elements(By.CSS_SELECTOR, "span")
                    for span in spans:
                        span_text = span.text.strip()
                        span_number = _extract_number_from_text(span_text)
                        if span_number > 0:
                            number = span_number
                            break

                # Also check button text
                if number == 0:
                    button_text = button.text.strip()
                    number = _extract_number_from_text(button_text)

                # Update the appropriate metric
                if metric_name == "replies":
                    replies = max(replies, number)
                elif metric_name == "retweets":
                    retweets = max(retweets, number)
                elif metric_name == "likes":
                    likes = max(likes, number)
        except Exception:
            continue

    # Fallback: try to find engagement buttons by aria-label
    button_selectors = [
        ('button[aria-label*="Reply"]', "replies"),
        ('button[aria-label*="Repost"]', "retweets"),
        ('button[aria-label*="Like"]', "likes"),
        ('button[aria-label*="Gefällt mir"]', "likes"),  # German
        ('button[aria-label*="Antworten"]', "replies"),  # German
    ]

    for selector, metric_name in button_selectors:
        try:
            buttons = post_element.find_elements(By.CSS_SELECTOR, selector)
            for button in buttons:
                aria_label = button.get_attribute("aria-label") or ""
                number = _extract_number_from_text(aria_label)

                if metric_name == "replies":
                    replies = max(replies, number)
                elif metric_name == "retweets":
                    retweets = max(retweets, number)
                elif metric_name == "likes":
                    likes = max(likes, number)
        except Exception:
            continue

    # Final fallback: try to find numbers near engagement words in text
    try:
        all_text = post_element.text
        lines = all_text.split("\n")
        for i, line in enumerate(lines):
            number = _extract_number_from_text(line)

            if number > 0:
                # Check context around the number
                context = " ".join(lines[max(0, i - 1) : i + 2]).lower()
                if "reply" in context or "antwort" in context or "repost" in context:
                    replies = max(replies, number)
                if "retweet" in context or "repost" in context:
                    retweets = max(retweets, number)
                if "like" in context or "gefällt" in context:
                    likes = max(likes, number)
    except Exception:
        pass

    return (likes, retweets, replies)


def _extract_post_id(post_element) -> str | None:
    """Extract post ID from a post element.

    Based on actual Twitter/X HTML structure:
    - Post ID is in timestamp link: /username/status/1990832010194010596
    - Or in time element's parent link

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        Post ID if found, None otherwise
    """
    # Try to find link to the post (most reliable - timestamp link)
    link_selectors = [
        'a[href*="/status/"]',
        'a[href*="/i/web/status/"]',
        "time[datetime]",
    ]

    for selector in link_selectors:
        try:
            elements = post_element.find_elements(By.CSS_SELECTOR, selector)
            for elem in elements:
                # Get href from element or parent link
                href = elem.get_attribute("href")
                if not href:
                    # Try parent link if this is a time element
                    try:
                        parent_link = elem.find_element(
                            By.XPATH, "./ancestor::a[@href]"
                        )
                        href = parent_link.get_attribute("href")
                    except Exception:
                        continue

                if href:
                    # Extract status ID from URL patterns:
                    # /username/status/1990832010194010596
                    # /i/web/status/1990832010194010596
                    match = re.search(r"/status/(\d+)", href)
                    if match:
                        return match.group(1)
        except Exception:
            continue

    # Try to extract from data attributes
    try:
        article = post_element.find_element(By.XPATH, "./ancestor::article")
        post_id = article.get_attribute("data-post-id")
        if post_id:
            return post_id
    except Exception:
        pass

    return None


def _extract_post_url(post_element) -> str | None:
    """Extract post URL from a post element.

    Based on actual Twitter/X HTML structure:
    - Prefer timestamp link (main status URL)
    - Avoid analytics links (/analytics)
    - Fallback to constructing from post ID and username

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        Post URL if found, None otherwise
    """
    # Try to find timestamp link first (most reliable, excludes analytics)
    try:
        time_elements = post_element.find_elements(By.CSS_SELECTOR, "time[datetime]")
        for time_elem in time_elements:
            try:
                parent_link = time_elem.find_element(By.XPATH, "./ancestor::a[@href]")
                href = parent_link.get_attribute("href")
                if href and "/status/" in href and "/analytics" not in href:
                    return href
            except Exception:
                continue
    except Exception:
        pass

    # Try to find status links, but exclude analytics
    link_selectors = [
        'a[href*="/status/"]',
        'a[href*="/i/web/status/"]',
    ]

    for selector in link_selectors:
        try:
            links = post_element.find_elements(By.CSS_SELECTOR, selector)
            for link in links:
                href = link.get_attribute("href")
                if href and "/status/" in href:
                    # Skip analytics links
                    if "/analytics" in href:
                        continue
                    # Prefer main status URL (no extra path after status ID)
                    if re.search(r"/status/\d+$", href) or re.search(
                        r"/status/\d+/?$", href
                    ):
                        return href
                    # Otherwise use it as fallback
                    return href
        except Exception:
            continue

    # Try to construct URL from post ID (most reliable)
    post_id = _extract_post_id(post_element)
    if post_id:
        username, _ = _extract_author_info(post_element)
        if username and username != "unknown":
            username_clean = username.replace("@", "")
            return f"https://x.com/{username_clean}/status/{post_id}"

    return None


def _extract_timestamp(post_element) -> datetime | None:
    """Extract post timestamp from a post element.

    Based on actual Twitter/X HTML structure:
    - Timestamp is in time[datetime] element with ISO datetime string

    Args:
        post_element: Selenium WebElement representing a post

    Returns:
        Post timestamp if found, None otherwise
    """
    try:
        time_elements = post_element.find_elements(By.CSS_SELECTOR, "time[datetime]")
        for time_elem in time_elements:
            datetime_str = time_elem.get_attribute("datetime")
            if datetime_str:
                try:
                    # Parse ISO format datetime: "2025-11-18T17:18:33.000Z"
                    # Replace 'Z' with '+00:00' for fromisoformat
                    if datetime_str.endswith("Z"):
                        datetime_str = datetime_str[:-1] + "+00:00"
                    elif "+" not in datetime_str and "-" not in datetime_str[-6:]:
                        # Assume UTC if no timezone indicator
                        datetime_str = datetime_str + "+00:00"

                    # Parse the ISO format datetime
                    return datetime.fromisoformat(datetime_str)
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        "timestamp_parse_failed",
                        extra={"datetime_str": datetime_str, "error": str(e)},
                    )
                    continue
    except Exception as e:
        logger.warning("timestamp_extraction_failed", extra={"error": str(e)})

    return None


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
                    post_id = _extract_post_id(post_element)
                    if post_id and post_id in seen_post_ids:
                        continue
                    if post_id:
                        seen_post_ids.add(post_id)

                    # Detect post type and skip non-text posts
                    post_type = _detect_post_type(post_element)
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
                    text = _extract_post_text(post_element)
                    if not text or len(text) < 10:  # Skip very short/invalid posts
                        logger.info(
                            "skipping_short_post",
                            extra={
                                "post_id": post_id,
                                "text_length": len(text) if text else 0,
                            },
                        )
                        continue

                    username, display_name = _extract_author_info(post_element)
                    if username == "unknown":
                        logger.warning(
                            "author_extraction_failed", extra={"post_id": post_id}
                        )
                        # Still include the post, but with unknown username

                    likes, retweets, replies = _extract_engagement_metrics(post_element)
                    url = _extract_post_url(post_element)
                    timestamp = _extract_timestamp(post_element)

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
