"""Test script to read first post from Twitter/X and save to JSON."""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import BotConfig
from src.x.auth import load_cookies, login, save_cookies
from src.x.driver import create_driver
from src.x.reading import read_frontpage_posts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_env_settings() -> dict[str, str | None]:
    """Load environment variables into a dictionary.

    Returns:
        Dictionary with environment variable values
    """
    return {
        "TWITTER_USERNAME": os.getenv("TWITTER_USERNAME"),
        "TWITTER_PASSWORD": os.getenv("TWITTER_PASSWORD"),
    }


def main():
    """Main entry point for test script."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config_path = Path("config/config.yaml")
    config = BotConfig.load(config_path)

    logger.info("test_script_starting", extra={"config_path": str(config_path)})

    # Load and validate environment settings
    env_settings = load_env_settings()
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    if not twitter_username:
        logger.error("TWITTER_USERNAME environment variable is required")
        return
    if not twitter_password:
        logger.error("TWITTER_PASSWORD environment variable is required")
        return

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
                return

            # Save cookies after successful login
            save_cookies(driver)

        # Read first post
        logger.info("reading_first_post")
        posts = read_frontpage_posts(driver, config, count=1)

        if not posts:
            logger.warning("no_posts_found")
            return

        post = posts[0]
        logger.info(
            "post_read",
            extra={
                "post_id": post.post_id,
                "username": post.username,
                "display_name": post.display_name,
                "post_type": post.post_type,
                "text_length": len(post.text),
                "likes": post.likes,
                "retweets": post.retweets,
                "replies": post.replies,
            },
        )

        # Convert post to dict for JSON serialization
        post_dict = {
            "text": post.text,
            "username": post.username,
            "display_name": post.display_name,
            "post_id": post.post_id,
            "post_type": post.post_type,
            "likes": post.likes,
            "retweets": post.retweets,
            "replies": post.replies,
            "timestamp": post.timestamp.isoformat() if post.timestamp else None,
            "url": post.url,
        }

        # Save to JSON file
        output_path = Path("test_post_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(post_dict, f, indent=2, ensure_ascii=False)

        logger.info("post_saved_to_json", extra={"output_path": str(output_path)})
        print(f"\n✅ Post saved to: {output_path}")
        print(f"\nPost details:")
        print(f"  Username: {post.username}")
        print(f"  Display Name: {post.display_name}")
        print(f"  Post ID: {post.post_id}")
        print(f"  Post Type: {post.post_type}")
        print(
            f"  Text: {post.text[:100]}..."
            if len(post.text) > 100
            else f"  Text: {post.text}"
        )
        print(
            f"  Likes: {post.likes}, Retweets: {post.retweets}, Replies: {post.replies}"
        )

    except Exception as e:
        logger.error("test_script_error", extra={"error": str(e)}, exc_info=True)
        print(f"\n❌ Error: {e}")

    finally:
        if driver:
            driver.quit()
            logger.info("browser_closed")


if __name__ == "__main__":
    main()
