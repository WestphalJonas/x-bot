"""Main entry point for autonomous Twitter/X posting bot."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError

from src.core.config import BotConfig
from src.core.llm import LLMClient
from src.state.manager import load_state, save_state
from src.x.auth import load_cookies, login, save_cookies
from src.x.driver import create_driver
from src.x.posting import post_tweet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the bot."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config_path = Path("config/config.yaml")
    config = BotConfig.load(config_path)

    logger.info("bot_starting", extra={"config_path": str(config_path)})

    # Get environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    twitter_username = os.getenv("TWITTER_USERNAME")
    twitter_password = os.getenv("TWITTER_PASSWORD")

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

    # Initialize LLM client with provider support
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
                "message": "Authentication failed. Please check your OPENAI_API_KEY in .env file.",
            },
        )
        return
    except RateLimitError as e:
        logger.error(
            "tweet_generation_failed_rate_limit",
            extra={
                "error": str(e),
                "message": "Rate limit or quota exceeded. Please check your OpenAI account billing or wait before retrying.",
            },
        )
        return
    except Exception as e:
        logger.error("tweet_generation_failed", extra={"error": str(e)}, exc_info=True)
        return

    # Validate tweet
    print(tweet_text)
    is_valid, error_message = await llm_client.validate_tweet(tweet_text)
    if not is_valid:
        logger.error("tweet_validation_failed", extra={"error": error_message})
        return

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
                return

            # Save cookies after successful login
            save_cookies(driver)

        # Post tweet
        logger.info("posting_tweet")
        post_success = post_tweet(driver, tweet_text, config)
        if not post_success:
            logger.error("tweet_post_failed")
            return

        # Update state
        state.counters["posts_today"] += 1
        state.last_post_time = datetime.now(datetime.UTC)

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


if __name__ == "__main__":
    asyncio.run(main())
