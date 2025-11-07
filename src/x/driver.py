"""Browser driver management with undetected-chromedriver."""

import logging
import random
import time

import undetected_chromedriver as uc
from selenium_stealth import stealth

from src.core.config import BotConfig

logger = logging.getLogger(__name__)


def create_driver(config: BotConfig) -> uc.Chrome:
    """Create undetected Chrome driver with stealth mode.

    Args:
        config: Bot configuration

    Returns:
        Configured Chrome driver instance
    """
    options = uc.ChromeOptions()

    if config.selenium.headless:
        options.add_argument("--headless=new")

    # Additional stealth options
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    # User-Agent rotation if enabled
    if config.selenium.user_agent_rotation:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        options.add_argument(f"--user-agent={random.choice(user_agents)}")

    try:
        driver = uc.Chrome(options=options, version_main=None)

        # Apply stealth techniques
        stealth(
            driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )

        logger.info("driver_created", extra={"headless": config.selenium.headless})

        return driver

    except Exception as e:
        logger.error("driver_creation_failed", extra={"error": str(e)}, exc_info=True)
        raise


def human_delay(config: BotConfig) -> None:
    """Sleep for a random delay between min and max delay seconds.

    Args:
        config: Bot configuration
    """
    delay = random.uniform(
        config.selenium.min_delay_seconds, config.selenium.max_delay_seconds
    )
    time.sleep(delay)
