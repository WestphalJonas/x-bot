"""Twitter/X session management with context manager pattern."""

import logging
from types import TracebackType

import undetected_chromedriver as uc

from src.core.config import BotConfig
from src.x.auth import load_cookies, login, save_cookies
from src.x.driver import create_driver

logger = logging.getLogger(__name__)


class TwitterSession:
    """Context manager for Twitter/X browser sessions.

    Handles driver creation, cookie-based login, and cleanup automatically.
    Reduces code duplication across job functions.

    Usage:
        with TwitterSession(config, username, password) as driver:
            post_tweet(driver, tweet_text, config)
    """

    def __init__(
        self,
        config: BotConfig,
        username: str,
        password: str,
    ):
        """Initialize Twitter session.

        Args:
            config: Bot configuration
            username: Twitter/X username
            password: Twitter/X password
        """
        self.config = config
        self.username = username
        self.password = password
        self.driver: uc.Chrome | None = None

    def __enter__(self) -> uc.Chrome:
        """Enter context: create driver and authenticate.

        Returns:
            Authenticated Chrome driver instance

        Raises:
            RuntimeError: If login fails
        """
        logger.info("twitter_session_starting")

        # Create driver
        self.driver = create_driver(self.config)

        # Try to load cookies first
        cookies_loaded = load_cookies(self.driver, self.config)

        if not cookies_loaded:
            # Login if cookies not available
            logger.info("twitter_session_logging_in")
            login_success = login(
                self.driver,
                self.username,
                self.password,
                self.config,
            )

            if not login_success:
                # Clean up driver on failure
                self.driver.quit()
                self.driver = None
                logger.error("twitter_session_login_failed")
                raise RuntimeError("Twitter login failed")

            # Save cookies after successful login
            save_cookies(self.driver)
            logger.info("twitter_session_cookies_saved")

        logger.info("twitter_session_authenticated")
        return self.driver

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit context: cleanup driver.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Traceback if an error occurred

        Returns:
            False to propagate any exceptions
        """
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("twitter_session_closed")

        # Don't suppress exceptions
        return False


class AsyncTwitterSession:
    """Async context manager for Twitter/X browser sessions.

    Same as TwitterSession but supports async with syntax.

    Usage:
        async with AsyncTwitterSession(config, username, password) as driver:
            post_tweet(driver, tweet_text, config)
    """

    def __init__(
        self,
        config: BotConfig,
        username: str,
        password: str,
    ):
        """Initialize async Twitter session.

        Args:
            config: Bot configuration
            username: Twitter/X username
            password: Twitter/X password
        """
        self.config = config
        self.username = username
        self.password = password
        self.driver: uc.Chrome | None = None

    async def __aenter__(self) -> uc.Chrome:
        """Enter async context: create driver and authenticate.

        Returns:
            Authenticated Chrome driver instance

        Raises:
            RuntimeError: If login fails
        """
        logger.info("async_twitter_session_starting")

        # Create driver (sync operation, but wrapped in async context)
        self.driver = create_driver(self.config)

        # Try to load cookies first
        cookies_loaded = load_cookies(self.driver, self.config)

        if not cookies_loaded:
            # Login if cookies not available
            logger.info("async_twitter_session_logging_in")
            login_success = login(
                self.driver,
                self.username,
                self.password,
                self.config,
            )

            if not login_success:
                # Clean up driver on failure
                self.driver.quit()
                self.driver = None
                logger.error("async_twitter_session_login_failed")
                raise RuntimeError("Twitter login failed")

            # Save cookies after successful login
            save_cookies(self.driver)
            logger.info("async_twitter_session_cookies_saved")

        logger.info("async_twitter_session_authenticated")
        return self.driver

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit async context: cleanup driver.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Traceback if an error occurred

        Returns:
            False to propagate any exceptions
        """
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("async_twitter_session_closed")

        # Don't suppress exceptions
        return False
