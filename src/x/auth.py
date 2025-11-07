"""Twitter/X authentication with cookie persistence."""

import json
import logging
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.x.driver import human_delay

logger = logging.getLogger(__name__)


def load_cookies(
    driver, config: BotConfig, cookie_path: str | Path = "config/cookie.json"
) -> bool:
    """Load cookies from JSON file if exists.

    Args:
        driver: Selenium driver instance
        config: Bot configuration
        cookie_path: Path to cookie JSON file

    Returns:
        True if cookies loaded successfully, False otherwise
    """
    path = Path(cookie_path)
    if not path.exists():
        logger.info("no_cookies_found", extra={"path": str(path)})
        return False

    # Check if file is empty
    if path.stat().st_size == 0:
        logger.info("cookie_file_empty", extra={"path": str(path)})
        return False

    try:
        driver.get("https://x.com")
        human_delay(config)

        with open(path) as f:
            content = f.read().strip()
            if not content:
                logger.info("cookie_file_empty", extra={"path": str(path)})
                return False
            cookies = json.loads(content)

        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                logger.warning(
                    "cookie_load_failed",
                    extra={
                        "error": str(e),
                        "cookie_name": cookie.get("name", "unknown"),
                    },
                )

        # Refresh to apply cookies
        driver.refresh()
        human_delay(config)

        # Check if logged in by looking for compose button or profile
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]')
                )
            )
            logger.info("cookies_loaded_successfully")
            return True
        except Exception:
            logger.warning("cookies_invalid_or_expired")
            return False

    except Exception as e:
        logger.error("cookie_load_error", extra={"error": str(e)}, exc_info=True)
        return False


def save_cookies(driver, cookie_path: str | Path = "config/cookie.json") -> None:
    """Save cookies to JSON file.

    Args:
        driver: Selenium driver instance
        cookie_path: Path to cookie JSON file
    """
    path = Path(cookie_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cookies = driver.get_cookies()
        with open(path, "w") as f:
            json.dump(cookies, f, indent=2)

        logger.info("cookies_saved", extra={"path": str(path), "count": len(cookies)})

    except Exception as e:
        logger.error("cookie_save_error", extra={"error": str(e)}, exc_info=True)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def login(driver, username: str, password: str, config: BotConfig) -> bool:
    """Login to Twitter/X with username and password.

    Args:
        driver: Selenium driver instance
        username: Twitter/X username
        password: Twitter/X password
        config: Bot configuration

    Returns:
        True if login successful, False otherwise

    Raises:
        Exception: If login fails after retries
    """
    try:
        driver.get("https://x.com/i/flow/login")
        human_delay(config)

        # Wait for username input
        wait = WebDriverWait(driver, 20)
        username_input = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'input[autocomplete="username"]')
            )
        )
        username_input.clear()
        username_input.send_keys(username)
        human_delay(config)

        # Click Next button
        next_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//span[text()="Next"]'))
        )
        next_button.click()
        human_delay(config)

        # Handle potential phone/email verification step
        # Twitter/X sometimes asks for phone/email verification after username
        try:
            # Wait a bit to see if verification challenge appears
            verification_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
                )
            )
            logger.info("verification_challenge_detected")
            # If username contains @, it's an email - might need to handle differently
            if "@" in username:
                # Try clicking "Forgot password?" to skip verification
                try:
                    forgot_link = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "Forgot")]'))
                    )
                    forgot_link.click()
                    human_delay(config)
                except Exception:
                    logger.warning("could_not_skip_verification")
            else:
                # For username, might need to enter phone number
                logger.warning("phone_verification_required")
        except Exception:
            # No verification step, continue normally
            logger.info("no_verification_challenge")

        # Wait for password input - try multiple selectors
        password_input = None
        password_selectors = [
            'input[name="password"]',
            'input[type="password"]',
            'input[autocomplete="current-password"]',
            'input[data-testid="ocfEnterTextTextInput"][type="password"]',
        ]
        
        for selector in password_selectors:
            try:
                password_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                logger.info("password_field_found", extra={"selector": selector})
                break
            except Exception:
                continue
        
        if not password_input:
            # Take screenshot for debugging
            try:
                screenshot_path = Path("login_debug.png")
                driver.save_screenshot(str(screenshot_path))
                logger.error(
                    "password_field_not_found",
                    extra={
                        "screenshot": str(screenshot_path),
                        "current_url": driver.current_url,
                        "page_source_length": len(driver.page_source),
                    }
                )
            except Exception:
                pass
            raise TimeoutException("Password input field not found. Twitter/X login flow may have changed.")
        password_input.clear()
        password_input.send_keys(password)
        human_delay(config)

        # Click Login button
        login_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//span[text()="Log in"]'))
        )
        login_button.click()
        human_delay(config)

        # Wait for successful login (check for compose button or home timeline)
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]')
                )
            )
            logger.info("login_successful", extra={"username": username})
            return True
        except Exception as e:
            logger.error("login_verification_failed", extra={"error": str(e)})
            return False

    except Exception as e:
        logger.error(
            "login_failed", extra={"error": str(e), "username": username}, exc_info=True
        )
        raise
