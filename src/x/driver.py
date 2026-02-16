"""Browser driver management with undetected-chromedriver."""

import logging
import platform
import random
import re
import subprocess
import time

import undetected_chromedriver as uc
from selenium.common.exceptions import SessionNotCreatedException
from selenium_stealth import stealth

from src.core.config import BotConfig

logger = logging.getLogger(__name__)


def _detect_browser_major_version() -> int | None:
    """Best-effort detection of installed Chrome major version."""
    system = platform.system()

    # Windows: read installed Chrome version from registry.
    if system == "Windows":
        try:
            import winreg

            reg_paths = [
                r"SOFTWARE\Google\Chrome\BLBeacon",
                r"SOFTWARE\WOW6432Node\Google\Chrome\BLBeacon",
            ]
            for reg_path in reg_paths:
                for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                    try:
                        with winreg.OpenKey(hive, reg_path) as key:
                            version, _ = winreg.QueryValueEx(key, "version")
                    except OSError:
                        continue

                    match = re.match(r"(\d+)\.", str(version))
                    if match:
                        return int(match.group(1))
        except Exception as exc:
            logger.debug("browser_major_detection_failed_windows", extra={"error": str(exc)})
        return None

    # macOS / Linux: query common Chrome executable names.
    commands = []
    if system == "Darwin":
        commands.append(
            ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"]
        )
    else:
        commands.extend(
            [
                ["google-chrome", "--version"],
                ["google-chrome-stable", "--version"],
                ["chromium-browser", "--version"],
                ["chromium", "--version"],
            ]
        )

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except Exception:
            continue

        output = (result.stdout or result.stderr or "").strip()
        match = re.search(r"(\d+)\.", output)
        if match:
            return int(match.group(1))

    return None


def create_driver(config: BotConfig) -> uc.Chrome:
    """Create undetected Chrome driver with stealth mode.

    Args:
        config: Bot configuration

    Returns:
        Configured Chrome driver instance
    """
    def _build_options() -> uc.ChromeOptions:
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

        return options

    def _create_uc_driver(version_main: int | None = None) -> uc.Chrome:
        return uc.Chrome(options=_build_options(), version_main=version_main)

    browser_major = _detect_browser_major_version()
    if browser_major is not None:
        logger.info("browser_major_detected", extra={"browser_major": browser_major})

    try:
        driver = _create_uc_driver(version_main=browser_major)
    except SessionNotCreatedException as exc:
        # Handle Chrome/ChromeDriver major-version mismatch by retrying with
        # the browser's major version extracted from the Selenium error.
        error_text = str(exc)
        match = re.search(r"Current browser version is\s+(\d+)\.", error_text)
        if not match:
            logger.error(
                "driver_creation_failed_version_mismatch_unresolved",
                extra={"error": error_text},
                exc_info=True,
            )
            raise

        browser_major = int(match.group(1))
        logger.warning(
            "driver_retry_with_browser_major",
            extra={"browser_major": browser_major},
        )
        driver = _create_uc_driver(version_main=browser_major)
    except Exception as e:
        logger.error("driver_creation_failed", extra={"error": str(e)}, exc_info=True)
        raise

    try:
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
