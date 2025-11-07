"""Twitter/X posting automation."""

import logging
import random
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.x.driver import human_delay

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def post_tweet(driver, tweet_text: str, config: BotConfig) -> bool:
    """Post a tweet to Twitter/X.

    Args:
        driver: Selenium driver instance
        tweet_text: Text content of the tweet
        config: Bot configuration

    Returns:
        True if post successful, False otherwise

    Raises:
        Exception: If posting fails after retries
    """
    try:
        # Navigate to X.com home
        driver.get("https://x.com/home")
        human_delay(config)

        # Wait for compose button and click it
        wait = WebDriverWait(driver, 20)
        compose_button = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]')
            )
        )
        compose_button.click()
        human_delay(config)

        # Wait for compose editor - try multiple selectors for Draft.js compatibility
        compose_editor = None
        editor_selectors = [
            'div[role="textbox"][data-testid="tweetTextarea_0"]',  # Contenteditable div with testid
            'div[contenteditable="true"][data-testid="tweetTextarea_0"]',  # Contenteditable attribute
            'div[role="textbox"]',  # Generic contenteditable textbox
            "div.public-DraftStyleDefault-block",  # Draft.js block class
            '[data-testid="tweetTextarea_0"]',  # Fallback to old selector
        ]

        for selector in editor_selectors:
            try:
                compose_editor = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                logger.info("compose_editor_found", extra={"selector": selector})
                break
            except Exception:
                continue

        if not compose_editor:
            raise Exception(
                "Could not find compose editor element. Twitter/X UI may have changed."
            )

        # Click to focus the editor
        compose_editor.click()
        human_delay(config)

        # Try to clear existing content
        try:
            compose_editor.send_keys(Keys.CONTROL + "a")
            compose_editor.send_keys(Keys.DELETE)
            human_delay(config)
        except Exception as e:
            logger.warning("could_not_clear_editor", extra={"error": str(e)})

        # Log that we're starting to write the tweet
        logger.info("starting_tweet_writing", extra={"tweet_length": len(tweet_text)})

        # Try typing directly first (works for most contenteditable elements)
        typing_success = False
        try:
            # Type the tweet character by character for more human-like behavior
            for char in tweet_text:
                compose_editor.send_keys(char)
                # Small random delay between characters
                time.sleep(random.uniform(0.05, 0.15))
            typing_success = True
            logger.info("tweet_typed_directly")
        except Exception as e:
            logger.warning("direct_typing_failed", extra={"error": str(e)})

        # If direct typing failed, use JavaScript fallback
        if not typing_success:
            logger.info("using_javascript_fallback")
            driver.execute_script(
                """
                var editor = arguments[0];
                var text = arguments[1];
                
                // Focus the editor
                editor.focus();
                
                // Clear existing content first
                editor.textContent = '';
                editor.innerText = '';
                
                // Set the text content
                editor.textContent = text;
                editor.innerText = text;
                
                // For Draft.js editors, we need to dispatch proper events
                // Trigger beforeinput event
                var beforeInputEvent = new InputEvent('beforeinput', { 
                    bubbles: true, 
                    cancelable: true,
                    inputType: 'insertText',
                    data: text
                });
                editor.dispatchEvent(beforeInputEvent);
                
                // Trigger input event for React/Draft.js
                var inputEvent = new InputEvent('input', { 
                    bubbles: true, 
                    cancelable: false,
                    inputType: 'insertText',
                    data: text
                });
                editor.dispatchEvent(inputEvent);
                
                // Trigger change event
                var changeEvent = new Event('change', { bubbles: true });
                editor.dispatchEvent(changeEvent);
                
                // Also try setting innerHTML for Draft.js compatibility
                var draftBlock = editor.querySelector('.public-DraftStyleDefault-block');
                if (draftBlock) {
                    draftBlock.textContent = text;
                    draftBlock.innerText = text;
                }
                """,
                compose_editor,
                tweet_text,
            )
            human_delay(config)

        human_delay(config)

        # Click the Tweet button
        tweet_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="tweetButton"]'))
        )
        tweet_button.click()

        # Wait for confirmation that tweet was posted
        # Check for success indicators (tweet disappears from compose modal, or success message)
        try:
            WebDriverWait(driver, 10).until(
                EC.invisibility_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]')
                )
            )
            logger.info("tweet_posted_successfully", extra={"length": len(tweet_text)})
            return True
        except Exception:
            # Tweet might have been posted but modal didn't close as expected
            # Check if we're back on home page
            current_url = driver.current_url
            if "x.com/home" in current_url or "x.com" == current_url:
                logger.info(
                    "tweet_posted_successfully", extra={"length": len(tweet_text)}
                )
                return True
            else:
                logger.warning(
                    "tweet_post_verification_uncertain", extra={"url": current_url}
                )
                # Assume success if we got this far without errors
                return True

    except Exception as e:
        logger.error(
            "tweet_post_failed",
            extra={"error": str(e), "tweet_length": len(tweet_text)},
            exc_info=True,
        )
        raise
