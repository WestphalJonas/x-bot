"""Constants and configuration limits for the X bot."""


class QueueLimits:
    """Maximum sizes for various queues and logs stored in state."""

    INTERESTING_POSTS = 50
    NOTIFICATIONS = 50
    PROCESSED_NOTIFICATION_IDS = 100
    REJECTED_TWEETS = 50
    TOKEN_USAGE_LOG = 100
    WRITTEN_TWEETS = 50


class RetryLimits:
    """Retry configuration for various operations."""

    MAX_ATTEMPTS = 3
    MIN_WAIT_SECONDS = 2
    MAX_WAIT_SECONDS = 10
    MULTIPLIER = 1


class BrowserConfig:
    """Browser automation constants."""

    # Default user agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # Wait timeouts
    DEFAULT_WAIT_TIMEOUT = 20
    SHORT_WAIT_TIMEOUT = 5
    LONG_WAIT_TIMEOUT = 30

    # Scroll settings
    SCROLL_AMOUNT = 500
    MAX_SCROLL_ATTEMPTS = 10


class TwitterURLs:
    """Twitter/X URLs used throughout the application."""

    BASE = "https://x.com"
    HOME = "https://x.com/home"
    LOGIN = "https://x.com/i/flow/login"
    NOTIFICATIONS = "https://x.com/notifications"


class DefaultCounts:
    """Default counts for various operations."""

    POSTS_TO_READ = 10
    NOTIFICATIONS_TO_CHECK = 20
    SIMILAR_POSTS_RESULTS = 5
