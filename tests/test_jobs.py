"""Integration tests for job functions."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.core.config import BotConfig
from src.scheduler.jobs import (
    post_autonomous_tweet,
    read_frontpage_posts,
    check_notifications,
)


@pytest.fixture
def config():
    """Create test config."""
    return BotConfig.load("config/config.yaml")


@pytest.fixture
def env_settings():
    """Create test environment settings."""
    return {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENROUTER_API_KEY": "test-openrouter-key",
        "TWITTER_USERNAME": "test_user",
        "TWITTER_PASSWORD": "test_password",
    }


@pytest.fixture
def mock_state():
    """Create mock state."""
    state = Mock()
    state.counters = {"posts_today": 0, "replies_today": 0}
    state.last_post_time = None
    return state


@patch("src.scheduler.jobs.save_state")
@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.post_tweet")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_success(
    mock_llm_client_class,
    mock_post_tweet,
    mock_load_cookies,
    mock_create_driver,
    mock_load_state,
    mock_save_state,
    config,
    env_settings,
    mock_state,
):
    """Test successful tweet posting."""
    # Setup mocks
    mock_load_state.return_value = mock_state

    mock_llm = Mock()
    mock_llm.generate_tweet = AsyncMock(return_value="Test tweet content")
    mock_llm.validate_tweet = AsyncMock(return_value=(True, ""))
    mock_llm_client_class.return_value = mock_llm

    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True
    mock_post_tweet.return_value = True

    # Run job
    post_autonomous_tweet(config, env_settings)

    # Verify
    mock_load_state.assert_called_once()
    mock_llm.generate_tweet.assert_called_once()
    mock_llm.validate_tweet.assert_called_once()
    mock_load_cookies.assert_called_once()
    mock_post_tweet.assert_called_once()
    mock_save_state.assert_called_once()
    assert mock_state.counters["posts_today"] == 1


@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_rate_limit_exceeded(
    mock_llm_client_class,
    mock_load_state,
    config,
    env_settings,
    mock_state,
):
    """Test rate limit prevents posting."""
    # Setup: rate limit exceeded
    mock_state.counters["posts_today"] = config.rate_limits.max_posts_per_day
    mock_load_state.return_value = mock_state

    # Run job
    post_autonomous_tweet(config, env_settings)

    # Verify: should return early without posting
    mock_load_state.assert_called_once()
    # LLM should not be called
    mock_llm_client_class.assert_not_called()


@patch("src.scheduler.jobs.save_state")
@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.login")
@patch("src.scheduler.jobs.save_cookies")
@patch("src.scheduler.jobs.post_tweet")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_login_required(
    mock_llm_client_class,
    mock_post_tweet,
    mock_save_cookies,
    mock_login,
    mock_load_cookies,
    mock_create_driver,
    mock_load_state,
    mock_save_state,
    config,
    env_settings,
    mock_state,
):
    """Test login when cookies not available."""
    # Setup mocks
    mock_load_state.return_value = mock_state

    mock_llm = Mock()
    mock_llm.generate_tweet = AsyncMock(return_value="Test tweet")
    mock_llm.validate_tweet = AsyncMock(return_value=(True, ""))
    mock_llm_client_class.return_value = mock_llm

    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = False  # Cookies not available
    mock_login.return_value = True
    mock_post_tweet.return_value = True

    # Run job
    post_autonomous_tweet(config, env_settings)

    # Verify login was called
    mock_login.assert_called_once()
    mock_save_cookies.assert_called_once()


@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_validation_failed(
    mock_llm_client_class,
    mock_load_state,
    config,
    env_settings,
    mock_state,
):
    """Test tweet validation failure."""
    # Setup mocks
    mock_load_state.return_value = mock_state

    mock_llm = Mock()
    mock_llm.generate_tweet = AsyncMock(return_value="Invalid tweet")
    mock_llm.validate_tweet = AsyncMock(return_value=(False, "Tweet too short"))
    mock_llm_client_class.return_value = mock_llm

    # Run job - should catch and log error, not crash scheduler
    # The ValueError is caught inside post_autonomous_tweet and logged
    post_autonomous_tweet(config, env_settings)

    # Verify validation was called
    mock_llm.validate_tweet.assert_called_once()


@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_llm_error(
    mock_llm_client_class,
    mock_load_state,
    config,
    env_settings,
    mock_state,
):
    """Test LLM error handling."""
    # Setup mocks
    mock_load_state.return_value = mock_state

    mock_llm = Mock()
    mock_llm.generate_tweet = AsyncMock(side_effect=Exception("LLM API error"))
    mock_llm_client_class.return_value = mock_llm

    # Run job - should catch and log error, not crash
    post_autonomous_tweet(config, env_settings)

    # Job should complete (error logged but doesn't crash scheduler)
    mock_load_state.assert_called_once()


@patch("src.scheduler.jobs.asyncio.run")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.login")
@patch("src.scheduler.jobs.save_cookies")
@patch("src.scheduler.jobs.read_posts_from_frontpage")
def test_read_frontpage_posts_success(
    mock_read_posts,
    mock_save_cookies,
    mock_login,
    mock_load_cookies,
    mock_create_driver,
    mock_asyncio_run,
    config,
    env_settings,
):
    """Test read_frontpage_posts job implementation."""
    from src.state.models import Post

    # Setup mocks
    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True
    mock_read_posts.return_value = [
        Post(
            text="Test tweet",
            username="@testuser",
            display_name="Test User",
            post_id="123456",
            likes=100,
            retweets=50,
            replies=25,
        )
    ]

    # Run job
    read_frontpage_posts(config, env_settings)

    # Verify
    mock_asyncio_run.assert_called_once()
    # The async function should be called by asyncio.run
    call_args = mock_asyncio_run.call_args[0][0]
    assert call_args is not None


def test_check_notifications_stub(config, env_settings):
    """Test check_notifications stub implementation."""
    # Should not raise exception
    check_notifications(config, env_settings)


@patch("src.scheduler.jobs.save_state")
@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.post_tweet")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_driver_cleanup(
    mock_llm_client_class,
    mock_post_tweet,
    mock_load_cookies,
    mock_create_driver,
    mock_load_state,
    mock_save_state,
    config,
    env_settings,
    mock_state,
):
    """Test driver is properly cleaned up."""
    # Setup mocks
    mock_load_state.return_value = mock_state

    mock_llm = Mock()
    mock_llm.generate_tweet = AsyncMock(return_value="Test tweet")
    mock_llm.validate_tweet = AsyncMock(return_value=(True, ""))
    mock_llm_client_class.return_value = mock_llm

    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True
    mock_post_tweet.return_value = True

    # Run job
    post_autonomous_tweet(config, env_settings)

    # Verify driver was quit
    mock_driver.quit.assert_called_once()


@patch("src.scheduler.jobs.save_state")
@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.post_tweet")
@patch("src.scheduler.jobs.LLMClient")
def test_post_autonomous_tweet_driver_cleanup_on_error(
    mock_llm_client_class,
    mock_post_tweet,
    mock_load_cookies,
    mock_create_driver,
    mock_load_state,
    mock_save_state,
    config,
    env_settings,
    mock_state,
):
    """Test driver is cleaned up even on error."""
    # Setup mocks
    mock_load_state.return_value = mock_state

    mock_llm = Mock()
    mock_llm.generate_tweet = AsyncMock(return_value="Test tweet")
    mock_llm.validate_tweet = AsyncMock(return_value=(True, ""))
    mock_llm_client_class.return_value = mock_llm

    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True
    mock_post_tweet.side_effect = Exception("Posting failed")

    # Run job - should handle error gracefully
    post_autonomous_tweet(config, env_settings)

    # Verify driver was still quit even on error
    mock_driver.quit.assert_called_once()

