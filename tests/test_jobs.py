"""Integration tests for job functions."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.core.config import BotConfig
from src.scheduler.jobs import (
    post_autonomous_tweet,
    read_frontpage_posts,
    check_notifications,
    _read_frontpage_posts_async,
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


@pytest.mark.asyncio
@patch("src.scheduler.jobs.save_state")
@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.read_posts_from_frontpage")
@patch("src.scheduler.jobs.check_interest")
@patch("src.scheduler.jobs.LLMClient")
async def test_read_frontpage_posts_with_interest_detection(
    mock_llm_client_class,
    mock_check_interest,
    mock_read_posts,
    mock_load_cookies,
    mock_create_driver,
    mock_load_state,
    mock_save_state,
    config,
    env_settings,
):
    """Test interest detection integration in reading job."""
    from src.state.models import Post, AgentState

    # Setup mocks
    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True

    # Create test posts
    post1 = Post(
        text="AI and machine learning are fascinating!",
        username="@techuser",
        display_name="Tech User",
        post_id="111",
        likes=100,
        retweets=50,
        replies=25,
    )
    post2 = Post(
        text="Check out this recipe for chocolate cake!",
        username="@cooking",
        display_name="Cooking Blog",
        post_id="222",
        likes=200,
        retweets=30,
        replies=15,
    )
    mock_read_posts.return_value = [post1, post2]

    # Mock LLM client
    mock_llm = Mock()
    mock_llm_client_class.return_value = mock_llm

    # Mock interest detection: first post interesting, second not
    async def check_interest_side_effect(post, config, llm_client):
        return post.post_id == "111"

    mock_check_interest.side_effect = check_interest_side_effect

    # Mock state
    mock_state = AgentState()
    mock_load_state.return_value = mock_state

    # Run async function
    await _read_frontpage_posts_async(config, env_settings)

    # Verify interest detection was called for each post
    assert mock_check_interest.call_count == 2

    # Verify posts have is_interesting flag set
    assert post1.is_interesting is True
    assert post2.is_interesting is False

    # Verify interesting posts were added to queue
    assert len(mock_state.interesting_posts_queue) == 1
    assert mock_state.interesting_posts_queue[0]["post_id"] == "111"

    # Verify state was saved
    mock_save_state.assert_called_once()


@pytest.mark.asyncio
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.read_posts_from_frontpage")
@patch("src.scheduler.jobs.check_interest")
@patch("src.scheduler.jobs.LLMClient")
async def test_read_frontpage_posts_interest_check_error_handling(
    mock_llm_client_class,
    mock_check_interest,
    mock_read_posts,
    mock_load_cookies,
    mock_create_driver,
    config,
    env_settings,
):
    """Test error handling when interest check fails."""
    from src.state.models import Post

    # Setup mocks
    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True

    post = Post(
        text="Test tweet",
        username="@testuser",
        display_name="Test User",
        post_id="123",
        likes=100,
    )
    mock_read_posts.return_value = [post]

    # Mock LLM client
    mock_llm = Mock()
    mock_llm_client_class.return_value = mock_llm

    # Mock interest check to raise exception
    mock_check_interest.side_effect = Exception("LLM API error")

    # Run async function - should handle error gracefully
    await _read_frontpage_posts_async(config, env_settings)

    # Verify post is marked as not interesting on error
    assert post.is_interesting is False

    # Verify driver was cleaned up
    mock_driver.quit.assert_called_once()


@pytest.mark.asyncio
@patch("src.scheduler.jobs.save_state")
@patch("src.scheduler.jobs.load_state")
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.read_posts_from_frontpage")
@patch("src.scheduler.jobs.check_interest")
@patch("src.scheduler.jobs.LLMClient")
async def test_read_frontpage_posts_queue_size_limit(
    mock_llm_client_class,
    mock_check_interest,
    mock_read_posts,
    mock_load_cookies,
    mock_create_driver,
    mock_load_state,
    mock_save_state,
    config,
    env_settings,
):
    """Test queue size limit enforcement."""
    from src.state.models import Post, AgentState

    # Setup mocks
    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True

    # Create 60 posts (more than max queue size of 50)
    posts = [
        Post(
            text=f"Interesting post {i}",
            username=f"@user{i}",
            post_id=str(i),
            likes=100,
        )
        for i in range(60)
    ]
    mock_read_posts.return_value = posts

    # Mock LLM client
    mock_llm = Mock()
    mock_llm_client_class.return_value = mock_llm

    # All posts are interesting
    mock_check_interest.return_value = True

    # Mock state with existing queue (40 posts)
    mock_state = AgentState()
    mock_state.interesting_posts_queue = [
        {"post_id": f"existing_{i}"} for i in range(40)
    ]
    mock_load_state.return_value = mock_state

    # Run async function
    await _read_frontpage_posts_async(config, env_settings)

    # Verify queue is trimmed to max size (50)
    assert len(mock_state.interesting_posts_queue) == 50

    # Verify state was saved
    mock_save_state.assert_called_once()


@pytest.mark.asyncio
@patch("src.scheduler.jobs.create_driver")
@patch("src.scheduler.jobs.load_cookies")
@patch("src.scheduler.jobs.read_posts_from_frontpage")
async def test_read_frontpage_posts_no_llm_provider(
    mock_read_posts,
    mock_load_cookies,
    mock_create_driver,
    config,
    env_settings,
):
    """Test reading job when no LLM provider is configured."""
    from src.state.models import Post

    # Setup mocks
    mock_driver = Mock()
    mock_create_driver.return_value = mock_driver
    mock_load_cookies.return_value = True

    post = Post(
        text="Test tweet",
        username="@testuser",
        post_id="123",
    )
    mock_read_posts.return_value = [post]

    # Remove LLM API keys from env_settings
    env_settings_no_llm = env_settings.copy()
    env_settings_no_llm["OPENAI_API_KEY"] = None
    env_settings_no_llm["OPENROUTER_API_KEY"] = None

    # Run async function
    await _read_frontpage_posts_async(config, env_settings_no_llm)

    # Verify posts are marked as not evaluated (None)
    assert post.is_interesting is None

    # Verify driver was cleaned up
    mock_driver.quit.assert_called_once()


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

