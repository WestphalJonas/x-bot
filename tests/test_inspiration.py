"""Unit tests for inspiration-based posting."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import BotConfig
from src.core.llm import LLMClient
from src.scheduler.jobs import process_inspiration_queue
from src.state.models import Post


@pytest.fixture
def config():
    """Create test config."""
    config = BotConfig.load("config/config.yaml")
    # Ensure scheduler config is initialized
    if not config.scheduler:
        from src.core.config import SchedulerConfig
        config.scheduler = SchedulerConfig()
    return config


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = Mock()
    state = Mock()
    state.interesting_posts_queue = []
    state.counters = {"posts_today": 0}
    manager.load_state = AsyncMock(return_value=state)
    manager.save_state = AsyncMock()
    return manager


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock(spec=LLMClient)
    client.generate_inspiration_tweet = AsyncMock(return_value="Inspired tweet")
    return client


@pytest.fixture
def mock_driver_manager():
    """Create mock driver manager."""
    return Mock()


@pytest.mark.asyncio
async def test_process_inspiration_queue_below_threshold(
    config, mock_state_manager, mock_llm_client, mock_driver_manager
):
    """Test that queue is not processed if below threshold."""
    # Setup state with few posts
    state = await mock_state_manager.load_state()
    state.interesting_posts_queue = [{"post_id": "1", "text": "test"}]
    
    await process_inspiration_queue(
        config, mock_state_manager, mock_llm_client, mock_driver_manager
    )
    
    # Should not generate tweet
    mock_llm_client.generate_inspiration_tweet.assert_not_called()


@pytest.mark.asyncio
async def test_process_inspiration_queue_success(
    config, mock_state_manager, mock_llm_client, mock_driver_manager
):
    """Test successful processing of inspiration queue."""
    # Setup state with enough posts
    state = await mock_state_manager.load_state()
    # Create 15 dummy posts
    state.interesting_posts_queue = [
        {
            "post_id": str(i),
            "text": f"Post {i}",
            "username": "user",
            "display_name": "User",
            "post_type": "text_only",
            "likes": 0,
            "retweets": 0,
            "replies": 0,
            "timestamp": None,
            "url": None,
            "is_interesting": True
        }
        for i in range(15)
    ]
    
    # Mock create_driver and post_tweet
    with patch("src.scheduler.jobs.create_driver") as mock_create_driver:
        with patch("src.scheduler.jobs.load_cookies", return_value=True):
            with patch("src.x.posting.post_tweet", return_value=True) as mock_post_tweet:
                
                await process_inspiration_queue(
                    config, mock_state_manager, mock_llm_client, mock_driver_manager
                )
                
                # Should generate tweet
                mock_llm_client.generate_inspiration_tweet.assert_called_once()
                
                # Should post tweet
                mock_post_tweet.assert_called_once()
                
                # Should update state
                mock_state_manager.save_state.assert_called_once()
                
                # Verify queue was trimmed (15 - 10 = 5 remaining)
                saved_state = mock_state_manager.save_state.call_args[0][0]
                assert len(saved_state.interesting_posts_queue) == 5
                assert saved_state.interesting_posts_queue[0]["post_id"] == "10"


@pytest.mark.asyncio
async def test_process_inspiration_queue_posting_failure(
    config, mock_state_manager, mock_llm_client, mock_driver_manager
):
    """Test handling of posting failure."""
    # Setup state with enough posts
    state = await mock_state_manager.load_state()
    state.interesting_posts_queue = [
        {
            "post_id": str(i),
            "text": f"Post {i}",
            "username": "user",
            "display_name": "User",
            "post_type": "text_only",
            "likes": 0,
            "retweets": 0,
            "replies": 0,
            "timestamp": None,
            "url": None,
            "is_interesting": True
        }
        for i in range(15)
    ]
    
    # Mock create_driver and post_tweet failure
    with patch("src.scheduler.jobs.create_driver"):
        with patch("src.scheduler.jobs.load_cookies", return_value=True):
            with patch("src.x.posting.post_tweet", return_value=False):
                
                await process_inspiration_queue(
                    config, mock_state_manager, mock_llm_client, mock_driver_manager
                )
                
                # Should generate tweet
                mock_llm_client.generate_inspiration_tweet.assert_called_once()
                
                # Should NOT save state (queue remains full)
                mock_state_manager.save_state.assert_not_called()
