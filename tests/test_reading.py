"""Unit tests for reading module."""

from unittest.mock import Mock, patch

import pytest

from src.core.config import BotConfig
from src.state.models import Post
from src.x.reading import read_frontpage_posts


@pytest.fixture
def config():
    """Create test config."""
    return BotConfig.load("config/config.yaml")


@pytest.fixture
def mock_driver():
    """Create mock Selenium driver."""
    driver = Mock()
    driver.get = Mock()
    driver.find_elements = Mock(return_value=[])
    driver.execute_script = Mock()
    driver.current_url = "https://x.com/home"
    return driver


@patch("src.x.reading.WebDriverWait")
@patch("src.x.reading.human_delay")
def test_read_frontpage_posts_success(mock_delay, mock_wait_class, mock_driver, config):
    """Test successful post reading."""
    # Mock WebDriverWait
    mock_wait = Mock()
    mock_wait_class.return_value = mock_wait

    # Mock post element
    post_element = Mock()
    post_element.text = "Test tweet\n@testuser\n1.2K\n500\n200"
    post_element.find_elements = Mock(return_value=[])
    post_element.get_attribute = Mock(return_value=None)
    post_element.find_element = Mock(side_effect=Exception("Not found"))

    # Mock driver find_elements to return posts
    mock_driver.find_elements = Mock(return_value=[post_element])

    # Mock expected conditions
    mock_wait.until = Mock(return_value=post_element)

    # Mock PostParser methods
    with patch("src.x.reading.PostParser") as MockParser:
        MockParser.extract_post_text.return_value = "Test tweet"
        MockParser.extract_author_info.return_value = ("@testuser", "Test User")
        MockParser.extract_post_id.return_value = "123456"
        MockParser.extract_engagement_metrics.return_value = (1200, 500, 200)
        MockParser.extract_post_url.return_value = "https://x.com/testuser/status/123456"
        MockParser.detect_post_type.return_value = "text_only"
        MockParser.extract_timestamp.return_value = None

        posts = read_frontpage_posts(mock_driver, config, count=1)

        assert len(posts) == 1
        assert isinstance(posts[0], Post)
        assert posts[0].text == "Test tweet"
        assert posts[0].username == "@testuser"
        assert posts[0].display_name == "Test User"
        assert posts[0].post_id == "123456"
        assert posts[0].likes == 1200
        assert posts[0].retweets == 500
        assert posts[0].replies == 200


@patch("src.x.reading.WebDriverWait")
@patch("src.x.reading.human_delay")
def test_read_frontpage_posts_timeout(mock_delay, mock_wait_class, mock_driver, config):
    """Test post reading with timeout."""
    from selenium.common.exceptions import TimeoutException

    # Mock WebDriverWait to raise TimeoutException
    mock_wait = Mock()
    mock_wait_class.return_value = mock_wait
    mock_wait.until = Mock(side_effect=TimeoutException("Timeout"))

    # Should return empty list on timeout
    posts = read_frontpage_posts(mock_driver, config, count=10)
    assert posts == []


@patch("src.x.reading.WebDriverWait")
@patch("src.x.reading.human_delay")
def test_read_frontpage_posts_scrolling(mock_delay, mock_wait_class, mock_driver, config):
    """Test post reading with scrolling."""
    # Mock WebDriverWait
    mock_wait = Mock()
    mock_wait_class.return_value = mock_wait

    # Mock post element
    post_element = Mock()
    post_element.text = "Test tweet"
    post_element.find_elements = Mock(return_value=[])
    post_element.get_attribute = Mock(return_value=None)
    post_element.find_element = Mock(side_effect=Exception("Not found"))

    # Mock driver to return fewer posts initially, then more after scroll
    call_count = [0]

    def mock_find_elements(selector):
        call_count[0] += 1
        if call_count[0] == 1:
            return [post_element]  # First call: 1 post
        else:
            return [post_element, post_element]  # After scroll: 2 posts

    mock_driver.find_elements = Mock(side_effect=mock_find_elements)
    mock_wait.until = Mock(return_value=post_element)

    # Mock PostParser methods
    with patch("src.x.reading.PostParser") as MockParser:
        MockParser.extract_post_text.return_value = "Test tweet"
        MockParser.extract_author_info.return_value = ("@testuser", None)
        MockParser.extract_post_id.return_value = "123456"
        MockParser.extract_engagement_metrics.return_value = (0, 0, 0)
        MockParser.extract_post_url.return_value = None
        MockParser.detect_post_type.return_value = "text_only"
        MockParser.extract_timestamp.return_value = None

        with patch("time.sleep"):  # Mock sleep to speed up test
            posts = read_frontpage_posts(mock_driver, config, count=2)

            # Should have scrolled and found posts
            assert mock_driver.execute_script.called


@patch("src.x.reading.WebDriverWait")
@patch("src.x.reading.human_delay")
def test_read_frontpage_posts_duplicate_filtering(mock_delay, mock_wait_class, mock_driver, config):
    """Test duplicate post filtering."""
    # Mock WebDriverWait
    mock_wait = Mock()
    mock_wait_class.return_value = mock_wait

    # Mock post element (same ID)
    post_element = Mock()
    post_element.text = "Test tweet"
    post_element.find_elements = Mock(return_value=[])
    post_element.get_attribute = Mock(return_value=None)
    post_element.find_element = Mock(side_effect=Exception("Not found"))

    mock_driver.find_elements = Mock(return_value=[post_element, post_element])
    mock_wait.until = Mock(return_value=post_element)

    # Mock PostParser methods - same post ID for both
    with patch("src.x.reading.PostParser") as MockParser:
        MockParser.extract_post_text.return_value = "Test tweet"
        MockParser.extract_author_info.return_value = ("@testuser", None)
        MockParser.extract_post_id.return_value = "123456"  # Same ID
        MockParser.extract_engagement_metrics.return_value = (0, 0, 0)
        MockParser.extract_post_url.return_value = None
        MockParser.detect_post_type.return_value = "text_only"
        MockParser.extract_timestamp.return_value = None

        posts = read_frontpage_posts(mock_driver, config, count=10)

        # Should filter duplicates
        assert len(posts) == 1
