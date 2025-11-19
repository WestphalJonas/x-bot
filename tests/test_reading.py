"""Unit tests for reading module."""

from unittest.mock import Mock, MagicMock, patch

import pytest

from src.core.config import BotConfig
from src.state.models import Post
from src.x.reading import (
    _extract_author_info,
    _extract_engagement_metrics,
    _extract_number_from_text,
    _extract_post_id,
    _extract_post_text,
    _extract_post_url,
    read_frontpage_posts,
)


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


@pytest.fixture
def mock_post_element():
    """Create mock post element."""
    element = Mock()
    element.text = "This is a test tweet\n@testuser\n1.2K\n500\n200"
    element.find_elements = Mock(return_value=[])
    element.get_attribute = Mock(return_value=None)
    return element


def test_extract_number_from_text():
    """Test number extraction from text."""
    assert _extract_number_from_text("1.2K") == 1200
    assert _extract_number_from_text("500") == 500
    assert _extract_number_from_text("2M") == 2000000
    assert _extract_number_from_text("1.5K") == 1500
    assert _extract_number_from_text("") == 0
    assert _extract_number_from_text(None) == 0
    assert _extract_number_from_text("abc") == 0


def test_extract_post_text(mock_post_element):
    """Test post text extraction."""
    # Mock text element
    text_element = Mock()
    text_element.text = "This is a test tweet"
    mock_post_element.find_elements = Mock(return_value=[text_element])

    text = _extract_post_text(mock_post_element)
    assert text == "This is a test tweet"


def test_extract_post_text_fallback(mock_post_element):
    """Test post text extraction fallback."""
    # No text elements found, use fallback
    mock_post_element.find_elements = Mock(return_value=[])
    mock_post_element.text = "This is a test tweet\n@testuser\n1.2K"

    text = _extract_post_text(mock_post_element)
    # Should filter out engagement metrics and author
    assert "test tweet" in text.lower()


def test_extract_author_info(mock_post_element):
    """Test author info extraction."""
    # Mock User-Name container with both display name and username
    user_name_container = Mock()
    display_name_elem = Mock()
    display_name_elem.text = "Test User"
    username_elem = Mock()
    username_elem.text = "@testuser"
    user_name_container.find_elements = Mock(
        side_effect=[
            [display_name_elem, username_elem],  # First call for spans
            [],  # Second call for links
        ]
    )
    mock_post_element.find_elements = Mock(return_value=[user_name_container])

    username, display_name = _extract_author_info(mock_post_element)
    assert username == "@testuser"
    assert display_name == "Test User"


def test_extract_author_info_from_link(mock_post_element):
    """Test author info extraction from link."""
    # Mock link element
    link_element = Mock()
    link_element.get_attribute = Mock(return_value="https://x.com/testuser")
    span_element = Mock()
    span_element.find_element = Mock(return_value=link_element)
    mock_post_element.find_elements = Mock(
        side_effect=[
            [],  # No User-Name container
            [span_element],  # Fallback selector
        ]
    )

    username, display_name = _extract_author_info(mock_post_element)
    assert username == "@testuser"
    assert display_name is None


def test_extract_author_info_fallback(mock_post_element):
    """Test author info extraction fallback."""
    mock_post_element.find_elements = Mock(return_value=[])
    mock_post_element.text = "Some text @testuser more text"

    username, display_name = _extract_author_info(mock_post_element)
    assert username == "@testuser"
    assert display_name is None


def test_extract_engagement_metrics(mock_post_element):
    """Test engagement metrics extraction."""
    # Mock button elements
    like_button = Mock()
    like_button.get_attribute = Mock(return_value="Like 1.2K")
    like_button.text = "1.2K"

    retweet_button = Mock()
    retweet_button.get_attribute = Mock(return_value="Retweet 500")
    retweet_button.text = "500"

    reply_button = Mock()
    reply_button.get_attribute = Mock(return_value="Reply 200")
    reply_button.text = "200"

    mock_post_element.find_elements = Mock(
        side_effect=[
            [like_button],  # First call for like buttons
            [retweet_button],  # Second call for retweet buttons
            [reply_button],  # Third call for reply buttons
        ]
    )

    likes, retweets, replies = _extract_engagement_metrics(mock_post_element)
    assert likes >= 1200
    assert retweets >= 500
    assert replies >= 200


def test_extract_post_id(mock_post_element):
    """Test post ID extraction."""
    # Mock link element
    link_element = Mock()
    link_element.get_attribute = Mock(return_value="https://x.com/user/status/1234567890")
    mock_post_element.find_elements = Mock(return_value=[link_element])

    post_id = _extract_post_id(mock_post_element)
    assert post_id == "1234567890"


def test_extract_post_url(mock_post_element):
    """Test post URL extraction."""
    # Mock link element
    link_element = Mock()
    link_element.get_attribute = Mock(return_value="https://x.com/user/status/1234567890")
    mock_post_element.find_elements = Mock(return_value=[link_element])

    url = _extract_post_url(mock_post_element)
    assert url == "https://x.com/user/status/1234567890"


def test_extract_post_url_from_id(mock_post_element):
    """Test post URL construction from ID."""
    # Mock post ID extraction
    with patch("src.x.reading._extract_post_id", return_value="1234567890"):
        with patch("src.x.reading._extract_author", return_value="@testuser"):
            url = _extract_post_url(mock_post_element)
            assert url == "https://x.com/testuser/status/1234567890"


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

    # Mock extraction functions
    with patch("src.x.reading._extract_post_text", return_value="Test tweet"):
        with patch("src.x.reading._extract_author_info", return_value=("@testuser", "Test User")):
            with patch("src.x.reading._extract_post_id", return_value="123456"):
                with patch("src.x.reading._extract_engagement_metrics", return_value=(1200, 500, 200)):
                    with patch("src.x.reading._extract_post_url", return_value="https://x.com/testuser/status/123456"):
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
    import time

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

    # Mock extraction functions
    with patch("src.x.reading._extract_post_text", return_value="Test tweet"):
        with patch("src.x.reading._extract_author_info", return_value=("@testuser", None)):
            with patch("src.x.reading._extract_post_id", return_value="123456"):
                with patch("src.x.reading._extract_engagement_metrics", return_value=(0, 0, 0)):
                    with patch("src.x.reading._extract_post_url", return_value=None):
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

    # Mock extraction functions - same post ID for both
    with patch("src.x.reading._extract_post_text", return_value="Test tweet"):
        with patch("src.x.reading._extract_author_info", return_value=("@testuser", None)):
            with patch("src.x.reading._extract_post_id", return_value="123456"):  # Same ID
                with patch("src.x.reading._extract_engagement_metrics", return_value=(0, 0, 0)):
                    with patch("src.x.reading._extract_post_url", return_value=None):
                        posts = read_frontpage_posts(mock_driver, config, count=10)

                        # Should filter duplicates
                        assert len(posts) == 1

