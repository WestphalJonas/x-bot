"""Unit tests for PostParser."""

from unittest.mock import Mock

import pytest

from src.x.parser import PostParser


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
    assert PostParser.extract_number_from_text("1.2K") == 1200
    assert PostParser.extract_number_from_text("500") == 500
    assert PostParser.extract_number_from_text("2M") == 2000000
    assert PostParser.extract_number_from_text("1.5K") == 1500
    assert PostParser.extract_number_from_text("") == 0
    assert PostParser.extract_number_from_text(None) == 0
    assert PostParser.extract_number_from_text("abc") == 0


def test_extract_post_text(mock_post_element):
    """Test post text extraction."""
    # Mock text element
    text_element = Mock()
    text_element.text = "This is a test tweet"
    mock_post_element.find_elements = Mock(return_value=[text_element])

    text = PostParser.extract_post_text(mock_post_element)
    assert text == "This is a test tweet"


def test_extract_post_text_fallback(mock_post_element):
    """Test post text extraction fallback."""
    # No text elements found, use fallback
    mock_post_element.find_elements = Mock(return_value=[])
    mock_post_element.text = "This is a test tweet\n@testuser\n1.2K"

    text = PostParser.extract_post_text(mock_post_element)
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
    # Configure links to return spans
    display_name_span = Mock()
    display_name_span.text = "Test User"
    display_name_elem.find_elements = Mock(return_value=[display_name_span])
    display_name_elem.get_attribute = Mock(return_value="https://x.com/testuser")

    username_span = Mock()
    username_span.text = "@testuser"
    username_elem.find_elements = Mock(return_value=[username_span])
    username_elem.get_attribute = Mock(return_value="https://x.com/testuser")

    user_name_container.find_elements = Mock(
        side_effect=[
            [display_name_elem, username_elem],  # First call for links
            [],  # Second call for username spans
        ]
    )
    def side_effect(by, value):
        if "User-Name" in value or "User-Names" in value:
            return [user_name_container]
        return []

    mock_post_element.find_elements = Mock(side_effect=side_effect)

    username, display_name = PostParser.extract_author_info(mock_post_element)
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

    username, display_name = PostParser.extract_author_info(mock_post_element)
    assert username == "@testuser"
    assert display_name is None


def test_extract_author_info_fallback(mock_post_element):
    """Test author info extraction fallback."""
    mock_post_element.find_elements = Mock(return_value=[])
    mock_post_element.text = "Some text @testuser more text"

    username, display_name = PostParser.extract_author_info(mock_post_element)
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
            [reply_button],  # First call for reply buttons
            [retweet_button],  # Second call for retweet buttons
            [like_button],  # Third call for like buttons
        ]
    )

    likes, retweets, replies = PostParser.extract_engagement_metrics(mock_post_element)
    assert likes >= 1200
    assert retweets >= 500
    assert replies >= 200


def test_extract_post_id(mock_post_element):
    """Test post ID extraction."""
    # Mock link element
    link_element = Mock()
    link_element.get_attribute = Mock(return_value="https://x.com/user/status/1234567890")
    mock_post_element.find_elements = Mock(return_value=[link_element])

    post_id = PostParser.extract_post_id(mock_post_element)
    assert post_id == "1234567890"


def test_extract_post_url(mock_post_element):
    """Test post URL extraction."""
    # Mock link element
    link_element = Mock()
    link_element.get_attribute = Mock(return_value="https://x.com/user/status/1234567890")
    mock_post_element.find_elements = Mock(return_value=[link_element])

    url = PostParser.extract_post_url(mock_post_element)
    assert url == "https://x.com/user/status/1234567890"
