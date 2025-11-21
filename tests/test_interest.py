"""Unit tests for interest detection."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import BotConfig
from src.core.interest import check_interest
from src.core.llm import LLMClient
from src.state.models import Post


@pytest.fixture
def config():
    """Create test config."""
    return BotConfig.load("config/config.yaml")


@pytest.fixture
def mock_llm_client(config):
    """Create mock LLM client."""
    client = Mock(spec=LLMClient)
    client.config = config
    client.model = config.llm.model
    client.openai_client = Mock()
    client.openrouter_client = None
    client._get_client = Mock(return_value=client.openai_client)
    return client


@pytest.fixture
def sample_post():
    """Create sample post for testing."""
    return Post(
        text="This is a great post about AI and machine learning!",
        username="@testuser",
        display_name="Test User",
        post_id="123456",
        post_type="text_only",
        likes=100,
        retweets=50,
        replies=25,
    )


@pytest.fixture
def non_matching_post():
    """Create non-matching post for testing."""
    return Post(
        text="Check out this amazing recipe for chocolate cake!",
        username="@cooking",
        display_name="Cooking Blog",
        post_id="789012",
        post_type="text_only",
        likes=200,
        retweets=30,
        replies=15,
    )


@pytest.mark.asyncio
async def test_check_interest_matches(mock_llm_client, sample_post, config):
    """Test interest check returns True for matching post."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "YES"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 50
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    result = await check_interest(sample_post, config, mock_llm_client)

    assert result is True
    mock_llm_client.openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_check_interest_no_match(mock_llm_client, non_matching_post, config):
    """Test interest check returns False for non-matching post."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "NO"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 50
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    result = await check_interest(non_matching_post, config, mock_llm_client)

    assert result is False
    mock_llm_client.openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_check_interest_case_insensitive(mock_llm_client, sample_post, config):
    """Test interest check handles case-insensitive YES/NO responses."""
    # Mock LLM response with lowercase
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "yes"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 50
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    result = await check_interest(sample_post, config, mock_llm_client)

    assert result is True


@pytest.mark.asyncio
async def test_check_interest_no_client_available(mock_llm_client, sample_post, config):
    """Test interest check returns False when no LLM client available."""
    mock_llm_client._get_client = Mock(return_value=None)
    mock_llm_client.openai_client = None
    mock_llm_client.openrouter_client = None

    result = await check_interest(sample_post, config, mock_llm_client)

    assert result is False


@pytest.mark.asyncio
async def test_check_interest_api_error(mock_llm_client, sample_post, config):
    """Test interest check returns False on API error."""
    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error")
    )

    result = await check_interest(sample_post, config, mock_llm_client)

    assert result is False


@pytest.mark.asyncio
async def test_check_interest_with_quoted_tweet(mock_llm_client, config):
    """Test interest check with quoted tweet."""
    quoted_post = Post(
        text="Original tweet about technology",
        username="@original",
        display_name="Original User",
        post_id="111111",
        post_type="text_only",
    )

    post_with_quote = Post(
        text="This is interesting!",
        username="@testuser",
        display_name="Test User",
        post_id="222222",
        post_type="quoted",
        likes=50,
        retweets=10,
        replies=5,
    )

    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "YES"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 50
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    result = await check_interest(post_with_quote, config, mock_llm_client)

    assert result is True


@pytest.mark.asyncio
async def test_check_interest_logs_token_usage(mock_llm_client, sample_post, config):
    """Test that token usage is logged."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "YES"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 100
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 20

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    with patch("src.core.interest.logger") as mock_logger:
        await check_interest(sample_post, config, mock_llm_client)

        # Check that info was logged with token usage
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args
        assert "interest_check_completed" in str(call_args)
        assert "tokens" in call_args.kwargs.get("extra", {})


@pytest.mark.asyncio
async def test_check_interest_uses_low_temperature(mock_llm_client, sample_post, config):
    """Test that interest check uses low temperature for consistent results."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "YES"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 50
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    await check_interest(sample_post, config, mock_llm_client)

    # Verify temperature is low (0.1)
    call_kwargs = mock_llm_client.openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["max_tokens"] == 10


@pytest.mark.asyncio
async def test_check_interest_includes_post_context(mock_llm_client, sample_post, config):
    """Test that prompt includes post context (author, engagement, type)."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "YES"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 50
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10

    mock_llm_client.openai_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    await check_interest(sample_post, config, mock_llm_client)

    # Verify prompt includes post information
    call_args = mock_llm_client.openai_client.chat.completions.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]

    assert sample_post.username in prompt
    assert sample_post.text in prompt
    assert str(sample_post.likes) in prompt
    assert sample_post.post_type in prompt

