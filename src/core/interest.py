"""Interest detection for evaluating if posts match bot's personality and topics."""

import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.core.llm import LLMClient, is_rate_limit_but_not_quota
from src.state.models import Post

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_rate_limit_but_not_quota),
    reraise=True,
)
async def check_interest(post: Post, config: BotConfig, llm_client: LLMClient) -> bool:
    """Check if post matches bot's interests and personality.

    Uses LLM to evaluate if a post aligns with the bot's configured personality,
    tone, style, and topics. Returns True if the post matches, False otherwise.

    Args:
        post: Post object to evaluate
        config: Bot configuration with personality settings
        llm_client: LLMClient instance for making API calls

    Returns:
        True if post matches bot's interests, False otherwise

    Raises:
        Exception: If LLM API call fails after retries
    """
    # Extract personality settings
    tone = config.personality.tone
    style = config.personality.style
    topics = ", ".join(config.personality.topics)

    # Build evaluation prompt
    prompt = f"""Evaluate if this Twitter/X post matches the bot's interests and personality:

Bot Personality:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Post:
- Author: {post.username} ({post.display_name or "N/A"})
- Content: "{post.text}"
- Engagement: {post.likes} likes, {post.retweets} retweets, {post.replies} replies
- Post Type: {post.post_type}

Does this post align with the bot's interests and personality?
Consider:
- Does the content relate to the bot's topics of interest?
- Does the tone/style match what the bot would engage with?
- Is this something the bot would find valuable or interesting?

Respond with only "YES" if it matches, or "NO" if it doesn't."""

    # Get LLM client (use primary provider)
    client = llm_client._get_client(llm_client.config.llm.provider)
    if not client:
        # Fallback to first available client
        client = llm_client.openai_client or llm_client.openrouter_client

    if not client:
        logger.warning(
            "no_client_for_interest_check",
            extra={"post_id": post.post_id, "username": post.username},
        )
        # Default to False on error (conservative approach)
        return False

    try:
        # Use low temperature for consistent binary classification
        response = await client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,  # Just need YES/NO
            temperature=0.1,  # Low temperature for consistent validation
        )

        result = response.choices[0].message.content.strip().upper()
        matches = result.startswith("YES")

        # Log token usage for analytics
        if response.usage:
            logger.info(
                "interest_check_completed",
                extra={
                    "post_id": post.post_id,
                    "username": post.username,
                    "matches": matches,
                    "provider": llm_client.config.llm.provider,
                    "model": llm_client.model,
                    "tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )

        return matches

    except Exception as e:
        logger.warning(
            "interest_check_failed",
            extra={
                "post_id": post.post_id,
                "username": post.username,
                "error": str(e),
            },
            exc_info=True,
        )
        # Default to False on error (conservative approach - don't react to unknown posts)
        return False
