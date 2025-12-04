"""Interest detection for evaluating if posts match bot's personality and topics."""

import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.core.llm import LLMClient, is_rate_limit_but_not_quota
from src.core.prompts import INTEREST_CHECK_PROMPT
from src.state.models import Post
from src.web.data_tracker import log_token_usage

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

    # Build evaluation prompt using template
    prompt = INTEREST_CHECK_PROMPT.format(
        tone=tone,
        style=style,
        topics=topics,
        username=post.username or "unknown",
        text=post.text,
        likes=post.likes or 0,
        retweets=post.retweets or 0,
    )

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
            status = "MATCH" if matches else "NO_MATCH"
            logger.info(
                f"interest_check_completed_{status}",
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

            # Log to dashboard tracker
            try:
                await log_token_usage(
                    provider=llm_client.config.llm.provider,
                    model=llm_client.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    operation="interest_check",
                )
            except Exception:
                pass  # Don't fail interest check if tracking fails

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
