"""Interest detection for evaluating if posts match bot's personality and topics."""

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.core.llm import LLMClient
from src.core.prompts import INTEREST_CHECK_PROMPT
from src.state.models import Post

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def check_interest(post: Post, config: BotConfig, llm_client: LLMClient) -> bool:
    """Check if a post matches configured interests using the LLM."""
    prompt = INTEREST_CHECK_PROMPT.format(
        tone=config.personality.tone,
        style=config.personality.style,
        topics=", ".join(config.personality.topics),
        username=post.username or "unknown",
        text=post.text,
        likes=post.likes or 0,
        retweets=post.retweets or 0,
    )

    try:
        result = await llm_client.chat(
            user_prompt=prompt,
            system_prompt=None,
            operation="interest_check",
            temperature=0.1,
            max_tokens=10,
        )
        matches = result.content.upper().startswith("YES")

        logger.info(
            "interest_check_completed",
            extra={
                "post_id": post.post_id,
                "username": post.username,
                "matches": matches,
                "provider": result.provider,
            },
        )
        return matches

    except Exception as exc:
        logger.warning(
            "interest_check_failed",
            extra={
                "post_id": post.post_id,
                "username": post.username,
                "error": str(exc),
            },
            exc_info=True,
        )
        return False
