"""Tweet re-evaluation before posting."""

import json
import logging

from src.core.config import BotConfig
from src.core.llm import LLMClient
from src.core.prompts import RE_EVALUATION_PROMPT

logger = logging.getLogger(__name__)


def _parse_evaluation_response(raw_response: str) -> tuple[bool, str]:
    """Parse LLM evaluation response into (approve, reason)."""
    cleaned = raw_response.strip()

    # Remove common code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "approve" in parsed:
            approve = bool(parsed.get("approve", False))
            reason = str(parsed.get("reason") or "").strip() or "No reason provided"
            return approve, reason
    except json.JSONDecodeError:
        pass

    # Fallback parsing for YES/NO style answers
    upper = cleaned.upper()
    if upper.startswith("YES") or upper.startswith("APPROVE"):
        reason = cleaned.partition("\n")[2].strip() or "Approved"
        return True, reason
    if upper.startswith("NO") or upper.startswith("REJECT"):
        reason = cleaned.partition("\n")[2].strip() or "Rejected"
        return False, reason

    return False, "Could not parse evaluation response"


async def re_evaluate_tweet(
    tweet_text: str,
    config: BotConfig,
    llm_client: LLMClient,
    operation: str = "autonomous",
) -> tuple[bool, str]:
    """Run a final LLM-based gatekeeper check before posting."""
    system_prompt = config.get_system_prompt()
    user_prompt = RE_EVALUATION_PROMPT.format(
        tone=config.personality.tone,
        style=config.personality.style,
        topics=", ".join(config.personality.topics),
        tweet=tweet_text,
    )

    try:
        result = await llm_client.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            operation="re_evaluate",
            max_tokens=config.llm.max_tokens,
            temperature=0.1,
        )
        approve, reason = _parse_evaluation_response(result.content)
        logger.info(
            "tweet_re_evaluated",
            extra={"provider": result.provider, "approved": approve},
        )
        return approve, reason
    except Exception as exc:
        logger.error(
            "re_evaluate_failed",
            extra={"error": str(exc)},
            exc_info=True,
        )
        raise

