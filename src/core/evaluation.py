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
    """Run a final LLM-based gatekeeper check before posting.

    Args:
        tweet_text: Tweet text to evaluate
        config: Bot configuration
        llm_client: Initialized LLM client with providers
        operation: Operation context for logging (autonomous, inspiration, reply)

    Returns:
        Tuple of (approve, reason)
    """
    system_prompt = config.get_system_prompt()
    user_prompt = RE_EVALUATION_PROMPT.format(
        tone=config.personality.tone,
        style=config.personality.style,
        topics=", ".join(config.personality.topics),
        tweet=tweet_text,
    )

    providers = [config.llm.provider]
    if config.llm.use_fallback:
        providers.extend(config.llm.fallback_providers)

    last_error: Exception | None = None

    for provider in providers:
        if not llm_client._has_provider(provider):
            logger.warning(
                "re_evaluate_provider_unavailable", extra={"provider": provider}
            )
            continue

        try:
            if provider == "google":
                raw_response = await llm_client._generate_with_google(
                    system_prompt, user_prompt, operation="re_evaluate"
                )
            elif provider == "anthropic":
                raw_response = await llm_client._generate_with_anthropic(
                    system_prompt, user_prompt, operation="re_evaluate"
                )
            else:
                client = llm_client._get_client(provider)
                if not client:
                    continue
                raw_response = await llm_client._generate_with_provider(
                    client=client,
                    provider=provider,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    operation="re_evaluate",
                )

            approve, reason = _parse_evaluation_response(raw_response)
            logger.info(
                "tweet_re_evaluated",
                extra={"provider": provider, "approved": approve},
            )
            return approve, reason

        except Exception as exc:
            last_error = exc
            logger.warning(
                "re_evaluate_attempt_failed",
                extra={"provider": provider, "error": str(exc)},
            )
            continue

    if last_error:
        raise last_error

    raise RuntimeError("No LLM providers available for re-evaluation.")

