"""LangChain-powered LLM orchestration with provider fallback."""

from __future__ import annotations

import json
import logging

from src.core.config import BotConfig, EnvSettings
from src.core.langchain_clients import ChatResult, LangChainLLM
from src.core.prompts import (
    BRAND_CHECK_PROMPT,
    INSPIRATION_TWEET_PROMPT,
    INSPIRATION_TWEET_WITH_CONTEXT_PROMPT,
    NOTIFICATION_INTENT_PROMPT,
    REPLY_GENERATION_PROMPT,
    TWEET_GENERATION_PROMPT,
    TWEET_GENERATION_WITH_CONTEXT_PROMPT,
)
from src.state.models import Notification, Post

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level LLM helper built on LangChain with fallback and token logging."""

    def __init__(self, config: BotConfig, env_settings: EnvSettings):
        self.config = config
        self.env_settings = env_settings
        self._client = LangChainLLM(config=config, env_settings=env_settings)

    async def chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        operation: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """General chat entrypoint that leverages provider fallback."""
        return await self._client.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            operation=operation,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_tweet(
        self, system_prompt: str, recent_tweets: list[str] | None = None
    ) -> str:
        """Generate a tweet with fallback providers."""
        if recent_tweets:
            user_prompt = TWEET_GENERATION_WITH_CONTEXT_PROMPT.format(
                recent_tweets=self._format_recent_tweets(recent_tweets),
                min_tweet_length=self.config.personality.min_tweet_length,
                max_tweet_length=self.config.personality.max_tweet_length,
            )
        else:
            user_prompt = TWEET_GENERATION_PROMPT.format(
                min_tweet_length=self.config.personality.min_tweet_length,
                max_tweet_length=self.config.personality.max_tweet_length,
            )

        result = await self._client.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            operation="generate",
            max_tokens=self.config.llm.max_tokens,
        )
        return result.content

    async def generate_inspiration_tweet(
        self, posts: list[Post], recent_tweets: list[str] | None = None
    ) -> str:
        """Generate a tweet inspired by a list of posts."""
        posts_context = "\n\n".join(
            [f"Post {i + 1}:\n{post.text}" for i, post in enumerate(posts)]
        )

        if recent_tweets:
            user_prompt = INSPIRATION_TWEET_WITH_CONTEXT_PROMPT.format(
                posts_context=posts_context,
                recent_tweets=self._format_recent_tweets(recent_tweets),
                tone=self.config.personality.tone,
                style=self.config.personality.style,
                topics=", ".join(self.config.personality.topics),
                min_tweet_length=self.config.personality.min_tweet_length,
                max_tweet_length=self.config.personality.max_tweet_length,
            )
        else:
            user_prompt = INSPIRATION_TWEET_PROMPT.format(
                posts_context=posts_context,
                tone=self.config.personality.tone,
                style=self.config.personality.style,
                topics=", ".join(self.config.personality.topics),
                min_tweet_length=self.config.personality.min_tweet_length,
                max_tweet_length=self.config.personality.max_tweet_length,
            )

        system_prompt = self.config.get_system_prompt()
        result = await self._client.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            operation="inspiration",
            max_tokens=self.config.llm.max_tokens,
        )
        return result.content.strip()

    async def validate_tweet(self, tweet: str) -> tuple[bool, str]:
        """Deterministic validation before expensive LLM checks."""
        tweet_length = len(tweet)
        min_length = self.config.personality.min_tweet_length
        max_length = self.config.personality.max_tweet_length

        if tweet_length < min_length:
            return False, f"Tweet too short: {tweet_length} chars (min: {min_length})"

        if tweet_length > max_length:
            return False, f"Tweet too long: {tweet_length} chars (max: {max_length})"

        if not tweet.strip():
            return False, "Tweet is empty"

        return True, ""

    async def check_brand_alignment(self, tweet: str) -> bool:
        """Check if a tweet aligns with configured tone/style/topics."""
        prompt = BRAND_CHECK_PROMPT.format(
            tone=self.config.personality.tone,
            style=self.config.personality.style,
            topics=", ".join(self.config.personality.topics),
            tweet=tweet,
        )

        result: ChatResult = await self._client.chat(
            user_prompt=prompt,
            system_prompt=None,
            operation="brand_check",
            max_tokens=10,
            temperature=0.1,
        )

        return result.content.upper().startswith("YES")

    async def embed_text(self, text: str) -> list[float]:
        """Get embedding for text using configured embedding provider."""
        return await self._client.embed_text(text)

    async def classify_notification_intent(
        self, notification: Notification
    ) -> tuple[bool, str]:
        """Classify whether a notification merits a reply."""
        prompt = NOTIFICATION_INTENT_PROMPT.format(
            tone=self.config.personality.tone,
            style=self.config.personality.style,
            topics=", ".join(self.config.personality.topics),
            notification_type=notification.type,
            from_username=notification.from_username,
            notification_text=notification.text,
            original_post_text=notification.original_post_text or "N/A",
        )

        try:
            result: ChatResult = await self._client.chat(
                user_prompt=prompt,
                system_prompt=None,
                operation="intent_check",
                temperature=0.2,
                max_tokens=80,
            )
            data = json.loads(result.content)
            positive = bool(data.get("positive", False))
            reason = str(data.get("reason", "")).strip() or "No reason provided"
            logger.info(
                "notification_intent_classified",
                extra={
                    "notification_type": notification.type,
                    "from_username": notification.from_username,
                    "positive": positive,
                },
            )
            return positive, reason
        except Exception as exc:
            logger.warning(
                "notification_intent_parse_failed",
                extra={
                    "error": str(exc),
                    "notification_type": notification.type,
                    "from_username": notification.from_username,
                },
                exc_info=True,
            )
            return False, "Intent classification failed"

    async def generate_reply(self, notification: Notification) -> str:
        """Generate a reply to a notification."""
        user_prompt = REPLY_GENERATION_PROMPT.format(
            from_username=notification.from_username,
            notification_type=notification.type,
            notification_text=notification.text,
            original_post_text=notification.original_post_text or "N/A",
            tone=self.config.personality.tone,
            style=self.config.personality.style,
            topics=", ".join(self.config.personality.topics),
            min_tweet_length=self.config.personality.min_tweet_length,
            max_tweet_length=self.config.personality.max_tweet_length,
        )

        system_prompt = self.config.get_system_prompt()
        result = await self._client.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            operation="reply",
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        reply_text = result.content.strip()
        logger.info(
            "reply_generated",
            extra={
                "from_username": notification.from_username,
                "length": len(reply_text),
            },
        )
        return reply_text

    @staticmethod
    def _format_recent_tweets(recent_tweets: list[str]) -> str:
        """Format recent tweets into a readable block."""
        return "\n".join([f"- {tweet}" for tweet in recent_tweets])

    async def close(self) -> None:
        """Close underlying HTTP clients to avoid loop shutdown warnings."""
        try:
            await self._client.aclose()
        except Exception:
            pass
