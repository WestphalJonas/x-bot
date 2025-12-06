"""LangChain-powered LLM orchestration with provider fallback."""

from __future__ import annotations

import logging

from src.core.config import BotConfig, EnvSettings
from src.core.langchain_clients import ChatResult, LangChainLLM
from src.core.prompts import (
    BRAND_CHECK_PROMPT,
    INSPIRATION_TWEET_PROMPT,
    TWEET_GENERATION_PROMPT,
)
from src.state.models import Post

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

    async def generate_tweet(self, system_prompt: str) -> str:
        """Generate a tweet with fallback providers."""
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

    async def generate_inspiration_tweet(self, posts: list[Post]) -> str:
        """Generate a tweet inspired by a list of posts."""
        posts_context = "\n\n".join(
            [f"Post {i + 1}:\n{post.text}" for i, post in enumerate(posts)]
        )

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

    async def close(self) -> None:
        """Compatibility close hook (LangChain clients are lazy)."""
        # Nothing to close explicitly; hook retained for API parity.
        return
