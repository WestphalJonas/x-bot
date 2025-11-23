"""LLM integration for tweet generation with multi-provider support."""

import logging

from openai import AsyncOpenAI, AuthenticationError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.state.models import Post

logger = logging.getLogger(__name__)


def is_rate_limit_but_not_quota(exception: Exception) -> bool:
    """Check if exception is a rate limit error but not a quota error.

    Quota errors should not be retried, but rate limit errors should be.
    """
    if isinstance(exception, RateLimitError):
        error_msg = str(exception).lower()
        # Don't retry on quota errors
        if "quota" in error_msg or "insufficient_quota" in error_msg:
            return False
        # Retry on other rate limit errors
        return True
    return False


class LLMClient:
    """Multi-provider LLM client with automatic fallback support."""

    def __init__(
        self,
        config: BotConfig,
        openai_api_key: str | None = None,
        openrouter_api_key: str | None = None,
    ):
        """Initialize LLM client with provider support.

        Args:
            config: Bot configuration
            openai_api_key: OpenAI API key (optional)
            openrouter_api_key: OpenRouter API key (optional)
        """
        self.config = config
        self.model = config.llm.model
        self.max_tokens = config.llm.max_tokens
        self.temperature = config.llm.temperature

        # Initialize clients for each provider
        self.openai_client = None
        self.openrouter_client = None

        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)

        if openrouter_api_key:
            self.openrouter_client = AsyncOpenAI(
                api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"
            )

    def _get_client(self, provider: str) -> AsyncOpenAI | None:
        """Get client for specific provider.

        Args:
            provider: Provider name ("openai" or "openrouter")

        Returns:
            AsyncOpenAI client or None if not available
        """
        if provider == "openai":
            return self.openai_client
        elif provider == "openrouter":
            return self.openrouter_client
        return None

    async def generate_tweet(self, system_prompt: str) -> str:
        """Generate a tweet with automatic fallback to backup providers.

        Args:
            system_prompt: System prompt with personality and guidelines

        Returns:
            Generated tweet text

        Raises:
            Exception: If all providers fail
        """
        providers = [self.config.llm.provider]
        if self.config.llm.use_fallback:
            providers.extend(self.config.llm.fallback_providers)

        last_error = None

        for provider in providers:
            client = self._get_client(provider)
            if not client:
                logger.warning(
                    "provider_not_available",
                    extra={
                        "provider": provider,
                        "message": f"No API key provided for {provider}",
                    },
                )
                continue

            try:
                tweet = await self._generate_with_provider(
                    client, provider, system_prompt
                )
                logger.info("llm_success", extra={"provider": provider})
                return tweet
            except (AuthenticationError, RateLimitError) as e:
                last_error = e
                logger.warning(
                    "llm_provider_failed",
                    extra={
                        "provider": provider,
                        "error": str(e),
                        "fallback_to": providers[providers.index(provider) + 1]
                        if providers.index(provider) < len(providers) - 1
                        else None,
                    },
                )
                continue
            except Exception as e:
                last_error = e
                logger.warning(
                    "llm_provider_error",
                    extra={
                        "provider": provider,
                        "error": str(e),
                        "fallback_to": providers[providers.index(provider) + 1]
                        if providers.index(provider) < len(providers) - 1
                        else None,
                    },
                )
                continue

        # All providers failed
        if last_error:
            raise last_error
        raise RuntimeError(
            "No LLM providers available. Please configure at least one provider API key."
        )

    async def generate_inspiration_tweet(self, posts: list[Post]) -> str:
        """Generate a tweet inspired by a list of posts.

        Args:
            posts: List of Post objects to use as inspiration

        Returns:
            Generated tweet text
        """
        from src.core.prompts import INSPIRATION_TWEET_PROMPT

        # Format posts context
        posts_context = "\n\n".join(
            [f"Post {i+1}:\n{post.text}" for i, post in enumerate(posts)]
        )

        tone = self.config.personality.tone
        style = self.config.personality.style
        topics = ", ".join(self.config.personality.topics)

        user_prompt = INSPIRATION_TWEET_PROMPT.format(
            posts_context=posts_context,
            tone=tone,
            style=style,
            topics=topics,
        )

        # For OpenRouter, we might need to adjust the model name
        model = self.model
        if self.config.llm.provider == "openrouter":
            model = self.config.llm.model

        try:
            tweet = await self._generate_with_provider(
                self._get_client(self.config.llm.provider),
                self.config.llm.provider,
                user_prompt,
            )
        except Exception as e:
            logger.error(
                "inspiration_generation_failed",
                extra={"error": str(e), "provider": self.config.llm.provider},
            )
            # Try fallback if available
            if self.config.llm.fallback_provider:
                logger.info(
                    "trying_fallback_provider",
                    extra={"provider": self.config.llm.fallback_provider},
                )
                try:
                    tweet = await self._generate_with_provider(
                        self._get_client(self.config.llm.fallback_provider),
                        self.config.llm.fallback_provider,
                        user_prompt,
                    )
                except Exception as fallback_error:
                    logger.error(
                        "fallback_inspiration_generation_failed",
                        extra={"error": str(fallback_error)},
                    )
                    raise
            else:
                raise

        # Validate the generated tweet
        return self._validate_tweet(tweet)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_rate_limit_but_not_quota),
        reraise=True,
    )
    async def _generate_with_provider(
        self, client: AsyncOpenAI, provider: str, system_prompt: str
    ) -> str:
        """Generate tweet using specific provider.

        Args:
            client: AsyncOpenAI client instance
            provider: Provider name for logging
            system_prompt: System prompt with personality and guidelines

        Returns:
            Generated tweet text
        """
        from src.core.prompts import TWEET_GENERATION_PROMPT

        user_prompt = TWEET_GENERATION_PROMPT

        # For OpenRouter, we might need to adjust the model name
        model = self.model
        if provider == "openrouter":
            # OpenRouter model names might need prefix, but many work as-is
            # You can customize this if needed
            pass

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        tweet = response.choices[0].message.content.strip()

        # Log token usage
        if response.usage:
            logger.info(
                "llm_call",
                extra={
                    "provider": provider,
                    "model": model,
                    "tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )

        return tweet

    async def validate_tweet(self, tweet: str) -> tuple[bool, str]:
        """Validate tweet quality and brand alignment.

        Checks:
        - Length (min/max from config)
        - Basic quality (not empty, reasonable content)
        - On-brand check (matches personality)

        Args:
            tweet: Tweet text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Length check
        tweet_length = len(tweet)
        min_length = self.config.personality.min_tweet_length
        max_length = self.config.personality.max_tweet_length

        if tweet_length < min_length:
            return False, f"Tweet too short: {tweet_length} chars (min: {min_length})"

        if tweet_length > max_length:
            return False, f"Tweet too long: {tweet_length} chars (max: {max_length})"

        # Basic quality checks
        if not tweet.strip():
            return False, "Tweet is empty"

        # On-brand check using LLM
        # TODO: Implement at 10. src/core/evaluation.py
        try:
            is_on_brand = await self._check_brand_alignment(tweet)
            if not is_on_brand:
                return False, "Tweet does not align with brand personality"
        except Exception as e:
            logger.warning(
                "brand_check_failed",
                extra={"error": str(e), "tweet": tweet[:50]},
            )
            # Don't fail validation if brand check fails, just log warning
            pass

        return True, ""

    async def _check_brand_alignment(self, tweet: str) -> bool:
        """Check if tweet aligns with brand personality using LLM.

        Args:
            tweet: Tweet text to check

        Returns:
            True if tweet aligns with brand, False otherwise
        """
        tone = self.config.personality.tone
        style = self.config.personality.style
        topics = ", ".join(self.config.personality.topics)

        from src.core.prompts import BRAND_CHECK_PROMPT

        prompt = BRAND_CHECK_PROMPT.format(
            tone=tone,
            style=style,
            topics=topics,
            tweet=tweet,
        )

        # Use the primary provider for brand check
        client = self._get_client(self.config.llm.provider)
        if not client:
            # Fallback to first available client
            client = self.openai_client or self.openrouter_client

        if not client:
            logger.warning("no_client_for_brand_check")
            return True  # Default to True if no client available

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,  # Low temperature for consistent validation
            )

            result = response.choices[0].message.content.strip().upper()
            return result.startswith("YES")

        except Exception as e:
            logger.warning(
                "brand_alignment_check_failed",
                extra={"error": str(e)},
            )
            # Default to True if check fails to avoid blocking valid tweets
            return True
