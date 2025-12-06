"""LLM integration for tweet generation with multi-provider support."""

import logging
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, AuthenticationError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core.config import BotConfig
from src.core.prompts import (
    BRAND_CHECK_PROMPT,
    INSPIRATION_TWEET_PROMPT,
    TWEET_GENERATION_PROMPT,
)
from src.state.models import Post
from src.web.data_tracker import log_token_usage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Optional imports for additional providers
try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


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
    """Multi-provider LLM client with automatic fallback support.

    Supports the following providers:
    - OpenAI (openai)
    - OpenRouter (openrouter)
    - Google Gemini (google) - requires google-generativeai package
    - Anthropic Claude (anthropic) - requires anthropic package
    """

    def __init__(
        self,
        config: BotConfig,
        openai_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        google_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        """Initialize LLM client with multi-provider support.

        Args:
            config: Bot configuration
            openai_api_key: OpenAI API key (optional)
            openrouter_api_key: OpenRouter API key (optional)
            google_api_key: Google AI API key (optional)
            anthropic_api_key: Anthropic API key (optional)
        """
        self.config = config
        self.model = config.llm.model
        self.max_tokens = config.llm.max_tokens
        self.temperature = config.llm.temperature

        # Create clients up front so fallback order reuses the same connections
        self.openai_client: AsyncOpenAI | None = None
        self.openrouter_client: AsyncOpenAI | None = None
        self.google_model: Any = None
        self.anthropic_client: Any = None

        self._google_api_key = google_api_key

        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)

        if openrouter_api_key:
            self.openrouter_client = AsyncOpenAI(
                api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"
            )

        if google_api_key and GOOGLE_AVAILABLE and genai is not None:
            genai.configure(api_key=google_api_key)
            # Resolve Gemini model once so we don't recompute it per request
            google_model_name = self._get_google_model_name()
            self.google_model = genai.GenerativeModel(google_model_name)
            logger.info("google_client_initialized", extra={"model": google_model_name})

        if anthropic_api_key and ANTHROPIC_AVAILABLE and AsyncAnthropic is not None:
            self.anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
            logger.info("anthropic_client_initialized")

    def _get_google_model_name(self) -> str:
        """Get the Google model name from config or default."""
        model = self.model
        # If model is in format "google/gemini-1.5-flash", extract the model name
        if model.startswith("google/"):
            return model.replace("google/", "")
        # If model is a Google model name, use it directly
        if model.startswith("gemini"):
            return model
        # Default to gemini-1.5-flash
        return "gemini-1.5-flash"

    def _get_anthropic_model_name(self) -> str:
        """Get the Anthropic model name from config or default."""
        model = self.model
        # If model is in format "anthropic/claude-3-haiku", extract the model name
        if model.startswith("anthropic/"):
            return model.replace("anthropic/", "")
        # If model is a Claude model name, use it directly
        if model.startswith("claude"):
            return model
        # Default to claude-3-haiku
        return "claude-3-haiku-20240307"

    def _get_client(self, provider: str) -> AsyncOpenAI | None:
        """Get OpenAI-compatible client for specific provider.

        Args:
            provider: Provider name ("openai", "openrouter", "google", "anthropic")

        Returns:
            AsyncOpenAI client or None if not available.
            For Google and Anthropic, returns None (handled separately).
        """
        if provider == "openai":
            return self.openai_client
        elif provider == "openrouter":
            return self.openrouter_client
        # Google and Anthropic use their own clients, not OpenAI-compatible
        return None

    def _has_provider(self, provider: str) -> bool:
        """Check if a provider is available.

        Args:
            provider: Provider name

        Returns:
            True if provider is configured and available
        """
        if provider == "openai":
            return self.openai_client is not None
        elif provider == "openrouter":
            return self.openrouter_client is not None
        elif provider == "google":
            return self.google_model is not None
        elif provider == "anthropic":
            return self.anthropic_client is not None
        return False

    async def close(self) -> None:
        """Close all async clients to prevent event loop errors."""
        if self.openai_client:
            await self.openai_client.close()
        if self.openrouter_client:
            await self.openrouter_client.close()
        if self.anthropic_client:
            await self.anthropic_client.close()

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

        # Format the generation prompt with character length requirements
        user_prompt = TWEET_GENERATION_PROMPT.format(
            min_tweet_length=self.config.personality.min_tweet_length,
            max_tweet_length=self.config.personality.max_tweet_length,
        )

        for provider in providers:
            if not self._has_provider(provider):
                logger.warning(
                    "provider_not_available",
                    extra={
                        "provider": provider,
                        "message": f"No API key provided for {provider}",
                    },
                )
                continue

            try:
                # Native clients use provider-specific paths; OpenAI-compatible ones share the same flow
                if provider == "google":
                    tweet = await self._generate_with_google(
                        system_prompt, user_prompt, "generate"
                    )
                elif provider == "anthropic":
                    tweet = await self._generate_with_anthropic(
                        system_prompt, user_prompt, "generate"
                    )
                else:
                    # OpenAI-compatible providers
                    client = self._get_client(provider)
                    if client:
                        tweet = await self._generate_with_provider(
                            client,
                            provider,
                            system_prompt,
                            user_prompt,
                            operation="generate",
                        )
                    else:
                        continue

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
        # Get system prompt with personality settings
        system_prompt = self.config.get_system_prompt()

        # Format posts context
        posts_context = "\n\n".join(
            [f"Post {i + 1}:\n{post.text}" for i, post in enumerate(posts)]
        )

        tone = self.config.personality.tone
        style = self.config.personality.style
        topics = ", ".join(self.config.personality.topics)

        user_prompt = INSPIRATION_TWEET_PROMPT.format(
            posts_context=posts_context,
            tone=tone,
            style=style,
            topics=topics,
            min_tweet_length=self.config.personality.min_tweet_length,
            max_tweet_length=self.config.personality.max_tweet_length,
        )

        # Try primary provider first
        providers = [self.config.llm.provider]
        if self.config.llm.use_fallback:
            providers.extend(self.config.llm.fallback_providers)

        last_error = None

        for provider in providers:
            if not self._has_provider(provider):
                logger.warning(
                    "provider_not_available",
                    extra={
                        "provider": provider,
                        "message": f"No API key provided for {provider}",
                    },
                )
                continue

            try:
                # Native clients use provider-specific paths; OpenAI-compatible ones share the same flow
                if provider == "google":
                    tweet = await self._generate_with_google(
                        system_prompt, user_prompt, "inspiration"
                    )
                elif provider == "anthropic":
                    tweet = await self._generate_with_anthropic(
                        system_prompt, user_prompt, "inspiration"
                    )
                else:
                    # OpenAI-compatible providers
                    client = self._get_client(provider)
                    if client:
                        tweet = await self._generate_with_provider(
                            client,
                            provider,
                            system_prompt,
                            user_prompt=user_prompt,
                            operation="inspiration",
                        )
                    else:
                        continue

                logger.info("inspiration_llm_success", extra={"provider": provider})
                return tweet.strip()
            except Exception as e:
                last_error = e
                logger.warning(
                    "inspiration_provider_failed",
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
            "No LLM providers available for inspiration generation. Please configure at least one provider API key."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_rate_limit_but_not_quota),
        reraise=True,
    )
    async def _generate_with_provider(
        self,
        client: AsyncOpenAI,
        provider: str,
        system_prompt: str,
        user_prompt: str | None = None,
        operation: str = "generate",
    ) -> str:
        """Generate tweet using specific provider.

        Args:
            client: AsyncOpenAI client instance
            provider: Provider name for logging
            system_prompt: System prompt with personality and guidelines
            user_prompt: Optional user prompt (defaults to formatted TWEET_GENERATION_PROMPT)
            operation: Operation type for tracking (generate, inspiration, etc.)

        Returns:
            Generated tweet text
        """
        if user_prompt is None:
            user_prompt = TWEET_GENERATION_PROMPT.format(
                min_tweet_length=self.config.personality.min_tweet_length,
                max_tweet_length=self.config.personality.max_tweet_length,
            )

        model = self.model

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
                    "operation": operation,
                },
            )

            # Log to dashboard tracker
            try:
                await log_token_usage(
                    provider=provider,
                    model=model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    operation=operation,
                )
            except Exception as e:
                # Don't fail the generation if tracking fails
                logger.debug("token_tracking_failed", extra={"error": str(e)})

        return tweet

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _generate_with_google(
        self,
        system_prompt: str,
        user_prompt: str,
        operation: str = "generate",
    ) -> str:
        """Generate tweet using Google Gemini.

        Args:
            system_prompt: System prompt with personality and guidelines
            user_prompt: User prompt for generation
            operation: Operation type for tracking

        Returns:
            Generated tweet text

        Raises:
            RuntimeError: If Google client is not available
        """
        if not self.google_model or not GOOGLE_AVAILABLE:
            raise RuntimeError("Google Gemini client not available")

        # Gemini expects a single prompt string; combine system + user content
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = await self.google_model.generate_content_async(
            full_prompt,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        tweet = response.text.strip()
        model_name = self._get_google_model_name()

        logger.info(
            "llm_call",
            extra={
                "provider": "google",
                "model": model_name,
                "operation": operation,
            },
        )

        try:
            from src.web.data_tracker import log_token_usage

            # Gemini doesn't return token counts; approximate for analytics only
            prompt_tokens = len(full_prompt) // 4
            completion_tokens = len(tweet) // 4

            await log_token_usage(
                provider="google",
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                operation=operation,
            )
        except Exception as e:
            logger.debug("token_tracking_failed", extra={"error": str(e)})

        return tweet

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _generate_with_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        operation: str = "generate",
    ) -> str:
        """Generate tweet using Anthropic Claude.

        Args:
            system_prompt: System prompt with personality and guidelines
            user_prompt: User prompt for generation
            operation: Operation type for tracking

        Returns:
            Generated tweet text

        Raises:
            RuntimeError: If Anthropic client is not available
        """
        if not self.anthropic_client or not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic client not available")

        model_name = self._get_anthropic_model_name()

        response = await self.anthropic_client.messages.create(
            model=model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        tweet = response.content[0].text.strip()

        logger.info(
            "llm_call",
            extra={
                "provider": "anthropic",
                "model": model_name,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "operation": operation,
            },
        )

        try:
            from src.web.data_tracker import log_token_usage

            await log_token_usage(
                provider="anthropic",
                model=model_name,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                operation=operation,
            )
        except Exception as e:
            logger.debug("token_tracking_failed", extra={"error": str(e)})

        return tweet

    async def validate_tweet(self, tweet: str) -> tuple[bool, str]:
        """Deterministic validation (length/empty) before LLM re-evaluation.

        Args:
            tweet: Tweet text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Fast length guard before running more expensive checks
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

        prompt = BRAND_CHECK_PROMPT.format(
            tone=tone,
            style=style,
            topics=topics,
            tweet=tweet,
        )

        # Prefer the configured provider; fall back to the first available client
        client = self._get_client(self.config.llm.provider)
        if not client:
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

            # Log token usage for validation
            if response.usage:
                try:
                    from src.web.data_tracker import log_token_usage

                    await log_token_usage(
                        provider=self.config.llm.provider,
                        model=self.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        operation="validate",
                    )
                except Exception:
                    pass  # Don't fail validation if tracking fails

            return result.startswith("YES")

        except Exception as e:
            logger.warning(
                "brand_alignment_check_failed",
                extra={"error": str(e)},
            )
            # Default to True if check fails to avoid blocking valid tweets
            return True
