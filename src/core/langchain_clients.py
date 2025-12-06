"""LangChain-based LLM and embedding clients with provider fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from src.core.config import BotConfig, EnvSettings
from src.web.data_tracker import log_token_usage

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Token usage metadata captured from provider responses."""

    provider: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    operation: str = "generate"


class ChatResult(BaseModel):
    """Normalized chat result with usage metadata."""

    content: str
    provider: str
    usage: TokenUsage | None = None


def _extract_str_content(content: Any) -> str:
    """Normalize LangChain message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        joined = " ".join([_extract_str_content(item) for item in content])
        return joined
    return str(content)


class LangChainLLM:
    """Unified LangChain client with provider fallback."""

    def __init__(self, config: BotConfig, env_settings: EnvSettings):
        self.config = config
        self.env_settings = env_settings

        # Optional LangSmith tracing
        if env_settings.get("LANGCHAIN_TRACING_V2"):
            os.environ["LANGCHAIN_TRACING_V2"] = str(
                env_settings.get("LANGCHAIN_TRACING_V2")
            )
        if env_settings.get("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = str(env_settings["LANGCHAIN_API_KEY"])
        if env_settings.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = str(env_settings["LANGCHAIN_PROJECT"])

        self._chat_clients: dict[str, BaseChatModel] = {}
        self._embedding_clients: dict[str, Embeddings] = {}

    # --- Provider factories -------------------------------------------------
    def _build_openai_chat(self) -> BaseChatModel | None:
        api_key = self.env_settings.get("OPENAI_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            api_key=api_key,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=30,
        )

    def _build_openrouter_chat(self) -> BaseChatModel | None:
        api_key = self.env_settings.get("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=30,
        )

    def _build_google_chat(self) -> BaseChatModel | None:
        api_key = self.env_settings.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        model_name = self._normalize_google_model(self.config.llm.model)
        return ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model_name,
            temperature=self.config.llm.temperature,
            max_output_tokens=self.config.llm.max_tokens,
        )

    def _build_anthropic_chat(self) -> BaseChatModel | None:
        api_key = self.env_settings.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        model_name = self._normalize_anthropic_model(self.config.llm.model)
        return ChatAnthropic(
            api_key=api_key,
            model=model_name,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=30,
        )

    def _get_chat(self, provider: str) -> BaseChatModel | None:
        if provider in self._chat_clients:
            return self._chat_clients[provider]

        builder_map = {
            "openai": self._build_openai_chat,
            "openrouter": self._build_openrouter_chat,
            "google": self._build_google_chat,
            "anthropic": self._build_anthropic_chat,
        }
        builder = builder_map.get(provider)
        if not builder:
            return None

        client = builder()
        if client:
            self._chat_clients[provider] = client
        return client

    # --- Embeddings ---------------------------------------------------------
    def _build_openai_embeddings(self) -> Embeddings | None:
        api_key = self.env_settings.get("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAIEmbeddings(
            api_key=api_key, model=self.config.llm.embedding_model, timeout=30
        )

    def _build_google_embeddings(self) -> Embeddings | None:
        api_key = self.env_settings.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        model_name = self._normalize_google_embedding_model(
            self.config.llm.embedding_model
        )
        return GoogleGenerativeAIEmbeddings(api_key=api_key, model=model_name)

    def _get_embeddings(self) -> Embeddings | None:
        provider = self.config.llm.embedding_provider
        if provider in self._embedding_clients:
            return self._embedding_clients[provider]

        builder_map = {
            "openai": self._build_openai_embeddings,
            "google": self._build_google_embeddings,
        }
        builder = builder_map.get(provider)
        if not builder:
            return None

        client = builder()
        if client:
            self._embedding_clients[provider] = client
        return client

    def get_embeddings(self) -> Embeddings | None:
        """Public accessor for embedding client (used by memory layer)."""
        return self._get_embeddings()

    # --- Public API ---------------------------------------------------------
    async def chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        operation: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Chat with fallback providers."""
        providers = [self.config.llm.provider]
        if self.config.llm.use_fallback:
            providers.extend(self.config.llm.fallback_providers)

        last_error: Exception | None = None

        for provider in providers:
            chat_client = self._get_chat(provider)
            if not chat_client:
                logger.warning(
                    "provider_unavailable",
                    extra={"provider": provider, "operation": operation},
                )
                continue

            try:
                messages: list[BaseMessage] = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=user_prompt))

                chat_kwargs: dict[str, Any] = {}
                tokens = max_tokens or self.config.llm.max_tokens
                if provider == "google":
                    chat_kwargs["max_output_tokens"] = tokens
                else:
                    chat_kwargs["max_tokens"] = tokens

                response = await chat_client.ainvoke(
                    messages,
                    temperature=temperature or self.config.llm.temperature,
                    **chat_kwargs,
                )

                content = _extract_str_content(response.content).strip()
                usage = self._extract_token_usage(provider, response, operation)

                if usage:
                    await self._log_tokens(usage)

                logger.info(
                    "llm_success",
                    extra={
                        "provider": provider,
                        "operation": operation,
                        "length": len(content),
                    },
                )
                return ChatResult(content=content, provider=provider, usage=usage)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "llm_provider_failed",
                    extra={
                        "provider": provider,
                        "operation": operation,
                        "error": str(exc),
                    },
                )
                continue

        if last_error:
            raise last_error
        raise RuntimeError("No LLM providers available")

    async def embed_text(self, text: str) -> list[float]:
        """Embed text using the configured embedding provider."""
        embeddings = self._get_embeddings()
        if not embeddings:
            raise RuntimeError("Embedding provider not configured")

        # Prefer async embedding when available
        if hasattr(embeddings, "aembed_query"):
            return await embeddings.aembed_query(text)  # type: ignore[attr-defined]

        return await asyncio.to_thread(embeddings.embed_query, text)

    async def _log_tokens(self, usage: TokenUsage) -> None:
        try:
            await log_token_usage(
                provider=usage.provider,
                model=usage.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                operation=usage.operation,
            )
        except Exception as exc:
            logger.debug("token_logging_failed", extra={"error": str(exc)})

    # --- Helpers ------------------------------------------------------------
    def _extract_token_usage(
        self,
        provider: str,
        response: Any,
        operation: str,
    ) -> TokenUsage | None:
        """Extract token usage from LangChain responses (best effort)."""
        metadata = getattr(response, "response_metadata", {}) or {}
        usage_meta = metadata.get("token_usage") or metadata.get("usage") or {}

        prompt_tokens = int(usage_meta.get("prompt_tokens", 0))
        completion_tokens = int(
            usage_meta.get("completion_tokens", usage_meta.get("output_tokens", 0))
        )
        total_tokens = int(
            usage_meta.get("total_tokens", prompt_tokens + completion_tokens)
        )

        # Gemini rarely returns usage; estimate if missing
        if provider == "google" and total_tokens == 0:
            prompt_tokens = max(len(str(response)) // 4, 0)
            completion_tokens = max(len(_extract_str_content(response.content)) // 4, 0)
            total_tokens = prompt_tokens + completion_tokens

        if total_tokens == 0:
            return None

        return TokenUsage(
            provider=provider,
            model=self._model_for_provider(provider),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            operation=operation,
        )

    def _normalize_google_model(self, model: str) -> str:
        if model.startswith("google/"):
            return model.split("google/", maxsplit=1)[1]
        if model.startswith("gemini"):
            return model
        return "gemini-1.5-flash"

    def _normalize_google_embedding_model(self, model: str) -> str:
        if model.startswith("models/"):
            return model
        return f"models/{model}"

    def _normalize_anthropic_model(self, model: str) -> str:
        if model.startswith("anthropic/"):
            return model.split("anthropic/", maxsplit=1)[1]
        if model.startswith("claude"):
            return model
        return "claude-3-haiku-20240307"

    def _model_for_provider(self, provider: str) -> str:
        if provider == "google":
            return self._normalize_google_model(self.config.llm.model)
        if provider == "anthropic":
            return self._normalize_anthropic_model(self.config.llm.model)
        return self.config.llm.model
