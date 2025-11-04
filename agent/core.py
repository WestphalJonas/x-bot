"""
LLM Integration Core Module

This module provides a unified interface for interacting with LLM providers
through OpenRouter API.
"""

import os
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    api_key: str = Field(..., description="API key for OpenRouter")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the API"
    )
    default_model: str = Field(
        default="deepseek/deepseek-v3.2-exp",
        description="Default model to use for completions"
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for completions"
    )
    default_max_tokens: Optional[int] = Field(
        default=None,
        description="Default max tokens for completions"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("API key cannot be empty")
        return v.strip()


class Message(BaseModel):
    """Represents a chat message."""

    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate message role."""
        valid_roles = ["system", "user", "assistant"]
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v


class LLMClient:
    """
    Client for interacting with LLM providers through OpenRouter.

    This class provides methods for chat completions, streaming responses,
    and managing conversations with language models.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.

        Args:
            config: Optional configuration object. If not provided, will load
                   from environment variables.
        """
        if config is None:
            config = self._load_config_from_env()

        self.config = config
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _load_config_from_env() -> LLMConfig:
        """Load configuration from environment variables."""
        load_dotenv(dotenv_path="./config/.env")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please create a .env file "
                "with OPENROUTER_API_KEY=your_key"
            )

        return LLMConfig(
            api_key=api_key,
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1"
            ),
            default_model=os.getenv(
                "DEFAULT_MODEL",
                "deepseek/deepseek-v3.2-exp"
            ),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "0")) or None,
            timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        )

    def chat_completion(
        self,
        messages: Union[List[Dict[str, str]], List[Message]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion.

        Args:
            messages: List of messages in the conversation
            model: Model to use (defaults to config default)
            temperature: Sampling temperature (defaults to config default)
            max_tokens: Maximum tokens to generate (defaults to config default)
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the API

        Returns:
            ChatCompletion object or iterator of ChatCompletionChunk objects
            if streaming
        """
        # Convert Message objects to dicts if needed
        if messages and isinstance(messages[0], Message):
            messages = [{"role": m.role, "content": m.content} for m in messages]

        # Use defaults from config if not provided
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.default_temperature
        max_tokens = max_tokens or self.config.default_max_tokens

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "extra_body": kwargs.pop("extra_body", {}),
            **kwargs,
        }

        # Only add max_tokens if it's set
        if max_tokens:
            params["max_tokens"] = max_tokens

        return self.client.chat.completions.create(**params)

    def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple chat interface for single-turn conversations.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt to set context
            **kwargs: Additional arguments to pass to chat_completion

        Returns:
            Response text from the model
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.chat_completion(messages=messages, **kwargs)

        if hasattr(response, "choices"):
            return response.choices[0].message.content

        return ""

    def stream_chat(
        self,
        messages: Union[List[Dict[str, str]], List[Message]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream chat completion responses.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional arguments to pass to chat_completion

        Yields:
            Text chunks from the streaming response
        """
        stream = self.chat_completion(messages=messages, stream=True, **kwargs)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def create_message(self, role: str, content: str) -> Message:
        """
        Create a Message object.

        Args:
            role: Role of the message sender
            content: Content of the message

        Returns:
            Message object
        """
        return Message(role=role, content=content)

    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """
        Create a new conversation with an optional system prompt.

        Args:
            system_prompt: Optional system prompt to set context

        Returns:
            List of Message objects representing the conversation
        """
        conversation = []

        if system_prompt:
            conversation.append(Message(role="system", content=system_prompt))

        return conversation

    def add_user_message(
        self,
        conversation: List[Message],
        content: str,
    ) -> List[Message]:
        """
        Add a user message to the conversation.

        Args:
            conversation: Existing conversation
            content: Message content

        Returns:
            Updated conversation
        """
        conversation.append(Message(role="user", content=content))
        return conversation

    def add_assistant_message(
        self,
        conversation: List[Message],
        content: str,
    ) -> List[Message]:
        """
        Add an assistant message to the conversation.

        Args:
            conversation: Existing conversation
            content: Message content

        Returns:
            Updated conversation
        """
        conversation.append(Message(role="assistant", content=content))
        return conversation


def create_llm_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        api_key: Optional API key. If not provided, will load from environment
        **kwargs: Additional configuration parameters

    Returns:
        Configured LLMClient instance
    """
    if api_key:
        config = LLMConfig(api_key=api_key, **kwargs)
    else:
        config = None  # Will load from environment

    return LLMClient(config=config)
