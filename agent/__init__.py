"""
Agent package for X Bot

This package provides LLM integration, agent functionality, and prompt engineering.
"""

from .core import (
    LLMClient,
    LLMConfig,
    Message,
    create_llm_client,
)

from .personality import (
    PromptBuilder,
    PersonalityConfig,
    ToneStyle,
    ContentType,
    PRESET_PERSONALITIES,
    get_prompt_builder,
)

__all__ = [
    # LLM Integration
    "LLMClient",
    "LLMConfig",
    "Message",
    "create_llm_client",
    # Prompt Engineering
    "PromptBuilder",
    "PersonalityConfig",
    "ToneStyle",
    "ContentType",
    "PRESET_PERSONALITIES",
    "get_prompt_builder",
]
