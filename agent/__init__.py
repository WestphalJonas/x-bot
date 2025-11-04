"""
Agent package for X Bot

This package contains the core agent logic, including personality and prompt engineering.
"""

from .personality import (
    PromptBuilder,
    PersonalityConfig,
    ToneStyle,
    ContentType,
    PRESET_PERSONALITIES,
    get_prompt_builder,
)

__all__ = [
    "PromptBuilder",
    "PersonalityConfig",
    "ToneStyle",
    "ContentType",
    "PRESET_PERSONALITIES",
    "get_prompt_builder",
]
