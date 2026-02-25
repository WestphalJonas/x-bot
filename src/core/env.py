"""Environment variable loading and validation helpers."""

from __future__ import annotations

import os

from src.core.config import EnvSettings


def load_env_settings() -> EnvSettings:
    """Load environment variables into a typed dictionary."""
    return EnvSettings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
        TWITTER_USERNAME=os.getenv("TWITTER_USERNAME"),
        TWITTER_PASSWORD=os.getenv("TWITTER_PASSWORD"),
        LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY"),
        LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT"),
        LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2"),
    )


def validate_env_settings(env_settings: EnvSettings) -> None:
    """Validate that required environment variables are set."""
    openai_api_key = env_settings.get("OPENAI_API_KEY")
    openrouter_api_key = env_settings.get("OPENROUTER_API_KEY")
    google_api_key = env_settings.get("GOOGLE_API_KEY")
    anthropic_api_key = env_settings.get("ANTHROPIC_API_KEY")
    twitter_username = env_settings.get("TWITTER_USERNAME")
    twitter_password = env_settings.get("TWITTER_PASSWORD")

    if not any([openai_api_key, openrouter_api_key, google_api_key, anthropic_api_key]):
        raise ValueError(
            "At least one LLM provider API key is required. "
            "Please set OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY."
        )
    if not twitter_username:
        raise ValueError("TWITTER_USERNAME environment variable is required")
    if not twitter_password:
        raise ValueError("TWITTER_PASSWORD environment variable is required")
