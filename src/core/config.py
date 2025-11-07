"""Configuration models and loading."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class RateLimitsConfig(BaseModel):
    """Rate limiting configuration."""

    max_posts_per_day: int = Field(
        default=5, ge=1, le=10, description="Max posts per day"
    )
    max_replies_per_day: int = Field(
        default=20, ge=1, le=50, description="Max replies per day"
    )
    reset_time_utc: str = Field(
        default="00:00", description="UTC time to reset counters (HH:MM)"
    )


class SeleniumConfig(BaseModel):
    """Selenium/browser automation configuration."""

    min_delay_seconds: float = Field(
        default=5.0, ge=1.0, description="Min delay between actions"
    )
    max_delay_seconds: float = Field(
        default=15.0, ge=5.0, description="Max delay between actions"
    )
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode (can be overridden with SELENIUM_HEADLESS env var)",
    )
    user_agent_rotation: bool = Field(
        default=True, description="Rotate User-Agent strings"
    )

    @field_validator("headless", mode="before")
    @classmethod
    def override_headless_from_env(cls, v):
        """Override headless mode from environment variable if set."""
        env_value = os.getenv("SELENIUM_HEADLESS")
        if env_value is not None:
            # Convert string to bool
            return env_value.lower() in ("true", "1", "yes", "on")
        return v


class LLMConfig(BaseModel):
    """LLM and AI configuration."""

    provider: Literal["openai", "google", "openrouter", "anthropic"] = Field(
        default="openai", description="Primary LLM provider"
    )
    fallback_providers: list[Literal["openai", "google", "openrouter", "anthropic"]] = (
        Field(
            default_factory=lambda: ["openrouter", "google"],
            description="Fallback providers in order of preference",
        )
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model name (provider-specific, e.g., 'gpt-4o-mini', 'gemini-1.5-flash', 'claude-3-haiku')",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model (OpenAI recommended for compatibility)",
    )
    embedding_provider: Literal["openai", "google"] = Field(
        default="openai", description="Provider for embeddings (OpenAI or Google)"
    )
    max_tokens: int = Field(
        default=150, ge=50, le=300, description="Max tokens per generation"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Duplicate detection threshold"
    )
    use_fallback: bool = Field(
        default=True, description="Enable automatic fallback to backup providers"
    )


class SchedulerConfig(BaseModel):
    """Task scheduling configuration."""

    post_interval_hours: float = Field(
        default=8.0, ge=1.0, description="Average hours between posts"
    )
    post_jitter_hours: float = Field(
        default=1.0, ge=0.0, description="Random jitter in hours (±)"
    )
    mention_check_minutes: int = Field(
        default=30, ge=5, description="Minutes between mention checks"
    )
    event_check_minutes: int = Field(
        default=15, ge=5, description="Minutes between event checks"
    )


class PersonalityConfig(BaseModel):
    """Bot personality configuration."""

    tone: Literal["professional", "casual", "humorous", "technical"] = Field(
        default="professional"
    )
    topics: list[str] = Field(
        default_factory=lambda: ["AI", "technology"], description="Preferred topics"
    )
    style: Literal["concise", "detailed", "conversational"] = Field(default="concise")
    min_tweet_length: int = Field(default=180, ge=1, le=280)
    max_tweet_length: int = Field(default=280, ge=180, le=280)


class BotConfig(BaseModel):
    """Main bot configuration."""

    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    selenium: SeleniumConfig = Field(default_factory=SeleniumConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "BotConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            # Create default config
            default_config = cls()
            default_config.save(path)
            return default_config

        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path: str | Path = "config/config.yaml") -> None:
        """Save configuration to YAML file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, indent=2, default_flow_style=False)

    def get_system_prompt(
        self, prompt_path: str | Path = "config/agent_prompt.txt"
    ) -> str:
        """Load and format the agent prompt template with personality settings."""
        prompt_file = Path(prompt_path)

        if not prompt_file.exists():
            # Return a basic prompt if template doesn't exist
            return self._generate_basic_prompt()

        with open(prompt_file, encoding="utf-8") as f:
            template = f.read()

        # Format template with personality config values
        formatted_prompt = template.format(
            tone=self.personality.tone,
            style=self.personality.style,
            topics=", ".join(self.personality.topics),
            min_tweet_length=self.personality.min_tweet_length,
            max_tweet_length=self.personality.max_tweet_length,
        )

        return formatted_prompt

    def _generate_basic_prompt(self) -> str:
        """Generate a basic system prompt if template file is missing."""
        return f"""You are an AI-powered Twitter/X bot.

Personality:
- Tone: {self.personality.tone}
- Style: {self.personality.style}
- Topics: {", ".join(self.personality.topics)}

Content Guidelines:
- Tweet length: {self.personality.min_tweet_length}-{self.personality.max_tweet_length} characters
- Create original, engaging content
- Stay within Twitter/X terms of service
- Include AI disclosure in profile bio"""
