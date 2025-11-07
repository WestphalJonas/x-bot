from pathlib import Path

from src.core.config import BotConfig


def main():
    """Main entry point for the bot."""
    # Load configuration
    config_path = Path("config/config.yaml")
    config = BotConfig.load(config_path)

    print("Hello from x-bot!")
    print(f"Loaded config: {config_path}")
    print(f"LLM Provider: {config.llm.provider}")
    print(f"Max posts per day: {config.rate_limits.max_posts_per_day}")
    print(f"Personality: {config.personality.tone} / {config.personality.style}")

    # Load and display the system prompt
    system_prompt = config.get_system_prompt()
    print("\n" + "=" * 60)
    print("System Prompt:")
    print("=" * 60)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)


if __name__ == "__main__":
    main()
