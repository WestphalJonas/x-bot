"""Token logging helper isolated from web dependencies."""

from src.state.database import get_database


async def log_token_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    operation: str,
) -> None:
    """Log token usage for analytics without importing web layers."""
    db = await get_database()
    await db.log_token_usage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        operation=operation,
    )

