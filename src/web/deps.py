"""Shared FastAPI dependencies for the web dashboard."""

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.core.config import BotConfig, EnvSettings

# ChromaDB is optional - may not be available on all Python versions
try:
    from src.memory.chroma_client import ChromaMemory

    CHROMA_AVAILABLE = True
except ImportError:
    ChromaMemory = None  # type: ignore
    CHROMA_AVAILABLE = False


@lru_cache
def get_config() -> BotConfig:
    """Get the loaded bot configuration."""
    return BotConfig.load()


@lru_cache
def get_chroma_memory() -> object | None:
    """Get the ChromaDB memory client if available."""
    if not CHROMA_AVAILABLE or ChromaMemory is None:
        return None

    try:
        from src.core.langchain_clients import LangChainLLM

        env_settings: EnvSettings = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "TWITTER_USERNAME": os.getenv("TWITTER_USERNAME"),
            "TWITTER_PASSWORD": os.getenv("TWITTER_PASSWORD"),
            "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
            "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"),
            "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        }
        config = get_config()
        llm_client = LangChainLLM(config=config, env_settings=env_settings)
        return ChromaMemory(config=config, llm_client=llm_client)
    except Exception:
        return None


ConfigDep = Annotated[BotConfig, Depends(get_config)]
ChromaMemoryDep = Annotated[object | None, Depends(get_chroma_memory)]

