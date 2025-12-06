"""LangGraph for tweet generation with duplicate checks and gatekeeping."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.core.evaluation import re_evaluate_tweet
from src.core.llm import LLMClient
from src.memory.chroma_client import ChromaMemory

logger = logging.getLogger(__name__)


class TweetGenerationState(BaseModel):
    """State for tweet generation graph."""

    system_prompt: str
    tweet_text: str | None = None
    attempts: int = 0
    duplicate_score: float | None = None
    approved: bool | None = None
    reason: str | None = None


@dataclass
class GenerationDependencies:
    """Dependencies injected into graph nodes."""

    llm_client: LLMClient
    memory: ChromaMemory | None = None


async def _generate_node(
    state: TweetGenerationState, deps: GenerationDependencies
) -> dict:
    tweet_text = await deps.llm_client.generate_tweet(state.system_prompt)
    logger.info(
        "graph_generate_tweet",
        extra={"length": len(tweet_text), "attempts": state.attempts + 1},
    )
    return {
        "tweet_text": tweet_text,
        "attempts": state.attempts + 1,
    }


async def _duplicate_check_node(
    state: TweetGenerationState, deps: GenerationDependencies
) -> dict:
    if not deps.memory or not state.tweet_text:
        return {}

    is_duplicate, score = await deps.memory.check_duplicate(state.tweet_text)
    return {
        "duplicate_score": score,
        "approved": None if is_duplicate else state.approved,
    }


def _duplicate_router(state: TweetGenerationState, threshold: float | None) -> str:
    should_retry = (
        threshold is not None
        and state.duplicate_score is not None
        and state.duplicate_score >= threshold
        and state.attempts < 3
    )
    return "regenerate" if should_retry else "proceed"


async def _gate_node(state: TweetGenerationState, deps: GenerationDependencies) -> dict:
    if not state.tweet_text:
        raise ValueError("No tweet text to evaluate")

    approved, reason = await re_evaluate_tweet(
        tweet_text=state.tweet_text,
        config=deps.llm_client.config,
        llm_client=deps.llm_client,
        operation="autonomous",
    )
    return {"approved": approved, "reason": reason}


def build_tweet_generation_graph(
    deps: GenerationDependencies,
) -> Callable[[TweetGenerationState], Any]:
    """Compile tweet generation graph."""
    graph = StateGraph(TweetGenerationState)

    async def generate_node(state: TweetGenerationState) -> dict:
        return await _generate_node(state, deps)

    async def duplicate_check_node(state: TweetGenerationState) -> dict:
        return await _duplicate_check_node(state, deps)

    async def gate_node(state: TweetGenerationState) -> dict:
        return await _gate_node(state, deps)

    graph.add_node("generate", generate_node)
    graph.add_node("duplicate_check", duplicate_check_node)
    graph.add_node("gate", gate_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "duplicate_check")
    graph.add_conditional_edges(
        "duplicate_check",
        lambda state: _duplicate_router(
            state, deps.memory.similarity_threshold if deps.memory else None
        ),
        {
            "regenerate": "generate",
            "proceed": "gate",
        },
    )
    graph.add_edge("gate", END)

    return graph.compile()
