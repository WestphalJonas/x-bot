"""ChromaDB integration via LangChain for embeddings and duplicate detection."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.core.config import BotConfig
from src.core.langchain_clients import LangChainLLM

logger = logging.getLogger(__name__)


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""

    text_hash: str
    embedding: list[float]
    cached: bool = False


class EmbeddingRateLimitError(Exception):
    """Kept for backward compatibility with previous API."""

    pass


class ChromaMemory:
    """Vector memory using LangChain Chroma store."""

    def __init__(
        self,
        config: BotConfig,
        llm_client: LangChainLLM,
        persist_directory: str = "./data/chroma",
    ):
        self.config = config
        self.similarity_threshold = config.llm.similarity_threshold
        self._embeddings = llm_client.get_embeddings()
        if not self._embeddings:
            raise RuntimeError("Embedding provider not configured")

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.tweets_store = Chroma(
            collection_name="tweets",
            embedding_function=self._embeddings,
            persist_directory=persist_directory,
        )
        self.posts_store = Chroma(
            collection_name="read_posts",
            embedding_function=self._embeddings,
            persist_directory=persist_directory,
        )

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def get_embedding(self, text: str) -> EmbeddingResult:
        """Embed text, using Chroma to cache by hash."""
        text_hash = self._hash_text(text)

        try:
            existing = self.tweets_store._collection.get(
                ids=[text_hash],
                include=["embeddings"],
            )
            if existing["ids"] and existing["embeddings"]:
                return EmbeddingResult(
                    text_hash=text_hash,
                    embedding=existing["embeddings"][0],
                    cached=True,
                )
        except Exception:
            # Cache miss, continue to embed
            pass

        # Embed with provider (async if available)
        if hasattr(self._embeddings, "aembed_query"):
            embedding = await self._embeddings.aembed_query(text)  # type: ignore[attr-defined]
        else:
            embedding = await asyncio.to_thread(self._embeddings.embed_query, text)

        return EmbeddingResult(text_hash=text_hash, embedding=embedding, cached=False)

    async def check_duplicate(self, text: str) -> tuple[bool, float | None]:
        """Check if text already exists in tweet memory."""
        if self.tweets_store._collection.count() == 0:  # type: ignore[attr-defined]
            return False, None

        result = await self.get_embedding(text)
        docs_with_scores = await asyncio.to_thread(
            self.tweets_store.similarity_search_by_vector_with_relevance_scores,
            result.embedding,
            1,
        )

        if not docs_with_scores:
            return False, None

        _, score = docs_with_scores[0]
        is_duplicate = score >= self.similarity_threshold
        logger.info(
            "duplicate_check",
            extra={
                "score": score,
                "threshold": self.similarity_threshold,
                "duplicate": is_duplicate,
            },
        )
        return is_duplicate, score

    async def store_tweet(
        self,
        text: str,
        tweet_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store tweet with embedding."""
        result = await self.get_embedding(text)
        doc_id = tweet_id or result.text_hash

        document = Document(
            page_content=text,
            metadata={
                "id": doc_id,
                "timestamp": datetime.now().isoformat(),
                "text_length": len(text),
                **(metadata or {}),
            },
        )

        await asyncio.to_thread(
            self.tweets_store.add_documents,
            documents=[document],
            ids=[doc_id],
        )

        return doc_id

    async def store_post(
        self,
        text: str,
        post_id: str,
        author: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store read post with embedding."""
        _ = await self.get_embedding(text)

        document = Document(
            page_content=text,
            metadata={
                "id": post_id,
                "timestamp": datetime.now().isoformat(),
                "author": author or "unknown",
                **(metadata or {}),
            },
        )

        await asyncio.to_thread(
            self.posts_store.add_documents,
            documents=[document],
            ids=[post_id],
        )

        return post_id

    async def find_similar_posts(
        self,
        text: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Find similar posts in memory."""
        if self.posts_store._collection.count() == 0:  # type: ignore[attr-defined]
            return []

        result = await self.get_embedding(text)
        docs_with_scores = await asyncio.to_thread(
            self.posts_store.similarity_search_by_vector_with_relevance_scores,
            result.embedding,
            n_results,
        )

        similar_posts: list[dict] = []
        for doc, score in docs_with_scores:
            similar_posts.append(
                {
                    "id": doc.metadata.get("id"),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": round(score, 4),
                }
            )
        return similar_posts

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "tweets_count": self.tweets_store._collection.count(),  # type: ignore[attr-defined]
            "posts_count": self.posts_store._collection.count(),  # type: ignore[attr-defined]
        }