"""ChromaDB integration for vector storage and duplicate detection."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import BotConfig

logger = logging.getLogger(__name__)


class EmbeddingRateLimitError(Exception):
    """Raised when rate limited on embedding API - should not be retried."""

    pass


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    
    text_hash: str
    embedding: list[float]
    cached: bool = False


class ChromaMemory:
    """Vector memory using ChromaDB with OpenAI embeddings."""

    def __init__(
        self,
        config: BotConfig,
        openai_api_key: str,
        persist_directory: str = "./data/chroma",
    ):
        """Initialize ChromaDB client.

        Args:
            config: Bot configuration
            openai_api_key: OpenAI API key for embeddings
            persist_directory: Directory to persist ChromaDB data
        """
        self.config = config
        self.embedding_model = config.llm.embedding_model
        self.similarity_threshold = config.llm.similarity_threshold
        
        # OpenAI client for embeddings
        # Disable built-in retries - we handle retries ourselves with tenacity
        self.openai_client = AsyncOpenAI(api_key=openai_api_key, max_retries=0)
        
        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collections
        self.tweets_collection = self.chroma_client.get_or_create_collection(
            name="tweets",
            metadata={"description": "All posted tweets"}
        )
        self.posts_collection = self.chroma_client.get_or_create_collection(
            name="read_posts",
            metadata={"description": "Posts read from timeline"}
        )

    async def close(self) -> None:
        """Close underlying async clients to avoid loop shutdown errors."""
        try:
            await self.openai_client.close()
        except Exception as exc:
            logger.debug("chroma_memory_close_failed", extra={"error": str(exc)})

    def _hash_text(self, text: str) -> str:
        """Generate consistent hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(EmbeddingRateLimitError),
        reraise=True,
    )
    async def _get_embedding_from_api(self, text: str) -> list[float]:
        """Get embedding from OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingRateLimitError: When rate limited (should not be retried)
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
        except RateLimitError as e:
            # Don't retry on rate limit - raise custom exception to stop retries
            logger.warning(
                "embedding_rate_limited",
                extra={
                    "model": self.embedding_model,
                    "error": str(e),
                },
            )
            raise EmbeddingRateLimitError(
                f"OpenAI embedding API rate limited: {e}"
            ) from e
        
        logger.info(
            "embedding_api_call",
            extra={
                "model": self.embedding_model,
                "text_length": len(text),
                "tokens": response.usage.total_tokens,
            },
        )
        
        return response.data[0].embedding

    async def get_embedding(self, text: str) -> EmbeddingResult:
        """Get embedding for text, using cache if available.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding and cache status
        """
        text_hash = self._hash_text(text)
        
        # Check cache first (in tweets collection)
        try:
            results = self.tweets_collection.get(
                ids=[text_hash],
                include=["embeddings"],
            )
            
            if results["ids"] and results["embeddings"]:
                logger.info("embedding_cache_hit", extra={"text_hash": text_hash})
                return EmbeddingResult(
                    text_hash=text_hash,
                    embedding=results["embeddings"][0],
                    cached=True,
                )
        except Exception:
            pass  # Cache miss, get from API
        
        # Get from API
        logger.info("embedding_cache_miss", extra={"text_hash": text_hash})
        embedding = await self._get_embedding_from_api(text)
        
        return EmbeddingResult(
            text_hash=text_hash,
            embedding=embedding,
            cached=False,
        )

    async def check_duplicate(self, text: str) -> tuple[bool, float | None]:
        """Check if text is a duplicate of existing content.

        Args:
            text: Text to check for duplicates

        Returns:
            Tuple of (is_duplicate, similarity_score)
        """
        if self.tweets_collection.count() == 0:
            return False, None
        
        # Get embedding for new text
        result = await self.get_embedding(text)
        
        # Query for similar tweets
        query_results = self.tweets_collection.query(
            query_embeddings=[result.embedding],
            n_results=1,
            include=["documents", "distances"],
        )
        
        if not query_results["distances"] or not query_results["distances"][0]:
            return False, None
        
        # ChromaDB returns L2 distance by default
        # Convert to similarity: similarity = 1 / (1 + distance)
        distance = query_results["distances"][0][0]
        similarity = 1 / (1 + distance)
        
        is_duplicate = similarity >= self.similarity_threshold
        
        logger.info(
            "duplicate_check",
            extra={
                "is_duplicate": is_duplicate,
                "similarity": round(similarity, 4),
                "threshold": self.similarity_threshold,
                "closest_match": query_results["documents"][0][0][:50] if query_results["documents"][0] else None,
            },
        )
        
        return is_duplicate, similarity

    async def store_tweet(
        self,
        text: str,
        tweet_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store a tweet with its embedding.

        Args:
            text: Tweet text
            tweet_id: Optional tweet ID (uses hash if not provided)
            metadata: Optional metadata dict

        Returns:
            ID of stored document
        """
        result = await self.get_embedding(text)
        doc_id = tweet_id or result.text_hash
        
        # Prepare metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
        }
        if metadata:
            meta.update(metadata)
        
        # Upsert to handle duplicates gracefully
        self.tweets_collection.upsert(
            ids=[doc_id],
            embeddings=[result.embedding],
            documents=[text],
            metadatas=[meta],
        )
        
        logger.info(
            "tweet_stored",
            extra={
                "doc_id": doc_id,
                "text_preview": text[:50],
                "cached_embedding": result.cached,
            },
        )
        
        return doc_id

    async def store_post(
        self,
        text: str,
        post_id: str,
        author: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store a read post with its embedding.

        Args:
            text: Post text
            post_id: Post ID
            author: Post author
            metadata: Optional metadata dict

        Returns:
            ID of stored document
        """
        result = await self.get_embedding(text)
        
        meta = {
            "timestamp": datetime.now().isoformat(),
            "author": author or "unknown",
        }
        if metadata:
            meta.update(metadata)
        
        self.posts_collection.upsert(
            ids=[post_id],
            embeddings=[result.embedding],
            documents=[text],
            metadatas=[meta],
        )
        
        logger.info(
            "post_stored",
            extra={"post_id": post_id, "author": author},
        )
        
        return post_id

    async def find_similar_posts(
        self,
        text: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Find posts similar to given text.

        Args:
            text: Text to find similar posts for
            n_results: Number of results to return

        Returns:
            List of similar posts with metadata
        """
        if self.posts_collection.count() == 0:
            return []
        
        result = await self.get_embedding(text)
        
        query_results = self.posts_collection.query(
            query_embeddings=[result.embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        similar_posts = []
        for i, doc_id in enumerate(query_results["ids"][0]):
            distance = query_results["distances"][0][i]
            similarity = 1 / (1 + distance)
            
            similar_posts.append({
                "id": doc_id,
                "text": query_results["documents"][0][i],
                "metadata": query_results["metadatas"][0][i],
                "similarity": round(similarity, 4),
            })
        
        return similar_posts

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "tweets_count": self.tweets_collection.count(),
            "posts_count": self.posts_collection.count(),
        }