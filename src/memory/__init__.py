"""Memory module for vector storage and duplicate detection."""

from src.memory.chroma_client import ChromaMemory, EmbeddingResult

__all__ = ["ChromaMemory", "EmbeddingResult"]