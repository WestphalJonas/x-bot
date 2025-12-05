"""SQLite database for persistent storage of tweets and token usage."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import aiosqlite

from src.state.models import Post

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = "data/bot.db"


class Database:
    """Async SQLite database client for bot data storage."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize database client.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        await self._create_tables()
        logger.info("database_initialized", extra={"db_path": self.db_path})

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("database_closed")

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get database connection, initializing if needed."""
        if self._connection is None:
            await self.init()
        assert self._connection is not None
        yield self._connection

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        async with self._get_connection() as conn:
            # Read posts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS read_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    username TEXT,
                    display_name TEXT,
                    post_type TEXT DEFAULT 'text_only',
                    url TEXT,
                    is_interesting INTEGER,
                    read_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    post_timestamp TIMESTAMP
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_read_posts_post_id ON read_posts(post_id)"
            )

            # Written tweets table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS written_tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    tweet_type TEXT DEFAULT 'autonomous',
                    tweet_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_written_tweets_created_at ON written_tweets(created_at)"
            )

            # Rejected tweets table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rejected_tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    operation TEXT DEFAULT 'autonomous',
                    rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rejected_tweets_rejected_at ON rejected_tweets(rejected_at)"
            )

            # Token usage table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    operation TEXT DEFAULT 'generate'
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(timestamp)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_token_usage_provider ON token_usage(provider)"
            )

            await conn.commit()

    # ===== Read Posts =====

    async def has_seen_post(self, post_id: str) -> bool:
        """Check if a post has already been seen/read.

        Args:
            post_id: The post ID to check

        Returns:
            True if post has been seen, False otherwise
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM read_posts WHERE post_id = ? LIMIT 1",
                (post_id,),
            )
            row = await cursor.fetchone()
            return row is not None

    async def store_read_post(self, post: Post) -> None:
        """Store a read post in the database.

        Args:
            post: Post object to store
        """
        async with self._get_connection() as conn:
            try:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO read_posts 
                    (post_id, text, username, display_name, post_type, url, is_interesting, post_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post.post_id,
                        post.text,
                        post.username,
                        post.display_name,
                        post.post_type,
                        post.url,
                        1
                        if post.is_interesting
                        else (0 if post.is_interesting is False else None),
                        post.timestamp.isoformat() if post.timestamp else None,
                    ),
                )
                await conn.commit()
                logger.info(
                    "read_post_stored",
                    extra={"post_id": post.post_id, "username": post.username},
                )
            except Exception as e:
                logger.error(
                    "failed_to_store_read_post",
                    extra={"post_id": post.post_id, "error": str(e)},
                )
                raise

    async def get_read_posts(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get recent read posts with pagination.

        Args:
            limit: Maximum number of posts to return
            offset: Number of posts to skip (for pagination)

        Returns:
            List of post dictionaries
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT post_id, text, username, display_name, post_type, url, 
                       is_interesting, read_at, post_timestamp
                FROM read_posts
                ORDER BY read_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_read_posts_count(self) -> int:
        """Get total count of read posts."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM read_posts")
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ===== Written Tweets =====

    async def store_written_tweet(
        self,
        text: str,
        tweet_type: str = "autonomous",
        tweet_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a written/posted tweet.

        Args:
            text: Tweet text
            tweet_type: Type of tweet (autonomous, inspiration, reply)
            tweet_id: Optional X/Twitter ID
            metadata: Optional additional metadata
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO written_tweets (text, tweet_type, tweet_id, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    text,
                    tweet_type,
                    tweet_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            await conn.commit()
            logger.info(
                "written_tweet_stored",
                extra={"tweet_type": tweet_type, "length": len(text)},
            )

    async def get_written_tweets(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get recent written tweets with pagination.

        Args:
            limit: Maximum number of tweets to return
            offset: Number of tweets to skip (for pagination)

        Returns:
            List of tweet dictionaries
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT text, tweet_type, tweet_id, created_at, metadata
                FROM written_tweets
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            result = []
            for row in rows:
                tweet = dict(row)
                if tweet.get("metadata"):
                    tweet["metadata"] = json.loads(tweet["metadata"])
                result.append(tweet)
            return result

    async def get_written_tweets_count(self) -> int:
        """Get total count of written tweets."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM written_tweets")
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ===== Rejected Tweets =====

    async def store_rejected_tweet(
        self,
        text: str,
        reason: str,
        operation: str = "autonomous",
    ) -> None:
        """Store a rejected tweet.

        Args:
            text: Tweet text that was rejected
            reason: Reason for rejection
            operation: Operation type (autonomous, inspiration, reply)
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO rejected_tweets (text, reason, operation)
                VALUES (?, ?, ?)
                """,
                (text, reason, operation),
            )
            await conn.commit()
            logger.info(
                "rejected_tweet_stored",
                extra={"operation": operation, "reason": reason[:50]},
            )

    async def get_rejected_tweets(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get recent rejected tweets with pagination.

        Args:
            limit: Maximum number of tweets to return
            offset: Number of tweets to skip (for pagination)

        Returns:
            List of rejected tweet dictionaries
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT text, reason, operation, rejected_at
                FROM rejected_tweets
                ORDER BY rejected_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_rejected_tweets_count(self) -> int:
        """Get total count of rejected tweets."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM rejected_tweets")
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ===== Token Usage =====

    async def log_token_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        operation: str = "generate",
    ) -> None:
        """Log token usage for analytics.

        Args:
            provider: LLM provider name
            model: Model name used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            operation: Operation type
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO token_usage 
                (provider, model, prompt_tokens, completion_tokens, total_tokens, operation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    operation,
                ),
            )
            await conn.commit()
            logger.info(
                "token_usage_logged",
                extra={
                    "provider": provider,
                    "model": model,
                    "total_tokens": total_tokens,
                    "operation": operation,
                },
            )

    async def get_token_usage(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get recent token usage entries with pagination.

        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip (for pagination)

        Returns:
            List of token usage dictionaries
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT timestamp, provider, model, prompt_tokens, 
                       completion_tokens, total_tokens, operation
                FROM token_usage
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_token_usage_stats(self) -> dict[str, Any]:
        """Get token usage statistics.

        Returns:
            Dictionary with usage stats
        """
        async with self._get_connection() as conn:
            # Total tokens
            cursor = await conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM token_usage"
            )
            row = await cursor.fetchone()
            total_tokens = row[0] if row else 0

            # Total entries
            cursor = await conn.execute("SELECT COUNT(*) FROM token_usage")
            row = await cursor.fetchone()
            total_entries = row[0] if row else 0

            # Tokens by provider
            cursor = await conn.execute(
                """
                SELECT provider, SUM(total_tokens) as tokens
                FROM token_usage
                GROUP BY provider
                """
            )
            rows = await cursor.fetchall()
            tokens_by_provider = {row["provider"]: row["tokens"] for row in rows}

            # Tokens by operation
            cursor = await conn.execute(
                """
                SELECT operation, SUM(total_tokens) as tokens
                FROM token_usage
                GROUP BY operation
                """
            )
            rows = await cursor.fetchall()
            tokens_by_operation = {row["operation"]: row["tokens"] for row in rows}

            return {
                "total_tokens": total_tokens,
                "total_entries": total_entries,
                "tokens_by_provider": tokens_by_provider,
                "tokens_by_operation": tokens_by_operation,
            }

    async def get_hourly_token_usage(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get token usage aggregated by hour for the last N hours.

        Args:
            hours: Number of hours to look back (default 24)

        Returns:
            List of dicts with hour and token count
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    SUM(total_tokens) as tokens
                FROM token_usage
                WHERE timestamp >= datetime('now', ? || ' hours')
                GROUP BY strftime('%Y-%m-%d %H', timestamp)
                ORDER BY hour ASC
                """,
                (f"-{hours}",),
            )
            rows = await cursor.fetchall()
            return [{"hour": row["hour"], "tokens": row["tokens"]} for row in rows]

    # ===== General Stats =====

    async def get_stats(self) -> dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with counts for each table
        """
        return {
            "read_posts_count": await self.get_read_posts_count(),
            "written_tweets_count": await self.get_written_tweets_count(),
            "rejected_tweets_count": await self.get_rejected_tweets_count(),
            "token_usage_entries": (await self.get_token_usage_stats())[
                "total_entries"
            ],
        }


# Global database instance
_db: Database | None = None


async def get_database(db_path: str = DEFAULT_DB_PATH) -> Database:
    """Get or create database instance.

    Args:
        db_path: Path to database file

    Returns:
        Database instance
    """
    global _db
    if _db is None:
        _db = Database(db_path)
        await _db.init()
    return _db


async def close_database() -> None:
    """Close the global database instance."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
