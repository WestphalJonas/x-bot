"""
Rate limiting utilities for API calls and other operations.
"""
import time
import threading
from collections import deque
from functools import wraps
from typing import Callable, Optional, Any
import asyncio


class RateLimiter:
    """
    A thread-safe rate limiter using a sliding window algorithm.

    This class limits the number of operations within a specified time window.
    It supports both synchronous and asynchronous operations.

    Args:
        max_calls: Maximum number of calls allowed in the time window
        time_window: Time window in seconds (default: 60)

    Example:
        >>> limiter = RateLimiter(max_calls=10, time_window=60)
        >>> limiter.wait_if_needed()  # Blocks if rate limit is exceeded
    """

    def __init__(self, max_calls: int, time_window: float = 60.0):
        """
        Initialize the rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
        """
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if time_window <= 0:
            raise ValueError("time_window must be positive")

        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()

    def _clean_old_calls(self, current_time: float) -> None:
        """Remove calls outside the current time window."""
        cutoff_time = current_time - self.time_window
        while self.calls and self.calls[0] < cutoff_time:
            self.calls.popleft()

    def _get_wait_time(self, current_time: float) -> float:
        """Calculate how long to wait before making the next call."""
        self._clean_old_calls(current_time)

        if len(self.calls) < self.max_calls:
            return 0.0

        # Need to wait until the oldest call expires
        oldest_call = self.calls[0]
        wait_time = (oldest_call + self.time_window) - current_time
        return max(0.0, wait_time)

    def wait_if_needed(self) -> float:
        """
        Wait if the rate limit has been exceeded.

        Returns:
            The time waited in seconds
        """
        with self.lock:
            current_time = time.time()
            wait_time = self._get_wait_time(current_time)

            if wait_time > 0:
                time.sleep(wait_time)
                current_time = time.time()

            self.calls.append(current_time)
            return wait_time

    async def async_wait_if_needed(self) -> float:
        """
        Asynchronously wait if the rate limit has been exceeded.

        Returns:
            The time waited in seconds
        """
        with self.lock:
            current_time = time.time()
            wait_time = self._get_wait_time(current_time)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        with self.lock:
            current_time = time.time()
            self.calls.append(current_time)

        return wait_time

    def can_proceed(self) -> bool:
        """
        Check if a call can proceed without waiting.

        Returns:
            True if the call can proceed, False otherwise
        """
        with self.lock:
            current_time = time.time()
            self._clean_old_calls(current_time)
            return len(self.calls) < self.max_calls

    def get_remaining_calls(self) -> int:
        """
        Get the number of remaining calls available in the current window.

        Returns:
            Number of calls that can be made without waiting
        """
        with self.lock:
            current_time = time.time()
            self._clean_old_calls(current_time)
            return max(0, self.max_calls - len(self.calls))

    def reset(self) -> None:
        """Reset the rate limiter, clearing all recorded calls."""
        with self.lock:
            self.calls.clear()


def rate_limit(max_calls: int, time_window: float = 60.0) -> Callable:
    """
    Decorator to apply rate limiting to a function.

    Args:
        max_calls: Maximum number of calls allowed in the time window
        time_window: Time window in seconds (default: 60)

    Example:
        >>> @rate_limit(max_calls=10, time_window=60)
        ... def api_call():
        ...     return "result"
    """
    limiter = RateLimiter(max_calls, time_window)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await limiter.async_wait_if_needed()
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                limiter.wait_if_needed()
                return func(*args, **kwargs)
            return sync_wrapper

    return decorator


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation.

    This algorithm allows bursts while maintaining an average rate.
    Tokens are added at a constant rate, and each operation consumes a token.

    Args:
        rate: Number of tokens added per second
        capacity: Maximum number of tokens the bucket can hold

    Example:
        >>> limiter = TokenBucketRateLimiter(rate=2.0, capacity=10)
        >>> limiter.wait_if_needed()  # Consumes one token
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize the token bucket rate limiter.

        Args:
            rate: Number of tokens added per second
            capacity: Maximum number of tokens in the bucket
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = current_time

    def wait_if_needed(self, tokens: int = 1) -> float:
        """
        Wait if not enough tokens are available.

        Args:
            tokens: Number of tokens to consume (default: 1)

        Returns:
            The time waited in seconds
        """
        if tokens > self.capacity:
            raise ValueError(f"Requested tokens ({tokens}) exceeds capacity ({self.capacity})")

        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate

            time.sleep(wait_time)
            self._refill()
            self.tokens -= tokens

            return wait_time

    async def async_wait_if_needed(self, tokens: int = 1) -> float:
        """
        Asynchronously wait if not enough tokens are available.

        Args:
            tokens: Number of tokens to consume (default: 1)

        Returns:
            The time waited in seconds
        """
        if tokens > self.capacity:
            raise ValueError(f"Requested tokens ({tokens}) exceeds capacity ({self.capacity})")

        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate

        await asyncio.sleep(wait_time)

        with self.lock:
            self._refill()
            self.tokens -= tokens

        return wait_time

    def can_proceed(self, tokens: int = 1) -> bool:
        """
        Check if enough tokens are available.

        Args:
            tokens: Number of tokens needed (default: 1)

        Returns:
            True if enough tokens are available, False otherwise
        """
        with self.lock:
            self._refill()
            return self.tokens >= tokens

    def get_available_tokens(self) -> float:
        """
        Get the number of available tokens.

        Returns:
            Number of available tokens
        """
        with self.lock:
            self._refill()
            return self.tokens

    def reset(self) -> None:
        """Reset the token bucket to full capacity."""
        with self.lock:
            self.tokens = self.capacity
            self.last_update = time.time()
