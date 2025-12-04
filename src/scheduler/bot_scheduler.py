"""Bot scheduler using APScheduler BackgroundScheduler."""

import logging
import threading
from collections import deque
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.core.config import BotConfig

logger = logging.getLogger(__name__)

# Global lock to prevent parallel job execution
_job_lock = threading.Lock()


def get_job_lock() -> threading.Lock:
    """Get the global job lock."""
    return _job_lock


class JobQueue:
    """Thread-safe job queue with deduplication.

    When a job can't run immediately (because another job is running),
    it gets added to this queue. After each job completes, the queue
    is processed to run any pending jobs.

    Jobs are deduplicated by job_id - only one pending execution per job type.
    """

    def __init__(self):
        """Initialize the job queue."""
        self._queue: deque[tuple[str, Callable]] = deque()
        self._pending_job_ids: set[str] = set()
        self._lock = threading.Lock()

    def add_job(self, job_id: str, func: Callable) -> bool:
        """Add a job to the queue if not already pending.

        Args:
            job_id: Unique identifier for the job type
            func: Callable to execute (already wrapped with config/env)

        Returns:
            True if job was added, False if already pending
        """
        with self._lock:
            if job_id in self._pending_job_ids:
                logger.info(
                    "job_already_queued",
                    extra={"job_id": job_id, "queue_size": len(self._queue)},
                )
                return False

            self._queue.append((job_id, func))
            self._pending_job_ids.add(job_id)
            logger.info(
                "job_queued",
                extra={"job_id": job_id, "queue_size": len(self._queue)},
            )
            return True

    def get_next(self) -> tuple[str, Callable] | None:
        """Get the next job from the queue.

        Returns:
            Tuple of (job_id, func) or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None

            job_id, func = self._queue.popleft()
            self._pending_job_ids.discard(job_id)
            return job_id, func

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """Get the current queue size."""
        with self._lock:
            return len(self._queue)


# Global job queue instance
_job_queue = JobQueue()


def get_job_queue() -> JobQueue:
    """Get the global job queue."""
    return _job_queue


class BotScheduler:
    """Scheduler for automated bot task execution."""

    def __init__(self, config: BotConfig):
        """Initialize bot scheduler.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.scheduler = BackgroundScheduler()
        self._is_running = False

    def add_job(
        self,
        func: Callable,
        job_id: str,
        trigger: IntervalTrigger,
        max_instances: int = 1,
        **kwargs,
    ) -> None:
        """Add a job to the scheduler.

        Args:
            func: Function to execute
            job_id: Unique identifier for the job
            trigger: APScheduler trigger (e.g., IntervalTrigger)
            max_instances: Maximum number of concurrent instances (default: 1)
            **kwargs: Additional arguments to pass to add_job
        """
        self.scheduler.add_job(
            func=func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            max_instances=max_instances,
            coalesce=True,  # Combine multiple pending executions into one
            **kwargs,
        )
        logger.info(
            "job_scheduled",
            extra={
                "job_id": job_id,
                "trigger": str(trigger),
                "max_instances": max_instances,
            },
        )

    def setup_posting_job(self, func: Callable, **kwargs) -> None:
        """Setup posting job with jitter.

        APScheduler's jitter adds random seconds (0 to jitter) to each execution.
        To achieve ±jitter_hours, we set:
        - interval = base_interval_hours - jitter_hours (minimum)
        - jitter = 2 * jitter_hours (so total range is ±jitter_hours)

        Args:
            func: Function to execute
            **kwargs: Additional arguments to pass to add_job
        """
        base_interval_hours = self.config.scheduler.post_interval_hours
        jitter_hours = self.config.scheduler.post_jitter_hours

        # Calculate interval with jitter: base - jitter (minimum)
        # Jitter will add up to 2*jitter_hours, giving range [base-jitter, base+jitter]
        interval_hours = max(1.0, base_interval_hours - jitter_hours)
        jitter_seconds = int(jitter_hours * 2 * 3600)

        trigger = IntervalTrigger(hours=interval_hours, jitter=jitter_seconds)
        self.add_job(func=func, job_id="post_tweet", trigger=trigger, **kwargs)

    def setup_reading_job(self, func: Callable, **kwargs) -> None:
        """Setup reading job.

        Args:
            func: Function to execute
            **kwargs: Additional arguments to pass to add_job
        """
        interval_minutes = self.config.scheduler.reading_check_minutes
        trigger = IntervalTrigger(minutes=interval_minutes)
        self.add_job(func=func, job_id="read_posts", trigger=trigger, **kwargs)

    def setup_notifications_job(self, func: Callable, **kwargs) -> None:
        """Setup notifications job.

        Args:
            func: Function to execute
            **kwargs: Additional arguments to pass to add_job
        """
        interval_minutes = self.config.scheduler.mention_check_minutes
        trigger = IntervalTrigger(minutes=interval_minutes)
        self.add_job(func=func, job_id="check_notifications", trigger=trigger, **kwargs)

    def setup_inspiration_job(self, func: Callable, **kwargs) -> None:
        """Setup inspiration queue processing job.

        Args:
            func: Function to execute
            **kwargs: Additional arguments to pass to add_job
        """
        interval_minutes = self.config.scheduler.inspiration_check_minutes
        trigger = IntervalTrigger(minutes=interval_minutes)
        self.add_job(
            func=func, job_id="process_inspiration_queue", trigger=trigger, **kwargs
        )

    def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            logger.warning("scheduler_already_running")
            return

        self.scheduler.start()
        self._is_running = True
        logger.info("scheduler_started")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._is_running:
            logger.warning("scheduler_not_running")
            return

        self.scheduler.shutdown(wait=True)
        self._is_running = False
        logger.info("scheduler_stopped")

    def shutdown(self) -> None:
        """Shutdown the scheduler (alias for stop)."""
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running
