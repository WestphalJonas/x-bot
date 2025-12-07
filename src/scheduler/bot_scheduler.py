"""Bot scheduler using APScheduler BackgroundScheduler."""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
import threading
from collections import deque
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.core.config import BotConfig
from src.state.manager import load_state, save_state
from src.state.models import AgentState

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

# Global scheduler instance (set via set_scheduler from main.py)
_scheduler: "BotScheduler | None" = None


def get_job_queue() -> JobQueue:
    """Get the global job queue."""
    return _job_queue


def get_scheduler() -> "BotScheduler | None":
    """Get the global scheduler instance."""
    return _scheduler


def set_scheduler(scheduler: "BotScheduler") -> None:
    """Set the global scheduler instance.

    Called from main.py after scheduler creation.
    """
    global _scheduler
    _scheduler = scheduler
    logger.info("scheduler_registered_globally")


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
        # Store job functions for reload capability
        self._job_funcs: dict[str, Callable] = {}

    def _load_state_sync(self) -> AgentState:
        """Load agent state synchronously."""
        return asyncio.run(load_state())

    def _save_state_sync(self, state: AgentState) -> None:
        """Save agent state synchronously."""
        asyncio.run(save_state(state))

    def _snapshot_next_runs(self) -> tuple[dict[str, float], dict[str, str]]:
        """Capture remaining seconds and absolute times until next run for each job."""
        now = datetime.now(timezone.utc)
        delays: dict[str, float] = {}
        times: dict[str, str] = {}

        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            if next_run is None:
                continue

            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)

            remaining = (next_run - now).total_seconds()
            delays[job.id] = remaining if remaining > 0 else 0.0
            times[job.id] = next_run.isoformat()

        return delays, times

    def pause_all(self) -> dict[str, object]:
        """Pause all jobs, persisting remaining time until next run."""
        next_run_delays, next_run_times = self._snapshot_next_runs()
        jobs = self.scheduler.get_jobs()

        for job in jobs:
            try:
                self.scheduler.pause_job(job.id)
            except Exception as e:
                logger.warning(
                    "job_pause_failed", extra={"job_id": job.id, "error": str(e)}
                )

        now = datetime.now(timezone.utc)
        state = self._load_state_sync()
        state.paused = True
        state.paused_at = now
        state.next_run_delays = next_run_delays
        state.next_run_times = next_run_times
        state.last_action = "Scheduler paused"
        state.last_action_time = now
        self._save_state_sync(state)

        logger.info(
            "scheduler_paused",
            extra={
                "jobs_paused": len(jobs),
                "next_runs_captured": len(next_run_delays),
            },
        )

        return {
            "status": "paused",
            "next_runs": next_run_delays,
            "next_run_times": next_run_times,
            "paused_at": now.isoformat(),
        }

    def resume_all(self) -> dict[str, object]:
        """Resume all jobs using captured next-run delays."""
        state = self._load_state_sync()
        delays = state.next_run_delays or {}
        times = state.next_run_times or {}
        jobs = self.scheduler.get_jobs()
        now = datetime.now(timezone.utc)
        paused_at = state.paused_at or now

        for job in jobs:
            delay = delays.get(job.id)
            time_str = times.get(job.id)

            if time_str:
                try:
                    parsed = datetime.fromisoformat(time_str)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    remaining_from_saved = (parsed - paused_at).total_seconds()
                    delay = max(remaining_from_saved, 0.0)
                except Exception as e:
                    logger.warning(
                        "job_next_run_parse_failed",
                        extra={"job_id": job.id, "error": str(e)},
                    )

            if delay is not None:
                new_time = now + timedelta(seconds=max(delay, 1.0))
                try:
                    self.scheduler.modify_job(job.id, next_run_time=new_time)
                except Exception as e:
                    logger.warning(
                        "job_modify_failed",
                        extra={"job_id": job.id, "error": str(e)},
                    )
            try:
                self.scheduler.resume_job(job.id)
            except Exception as e:
                logger.warning(
                    "job_resume_failed", extra={"job_id": job.id, "error": str(e)}
                )

        state.paused = False
        state.paused_at = None
        state.next_run_delays = {}
        state.next_run_times = {}
        state.last_action = "Scheduler resumed"
        state.last_action_time = now
        self._save_state_sync(state)

        logger.info(
            "scheduler_resumed",
            extra={
                "jobs_resumed": len(jobs),
                "delays_applied": len(delays),
                "times_applied": len(times),
            },
        )

        return {
            "status": "resumed",
            "jobs_resumed": len(jobs),
            "delays_applied": len(delays),
            "times_applied": len(times),
        }

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
        self._job_funcs["post_tweet"] = func
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
        self._job_funcs["read_posts"] = func
        interval_minutes = self.config.scheduler.reading_check_minutes
        trigger = IntervalTrigger(minutes=interval_minutes)
        self.add_job(func=func, job_id="read_posts", trigger=trigger, **kwargs)

    def setup_notifications_job(self, func: Callable, **kwargs) -> None:
        """Setup notifications job.

        Args:
            func: Function to execute
            **kwargs: Additional arguments to pass to add_job
        """
        self._job_funcs["check_notifications"] = func
        interval_minutes = self.config.scheduler.mention_check_minutes
        trigger = IntervalTrigger(minutes=interval_minutes)
        self.add_job(func=func, job_id="check_notifications", trigger=trigger, **kwargs)

    def setup_inspiration_job(self, func: Callable, **kwargs) -> None:
        """Setup inspiration queue processing job.

        Args:
            func: Function to execute
            **kwargs: Additional arguments to pass to add_job
        """
        self._job_funcs["process_inspiration_queue"] = func
        interval_minutes = self.config.scheduler.inspiration_check_minutes
        trigger = IntervalTrigger(minutes=interval_minutes)
        self.add_job(
            func=func, job_id="process_inspiration_queue", trigger=trigger, **kwargs
        )

    def reload_config(self) -> dict[str, str]:
        """Reload configuration and reschedule all jobs with updated intervals.

        Returns:
            Dict with status and details about what was reloaded
        """
        logger.info("config_reload_started")

        # Load fresh config
        new_config = BotConfig.load()
        old_config = self.config
        self.config = new_config

        changes = []

        # Reschedule posting job if interval changed
        if "post_tweet" in self._job_funcs:
            old_interval = old_config.scheduler.post_interval_hours
            new_interval = new_config.scheduler.post_interval_hours
            if old_interval != new_interval:
                changes.append(f"post_interval: {old_interval}h → {new_interval}h")
            self.setup_posting_job(self._job_funcs["post_tweet"])

        # Reschedule reading job if interval changed
        if "read_posts" in self._job_funcs:
            old_interval = old_config.scheduler.reading_check_minutes
            new_interval = new_config.scheduler.reading_check_minutes
            if old_interval != new_interval:
                changes.append(f"reading_interval: {old_interval}m → {new_interval}m")
            self.setup_reading_job(self._job_funcs["read_posts"])

        # Reschedule notifications job if interval changed
        if "check_notifications" in self._job_funcs:
            old_interval = old_config.scheduler.mention_check_minutes
            new_interval = new_config.scheduler.mention_check_minutes
            if old_interval != new_interval:
                changes.append(f"mention_interval: {old_interval}m → {new_interval}m")
            self.setup_notifications_job(self._job_funcs["check_notifications"])

        # Reschedule inspiration job if interval changed
        if "process_inspiration_queue" in self._job_funcs:
            old_interval = old_config.scheduler.inspiration_check_minutes
            new_interval = new_config.scheduler.inspiration_check_minutes
            if old_interval != new_interval:
                changes.append(
                    f"inspiration_interval: {old_interval}m → {new_interval}m"
                )
            self.setup_inspiration_job(self._job_funcs["process_inspiration_queue"])

        logger.info(
            "config_reload_completed",
            extra={"changes": changes, "jobs_rescheduled": len(self._job_funcs)},
        )

        return {
            "status": "ok",
            "jobs_rescheduled": len(self._job_funcs),
            "changes": changes,
        }

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
