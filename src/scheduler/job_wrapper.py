"""Scheduler job wrapper helpers with lock/queue orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from src.core.config import BotConfig, EnvSettings
from src.core.env import load_env_settings
from src.scheduler.bot_scheduler import get_job_lock, get_job_queue

logger = logging.getLogger(__name__)

JobFunc = Callable[[BotConfig, EnvSettings], Any]


def create_job_wrapper(
    job_func: JobFunc, config: BotConfig, env_settings: EnvSettings
) -> Callable[[], None]:
    """Create a wrapper that serializes jobs and queues overlapping runs."""
    job_id = job_func.__name__

    def run_job() -> None:
        """Execute the actual job with live scheduler config and reloaded env."""
        scheduler = None
        try:
            from src.scheduler.bot_scheduler import get_scheduler

            scheduler = get_scheduler()
        except Exception:
            scheduler = None

        active_config = scheduler.config if scheduler is not None else config
        active_env_settings = load_env_settings()
        job_func(active_config, active_env_settings or env_settings)

    def process_queue() -> None:
        """Process any pending jobs in the queue."""
        job_queue = get_job_queue()
        while not job_queue.is_empty():
            next_job = job_queue.get_next()
            if next_job is None:
                continue
            next_job_id, next_func = next_job
            logger.info(
                "processing_queued_job",
                extra={"job_id": next_job_id, "remaining": job_queue.size()},
            )
            try:
                started = time.perf_counter()
                next_func()
                logger.info(
                    "job_wrapper_completed",
                    extra={
                        "job_id": next_job_id,
                        "source": "queue",
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
            except Exception as exc:
                logger.error(
                    "queued_job_failed",
                    extra={"job_id": next_job_id, "error": str(exc)},
                    exc_info=True,
                )

    def wrapper() -> None:
        lock = get_job_lock()
        job_queue = get_job_queue()

        if not lock.acquire(blocking=False):
            job_queue.add_job(job_id, run_job)
            return

        try:
            started = time.perf_counter()
            run_job()
            process_queue()
            logger.info(
                "job_wrapper_completed",
                extra={
                    "job_id": job_id,
                    "source": "scheduler_wrapper",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        finally:
            lock.release()

    return wrapper
