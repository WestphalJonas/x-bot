"""Scheduler bootstrap helpers for job registration and runtime startup."""

from __future__ import annotations

from src.core.config import BotConfig, EnvSettings
from src.scheduler.bot_scheduler import BotScheduler
from src.scheduler.job_wrapper import create_job_wrapper
from src.scheduler.jobs import (
    check_notifications,
    post_autonomous_tweet,
    process_inspiration_queue,
    process_replies,
    read_frontpage_posts,
)


def register_scheduler_jobs(
    scheduler: BotScheduler,
    config: BotConfig,
    env_settings: EnvSettings,
) -> None:
    """Register all scheduled jobs using the shared wrapper."""
    scheduler.setup_posting_job(
        create_job_wrapper(post_autonomous_tweet, config, env_settings)
    )
    scheduler.setup_reading_job(
        create_job_wrapper(read_frontpage_posts, config, env_settings)
    )
    scheduler.setup_notifications_job(
        create_job_wrapper(check_notifications, config, env_settings)
    )
    scheduler.setup_replies_job(
        create_job_wrapper(process_replies, config, env_settings)
    )
    scheduler.setup_inspiration_job(
        create_job_wrapper(process_inspiration_queue, config, env_settings)
    )
