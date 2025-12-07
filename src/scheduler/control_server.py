"""Lightweight control server for cross-process scheduler commands."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI

from src.scheduler.bot_scheduler import BotScheduler

logger = logging.getLogger(__name__)
app = FastAPI()

# Stored scheduler instance for control commands
_scheduler: BotScheduler | None = None


def _get_scheduler() -> BotScheduler:
    """Return the registered scheduler or raise if unavailable."""
    if _scheduler is None:
        raise RuntimeError("scheduler_not_registered")
    return _scheduler


@app.post("/reload")
async def reload_config() -> dict[str, Any]:
    """Trigger config reload on the scheduler process."""
    scheduler = _get_scheduler()
    result = await asyncio.to_thread(scheduler.reload_config)
    logger.info(
        "control_reload_completed",
        extra={
            "changes": result.get("changes"),
            "jobs_rescheduled": result.get("jobs_rescheduled"),
        },
    )
    return result


@app.get("/health")
async def health() -> dict[str, str]:
    """Health endpoint for control server."""
    return {"status": "ok"}


@app.post("/pause")
async def pause_scheduler() -> dict[str, Any]:
    """Pause all scheduler jobs and persist remaining next-run delays."""
    scheduler = _get_scheduler()
    result = await asyncio.to_thread(scheduler.pause_all)
    return result


@app.post("/resume")
async def resume_scheduler() -> dict[str, Any]:
    """Resume scheduler jobs using persisted next-run delays."""
    scheduler = _get_scheduler()
    result = await asyncio.to_thread(scheduler.resume_all)
    return result


def start_control_server(
    scheduler: BotScheduler, host: str | None = None, port: int | None = None
) -> threading.Thread:
    """Start the control server in a background thread."""
    global _scheduler
    _scheduler = scheduler

    host = host or os.getenv("SCHEDULER_CONTROL_HOST", "127.0.0.1")
    port = port or int(os.getenv("SCHEDULER_CONTROL_PORT", "8790"))

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config=config)

    thread = threading.Thread(target=server.run, name="scheduler-control", daemon=True)
    thread.start()

    logger.info("control_server_started", extra={"host": host, "port": port})
    return thread
