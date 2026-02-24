"""Logging configuration with human-readable console output and JSONL file logs."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RESERVED_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}
_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "cookie",
    "api_key",
    "apikey",
    "authorization",
)
_PREFERRED_CONTEXT_KEYS = [
    "job_id",
    "operation",
    "provider",
    "model",
    "post_id",
    "notification_id",
    "username",
    "from_username",
    "length",
    "tweet_length",
    "attempt",
    "max_attempts",
    "duration_ms",
    "status",
    "error",
    "error_message",
]

_LEVEL_EMOJI = {
    "DEBUG": "🔍",
    "INFO": "ℹ️",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "🔥",
}

_TASK_EMOJI = {
    "scheduler": "⏰",
    "posting": "📝",
    "reading": "📖",
    "notifications": "🔔",
    "replies": "💬",
    "llm": "🧠",
    "auth": "🔐",
    "web": "🌐",
    "db": "💾",
    "state": "💾",
    "browser": "🧭",
    "driver": "🧭",
    "x": "🧭",
    "memory": "🧠",
    "system": "🚀",
}

_STATUS_BY_SUFFIX = {
    "_started": "start",
    "_starting": "start",
    "_completed": "success",
    "_success": "success",
    "_successful": "success",
    "_failed": "error",
    "_error": "error",
}

_EVENT_LABELS = {
    "bot_starting": "Bot starting",
    "bot_shutdown_complete": "Bot shutdown complete",
    "scheduler_running": "Scheduler running",
    "scheduler_started": "Scheduler started",
    "scheduler_stopped": "Scheduler stopped",
    "scheduler_paused": "Scheduler paused",
    "scheduler_resumed": "Scheduler resumed",
    "job_started": "Job started",
    "job_completed": "Job completed",
    "job_wrapper_completed": "Job wrapper completed",
    "job_failed": "Job failed",
    "processing_queued_job": "Running queued job",
    "queued_job_failed": "Queued job failed",
    "job_queued": "Job queued",
    "job_already_queued": "Job already queued",
    "configuration_error": "Configuration error",
    "configuration_error: %s": "Configuration error",
    "rate_limit_exceeded": "Rate limit reached",
    "posting_tweet": "Posting tweet",
    "tweet_posted_successfully": "Tweet posted",
    "reply_posted_successfully": "Reply posted",
    "llm_success": "LLM request succeeded",
    "llm_provider_failed": "LLM provider failed",
    "provider_unavailable": "LLM provider unavailable",
    "interest_check_completed": "Interest check completed",
    "tweet_re_evaluated": "Tweet re-evaluated",
    "re_evaluate_failed": "Tweet re-evaluation failed",
}

_ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

_LEVEL_COLORS = {
    "DEBUG": "cyan",
    "INFO": "blue",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "magenta",
}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _redact_value(key: str, value: Any) -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(key, v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, v) for v in value)
    return value


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def _safe_compact(value: Any, max_len: int) -> str:
    if isinstance(value, str):
        text = value.replace("\n", "\\n")
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        text = text.replace("\n", "\\n")
    return _truncate_text(text, max_len)


def _extract_extra(record: logging.LogRecord) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _RESERVED_RECORD_KEYS:
            continue
        extra[key] = _redact_value(key, value)
    return extra


def _classify_task(logger_name: str) -> str:
    name = logger_name.lower()
    if ".jobs.posting" in name or ".x.posting" in name:
        return "posting"
    if ".jobs.reading" in name or ".x.reading" in name or ".graph.reading" in name:
        return "reading"
    if ".jobs.notifications" in name or ".x.notifications" in name or ".graph.notifications" in name:
        return "notifications"
    if ".jobs.replies" in name or ".x.replies" in name or ".graph.replies" in name:
        return "replies"
    if ".langchain_clients" in name or ".core.llm" in name:
        return "llm"
    if ".web.routes" in name or ".web.app" in name:
        return "web"
    if ".web.auth" in name:
        return "auth"
    if ".scheduler" in name or "apscheduler" in name:
        return "scheduler"
    if ".state.database" in name:
        return "db"
    if ".state.manager" in name:
        return "state"
    if ".x.driver" in name:
        return "driver"
    if ".x." in name:
        return "x"
    if ".memory." in name:
        return "memory"
    if logger_name == "__main__":
        return "system"
    return "system"


def _status_for(record: logging.LogRecord, event: str) -> str:
    if record.levelno >= logging.ERROR:
        return "error"
    if record.levelno >= logging.WARNING:
        return "warn"
    for suffix, status in _STATUS_BY_SUFFIX.items():
        if event.endswith(suffix):
            return status
    return "info" if record.levelno == logging.INFO else "debug"


def _humanize_event(event: str, rendered_message: str) -> str:
    if event in _EVENT_LABELS:
        return _EVENT_LABELS[event]
    # If the original event is already a readable sentence, use the rendered message.
    if " " in event or ":" in event or "%" in event:
        return rendered_message
    words = event.replace("-", "_").split("_")
    if not words:
        return rendered_message
    label = " ".join(w for w in words if w)
    if not label:
        return rendered_message
    return label.capitalize()


def _ordered_context_items(context: dict[str, Any]) -> list[tuple[str, Any]]:
    seen: set[str] = set()
    items: list[tuple[str, Any]] = []
    for key in _PREFERRED_CONTEXT_KEYS:
        if key in context:
            items.append((key, context[key]))
            seen.add(key)
    for key in sorted(context.keys()):
        if key not in seen:
            items.append((key, context[key]))
    return items


def _colorize(text: str, color_name: str | None, enabled: bool) -> str:
    if not enabled or not color_name or color_name not in _ANSI:
        return text
    return f"{_ANSI[color_name]}{text}{_ANSI['reset']}"


class _BaseEventFormatter(logging.Formatter):
    """Common log normalization for human and JSONL formatters."""

    def _normalize(self, record: logging.LogRecord) -> dict[str, Any]:
        rendered_message = record.getMessage()
        raw_event = record.msg if isinstance(record.msg, str) else rendered_message
        event = raw_event or rendered_message
        context = _extract_extra(record)
        task = _classify_task(record.name)
        status = _status_for(record, event)

        exception: dict[str, Any] | None = None
        if record.exc_info:
            exc_type = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            exc_message = str(record.exc_info[1]) if record.exc_info[1] else ""
            exception = {
                "type": exc_type,
                "message": exc_message,
                "traceback": "".join(traceback.format_exception(*record.exc_info)).rstrip(),
            }

        created_utc = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return {
            "timestamp": created_utc.isoformat(),
            "timestamp_local": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
            "time_local": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "event": event,
            "message": _humanize_event(event, rendered_message),
            "rendered_message": rendered_message,
            "task": task,
            "status": status,
            "context": context,
            "exception": exception,
        }


class HumanConsoleFormatter(_BaseEventFormatter):
    """Human-readable formatter for terminal output."""

    def __init__(
        self,
        *,
        use_emoji: bool = True,
        use_color: bool = True,
        context_max_len: int = 120,
        context_max_items: int = 6,
    ) -> None:
        super().__init__()
        self.use_emoji = use_emoji
        self.use_color = use_color
        self.context_max_len = context_max_len
        self.context_max_items = context_max_items

    def format(self, record: logging.LogRecord) -> str:
        event = self._normalize(record)

        severity_icon = _LEVEL_EMOJI.get(event["level"], "")
        task_icon = _TASK_EMOJI.get(event["task"], "")
        icons = " ".join(i for i in [severity_icon, task_icon] if i and self.use_emoji).strip()
        icons = f"{icons} " if icons else ""

        level_color = _LEVEL_COLORS.get(event["level"])
        task_label = f"[{event['task']}]"
        task_label = _colorize(task_label, level_color, self.use_color)

        rendered = event["rendered_message"]
        message = event["message"]
        # If rendered message contains useful parameterized detail, prefer it.
        if rendered and rendered != event["event"] and ":" in rendered:
            message = rendered

        parts = [f"{event['time_local']} {icons}{task_label} {message}"]

        context_items = _ordered_context_items(event["context"])
        if context_items:
            compact_items: list[str] = []
            for idx, (key, value) in enumerate(context_items):
                if idx >= self.context_max_items:
                    compact_items.append(f"+{len(context_items) - self.context_max_items} more")
                    break
                compact_items.append(f"{key}={_safe_compact(value, self.context_max_len)}")
            context_text = " ".join(compact_items)
            if context_text:
                parts[0] = f"{parts[0]} | {context_text}"

        if event["exception"]:
            exc = event["exception"]
            parts.append(f"    {exc['type']}: {exc['message']}")
            if exc.get("traceback") and record.levelno >= logging.WARNING:
                parts.append(exc["traceback"])

        return "\n".join(parts)


class JsonLinesFormatter(_BaseEventFormatter):
    """JSON Lines formatter for file logs."""

    def format(self, record: logging.LogRecord) -> str:
        event = self._normalize(record)
        # rendered_message is redundant in file logs but useful when message is humanized
        payload = {
            "timestamp": event["timestamp"],
            "level": event["level"],
            "logger": event["logger"],
            "module": event["module"],
            "event": event["event"],
            "message": event["rendered_message"],
            "message_human": event["message"],
            "task": event["task"],
            "status": event["status"],
            "context": event["context"],
            "exception": event["exception"],
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    log_level: str | int = logging.INFO,
    log_dir: str | Path = "logs",
    log_file: str = "bot.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    console_format: str = "human",
    file_format: str = "jsonl",
    use_color: bool = True,
    use_emoji: bool = True,
    console_context_max_len: int = 120,
    console_context_max_items: int = 6,
) -> None:
    """Configure root logging with human console output and JSONL file logs."""
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_path / log_file

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    if file_format == "jsonl":
        file_handler.setFormatter(JsonLinesFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        color_enabled = use_color and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        if console_format == "human":
            console_handler.setFormatter(
                HumanConsoleFormatter(
                    use_emoji=use_emoji,
                    use_color=color_enabled,
                    context_max_len=console_context_max_len,
                    context_max_items=console_context_max_items,
                )
            )
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        root_logger.addHandler(console_handler)

    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
