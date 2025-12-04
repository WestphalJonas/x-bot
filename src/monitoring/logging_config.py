"""Logging configuration for the bot."""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Formatter that handles extra context dictionaries."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured extra data."""
        # Extract extra fields
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                extra_data[key] = value

        # Format base message
        base_msg = super().format(record)

        # Add structured extra data as JSON if present
        if extra_data:
            try:
                extra_json = json.dumps(extra_data, default=str, ensure_ascii=False)
                return f"{base_msg} | {extra_json}"
            except Exception:
                # Fallback if JSON serialization fails
                return f"{base_msg} | {extra_data}"

        return base_msg


def setup_logging(
    log_level: str | int = logging.INFO,
    log_dir: str | Path = "logs",
    log_file: str = "bot.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """Configure logging with file and console handlers.

    Args:
        log_level: Logging level (string like 'INFO' or logging constant)
        log_dir: Directory to store log files
        log_file: Name of the log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to also log to console
    """
    # Convert string level to logging constant
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Full path to log file
    log_file_path = log_path / log_file

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # File handler with rotation - uses structured formatter
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        StructuredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    # Console handler (optional) - simpler format
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

