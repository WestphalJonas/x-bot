"""Helpers for robustly parsing LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object_text(raw_response: str) -> str:
    """Extract a JSON object substring from a raw LLM response."""
    cleaned = (raw_response or "").strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def parse_json_object(raw_response: str) -> dict[str, Any] | None:
    """Parse a JSON object from an LLM response, returning None on failure."""
    try:
        parsed = json.loads(extract_json_object_text(raw_response))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


def parse_structured_bool_response(
    raw_response: str,
    *,
    bool_key: str,
    reason_key: str = "reason",
    legacy_true_prefixes: tuple[str, ...] = (),
    legacy_false_prefixes: tuple[str, ...] = (),
) -> tuple[bool | None, str | None]:
    """Parse a JSON bool field with optional fallback for truncated/legacy responses."""
    parsed = parse_json_object(raw_response)
    if parsed is not None and bool_key in parsed:
        value = parsed.get(bool_key)
        reason = str(parsed.get(reason_key) or "").strip() or None
        return bool(value), reason

    lowered = (raw_response or "").lower()
    escaped_key = re.escape(bool_key)
    if re.search(rf'"{escaped_key}"\s*:\s*true', lowered):
        return True, None
    if re.search(rf'"{escaped_key}"\s*:\s*false', lowered):
        return False, None

    upper = (raw_response or "").strip().upper()
    if legacy_true_prefixes and upper.startswith(legacy_true_prefixes):
        return True, None
    if legacy_false_prefixes and upper.startswith(legacy_false_prefixes):
        return False, None

    return None, None
