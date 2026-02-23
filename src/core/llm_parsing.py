"""Helpers for robustly parsing LLM responses."""

from __future__ import annotations

import json
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
