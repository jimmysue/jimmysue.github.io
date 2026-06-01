"""YAML frontmatter parsing + validation."""
from __future__ import annotations

import datetime
import re
from typing import Any

import yaml


REQUIRED_KEYS = {"type", "slug", "title", "date", "tldr", "tags"}
VALID_TYPES = {"paper", "tutorial"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class ValidationError(ValueError):
    """Frontmatter validation failure."""


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into (frontmatter_dict, body).

    Returns ({}, original_text) when no frontmatter is found.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    yaml_block = m.group(1)
    body = text[m.end():]
    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as e:
        raise ValidationError(f"YAML parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError(f"Frontmatter must be a mapping, got {type(data).__name__}")
    # Normalize date objects to ISO strings so callers always see plain strings
    if "date" in data and isinstance(data["date"], (datetime.date, datetime.datetime)):
        data["date"] = data["date"].isoformat()[:10]
    return data, body


def validate(meta: dict[str, Any], expected_slug: str, expected_dir: str) -> list[str]:
    """Return a list of human-readable error strings. Empty = valid.

    expected_dir is "papers" or "tutorials" — used to cross-check `type`.
    """
    errors: list[str] = []

    missing = REQUIRED_KEYS - set(meta.keys())
    for key in sorted(missing):
        errors.append(f"missing required key: {key}")

    if "type" in meta and meta["type"] not in VALID_TYPES:
        errors.append(f"invalid type: {meta['type']!r} (must be paper|tutorial)")

    if "type" in meta and expected_dir:
        expected_type = "paper" if expected_dir == "papers" else "tutorial"
        if meta["type"] != expected_type:
            errors.append(
                f"type {meta['type']!r} does not match directory {expected_dir!r} "
                f"(expected type={expected_type!r})"
            )

    if "slug" in meta and meta["slug"] != expected_slug:
        errors.append(f"slug {meta['slug']!r} does not match parent dir name {expected_slug!r}")

    if "date" in meta:
        date = meta["date"]
        # PyYAML may parse 2026-05-22 as a date object; coerce to ISO string
        date_str = str(date) if not isinstance(date, str) else date
        if not ISO_DATE_RE.match(date_str):
            errors.append(f"date {date!r} is not ISO 8601 (YYYY-MM-DD)")

    if "tags" in meta and not isinstance(meta["tags"], list):
        errors.append(f"tags must be a list, got {type(meta['tags']).__name__}")

    return errors
