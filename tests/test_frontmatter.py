"""Tests for build_lib/frontmatter.py."""
from __future__ import annotations

import pytest
from build_lib.frontmatter import parse, validate, ValidationError


def test_parse_extracts_metadata_and_body(sample_frontmatter_paper):
    meta, body = parse(sample_frontmatter_paper)
    assert meta["type"] == "paper"
    assert meta["slug"] == "l2p-2026"
    assert meta["title"] == "L2P: example"
    assert meta["date"] == "2026-05-22"
    assert "A multi-line" in meta["tldr"]
    assert meta["tags"] == ["diffusion", "flow-matching"]
    assert meta["paper"]["arxiv_id"] == "2605.12013"
    assert body.lstrip().startswith("# L2P")


def test_parse_no_frontmatter_returns_empty_meta():
    text = "# Just markdown\n\nNo frontmatter here."
    meta, body = parse(text)
    assert meta == {}
    assert body == text


def test_parse_empty_frontmatter():
    text = "---\n---\n\n# Body"
    meta, body = parse(text)
    assert meta == {} or meta is None or meta == {}
    assert "# Body" in body


def test_validate_passes_on_valid_paper(sample_frontmatter_paper):
    meta, _ = parse(sample_frontmatter_paper)
    errors = validate(meta, expected_slug="l2p-2026", expected_dir="papers")
    assert errors == []


def test_validate_catches_missing_required():
    meta = {"type": "paper", "slug": "x", "title": "X"}  # missing date/tldr/tags
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("date" in e for e in errors)
    assert any("tldr" in e for e in errors)
    assert any("tags" in e for e in errors)


def test_validate_catches_slug_mismatch(sample_frontmatter_paper):
    meta, _ = parse(sample_frontmatter_paper)
    errors = validate(meta, expected_slug="WRONG", expected_dir="papers")
    assert any("slug" in e.lower() for e in errors)


def test_validate_catches_bad_type():
    meta = {"type": "blog-post", "slug": "x", "title": "X", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("type" in e for e in errors)


def test_validate_catches_bad_date_format():
    meta = {"type": "paper", "slug": "x", "title": "X", "date": "May 22 2026",
            "tldr": "y", "tags": []}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("date" in e for e in errors)


def test_validate_catches_tutorial_under_papers():
    meta = {"type": "tutorial", "slug": "x", "title": "X", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("type" in e.lower() or "directory" in e.lower() for e in errors)
