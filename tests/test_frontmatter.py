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
    assert meta["concepts"] == ["diffusion", "flow-matching"]
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
    assert meta == {}
    assert "# Body" in body


def test_validate_passes_on_valid_paper(sample_frontmatter_paper):
    meta, _ = parse(sample_frontmatter_paper)
    errors = validate(meta, expected_slug="l2p-2026", expected_dir="papers")
    assert errors == []


def test_validate_catches_missing_required():
    meta = {"type": "paper", "slug": "x", "title": "X"}  # missing date/tldr/concepts
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("date" in e for e in errors)
    assert any("tldr" in e for e in errors)
    assert any("concepts" in e for e in errors)


def test_validate_catches_slug_mismatch(sample_frontmatter_paper):
    meta, _ = parse(sample_frontmatter_paper)
    errors = validate(meta, expected_slug="WRONG", expected_dir="papers")
    assert any("slug" in e.lower() for e in errors)


def test_validate_catches_bad_type():
    meta = {"type": "blog-post", "slug": "x", "title": "X", "date": "2026-01-01",
            "tldr": "y", "concepts": []}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("type" in e for e in errors)


def test_validate_catches_bad_date_format():
    meta = {"type": "paper", "slug": "x", "title": "X", "date": "May 22 2026",
            "tldr": "y", "concepts": []}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("date" in e for e in errors)


def test_parse_raises_on_malformed_yaml():
    with pytest.raises(ValidationError, match="YAML parse error"):
        parse("---\nkey: : invalid\n---\nbody\n")


def test_parse_raises_on_non_mapping_root():
    with pytest.raises(ValidationError, match="must be a mapping"):
        parse("---\n- item1\n- item2\n---\nbody\n")


def test_parse_normalizes_yaml_date_to_string():
    """PyYAML parses '2026-05-22' as datetime.date; parse() must coerce to str."""
    text = "---\ndate: 2026-05-22\n---\nbody\n"
    meta, _ = parse(text)
    assert meta["date"] == "2026-05-22"
    assert isinstance(meta["date"], str)


def test_validate_catches_tutorial_under_papers():
    meta = {"type": "tutorial", "slug": "x", "title": "X", "date": "2026-01-01",
            "tldr": "y", "concepts": []}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("type" in e.lower() or "directory" in e.lower() for e in errors)


def test_validate_warns_on_legacy_tags():
    meta = {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
            "tldr": "y", "concepts": [], "tags": ["legacy"]}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("legacy" in e.lower() and "tags" in e.lower() for e in errors)
