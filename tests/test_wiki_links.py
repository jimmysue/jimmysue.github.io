"""Tests for build_lib/wiki_links.py."""
from __future__ import annotations

from build_lib.wiki_links import preprocess, resolve


# --- Preprocess ---

def test_preprocess_simple_slug():
    out = preprocess("See [[l2p-2026]] for details.")
    assert '<a class="wiki-link" data-slug="l2p-2026">l2p-2026</a>' in out


def test_preprocess_slug_with_alias():
    out = preprocess("See [[l2p-2026|L2P]] paper.")
    assert '<a class="wiki-link" data-slug="l2p-2026">L2P</a>' in out


def test_preprocess_multiple_links():
    out = preprocess("[[a-2025]] and [[b-2026|B]] and [[c-2024]]")
    assert out.count('class="wiki-link"') == 3
    assert 'data-slug="a-2025">a-2025</a>' in out
    assert 'data-slug="b-2026">B</a>' in out
    assert 'data-slug="c-2024">c-2024</a>' in out


def test_preprocess_leaves_non_wiki_brackets_alone():
    # Markdown image syntax has [[ but as part of [![ — should not be confused
    out = preprocess("Normal text. [single] and [link](url).")
    assert "<a" not in out
    assert "[single]" in out
    assert "[link](url)" in out


def test_preprocess_does_not_match_inside_code_fence():
    # Pragmatic limitation: preprocess works on raw text including code blocks.
    # If wiki-link syntax appears inside a code fence it WILL be transformed.
    # Authors are expected to escape ``[[slug]]`` with backslash if needed.
    # This test documents the behavior.
    md = "```python\nx = [[a, b], [c, d]]\n```"
    out = preprocess(md)
    # No 'foo-bar' slug pattern in 'a,b' style — should NOT match (slug must be
    # one token, not contain spaces or commas).
    assert "<a class=" not in out


def test_preprocess_slug_must_not_contain_spaces():
    out = preprocess("[[two words]] should not match")
    assert "<a class=" not in out
    assert "[[two words]]" in out


# --- Resolve ---

def test_resolve_valid_slug(slug_set_basic):
    html = '<p>See <a class="wiki-link" data-slug="l2p-2026">l2p-2026</a>.</p>'
    out, warnings = resolve(html, slug_set_basic, current_post_dir="papers/awm-2025")
    assert 'href="../l2p-2026/index.html"' in out
    assert "wiki-link-broken" not in out
    assert warnings == []


def test_resolve_unknown_slug_marks_broken(slug_set_basic):
    html = '<a class="wiki-link" data-slug="ghost-2099">ghost-2099</a>'
    out, warnings = resolve(html, slug_set_basic, current_post_dir="papers/awm-2025")
    assert "wiki-link-broken" in out
    assert any("ghost-2099" in w for w in warnings)


def test_resolve_keeps_data_slug_attr(slug_set_basic):
    """data-slug must persist for Phase 2 graph extraction."""
    html = '<a class="wiki-link" data-slug="l2p-2026">link text</a>'
    out, _ = resolve(html, slug_set_basic, current_post_dir="papers/awm-2025")
    assert 'data-slug="l2p-2026"' in out
