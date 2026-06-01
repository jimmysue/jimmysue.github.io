# Markdown-Source Blog — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert per-paper / per-tutorial pages from hand-written HTML to markdown source + build.py render, as the foundation for Phase 2 knowledge graph.

**Architecture:** Markdown source (`papers/<slug>/index.md`, `tutorials/<slug>/index.md`) with YAML frontmatter replaces the existing `meta.json + index.html` split. A new `build_lib/` package handles parsing/rendering; the existing `build.py` stays as the CLI orchestrator. Generated HTML is committed alongside markdown.

**Tech Stack:**
- Python 3.9 (`from __future__ import annotations` everywhere for `list[dict]`-style hints)
- markdown-it-py + mdit-py-plugins (markdown rendering)
- PyYAML (frontmatter parsing)
- beautifulsoup4 + html5lib (one-shot migration tool only)
- pytest (test harness)

Spec reference: `docs/superpowers/specs/2026-05-29-md-blog-phase1-design.md`

---

## File Structure

```
paper-reading/
  requirements.txt              # NEW — markdown-it-py, mdit-py-plugins, PyYAML, BS4, html5lib
  build.py                      # MODIFY — orchestrator; calls into build_lib/ for MD rendering
  migrate-md.py                 # NEW — one-shot HTML→MD migrator (Phase 1 only)
  build_lib/                    # NEW package
    __init__.py
    frontmatter.py              # parse + validate YAML frontmatter
    wiki_links.py               # [[slug]] preprocess + resolve
    headings.py                 # sec-N / sec-N-M ID injection + TOC HTML
    figures.py                  # lightbox class tagging on <figure> > img
    markdown.py                 # tie everything: MD text → rendered HTML body
    post_assembly.py            # MD body → full HTML page (head + nav + body + toc + scripts)
  tests/
    __init__.py
    conftest.py                 # shared fixtures
    test_frontmatter.py
    test_wiki_links.py
    test_headings.py
    test_figures.py
    test_markdown.py
    test_post_assembly.py
    test_migrate_md.py
  papers/<slug>/                # 22 dirs: each gets new index.md (migrated), meta.json removed
  tutorials/<slug>/              # 3 dirs: same migration
  .claude/skills/reading-papers/
    SKILL.md                    # MODIFY — emit MD instead of HTML
    templates/index.md           # NEW — replaces templates/index.html
  .claude/skills/writing-tutorial/
    SKILL.md                    # MODIFY
    templates/skeleton.md       # NEW
    templates/spiral-section.md # NEW
  CLAUDE.md                     # MODIFY — update per-paper/tutorial layout sections
```

**Independence map (for subagent dispatch):**

- Tasks 2, 3, 4, 5 are independent (different module each) — can dispatch in parallel
- Task 6 depends on 2-5
- Task 7 depends on 2
- Task 8 depends on 6, 7
- Task 9 depends on 8
- Task 10 is independent (only uses BS4) — can dispatch anytime
- Task 11 depends on 9, 10
- Tasks 12, 13, 14 depend on 11 (need at least one example MD to reference)
- Task 15 is final

---

## Task 1: Setup — Dependencies, package skeleton, pytest harness

**Files:**
- Create: `requirements.txt`
- Create: `build_lib/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore` (add `__pycache__/`, `.pytest_cache/`)

- [ ] **Step 1.1: Create `requirements.txt`**

```
markdown-it-py>=3.0
mdit-py-plugins>=0.4
PyYAML>=6.0
# Migration-only (can remove after Phase 1):
beautifulsoup4>=4.12
html5lib>=1.1
# Dev:
pytest>=7.0
```

- [ ] **Step 1.2: Install dependencies**

Run: `pip3 install -r requirements.txt`
Expected: clean install (or already-satisfied).

- [ ] **Step 1.3: Create `build_lib/__init__.py`**

```python
"""Building blocks for the markdown-source blog renderer.

This package is imported by build.py. Each module has a single
responsibility:

  frontmatter   — parse + validate YAML frontmatter
  wiki_links    — [[slug]] preprocess + resolve
  headings      — sec-N IDs + TOC generation
  figures       — lightbox class tagging
  markdown      — orchestrate the full markdown → HTML body pipeline
  post_assembly — wrap body in full HTML page (head/nav/toc/scripts)
"""
```

- [ ] **Step 1.4: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 1.5: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_frontmatter_paper() -> str:
    return """---
type: paper
slug: l2p-2026
title: "L2P: example"
date: 2026-05-22
tldr: |
  A multi-line
  summary.
tags: [diffusion, flow-matching]
paper:
  arxiv_id: "2605.12013"
  authors: "Author A, Author B"
---

# L2P

Body text.
"""


@pytest.fixture
def slug_set_basic() -> set[str]:
    return {"l2p-2026", "asymflow-2026", "awm-2025"}
```

- [ ] **Step 1.6: Update `.gitignore`**

Append to existing file:
```
__pycache__/
.pytest_cache/
*.pyc
```

- [ ] **Step 1.7: Verify pytest harness works**

Run: `python3 -m pytest tests/ -v`
Expected: `no tests ran` (collected 0 items) — but NO errors about missing dependencies or import errors.

- [ ] **Step 1.8: Commit**

```bash
git add requirements.txt build_lib/__init__.py tests/__init__.py tests/conftest.py .gitignore
git commit -m "$(cat <<'EOF'
Phase 1 scaffold: requirements + build_lib/ package + pytest harness

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Frontmatter parser + validator

**Files:**
- Create: `build_lib/frontmatter.py`
- Create: `tests/test_frontmatter.py`

- [ ] **Step 2.1: Write failing tests**

`tests/test_frontmatter.py`:

```python
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
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_frontmatter.py -v`
Expected: ImportError on `build_lib.frontmatter`.

- [ ] **Step 2.3: Implement `build_lib/frontmatter.py`**

```python
"""YAML frontmatter parsing + validation."""
from __future__ import annotations

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
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_frontmatter.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add build_lib/frontmatter.py tests/test_frontmatter.py
git commit -m "$(cat <<'EOF'
Add frontmatter parser + validator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wiki-link preprocess + resolve

**Files:**
- Create: `build_lib/wiki_links.py`
- Create: `tests/test_wiki_links.py`

Wiki links are a two-step process:
1. **Preprocess** on raw markdown text — regex replaces `[[slug]]` / `[[slug|alias]]` with `<a class="wiki-link" data-slug="slug">slug-or-alias</a>` (markdown-it-py passes raw HTML through unchanged).
2. **Resolve** on rendered HTML — find every `<a class="wiki-link" data-slug=...>` and add `href="../<slug>/index.html"` if slug is valid, or add `wiki-link-broken` class if not.

- [ ] **Step 3.1: Write failing tests**

`tests/test_wiki_links.py`:

```python
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
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_wiki_links.py -v`
Expected: ImportError.

- [ ] **Step 3.3: Implement `build_lib/wiki_links.py`**

```python
"""Wiki-link [[slug]] preprocess (MD text) and resolve (HTML).

Two-step design:
  1. preprocess(md_text)  — regex replaces [[slug]] / [[slug|alias]] with
                           raw HTML; markdown-it-py passes it through.
  2. resolve(html, slug_set, current_post_dir)
                          — adds href= to known slugs, marks unknowns broken.
"""
from __future__ import annotations

import re


# slug = ASCII letters/digits/hyphens, no whitespace, no | or ]
WIKI_LINK_RE = re.compile(r"\[\[([A-Za-z0-9][A-Za-z0-9\-_]*)(?:\|([^\]]+))?\]\]")

# Find <a class="wiki-link" ... data-slug="x">text</a> for resolve()
WIKI_ANCHOR_RE = re.compile(
    r'<a class="wiki-link" data-slug="([^"]+)">([^<]*)</a>'
)


def preprocess(md_text: str) -> str:
    """Replace [[slug]] and [[slug|alias]] in markdown text with raw HTML anchors.

    The output anchor keeps `data-slug` so the post-process pass can
    resolve href= later.
    """
    def _sub(m: re.Match) -> str:
        slug = m.group(1)
        alias = m.group(2) if m.group(2) else slug
        return f'<a class="wiki-link" data-slug="{slug}">{alias}</a>'
    return WIKI_LINK_RE.sub(_sub, md_text)


def resolve(html: str, slug_set: set[str], current_post_dir: str) -> tuple[str, list[str]]:
    """Resolve wiki-link anchors. Returns (rewritten_html, warnings).

    `current_post_dir` is like "papers/awm-2025" — used to compute relative
    href (the target is always "../<target-slug>/index.html").

    Unknown slugs get class="wiki-link wiki-link-broken" and no href.
    """
    warnings: list[str] = []

    def _sub(m: re.Match) -> str:
        slug = m.group(1)
        text = m.group(2)
        if slug in slug_set:
            href = f"../{slug}/index.html"
            return (
                f'<a class="wiki-link" data-slug="{slug}" href="{href}">{text}</a>'
            )
        else:
            warnings.append(f"{current_post_dir}: broken wiki-link [[{slug}]]")
            return (
                f'<a class="wiki-link wiki-link-broken" data-slug="{slug}">{text}</a>'
            )

    return WIKI_ANCHOR_RE.sub(_sub, html), warnings
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_wiki_links.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add build_lib/wiki_links.py tests/test_wiki_links.py
git commit -m "$(cat <<'EOF'
Add wiki-link [[slug]] preprocess + resolve

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Heading IDs + TOC extraction

**Files:**
- Create: `build_lib/headings.py`
- Create: `tests/test_headings.py`

The rule (from spec §6.3):
- 1st h2 → `id="sec-1"`, 2nd h2 → `id="sec-2"`, ...
- Within h2 #N, 1st h3 → `id="sec-N-1"`, 2nd h3 → `id="sec-N-2"`, ...
- h1/h4/h5/h6 are not assigned IDs (h1 = page title; h4+ not in TOC)

TOC is generated by walking the resulting `(h2|h3)[id^="sec-"]` set.

- [ ] **Step 4.1: Write failing tests**

`tests/test_headings.py`:

```python
"""Tests for build_lib/headings.py."""
from __future__ import annotations

from build_lib.headings import inject_ids, build_toc_html


def test_inject_ids_single_h2():
    html = "<h2>First</h2>"
    out = inject_ids(html)
    assert '<h2 id="sec-1">First</h2>' in out


def test_inject_ids_h2_increments():
    html = "<h2>A</h2><p>x</p><h2>B</h2>"
    out = inject_ids(html)
    assert 'id="sec-1">A</h2>' in out
    assert 'id="sec-2">B</h2>' in out


def test_inject_ids_h3_resets_per_h2():
    html = "<h2>A</h2><h3>a1</h3><h3>a2</h3><h2>B</h2><h3>b1</h3>"
    out = inject_ids(html)
    assert 'id="sec-1">A</h2>' in out
    assert 'id="sec-1-1">a1</h3>' in out
    assert 'id="sec-1-2">a2</h3>' in out
    assert 'id="sec-2">B</h2>' in out
    assert 'id="sec-2-1">b1</h3>' in out


def test_inject_ids_h3_before_first_h2_skipped():
    """An h3 with no enclosing h2 has no parent number — skip ID injection."""
    html = "<h3>orphan</h3><h2>A</h2><h3>child</h3>"
    out = inject_ids(html)
    # Orphan h3 stays unchanged
    assert "<h3>orphan</h3>" in out
    # h2 gets sec-1, child h3 gets sec-1-1
    assert 'id="sec-1">A</h2>' in out
    assert 'id="sec-1-1">child</h3>' in out


def test_inject_ids_preserves_existing_attributes():
    """If author writes <h2 class="foo">X</h2>, the class should be kept."""
    html = '<h2 class="custom">X</h2>'
    out = inject_ids(html)
    assert 'id="sec-1"' in out
    assert 'class="custom"' in out
    assert ">X</h2>" in out


def test_inject_ids_skips_h1_h4():
    html = "<h1>Title</h1><h2>A</h2><h4>note</h4>"
    out = inject_ids(html)
    # h1 unchanged
    assert "<h1>Title</h1>" in out
    # h4 unchanged
    assert "<h4>note</h4>" in out
    # h2 gets id
    assert 'id="sec-1"' in out


# --- TOC ---

def test_build_toc_html_basic():
    html = (
        '<h2 id="sec-1">Intro</h2>'
        '<h3 id="sec-1-1">Setup</h3>'
        '<h2 id="sec-2">Method</h2>'
    )
    toc = build_toc_html(html)
    assert 'class="toc"' in toc
    assert '<a href="#sec-1">Intro</a>' in toc
    assert '<a href="#sec-1-1">Setup</a>' in toc
    assert '<a href="#sec-2">Method</a>' in toc
    # h2 entries should have class h2, h3 entries class h3 (spec §TOC contract)
    assert 'class="h2"' in toc
    assert 'class="h3"' in toc


def test_build_toc_html_strips_inline_markup():
    """If h2 contains <strong>, the TOC link text is plain."""
    html = '<h2 id="sec-1">Intro <strong>important</strong></h2>'
    toc = build_toc_html(html)
    assert "Intro important" in toc
    # No nested <strong> in the TOC link
    assert "<strong>" not in toc


def test_build_toc_html_empty_when_no_headings():
    toc = build_toc_html("<p>no headings here</p>")
    assert toc == "" or 'class="toc"' not in toc
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_headings.py -v`
Expected: ImportError.

- [ ] **Step 4.3: Implement `build_lib/headings.py`**

```python
"""Heading ID injection (sec-N / sec-N-M) + right-rail TOC HTML."""
from __future__ import annotations

import re


# Match <h2> or <h3> open tag with optional existing attributes.
# Capture: 1=tag name (h2 or h3), 2=existing attrs (may be empty), 3=text content
H2_OR_H3_RE = re.compile(
    r"<(h[23])([^>]*)>(.*?)</\1>",
    re.DOTALL,
)

# For TOC: match injected <h2 id="sec-N">text</h2> and <h3 id="sec-N-M">text</h3>
TOC_HEADING_RE = re.compile(
    r'<(h[23])\s+id="(sec-[\d\-]+)"[^>]*>(.*?)</\1>',
    re.DOTALL,
)

# Strip inline HTML for TOC link text
INLINE_TAG_RE = re.compile(r"<[^>]+>")


def inject_ids(html: str) -> str:
    """Walk h2/h3 in document order, assign sec-N / sec-N-M IDs."""
    h2_count = 0
    h3_count = 0

    def _sub(m: re.Match) -> str:
        nonlocal h2_count, h3_count
        tag = m.group(1)
        attrs = m.group(2) or ""
        text = m.group(3)
        if tag == "h2":
            h2_count += 1
            h3_count = 0
            heading_id = f"sec-{h2_count}"
        else:  # h3
            if h2_count == 0:
                # Orphan h3 before any h2 — leave unchanged
                return m.group(0)
            h3_count += 1
            heading_id = f"sec-{h2_count}-{h3_count}"
        # Preserve any existing attributes (class, etc.), prepend id=
        attrs_str = attrs.strip()
        if attrs_str:
            new_open = f'<{tag} id="{heading_id}" {attrs_str}>'
        else:
            new_open = f'<{tag} id="{heading_id}">'
        return f"{new_open}{text}</{tag}>"

    return H2_OR_H3_RE.sub(_sub, html)


def build_toc_html(html: str) -> str:
    """Build the right-rail <nav class='toc'> block from already-injected IDs.

    Returns "" if no headings found.
    """
    items: list[tuple[str, str, str]] = []
    for m in TOC_HEADING_RE.finditer(html):
        tag, heading_id, text = m.group(1), m.group(2), m.group(3)
        clean_text = INLINE_TAG_RE.sub("", text).strip()
        items.append((tag, heading_id, clean_text))

    if not items:
        return ""

    lines = ['<nav class="toc" aria-label="目录">']
    lines.append('  <div class="toc-title">目录 / TOC</div>')
    lines.append('  <ul>')
    for tag, heading_id, text in items:
        lines.append(
            f'    <li class="{tag}"><a href="#{heading_id}">{text}</a></li>'
        )
    lines.append("  </ul>")
    lines.append("</nav>")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_headings.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add build_lib/headings.py tests/test_headings.py
git commit -m "$(cat <<'EOF'
Add heading ID injection + TOC generation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Figure lightbox class tagging

**Files:**
- Create: `build_lib/figures.py`
- Create: `tests/test_figures.py`

- [ ] **Step 5.1: Write failing tests**

`tests/test_figures.py`:

```python
"""Tests for build_lib/figures.py."""
from __future__ import annotations

from build_lib.figures import tag_for_lightbox


def test_tag_adds_zoomable_class():
    html = '<figure><img src="x.png" alt="a"><figcaption>cap</figcaption></figure>'
    out = tag_for_lightbox(html)
    assert 'class="zoomable"' in out
    assert '<img class="zoomable" src="x.png" alt="a">' in out


def test_tag_preserves_existing_img_class():
    html = '<figure><img class="big" src="x.png"></figure>'
    out = tag_for_lightbox(html)
    assert "big" in out
    assert "zoomable" in out


def test_tag_skips_img_outside_figure():
    html = '<p><img src="inline.png" alt="x"></p>'
    out = tag_for_lightbox(html)
    assert "zoomable" not in out
    assert '<img src="inline.png" alt="x">' in out


def test_tag_multiple_figures():
    html = (
        '<figure><img src="a.png"><figcaption>A</figcaption></figure>'
        '<p>between</p>'
        '<figure><img src="b.png"></figure>'
    )
    out = tag_for_lightbox(html)
    # both <img> inside figures should get zoomable
    assert out.count('class="zoomable"') == 2


def test_tag_already_zoomable_unchanged():
    """If a <figure> > img already has class='zoomable', don't duplicate."""
    html = '<figure><img class="zoomable" src="x.png"></figure>'
    out = tag_for_lightbox(html)
    assert out.count("zoomable") == 1
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_figures.py -v`
Expected: ImportError.

- [ ] **Step 5.3: Implement `build_lib/figures.py`**

```python
"""Add class='zoomable' to <img> inside <figure>, for the lightbox JS."""
from __future__ import annotations

import re


# Match <figure ...>...</figure> non-greedy, then transform <img> tags inside.
FIGURE_RE = re.compile(r"<figure([^>]*)>(.*?)</figure>", re.DOTALL)


def tag_for_lightbox(html: str) -> str:
    """Add class='zoomable' to every <img> inside <figure>.

    Idempotent: if class='zoomable' already present, leave it alone.
    """
    def _fig_sub(m: re.Match) -> str:
        fig_attrs = m.group(1)
        inner = m.group(2)
        new_inner = _retag_imgs(inner)
        return f"<figure{fig_attrs}>{new_inner}</figure>"

    return FIGURE_RE.sub(_fig_sub, html)


_IMG_OPEN_RE = re.compile(r"<img\b([^>]*?)/?>", re.DOTALL)
_CLASS_ATTR_RE = re.compile(r'\bclass\s*=\s*"([^"]*)"')


def _retag_imgs(fragment: str) -> str:
    def _sub(m: re.Match) -> str:
        attrs = m.group(1)
        cm = _CLASS_ATTR_RE.search(attrs)
        if cm:
            classes = cm.group(1).split()
            if "zoomable" in classes:
                return m.group(0)  # already has it
            classes.append("zoomable")
            new_attrs = _CLASS_ATTR_RE.sub(f'class="{" ".join(classes)}"', attrs)
        else:
            new_attrs = ' class="zoomable"' + attrs
        return f"<img{new_attrs}>"

    return _IMG_OPEN_RE.sub(_sub, fragment)
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_figures.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add build_lib/figures.py tests/test_figures.py
git commit -m "$(cat <<'EOF'
Add lightbox class tagging for <figure> > img

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Markdown renderer (tie everything)

**Files:**
- Create: `build_lib/markdown.py`
- Create: `tests/test_markdown.py`

This module orchestrates: preprocess wiki-links → markdown-it-py parse → post-process (inject IDs, resolve wiki-links, lightbox tag) → also returns the TOC HTML.

- [ ] **Step 6.1: Write failing tests**

`tests/test_markdown.py`:

```python
"""Tests for build_lib/markdown.py (end-to-end MD→HTML pipeline)."""
from __future__ import annotations

from build_lib.markdown import render_post_body


def test_render_basic_markdown(slug_set_basic):
    md = "# Title\n\nA paragraph with **bold**.\n\n## Section 1\n\nText."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert '<h2 id="sec-1">Section 1</h2>' in html
    assert warnings == []


def test_render_handles_inline_math(slug_set_basic):
    md = "Inline math: $x_t = \\sigma$ here."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    # markdown-it-py with dollarmath plugin emits math; specific markup varies.
    # We assert that the dollar signs are NOT escaped and the content is preserved.
    assert "x_t" in html
    assert "\\sigma" in html or "σ" in html or "sigma" in html


def test_render_handles_display_math(slug_set_basic):
    md = "Before.\n\n$$ x_t = (1-\\sigma) x_0 $$\n\nAfter."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    # Display math should survive as $$ ... $$ for MathJax to pick up, OR be wrapped
    # in some math container. Either way the LaTeX source must be present.
    assert "x_t" in html
    assert "(1-\\sigma)" in html or "1-σ" in html


def test_render_wiki_links_resolved(slug_set_basic):
    md = "See [[l2p-2026]] paper."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/awm-2025")
    assert 'href="../l2p-2026/index.html"' in html
    assert "wiki-link-broken" not in html


def test_render_broken_wiki_link_warned(slug_set_basic):
    md = "See [[ghost-2099]] paper."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/awm-2025")
    assert "wiki-link-broken" in html
    assert any("ghost-2099" in w for w in warnings)


def test_render_embedded_html_passes_through(slug_set_basic):
    md = '<figure>\n<img src="x.png" alt="a">\n<figcaption>cap</figcaption>\n</figure>\n'
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert "<figure>" in html
    assert "<figcaption>cap</figcaption>" in html
    # Lightbox class added
    assert 'class="zoomable"' in html


def test_render_code_block_with_language(slug_set_basic):
    md = "```python\ndef hello():\n    pass\n```"
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert 'class="language-python"' in html
    assert "def hello():" in html


def test_render_gfm_table(slug_set_basic):
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert "<table" in html
    assert "<th>A</th>" in html
    assert "<td>1</td>" in html


def test_render_emits_toc_for_sections(slug_set_basic):
    md = "## A\n\ntext\n\n### A1\n\ntext\n\n## B\n\ntext\n"
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert 'class="toc"' in toc
    assert '<a href="#sec-1">A</a>' in toc
    assert '<a href="#sec-1-1">A1</a>' in toc
    assert '<a href="#sec-2">B</a>' in toc
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_markdown.py -v`
Expected: ImportError on `build_lib.markdown`.

- [ ] **Step 6.3: Implement `build_lib/markdown.py`**

```python
"""End-to-end markdown → HTML body pipeline.

Flow:
  1. preprocess: [[slug]] → raw HTML anchors
  2. parse: markdown-it-py with plugins (front_matter, dollarmath, tables)
  3. post-process: inject sec-N IDs → resolve wiki-link href → tag figures
  4. extract TOC HTML separately

Returns: (body_html, toc_html, warnings)
"""
from __future__ import annotations

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

from build_lib.figures import tag_for_lightbox
from build_lib.headings import build_toc_html, inject_ids
from build_lib.wiki_links import preprocess as preprocess_wiki, resolve as resolve_wiki


def _make_parser() -> MarkdownIt:
    md = (
        MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False})
        .enable("table")
        .enable("strikethrough")
        .use(front_matter_plugin)          # tolerates frontmatter if present (already stripped by caller)
        .use(
            dollarmath_plugin,
            allow_labels=False,
            allow_space=True,
            allow_digits=True,
            double_inline=False,
        )
    )
    return md


_PARSER = _make_parser()


def render_post_body(
    md_text: str,
    slug_set: set[str],
    current_post_dir: str,
) -> tuple[str, str, list[str]]:
    """Render markdown body (frontmatter already stripped) to HTML.

    Returns:
        body_html — the main content HTML
        toc_html  — the <nav class="toc"> block (may be "")
        warnings  — list of human-readable warnings (e.g. broken wiki links)
    """
    # 1. Preprocess wiki-links on raw text
    md_with_wiki_html = preprocess_wiki(md_text)

    # 2. Parse markdown
    body_html = _PARSER.render(md_with_wiki_html)

    # 3a. Inject heading IDs
    body_html = inject_ids(body_html)

    # 3b. Resolve wiki-link hrefs
    body_html, warnings = resolve_wiki(body_html, slug_set, current_post_dir)

    # 3c. Tag figures for lightbox
    body_html = tag_for_lightbox(body_html)

    # 4. Build TOC from injected IDs
    toc_html = build_toc_html(body_html)

    return body_html, toc_html, warnings
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_markdown.py -v`
Expected: All 9 tests PASS.

If `dollarmath_plugin` is missing from `mdit-py-plugins` < 0.4, upgrade: `pip3 install --upgrade mdit-py-plugins`.

- [ ] **Step 6.5: Commit**

```bash
git add build_lib/markdown.py tests/test_markdown.py
git commit -m "$(cat <<'EOF'
Add end-to-end markdown render pipeline

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Per-post HTML page assembly

**Files:**
- Create: `build_lib/post_assembly.py`
- Create: `tests/test_post_assembly.py`

`assemble_post_page(meta, body_html, toc_html, nav_tmpl, asset_path_prefix)` returns the full HTML document with `<head>`, nav, body, TOC, and the CDN script tags from the existing per-paper template (MathJax, highlight.js, lightbox.js).

- [ ] **Step 7.1: Write failing tests**

`tests/test_post_assembly.py`:

```python
"""Tests for build_lib/post_assembly.py."""
from __future__ import annotations

from build_lib.post_assembly import assemble_post_page


def test_assembled_page_has_doctype():
    out = assemble_post_page(
        meta={"title": "Test", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<h1>X</h1>",
        toc_html="",
        nav_html='<nav class="site-nav"></nav>',
        asset_prefix="../../",
    )
    assert out.startswith("<!doctype html>") or out.startswith("<!DOCTYPE html>")
    assert "<html" in out and "</html>" in out


def test_assembled_page_includes_title():
    out = assemble_post_page(
        meta={"title": "My Paper Title", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<h1>X</h1>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert "<title>My Paper Title" in out


def test_assembled_page_loads_mathjax_highlightjs_lightbox():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert "mathjax" in out.lower()
    assert "highlight" in out.lower()
    assert "lightbox" in out.lower()


def test_assembled_page_uses_asset_prefix():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert '"../../assets/style.css"' in out
    assert '"../../assets/lightbox.js"' in out


def test_assembled_page_includes_toc_when_provided():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html='<nav class="toc"><ul><li><a href="#sec-1">A</a></li></ul></nav>',
        nav_html="",
        asset_prefix="../../",
    )
    assert 'class="toc"' in out


def test_assembled_page_includes_nav():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html="",
        nav_html='<nav class="site-nav">HEADER</nav>',
        asset_prefix="../../",
    )
    assert 'class="site-nav"' in out
    assert "HEADER" in out
```

- [ ] **Step 7.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_post_assembly.py -v`
Expected: ImportError.

- [ ] **Step 7.3: Implement `build_lib/post_assembly.py`**

The HTML shell is lifted verbatim from `papers/l2p-2026/index.html` head/scripts (it's the canonical per-paper template — refer to that file if anything is ambiguous).

```python
"""Wrap a rendered MD body in the full per-post HTML shell.

Shell components (in order):
  <head>  — title, viewport, style.css, MathJax + highlight.js CDN, lightbox.js,
            TOC IntersectionObserver init script
  <body>
    <nav class="site-nav">  — injected by build.py from assets/nav-header.html
    <nav class="toc">       — generated by build_lib.headings, optional
    <main>                  — body_html
"""
from __future__ import annotations

import html as _html
from typing import Any


HEAD_TMPL = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="{prefix}assets/style.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/atom-one-dark.min.css">
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['$','$'],['\\\\(','\\\\)']] }} }};
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
  <script>document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());</script>
  <script defer src="{prefix}assets/lightbox.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', () => {{
      const links = document.querySelectorAll('.toc a');
      if (!links.length) return;
      const map = new Map();
      links.forEach(a => {{
        const el = document.getElementById(a.getAttribute('href').slice(1));
        if (el) map.set(el, a);
      }});
      const observer = new IntersectionObserver((entries) => {{
        entries.forEach(e => {{
          const a = map.get(e.target);
          if (!a) return;
          if (e.isIntersecting) {{
            links.forEach(l => l.classList.remove('active'));
            a.classList.add('active');
            const toc = document.querySelector('.toc');
            if (toc) toc.scrollTo({{ top: a.offsetTop - toc.clientHeight / 2, behavior: 'smooth' }});
          }}
        }});
      }}, {{ rootMargin: '-20% 0px -70% 0px' }});
      map.forEach((_, el) => observer.observe(el));
    }});
  </script>
</head>
<body>
"""

TAIL = "</main>\n</body>\n</html>\n"


def assemble_post_page(
    meta: dict[str, Any],
    body_html: str,
    toc_html: str,
    nav_html: str,
    asset_prefix: str,
) -> str:
    """Compose the full per-post HTML page.

    asset_prefix: relative path from the post's own dir to repo root.
                  For papers/<slug>/index.html that's "../../".
    """
    title = _html.escape(str(meta.get("title", "Untitled")), quote=True)
    head = HEAD_TMPL.format(title=title, prefix=asset_prefix)
    parts = [head]
    if nav_html:
        parts.append(nav_html)
    if toc_html:
        parts.append(toc_html)
    parts.append("<main>\n")
    parts.append(body_html)
    parts.append(TAIL)
    return "".join(parts)
```

- [ ] **Step 7.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_post_assembly.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 7.5: Commit**

```bash
git add build_lib/post_assembly.py tests/test_post_assembly.py
git commit -m "$(cat <<'EOF'
Add per-post HTML page assembly (head + nav + body + toc + scripts)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire build.py — discover from frontmatter, render per-post HTML

**Files:**
- Modify: `build.py`

Three changes to `build.py`:

1. `discover_posts()` (lines ~74–119) — replace `meta.json` reading with markdown frontmatter reading. The function should still return the same dict shape (so `build_site()` keeps working).
2. New function `build_posts(posts, root)` — iterate posts, render each MD body to HTML, write to `<post_dir>/index.html`.
3. `main()` — call `build_posts()` after `build_site()`.

- [ ] **Step 8.1: Read current `discover_posts()` and `main()` to understand state**

Run: `sed -n '74,120p' build.py` and `sed -n '480,516p' build.py` to refresh.

- [ ] **Step 8.2: Replace `discover_posts()` body**

Find the existing function (`def discover_posts(root: Path) -> list[dict]:`). Replace its body so it reads markdown frontmatter from `papers/<slug>/index.md` and `tutorials/<slug>/index.md` instead of `meta.json`. Keep the same output shape (existing site-page builders use `post["title"]`, `post["tldr"]`, `post["tags"]`, `post["type"]`, `post["slug"]`, `post["date"]`, `post["_url"]`, `post["tutorial_meta"]`).

```python
from build_lib.frontmatter import parse as parse_frontmatter
from build_lib.frontmatter import validate as validate_frontmatter


def discover_posts(root: Path) -> list[dict]:
    """Walk papers/<slug>/index.md and tutorials/<slug>/index.md."""
    posts: list[dict] = []
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            md_path = sub / "index.md"
            if not md_path.is_file():
                print(f"WARN: skipping {sub} (no index.md)", file=sys.stderr)
                continue
            try:
                text = md_path.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except Exception as e:
                print(f"WARN: skipping {md_path} (frontmatter parse error: {e})", file=sys.stderr)
                continue
            errs = validate_frontmatter(meta, expected_slug=sub.name, expected_dir=dirname)
            if errs:
                print(f"WARN: skipping {md_path}:", file=sys.stderr)
                for e in errs:
                    print(f"  - {e}", file=sys.stderr)
                continue
            # canonicalise tags (lowercase, hyphenated)
            meta["tags"] = [slugify_tag(t) for t in meta.get("tags", [])]
            # `date` may have been parsed as datetime.date — normalise to ISO string
            if not isinstance(meta.get("date"), str):
                meta["date"] = str(meta["date"])
            # URL
            meta["_url"] = f"{dirname}/{sub.name}/index.html"
            meta["_md_path"] = md_path
            meta["_body_md"] = body
            # tutorial_meta — map from frontmatter `tutorial:` block (if any)
            meta["tutorial_meta"] = meta.get("tutorial") or None
            posts.append(meta)
    return posts
```

- [ ] **Step 8.3: Add `build_posts(posts, root, nav_tmpl)` function**

Add right above the existing `def build_site(...)`:

```python
from build_lib.markdown import render_post_body
from build_lib.post_assembly import assemble_post_page


def build_posts(posts: list[dict], root: Path, nav_tmpl: str) -> int:
    """Render each post's index.md to index.html. Returns count rendered."""
    slug_set = {p["slug"] for p in posts}
    n = 0
    total_warnings: list[str] = []
    for p in posts:
        md_path: Path = p["_md_path"]
        body_md: str = p["_body_md"]
        slug: str = p["slug"]
        # papers/<slug>/ → asset prefix "../../"
        current_dir = f"{md_path.parent.parent.name}/{slug}"
        body_html, toc_html, warnings = render_post_body(
            body_md, slug_set, current_post_dir=current_dir,
        )
        total_warnings.extend(warnings)
        # Per-post nav: papers/x/ is depth=1 below root (in URL terms papers/x/index.html)
        # nav links need ../ prefix
        nav_html = render_nav(nav_tmpl, active="papers" if p["type"] == "paper" else "tutorials",
                              depth=1)
        # Fix nav links: depth-1 logic in render_nav gives "../" — but we're at depth=2 here.
        # render_nav was originally written for tags/<tag>.html (depth=1).
        # For papers/<slug>/index.html we need "../../" prefix in nav.
        # Easiest: post-process the nav HTML to bump "../" to "../../".
        nav_html = nav_html.replace('href="../', 'href="../../')
        html = assemble_post_page(
            meta=p, body_html=body_html, toc_html=toc_html,
            nav_html=nav_html, asset_prefix="../../",
        )
        out_path = md_path.parent / "index.html"
        out_path.write_text(html, encoding="utf-8")
        n += 1
    if total_warnings:
        print(f"WARN: {len(total_warnings)} build warnings:", file=sys.stderr)
        for w in total_warnings:
            print(f"  - {w}", file=sys.stderr)
    return n
```

- [ ] **Step 8.4: Wire into `main()`**

Find `def main(argv ...)` near the bottom. After the existing `summary = build_site(posts, root, nav_tmpl)` line, add:

```python
    n_posts = build_posts(posts, root, nav_tmpl)
    print(f"Rendered {n_posts} per-post HTML pages from markdown.")
```

- [ ] **Step 8.5: Manual sanity test with one already-migrated paper**

Since no `index.md` files exist yet, create a temporary minimal one to test the pipeline. Pick `papers/l2p-2026/` (it has rich content) and create a tiny `index.md` next to the existing `index.html`:

```bash
cat > /tmp/test-l2p.md <<'EOF'
---
type: paper
slug: l2p-2026
title: "L2P test"
date: 2026-05-22
tldr: |
  Test summary.
tags: [diffusion]
---

# L2P test

A paragraph.

## Section 1

Text with $x = 1$.

See [[asymflow-2026]] paper.
EOF

# Don't overwrite the real index.html. Move it aside first.
cp papers/l2p-2026/index.html papers/l2p-2026/index.html.bak
mv /tmp/test-l2p.md papers/l2p-2026/index.md

python3 build.py 2>&1 | head -20

# Inspect
head -50 papers/l2p-2026/index.html

# Restore
mv papers/l2p-2026/index.html.bak papers/l2p-2026/index.html
rm papers/l2p-2026/index.md
```

Expected: `python3 build.py` runs without errors, generates HTML with the test content, wiki-link to asymflow-2026 resolves to `../asymflow-2026/index.html`.

- [ ] **Step 8.6: Commit**

```bash
git add build.py
git commit -m "$(cat <<'EOF'
Wire build.py to read markdown frontmatter + render per-post HTML

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: CLI flags `--post` and `--check`

**Files:**
- Modify: `build.py`

- [ ] **Step 9.1: Extend argument parser**

In `main()`, add to the existing `argparse.ArgumentParser`:

```python
    parser.add_argument(
        "--post",
        metavar="SLUG",
        default=None,
        help="Render only a single post (papers/<slug> or tutorials/<slug>).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate frontmatter + wiki-links across all posts; "
             "exit non-zero on any warning. Don't write HTML.",
    )
```

- [ ] **Step 9.2: Implement `--post <slug>` branch**

After `posts = discover_posts(root)` in `main()`, add:

```python
    if args.post:
        posts = [p for p in posts if p["slug"] == args.post]
        if not posts:
            print(f"ERROR: no post with slug {args.post!r}", file=sys.stderr)
            return 2
        # Build only this post, skip site-level pages
        nav_tmpl = load_nav_header(root)
        n = build_posts(posts, root, nav_tmpl)
        print(f"Rendered {n} post(s).")
        return 0
```

This branch returns early — `build_site()` is not called.

- [ ] **Step 9.3: Implement `--check` branch**

After the `--post` branch, add (before the normal full-build path):

```python
    if args.check:
        slug_set = {p["slug"] for p in posts}
        all_warnings: list[str] = []
        for p in posts:
            _, _, warns = render_post_body(
                p["_body_md"], slug_set,
                current_post_dir=f"{Path(p['_url']).parent}",
            )
            all_warnings.extend(warns)
        if all_warnings:
            for w in all_warnings:
                print(f"WARN: {w}", file=sys.stderr)
            print(f"FAIL: {len(all_warnings)} warning(s).", file=sys.stderr)
            return 1
        print(f"OK: {len(posts)} posts validated.")
        return 0
```

Add `from build_lib.markdown import render_post_body` near the top of the file.

- [ ] **Step 9.4: Manual test of new flags**

Run: `python3 build.py --check`
Expected (no posts migrated yet): `OK: 0 posts validated.` or warnings about empty discovery.

Run: `python3 build.py --post nonexistent-slug`
Expected: `ERROR: no post with slug 'nonexistent-slug'` (return code 2).

- [ ] **Step 9.5: Commit**

```bash
git add build.py
git commit -m "$(cat <<'EOF'
Add --post and --check CLI flags

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: HTML → MD migrator

**Files:**
- Create: `migrate-md.py`
- Create: `tests/test_migrate_md.py`

This is a one-shot tool to convert all 22+3 existing HTML pages to markdown. It must preserve:
- frontmatter (built from `meta.json` + body's `.meta` div which has authors/arxiv/etc)
- `<figure>` blocks (whole HTML preserved as HTML island)
- `<p class="math-translation">` and `<p class="code-source">` (whole `<p>` preserved as HTML island)
- `<pre><code class="language-X">` → `` ```X ... ``` ``
- Inline `<strong>` `<em>` `<code>` `<a>` → markdown
- Lists, blockquotes, tables → markdown
- Math `$...$` and `$$...$$` → as-is (already markdown-ready in the HTML body)

It must drop:
- `<nav class="toc">` (build.py regenerates)
- `<script>` tags (build.py injects)
- HTML comments

- [ ] **Step 10.1: Write failing tests**

`tests/test_migrate_md.py`:

```python
"""Tests for migrate-md.py (HTML → markdown migration)."""
from __future__ import annotations

# Note: migrate-md.py has a hyphen, so import via importlib.
import importlib.util
import pathlib
import pytest


def _load_migrate():
    path = pathlib.Path(__file__).parent.parent / "migrate-md.py"
    spec = importlib.util.spec_from_file_location("migrate_md", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def migrate_mod():
    return _load_migrate()


def test_html_to_md_keeps_h1(migrate_mod):
    html = '<html><body><main><h1>Title</h1></main></body></html>'
    meta = {"type": "paper", "slug": "x", "title": "Title", "date": "2026-01-01",
            "tldr": "y", "tags": ["a"]}
    md = migrate_mod.html_to_md(html, meta)
    assert "# Title" in md


def test_html_to_md_keeps_figures_as_html(migrate_mod):
    html = (
        '<main>'
        '<figure>'
        '<img src="figures/a.png" alt="X">'
        '<figcaption>Cap with <strong>bold</strong>.</figcaption>'
        '</figure>'
        '</main>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "<figure>" in md
    assert "<figcaption>" in md
    assert "<strong>bold</strong>" in md


def test_html_to_md_keeps_math_translation_class(migrate_mod):
    html = (
        '<main>'
        '<p>$$x = 1$$</p>'
        '<p class="math-translation">—— 翻译: x is one.</p>'
        '</main>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert 'class="math-translation"' in md


def test_html_to_md_keeps_code_source_class(migrate_mod):
    html = (
        '<main>'
        '<p class="code-source">repo/a.py:L1-L10 — role</p>'
        '<pre><code class="language-python">x = 1\n</code></pre>'
        '</main>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert 'class="code-source"' in md
    assert "```python" in md
    assert "x = 1" in md


def test_html_to_md_drops_toc(migrate_mod):
    html = (
        '<html><body>'
        '<nav class="toc"><ul><li>a</li></ul></nav>'
        '<main><h1>T</h1></main>'
        '</body></html>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "class=\"toc\"" not in md
    assert "# T" in md


def test_html_to_md_emits_frontmatter(migrate_mod):
    html = '<main><h1>T</h1></main>'
    meta = {"type": "paper", "slug": "x-2026", "title": "Hello",
            "date": "2026-01-01", "tldr": "summary", "tags": ["a", "b"]}
    md = migrate_mod.html_to_md(html, meta)
    assert md.startswith("---\n")
    assert "type: paper" in md
    assert "slug: x-2026" in md
    assert "tags:" in md


def test_html_to_md_converts_list(migrate_mod):
    html = '<main><ul><li>one</li><li>two</li></ul></main>'
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    # Either '- one' or '* one' is acceptable
    assert ("- one" in md) or ("* one" in md)


def test_html_to_md_converts_inline_strong_em(migrate_mod):
    html = '<main><p>This is <strong>bold</strong> and <em>italic</em>.</p></main>'
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "**bold**" in md
    assert "*italic*" in md or "_italic_" in md
```

- [ ] **Step 10.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_migrate_md.py -v`
Expected: FileNotFoundError on `migrate-md.py`.

- [ ] **Step 10.3: Implement `migrate-md.py`**

```python
#!/usr/bin/env python3
"""One-shot HTML → markdown migrator for paper-reading blog.

Reads each papers/<slug>/index.html + meta.json, emits papers/<slug>/index.md.
Same for tutorials/<slug>/. Backs up the original index.html → index.html.pre-migrate.

Usage:
    python3 migrate-md.py --convert         # write .md, keep .html.pre-migrate backups
    python3 migrate-md.py --cleanup --yes   # delete meta.json + .pre-migrate backups
    python3 migrate-md.py --dry-run         # show what would be written, don't write
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag


# Tags whose entire HTML we keep as a raw island (the markdown is HTML).
PRESERVE_AS_HTML = {
    ("figure", None),
    ("p", "math-translation"),
    ("p", "code-source"),
    ("table", None),  # markdown tables don't handle multi-line cells; keep HTML
}


def _matches_preserve(el: Tag) -> bool:
    name = el.name
    classes = el.get("class") or []
    for tag, cls in PRESERVE_AS_HTML:
        if name == tag and (cls is None or cls in classes):
            return True
    return False


def html_to_md(html: str, meta: dict[str, Any]) -> str:
    soup = BeautifulSoup(html, "html5lib")

    # Drop scripts and the right-rail TOC
    for s in soup.find_all("script"):
        s.decompose()
    for nav in soup.find_all("nav", class_="toc"):
        nav.decompose()
    # Drop site nav (build.py reinjects)
    for nav in soup.find_all("nav", class_="site-nav"):
        nav.decompose()

    main = soup.find("main")
    if main is None:
        # Some pages may not have <main>; fall back to <body>
        main = soup.find("body") or soup
    # Drop any .meta sidebar div inside main (we already have meta.json)
    for d in main.find_all("div", class_="meta"):
        d.decompose()

    blocks: list[str] = []
    for el in list(main.children):
        if isinstance(el, NavigableString):
            txt = str(el).strip()
            if txt:
                blocks.append(txt)
            continue
        if not isinstance(el, Tag):
            continue
        blocks.append(_convert_block(el))

    body_md = "\n\n".join(b for b in blocks if b)
    fm = _build_frontmatter(meta)
    return fm + "\n\n" + body_md + "\n"


def _convert_block(el: Tag) -> str:
    """Convert a top-level block element to markdown (or preserve as HTML)."""
    if _matches_preserve(el):
        # Lightly normalise: ensure surrounding whitespace
        return str(el)

    name = el.name

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = _inline_md(el)
        return f"{'#' * level} {text}"

    if name == "p":
        return _inline_md(el)

    if name == "ul":
        return _convert_list(el, ordered=False)
    if name == "ol":
        return _convert_list(el, ordered=True)

    if name == "blockquote":
        inner = "\n\n".join(_inline_md(child) for child in el.find_all(["p"], recursive=False))
        if not inner:
            inner = _inline_md(el)
        return "\n".join(f"> {line}" for line in inner.splitlines())

    if name == "pre":
        return _convert_code_block(el)

    if name == "section":
        # <section> wraps groups of headings — recurse into children
        return "\n\n".join(_convert_block(c) for c in el.children
                           if isinstance(c, Tag))

    if name == "hr":
        return "---"

    # Fallback: preserve as HTML
    return str(el)


def _convert_list(el: Tag, ordered: bool) -> str:
    lines = []
    for i, li in enumerate(el.find_all("li", recursive=False), start=1):
        prefix = f"{i}." if ordered else "-"
        # Children of li might be paragraphs/sub-lists
        text = _inline_md(li).strip()
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def _convert_code_block(pre: Tag) -> str:
    code = pre.find("code")
    if code is None:
        return str(pre)
    classes = code.get("class") or []
    lang = ""
    for c in classes:
        if c.startswith("language-"):
            lang = c[len("language-"):]
            break
    text = code.get_text()
    # Strip trailing newline duplication
    text = text.rstrip("\n")
    return f"```{lang}\n{text}\n```"


_INLINE_MAP = {
    "strong": ("**", "**"),
    "b": ("**", "**"),
    "em": ("*", "*"),
    "i": ("*", "*"),
    "code": ("`", "`"),
}


def _inline_md(el: Tag | NavigableString) -> str:
    """Convert inline HTML to markdown text."""
    if isinstance(el, NavigableString):
        return str(el)
    if not isinstance(el, Tag):
        return ""
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in _INLINE_MAP:
                open_, close_ = _INLINE_MAP[child.name]
                parts.append(f"{open_}{_inline_md(child)}{close_}")
            elif child.name == "a":
                href = child.get("href", "")
                text = _inline_md(child)
                parts.append(f"[{text}]({href})")
            elif child.name == "br":
                parts.append("\n")
            elif child.name == "img":
                src = child.get("src", "")
                alt = child.get("alt", "")
                parts.append(f"![{alt}]({src})")
            else:
                # Fallback: keep nested HTML
                parts.append(str(child))
    return "".join(parts).strip()


def _build_frontmatter(meta: dict[str, Any]) -> str:
    """Render frontmatter YAML from a meta.json dict."""
    out = {
        "type": meta["type"],
        "slug": meta["slug"],
        "title": meta["title"],
        "date": meta["date"],
        "tldr": meta["tldr"],
        "tags": meta.get("tags", []),
    }
    if meta.get("tutorial_meta"):
        out["tutorial"] = meta["tutorial_meta"]
    yaml_text = yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=1000)
    return f"---\n{yaml_text}---"


# ---------------------------------------------------------------------------
# CLI

def migrate_all(repo_root: Path, dry_run: bool = False) -> int:
    n = 0
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = repo_root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            html_path = sub / "index.html"
            meta_path = sub / "meta.json"
            md_out = sub / "index.md"
            if not html_path.is_file():
                continue
            if not meta_path.is_file():
                print(f"SKIP: {sub} (no meta.json)", file=sys.stderr)
                continue
            html_text = html_path.read_text(encoding="utf-8")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            md = html_to_md(html_text, meta)
            if dry_run:
                print(f"DRY: would write {md_out} ({len(md)} bytes)")
                n += 1
                continue
            # Backup the original HTML
            backup = sub / "index.html.pre-migrate"
            if not backup.exists():
                shutil.copy2(html_path, backup)
            md_out.write_text(md, encoding="utf-8")
            print(f"WROTE: {md_out}")
            n += 1
    return n


def cleanup_all(repo_root: Path) -> int:
    n = 0
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = repo_root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            for fname in ("meta.json", "index.html.pre-migrate"):
                f = sub / fname
                if f.is_file():
                    f.unlink()
                    print(f"DELETED: {f}")
                    n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate paper-reading HTML → markdown.")
    ap.add_argument("--root", default=".", help="Repo root (default: CWD)")
    ap.add_argument("--convert", action="store_true", help="Write index.md files.")
    ap.add_argument("--cleanup", action="store_true",
                    help="Delete meta.json and .pre-migrate backups.")
    ap.add_argument("--dry-run", action="store_true", help="Don't write, just report.")
    ap.add_argument("--yes", action="store_true",
                    help="Confirm destructive --cleanup.")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.convert:
        n = migrate_all(root, dry_run=args.dry_run)
        print(f"Migrated {n} posts ({'dry-run' if args.dry_run else 'written'}).")
        return 0
    if args.cleanup:
        if not args.yes:
            print("Refusing to --cleanup without --yes (destructive).", file=sys.stderr)
            return 2
        n = cleanup_all(root)
        print(f"Deleted {n} files.")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_migrate_md.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 10.5: Commit**

```bash
git add migrate-md.py tests/test_migrate_md.py
git commit -m "$(cat <<'EOF'
Add HTML → markdown migrator (one-shot, Phase 1 only)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Run migration on real data + verify

**Files (touched by automation):**
- 22× `papers/<slug>/index.md` (created)
- 22× `papers/<slug>/index.html.pre-migrate` (backup created)
- 3× `tutorials/<slug>/index.md` (created)
- 3× `tutorials/<slug>/index.html.pre-migrate` (backup created)

- [ ] **Step 11.1: Dry-run migration**

Run: `python3 migrate-md.py --convert --dry-run`
Expected: 25 lines `DRY: would write papers/.../index.md (... bytes)`.

- [ ] **Step 11.2: Real migration**

Run: `python3 migrate-md.py --convert`
Expected: 25 lines `WROTE: papers/.../index.md`. Each slug should have a backup `index.html.pre-migrate`.

- [ ] **Step 11.3: Spot-check 5 random `index.md` files for sanity**

Pick 5: `papers/l2p-2026/index.md`, `papers/asymflow-2026/index.md`, `papers/awm-2025/index.md`, `tutorials/rl-for-diffusion-2023/index.md`, `papers/dapo-2025/index.md` (the last has known MathJax-tricky math).

For each:
- Verify frontmatter at top is valid YAML
- Verify `# Title` is the first heading
- Spot any HTML islands that look right (`<figure>`, `<p class="...">`)
- Spot any code blocks that have language hints

Run: `head -40 papers/l2p-2026/index.md` and similar for the other 4.

- [ ] **Step 11.4: Run `--check` on migrated posts**

Run: `python3 build.py --check`
Expected: `OK: 25 posts validated.` — no warnings.

If there are warnings, list each and either:
- Hand-fix the offending frontmatter
- Update the migrator and re-run (delete the broken `index.md` first)

- [ ] **Step 11.5: Full rebuild**

Run: `python3 build.py`
Expected:
- `Generated: N pages from 22 papers + 3 tutorials, with K unique tags`
- `Rendered 25 per-post HTML pages from markdown.`

- [ ] **Step 11.6: Visual diff spot-check (~5 papers)**

For 5 randomly-picked papers:
```bash
diff papers/l2p-2026/index.html papers/l2p-2026/index.html.pre-migrate | head -100
```

Look for:
- ✅ Whitespace/attribute-order diffs: acceptable
- ✅ TOC content reordering (e.g., if h3 text varied slightly from the old TOC link text)
- ❌ Missing content blocks (sections, paragraphs, figures, code) — abort and debug

- [ ] **Step 11.7: Browser smoke test**

```bash
./serve.sh 8765 -bg
open http://127.0.0.1:8765/                    # homepage card grid
open http://127.0.0.1:8765/papers/l2p-2026/    # paper page
open http://127.0.0.1:8765/tutorials/rl-for-diffusion-2023/   # tutorial page
```

Verify in browser:
- Card grid loads, links work
- Paper page: math renders, code highlights, figures open in lightbox, TOC highlights current section
- Tutorial page: same

If anything is broken, debug + fix + re-run `python3 build.py` + re-test.

- [ ] **Step 11.8: Cleanup**

After all the above passes:

Run: `python3 migrate-md.py --cleanup --yes`
Expected: 50 lines `DELETED: ...` (25 meta.json + 25 index.html.pre-migrate).

- [ ] **Step 11.9: Commit**

```bash
git add papers/ tutorials/
git status   # verify nothing surprising — only papers/*/index.md and papers/*/index.html changes
git commit -m "$(cat <<'EOF'
Migrate 22 papers + 3 tutorials to markdown source

- Each papers/<slug>/ and tutorials/<slug>/ now has index.md as source
- index.html is regenerated by build.py from the markdown
- meta.json deleted (content moved into frontmatter)
- HTML preserved verbatim where markdown can't express it cleanly
  (figures, math-translations, code-source citations)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Update reading-papers skill

**Files:**
- Modify: `.claude/skills/reading-papers/SKILL.md`
- Create: `.claude/skills/reading-papers/templates/index.md`

- [ ] **Step 12.1: Read current SKILL.md to know what to change**

Run: `cat .claude/skills/reading-papers/SKILL.md`

- [ ] **Step 12.2: Create `.claude/skills/reading-papers/templates/index.md`**

```markdown
---
type: paper
slug: REPLACE-SLUG-YEAR
title: "REPLACE TITLE"
date: REPLACE-YYYY-MM-DD
tldr: |
  REPLACE multi-line summary (1-3 sentences).
tags: [REPLACE, COMMA, SEPARATED]
paper:
  arxiv_id: ""
  authors: ""
  venue: ""
  project_page: ""
  code_url: ""
  weights_url: ""
---

# REPLACE: full title with subtitle

<figure>
  <img src="figures/fig1-teaser.png" alt="Teaser caption (accessibility)">
  <figcaption>
    <strong>Fig. 1</strong> — Caption with full markdown / HTML allowed inside.
  </figcaption>
</figure>

## 1. 出发点 (Motivation)

Why does this paper exist? What's the core problem?

## 2. 方法 (Method)

How does it work?

### 2.1 Core idea

Math display:

$$ x = (1-\sigma) y $$
<p class="math-translation">—— 翻译: explain what the equation means in plain language.</p>

Code citation:

<p class="code-source">repo/path/to/file.py:L10-L20 — one-line role description</p>

```python
def example():
    pass
```

## 3. 结论 (Key findings)

Headline results.

## 4. 实现细节 (Implementation notes)

Gotchas, undocumented things, paper-vs-code gaps.

## 5. 批判性总结 (Critical assessment)

### Strengths

### Limitations / open questions

### When to use / not use

### Further reading

- Related work: [[other-slug-2025]]
```

- [ ] **Step 12.3: Update `SKILL.md` — §5b "HTML skeleton" → "MD skeleton"**

Find the section in SKILL.md that has the HTML template (it's labeled §5b or similar). Replace its content with a pointer to `templates/index.md` and the new convention summary. Specifically:

In `SKILL.md`, locate the section describing the HTML skeleton (search for `<!doctype html>` or `<nav class="toc">` inside SKILL.md). Replace that whole section with:

```markdown
### 5b. Markdown skeleton

The source of truth for a paper is `papers/<slug>/index.md`. Use
`templates/index.md` as the starting point. Replace placeholders, then
fill the 5 sections.

**Conventions:**
- Top-of-file YAML frontmatter (required: type, slug, title, date, tldr, tags)
- Figures: use `<figure><img><figcaption>` HTML (caption can be rich)
- Math translation: `<p class="math-translation">—— 翻译: ...</p>` right after a `$$...$$` block
- Code citation: `<p class="code-source">repo/path:Lstart-Lend — role</p>` right before a fenced code block
- Cross-paper references: `[[other-slug]]` or `[[other-slug|alias]]`
- TOC: do NOT write `<nav class="toc">` — build.py auto-generates it from h2/h3
- Heading IDs: do NOT write `id="sec-N"` — build.py assigns them

After writing `index.md`, run `python3 build.py --post <slug>` to render the
HTML. Then `git add papers/<slug>/index.md papers/<slug>/index.html`.
```

- [ ] **Step 12.4: Update SKILL.md — workflow steps**

Find the section listing steps for adding a new paper (search for "mkdir -p papers" in SKILL.md). Update step 6 from "write index.html" to "write index.md", and ADD a step 6.5: "render HTML: `python3 build.py --post <slug>`".

- [ ] **Step 12.5: Sanity-check the updated SKILL.md**

Run: `grep -n 'index\.html\|meta\.json\|<nav class="toc">' .claude/skills/reading-papers/SKILL.md`
Expected: any remaining hits are either deliberate (referencing the generated HTML output) or should be cleaned up. The string `<nav class="toc">` should not appear as something the author writes.

- [ ] **Step 12.6: Commit**

```bash
git add .claude/skills/reading-papers/SKILL.md .claude/skills/reading-papers/templates/index.md
git commit -m "$(cat <<'EOF'
Update reading-papers skill: emit markdown instead of HTML

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update writing-tutorial skill

**Files:**
- Modify: `.claude/skills/writing-tutorial/SKILL.md`
- Create: `.claude/skills/writing-tutorial/templates/skeleton.md`
- Create: `.claude/skills/writing-tutorial/templates/spiral-section.md`

- [ ] **Step 13.1: Read current SKILL.md + existing templates**

Run:
```bash
cat .claude/skills/writing-tutorial/SKILL.md | head -80
cat .claude/skills/writing-tutorial/templates/skeleton.html
cat .claude/skills/writing-tutorial/templates/spiral-section.html
```

- [ ] **Step 13.2: Create `templates/skeleton.md`**

Based on the existing `skeleton.html`, port to markdown:

```markdown
---
type: tutorial
slug: REPLACE-DOMAIN-YEAR
title: "REPLACE: tutorial title"
date: REPLACE-YYYY-MM-DD
tldr: |
  REPLACE: 1-3 sentences summarising what this tutorial covers.
tags: [REPLACE, COMMA, SEPARATED]
tutorial:
  word_count: "REPLACE"
  reading_minutes: "REPLACE"
---

# REPLACE: tutorial title

Introduction paragraph.

<!-- Per-section content follows. Use templates/spiral-section.md per section. -->

## 1. REPLACE: section 1 title

<!-- 5-step spiral content; see templates/spiral-section.md -->

## 2. REPLACE: section 2 title

<!-- ... -->
```

- [ ] **Step 13.3: Create `templates/spiral-section.md`**

```markdown
## N. REPLACE: section title

### N.1 直觉 (Intuition)

REPLACE: kid-friendly intuition (1-2 paragraphs, optional analogy).

### N.2 最小 demo (Minimal demo)

REPLACE: 5-30 line hand-written, NON-production demo code.

```python
# class="teaching-demo" is added by build.py for hand-written demos.
# Use a fenced code block; the build does not yet tag this automatically —
# so explicitly write the class via an HTML wrapper if you need that styling:
# <pre class="teaching-demo"><code class="language-python">...</code></pre>
def toy_example():
    pass
```

### N.3 正式化 (Formalization)

REPLACE: full math derivation.

$$ \text{display equation} $$
<p class="math-translation">—— 翻译: explain in 1-2 lines.</p>

### N.4 代码引用 (Code reference)

REPLACE: VERBATIM code citation from `sources/repos/<repo>/...`.

<p class="code-source">sources/repos/org-name/path/file.py:Lstart-Lend — what this snippet shows</p>

```python
# Paste verbatim from the cited file:line range. Do NOT modify.
```

### N.5 洞察 (Insight)

REPLACE: 1-2 takeaways, comparisons, caveats.
```

- [ ] **Step 13.4: Update SKILL.md**

Find the section in `writing-tutorial/SKILL.md` that describes the per-tutorial page invariants (look for "<h2 id=" or "spiral structure"). Update to drop manual ID writing requirement:

- "Every `<h2 id="sec-N">` has exactly 5 `<h3 id="sec-N-1..5">`" → "Every `## N. Title` has exactly 5 `### N.M Title` subsections. IDs are auto-assigned by `build.py` — do NOT write them yourself."
- "Step 2 demo code blocks tagged `class="teaching-demo"`" — note that this currently requires HTML wrapping (no markdown convention for class on a fence yet); document explicitly in SKILL.md.
- Update phase 6 (Assemble + Verify + Publish): change "main agent stitches sections" to "main agent writes the markdown file index.md, then runs `python3 build.py --post <slug>`".

- [ ] **Step 13.5: Commit**

```bash
git add .claude/skills/writing-tutorial/SKILL.md .claude/skills/writing-tutorial/templates/skeleton.md .claude/skills/writing-tutorial/templates/spiral-section.md
git commit -m "$(cat <<'EOF'
Update writing-tutorial skill: emit markdown instead of HTML

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 14.1: Read current CLAUDE.md to find sections to update**

Run: `cat CLAUDE.md | head -100`

- [ ] **Step 14.2: Update "Per-paper layout" section**

Find the section starting with `## Per-paper layout`. Replace the layout block to add `index.md`:

```
papers/<slug>/
  index.md                # source of truth (NEW)
  index.html              # generated by build.py
  figures/                # final cropped PNGs
  figures-raw/            # gitignored intermediate
  raw/                    # gitignored, original PDF
  repo/                   # gitignored, cloned source
```

Add a note: "meta.json no longer exists — its content lives in YAML frontmatter at the top of index.md."

- [ ] **Step 14.3: Update "Per-tutorial layout" section**

Same treatment: add `index.md`, note meta.json removal.

- [ ] **Step 14.4: Update "The rendering stack" section**

Add a paragraph before the MathJax bullet:

> The per-paper / per-tutorial `index.html` is **generated** by `build.py` from
> `index.md`. Source = markdown + YAML frontmatter; output = HTML. Both are
> committed. Do not hand-edit `index.html` — your changes will be wiped on the
> next `python3 build.py`.

- [ ] **Step 14.5: Update "TOC contract" section**

Replace the bullet list (which says authors must write `id="sec-N"`) with:

> The right-side floating TOC is required on every per-paper page. **It is
> generated automatically by `build.py`**. Authors write `## ` and `### `
> headings in markdown; `build.py` assigns `id="sec-N"` to h2 and `id="sec-N-M"`
> to h3 (h2-scoped numbering). Do not write IDs yourself.

- [ ] **Step 14.6: Update "Adding a new paper" section**

In the step list, change step 6 from "Write `papers/<slug>/index.html`" to "Write `papers/<slug>/index.md`", and insert a new step 6.5: "Run `python3 build.py --post <slug>` to generate `index.html`."

- [ ] **Step 14.7: Update "Editing existing pages" section**

Replace any references to editing index.html with editing index.md + rebuilding:

> Edit `index.md`. Then run `python3 build.py --post <slug>` to regenerate
> `index.html`. Commit both.

- [ ] **Step 14.8: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Update CLAUDE.md for markdown-source workflow

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: End-to-end verification + dummy paper test

**Files:**
- Temporary: `papers/_smoke-test-2026/index.md` (created, then deleted)

This task verifies acceptance criteria from spec §10.

- [ ] **Step 15.1: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass (no skips, no failures).

- [ ] **Step 15.2: Run `--check` clean**

Run: `python3 build.py --check`
Expected: `OK: 25 posts validated.` (exit code 0).

- [ ] **Step 15.3: Full clean rebuild**

Run:
```bash
python3 build.py 2>&1 | tee /tmp/build.log
grep -i "warn\|error" /tmp/build.log || echo "no warnings"
```
Expected: "no warnings".

- [ ] **Step 15.4: Smoke-test mode still works**

Run: `python3 build.py --smoke-test`
Expected: `OK\nGenerated: ...` (the existing smoke test).

- [ ] **Step 15.5: Dummy paper end-to-end (skill workflow)**

Create a minimal dummy paper to verify the end-to-end authoring path:

```bash
mkdir -p papers/_smoke-test-2026/figures
cp .claude/skills/reading-papers/templates/index.md papers/_smoke-test-2026/index.md
# Edit placeholder fields:
sed -i.bak \
    -e 's/REPLACE-SLUG-YEAR/_smoke-test-2026/' \
    -e 's/REPLACE TITLE/Smoke Test Paper/' \
    -e 's/REPLACE-YYYY-MM-DD/2026-05-29/' \
    -e 's/REPLACE multi-line summary.*/Smoke test./' \
    -e 's/\[REPLACE, COMMA, SEPARATED\]/[test]/' \
    papers/_smoke-test-2026/index.md
rm papers/_smoke-test-2026/index.md.bak

# Build it
python3 build.py --post _smoke-test-2026

# Verify HTML output
test -f papers/_smoke-test-2026/index.html && echo "OK: HTML exists"
grep -c "Smoke Test Paper" papers/_smoke-test-2026/index.html
```

Expected: HTML file generated, title visible.

- [ ] **Step 15.6: Cleanup dummy**

```bash
rm -rf papers/_smoke-test-2026/
# Rebuild to make sure the dummy is purged from any site-level pages
python3 build.py
```

- [ ] **Step 15.7: Browser visual check (final)**

```bash
./serve.sh 8765 -bg
# Open and click around:
open http://127.0.0.1:8765/
open http://127.0.0.1:8765/papers/l2p-2026/
open http://127.0.0.1:8765/tutorials/rlhf-evolution-2024/
open http://127.0.0.1:8765/tags/diffusion.html
```

Verify:
- Homepage cards display correctly
- L2P paper page: TOC right rail, math renders, code highlights, figures clickable (lightbox)
- Tutorial page: same plus spiral-section structure intact
- Tag page: list of papers + tutorials with that tag

- [ ] **Step 15.8: Verify `publish.sh` still works (dry-run)**

Don't actually push. Just check the script:

Run: `bash -n publish.sh && cat publish.sh | head -30`
Expected: syntax OK; no obvious assumption-breakage.

- [ ] **Step 15.9: Final commit (only if anything outstanding)**

```bash
git status
# If anything is uncommitted: commit it.
# Otherwise: this task ends with no commit.
```

- [ ] **Step 15.10: Verify acceptance criteria from spec §10**

Confirm each:
1. ✅ requirements.txt exists, `pip install -r requirements.txt` succeeded
2. ✅ `python3 build.py` clean (no warnings)
3. ✅ `python3 build.py --check` exit 0
4. ✅ All 25 index.md present + frontmatter valid
5. ✅ All 25 index.html rendered (spot-checked 5+ visually)
6. ✅ All meta.json deleted
7. ✅ Both skill SKILL.md updated, templates/index.md (etc.) present
8. ✅ Dummy paper workflow tested + cleaned
9. ✅ `[[slug]]` cross-references render with valid hrefs (at least one tested)
10. ✅ publish.sh unchanged + still bash-syntactically valid

If all 10 ✅: Phase 1 is complete.

---

## Self-Review

### Spec coverage

Re-scanning the spec against the plan:

| Spec section | Plan task |
|---|---|
| §3 Directory structure | Tasks 8 + 11 (HTML coexists with MD) |
| §4 Frontmatter schema | Task 2 (parser + validator) |
| §5.1 Pure markdown | Task 6 (markdown-it-py with table + dollarmath) |
| §5.2(a) Figure HTML | Task 5 (lightbox tagging) + Task 10 (migration preserves) |
| §5.2(b) math-translation | Task 10 preserve; Task 6 passes through |
| §5.2(c) code-source | Task 10 preserve; Task 6 passes through |
| §5.3 Auto heading IDs | Task 4 |
| §5.3 Auto TOC | Task 4 |
| §5.3 Lightbox tagging | Task 5 |
| §5.3 MathJax/highlight.js injection | Task 7 (assemble_post_page) |
| §6.1 Dependencies | Task 1 |
| §6.2 3-pass build flow | Tasks 8 (pass 1+2 in build_posts) + existing build.py (pass 3 unchanged) |
| §6.3 Post-process rules | Tasks 3, 4, 5, 6 |
| §6.4 CLI --post/--check | Task 9 |
| §6.5 Error handling | Tasks 2 (validation), 6 (warnings) |
| §7 Skill changes | Tasks 12, 13 |
| §7.3 CLAUDE.md update | Task 14 |
| §8 Migration | Task 10 (migrator) + Task 11 (apply) |
| §9 Testing | Test files in every task + Task 15 |
| §10 Acceptance | Task 15.10 |

No spec section is uncovered.

### Type/signature consistency

Checked: `parse(text) → (dict, str)` (Task 2) matches usage in `discover_posts` (Task 8). `render_post_body(md, slug_set, current_post_dir) → (body_html, toc_html, warnings)` (Task 6) matches usage in `build_posts` (Task 8) and `--check` branch (Task 9). `assemble_post_page(meta, body_html, toc_html, nav_html, asset_prefix)` (Task 7) matches usage in `build_posts` (Task 8).

`preprocess()` and `resolve()` in wiki_links.py are imported by Task 6 with the names used in their definitions in Task 3. ✓

### Placeholder scan

No `TBD`, `TODO`, or "implement later" in the plan body. Each step has either concrete code or concrete bash to run. Spot-checks pass.

### Risks not in the plan

- The 22-paper migration may surface markdown-it-py rendering edge cases that none of the unit tests catch (e.g., a paper with a specific LaTeX construct that conflicts with $ tokenisation). Mitigation: Step 11.6 visual diff catches content loss; the user can hand-fix the migrator + re-run on the offending paper.
- markdown-it-py's `dollarmath_plugin` API may have shifted across versions. Pinned `>=0.4` in requirements; if API mismatch, the test in Task 6 will fail and surface it before migration.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-29-md-blog-phase1.md`.

User has already chosen subagent-driven execution. Next step: invoke `superpowers:subagent-driven-development`.
