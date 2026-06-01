# Markdown-Source Blog — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Add structured metadata (concepts/citations/repos) + knowledge graph extraction + interactive viz on top of Phase 1's markdown infrastructure.

**Architecture:** Rename `tags:` → `concepts:` in frontmatter; add `citations:` and `repos:` lists; new `build_lib/graph.py` walks posts + concept files to produce `graph.json` + `/graph.html` (cytoscape.js) + `concepts/<slug>.html` aggregation pages.

**Tech Stack:**
- Python 3.9 (`from __future__ import annotations`)
- markdown-it-py + mdit-py-plugins + PyYAML (already installed Phase 1)
- cytoscape.js + cytoscape-cose-bilkent (CDN, no Python dep)
- pytest

Spec reference: `docs/superpowers/specs/2026-05-29-md-blog-phase2-design.md`

---

## File Structure

```
paper-reading/
  build.py                          # MODIFY — add Pass 4/5/6 calls
  migrate-concepts.py               # NEW — one-shot frontmatter migration
  build_lib/
    frontmatter.py                  # MODIFY — switch required key tags→concepts
    graph.py                        # NEW — graph extraction + concept/graph page rendering
  tests/
    test_frontmatter.py             # MODIFY — adapt to concepts: requirement
    test_graph.py                   # NEW
    test_migrate_concepts.py        # NEW
  papers/<slug>/index.md            # MODIFY (×19) — rename tags: to concepts:, code_url to repos:
  tutorials/<slug>/index.md         # MODIFY (×3) — same
  concepts/                         # NEW DIRECTORY
    <slug>.md                       # NEW (optional, user-authored per concept)
    <slug>.html                     # NEW (build product, ×N)
  concepts.html                     # NEW (build product, replaces tags.html)
  tags.html, tags/                  # DELETE (replaced)
  graph.html                        # NEW (build product, cytoscape.js viz)
  graph.json                        # NEW (build product, AI/RAG consumption)
  assets/style.css                  # MODIFY — small additions for graph page + concept page
  .claude/skills/reading-papers/
    templates/index.md              # MODIFY — replace tags with concepts, add citations/repos
    SKILL.md                        # MODIFY — describe new fields
  .claude/skills/writing-tutorial/
    templates/skeleton.md           # MODIFY
    SKILL.md                        # MODIFY
  CLAUDE.md                         # MODIFY — frontmatter sections + new "Knowledge graph" section
```

---

## Task 1: Migration script — `migrate-concepts.py`

**Files:**
- Create: `migrate-concepts.py`
- Create: `tests/test_migrate_concepts.py`

This is a small one-shot that walks `papers/<slug>/index.md` and `tutorials/<slug>/index.md`, and in the YAML frontmatter:
1. Renames key `tags:` → `concepts:`
2. If `paper.code_url` exists, moves its value into a new `repos: [<value>]` list and removes `paper.code_url`
3. Leaves everything else untouched
4. Idempotent: running twice produces no diff

Backs up original to `index.md.pre-concepts` before write.

- [ ] **Step 1.1: Failing tests**

`tests/test_migrate_concepts.py`:

```python
"""Tests for migrate-concepts.py."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load():
    path = pathlib.Path(__file__).parent.parent / "migrate-concepts.py"
    spec = importlib.util.spec_from_file_location("migrate_concepts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mig():
    return _load()


def test_rename_tags_to_concepts(mig):
    src = """---
type: paper
slug: x
title: Test
date: 2026-01-01
tldr: y
tags:
- a
- b
---

body
"""
    out = mig.transform(src)
    assert "tags:" not in out
    assert "concepts:" in out
    # Both items survived
    assert "- a" in out and "- b" in out
    # Body unchanged
    assert "\nbody\n" in out


def test_fold_code_url_into_repos(mig):
    src = """---
type: paper
slug: x
title: Test
date: 2026-01-01
tldr: y
tags: [a]
paper:
  arxiv_id: '123'
  code_url: 'https://github.com/foo/bar'
  weights_url: 'https://huggingface.co/x'
---

body
"""
    out = mig.transform(src)
    assert "code_url" not in out
    assert "repos:" in out
    assert "https://github.com/foo/bar" in out
    assert "weights_url" in out  # other paper.* fields preserved


def test_idempotent_on_already_migrated(mig):
    src = """---
type: paper
slug: x
title: Test
date: 2026-01-01
tldr: y
concepts: [a]
repos:
- https://github.com/foo/bar
---

body
"""
    out = mig.transform(src)
    # Already in target shape; should match src closely
    assert "concepts:" in out
    assert "tags:" not in out
    assert "repos:" in out


def test_preserves_other_frontmatter_keys(mig):
    src = """---
type: tutorial
slug: x
title: T
date: 2026-01-01
tldr: y
tags: [a]
tutorial:
  word_count: '5k'
  reading_minutes: '30'
---

body
"""
    out = mig.transform(src)
    assert "tutorial:" in out
    assert "word_count" in out
    assert "5k" in out


def test_no_frontmatter_returns_unchanged(mig):
    src = "# Just markdown\n\nNo frontmatter.\n"
    out = mig.transform(src)
    assert out == src


def test_missing_tags_field_no_op_for_rename(mig):
    """If a file has no `tags:` field, the rename is a no-op but other rules still run."""
    src = """---
type: paper
slug: x
title: T
date: 2026-01-01
tldr: y
concepts: [a]
paper:
  code_url: 'https://github.com/foo/bar'
---

body
"""
    out = mig.transform(src)
    assert "code_url" not in out
    assert "repos:" in out
    assert "https://github.com/foo/bar" in out
```

- [ ] **Step 1.2: Verify fail**

`python3 -m pytest tests/test_migrate_concepts.py -v` → ImportError.

- [ ] **Step 1.3: Implement `migrate-concepts.py`**

```python
#!/usr/bin/env python3
"""One-shot migration: tags→concepts + paper.code_url→repos in markdown frontmatter.

Usage:
    python3 migrate-concepts.py --convert   # write changes, backup .pre-concepts
    python3 migrate-concepts.py --cleanup --yes  # delete .pre-concepts backups
    python3 migrate-concepts.py --dry-run   # just report
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


FRONTMATTER_RE = re.compile(r"\A(---\s*\n)(.*?)(\n---\s*\n)", re.DOTALL)


def transform(text: str) -> str:
    """Apply the rename + code_url fold transform to a single markdown file's text."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return text
    head, yaml_block, tail = m.group(1), m.group(2), m.group(3)
    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text

    changed = False

    # Rule 1: rename tags → concepts (only if concepts not already set)
    if "tags" in data:
        if "concepts" not in data:
            data["concepts"] = data["tags"]
        del data["tags"]
        changed = True

    # Rule 2: fold paper.code_url into repos
    paper = data.get("paper")
    if isinstance(paper, dict) and "code_url" in paper:
        code_url = paper.pop("code_url")
        if code_url:
            repos = data.get("repos") or []
            if not isinstance(repos, list):
                repos = []
            if code_url not in repos:
                repos.append(code_url)
            data["repos"] = repos
        changed = True

    if not changed:
        return text

    new_yaml = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)
    return f"{head}{new_yaml.rstrip()}{tail}{text[m.end():]}"


def migrate_all(root: Path, dry_run: bool) -> int:
    n = 0
    for kind in ("papers", "tutorials"):
        base = root / kind
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            md = sub / "index.md"
            if not md.is_file():
                continue
            src = md.read_text(encoding="utf-8")
            out = transform(src)
            if out == src:
                continue
            if dry_run:
                print(f"DRY: would update {md}")
                n += 1
                continue
            backup = sub / "index.md.pre-concepts"
            if not backup.exists():
                shutil.copy2(md, backup)
            md.write_text(out, encoding="utf-8")
            print(f"WROTE: {md}")
            n += 1
    return n


def cleanup_all(root: Path) -> int:
    n = 0
    for kind in ("papers", "tutorials"):
        base = root / kind
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            backup = sub / "index.md.pre-concepts"
            if backup.is_file():
                backup.unlink()
                print(f"DELETED: {backup}")
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    if args.convert:
        n = migrate_all(root, args.dry_run)
        print(f"Migrated {n} files.")
        return 0
    if args.cleanup:
        if not args.yes:
            print("Refusing to --cleanup without --yes (destructive).", file=sys.stderr)
            return 2
        n = cleanup_all(root)
        print(f"Deleted {n} backups.")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 1.4: Verify pass**

`python3 -m pytest tests/test_migrate_concepts.py -v` → all 6 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add migrate-concepts.py tests/test_migrate_concepts.py
git commit -m "$(cat <<'EOF'
Phase 2: migrate-concepts.py for tags→concepts + code_url→repos

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update `build_lib/frontmatter.py`

**Files:**
- Modify: `build_lib/frontmatter.py`
- Modify: `tests/test_frontmatter.py`

Switch the required-key check from `tags` to `concepts`. Allow legacy `tags:` to coexist with a warning (Phase 2 transitional grace period). After migration in Task 4, all real files will use `concepts:`.

- [ ] **Step 2.1: Update test fixture in `conftest.py`**

In `tests/conftest.py`, change the `sample_frontmatter_paper` fixture to use `concepts:` instead of `tags:`:

```python
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
concepts: [diffusion, flow-matching]
paper:
  arxiv_id: "2605.12013"
  authors: "Author A, Author B"
---

# L2P

Body text.
"""
```

- [ ] **Step 2.2: Update `REQUIRED_KEYS` + add legacy tag warning**

In `build_lib/frontmatter.py`:

```python
REQUIRED_KEYS = {"type", "slug", "title", "date", "tldr", "concepts"}
# `tags` is legacy from Phase 1; readers should migrate to `concepts`.
LEGACY_KEYS = {"tags"}
```

In `validate()`, add after the existing required-key check:

```python
    # Legacy-key warnings (don't fail, just signal)
    legacy_present = LEGACY_KEYS & set(meta.keys())
    for k in sorted(legacy_present):
        errors.append(f"legacy key {k!r} present; run migrate-concepts.py to rename to 'concepts'")

    # concepts must be a list (mirror old tags check)
    if "concepts" in meta and not isinstance(meta["concepts"], list):
        errors.append(f"concepts must be a list, got {type(meta['concepts']).__name__}")

    # Old tags-is-list check stays but now applies to legacy field if present
    if "tags" in meta and not isinstance(meta["tags"], list):
        errors.append(f"tags must be a list, got {type(meta['tags']).__name__}")
```

Remove the OLD `tags` from required keys (now in legacy).

- [ ] **Step 2.3: Update existing tests to match new contract**

In `tests/test_frontmatter.py`:

- `test_validate_passes_on_valid_paper` already uses `sample_frontmatter_paper` which is now `concepts:` — should still pass.
- `test_validate_catches_missing_required` references `tags` — update to reference `concepts` (the new required key).
- Add new test:

```python
def test_validate_warns_on_legacy_tags():
    meta = {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
            "tldr": "y", "concepts": [], "tags": ["legacy"]}
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("legacy" in e.lower() and "tags" in e.lower() for e in errors)
```

In the `test_validate_catches_missing_required` test, change `tags` to `concepts`:

```python
def test_validate_catches_missing_required():
    meta = {"type": "paper", "slug": "x", "title": "X"}  # missing date/tldr/concepts
    errors = validate(meta, expected_slug="x", expected_dir="papers")
    assert any("date" in e for e in errors)
    assert any("tldr" in e for e in errors)
    assert any("concepts" in e for e in errors)
```

- [ ] **Step 2.4: Verify all tests pass**

`python3 -m pytest tests/ -v` → all pass (including the new test).

- [ ] **Step 2.5: Commit**

```bash
git add build_lib/frontmatter.py tests/test_frontmatter.py tests/conftest.py
git commit -m "$(cat <<'EOF'
Phase 2: switch frontmatter required key from tags to concepts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `build.py` to read `concepts:`

**Files:**
- Modify: `build.py`

The existing code uses `meta["tags"]` in several places. Switch to `meta["concepts"]` with a fallback to `meta["tags"]` for graceful behavior during the transition.

- [ ] **Step 3.1: Locate all `tags` references in build.py**

Run: `grep -n '"tags"\|tags\.html\|tags/' build.py`

Each site-level page builder (`build_tags_cloud`, `build_tag_page`) and `_tag_index` references `tags`. The card renderer (`render_card`) renders `tag-chip` links to `tags/<x>.html`. And `discover_posts` slugifies `meta["tags"]`.

- [ ] **Step 3.2: Make `discover_posts` read concepts with tags fallback**

Replace:

```python
meta["tags"] = [slugify_tag(t) for t in meta.get("tags", [])]
```

with:

```python
# Phase 2: `concepts` is the canonical field; fall back to legacy `tags`.
raw = meta.get("concepts") or meta.get("tags") or []
meta["tags"] = [slugify_tag(t) for t in raw]  # internal canonical name; keep `tags` for current builders
```

Note: We keep the INTERNAL field name `meta["tags"]` for now (used by existing builders). Task 4 (post-migration) and Task 5 (graph builder) will switch fully.

- [ ] **Step 3.3: Switch link paths from `tags/<x>.html` to `concepts/<x>.html`**

In `render_card`:

```python
# Old:
f'<a class="tag-chip" href="{prefix}tags/{esc(t)}.html">{esc(t)}</a>'
# New:
f'<a class="tag-chip" href="{prefix}concepts/{esc(t)}.html">{esc(t)}</a>'
```

In `build_tags_cloud`:

```python
# Old:
f'  <a class="tag-cloud__item" href="tags/{esc(tag)}.html" ...
# New:
f'  <a class="tag-cloud__item" href="concepts/{esc(tag)}.html" ...
```

In `build_tag_page` (the per-tag page builder): note this function is called for each tag and writes `tags/<tag>.html`. We will REPLACE this whole function with a concept page renderer in Task 5. For now, just adjust the back-link inside:

```python
# Old:
body += '<p class="back-link"><a href="../tags.html">← 返回所有标签</a></p>\n'
# New (still using tags.html since build_tags_cloud writes to that name for now):
body += '<p class="back-link"><a href="../concepts.html">← 返回所有概念</a></p>\n'
```

And the file path it writes (in `build_site`):

```python
# Old:
(out_root / "tags").mkdir(parents=True, exist_ok=True)
... (out_root / "tags" / f"{tag}.html").write_text(page, encoding="utf-8")
# New:
(out_root / "concepts").mkdir(parents=True, exist_ok=True)
... (out_root / "concepts" / f"{tag}.html").write_text(page, encoding="utf-8")
```

And the tag-cloud page itself:

```python
# Old:
(out_root / "tags.html").write_text(build_tags_cloud(posts, nav_tmpl), encoding="utf-8")
# New:
(out_root / "concepts.html").write_text(build_tags_cloud(posts, nav_tmpl), encoding="utf-8")
```

Also update `render_nav` if `tags` appears as an `active=` value — find any hard-coded `"tags"`:

```python
# In build_tags_cloud and build_tag_page:
body += render_nav(nav_tmpl, active="concepts", depth=0)  # was "tags"
```

And in `assets/nav-header.html`, find:

```html
<a class="site-nav__link{ACTIVE_TAGS}" href="{TAGS_URL}">标签</a>
```

Replace with:

```html
<a class="site-nav__link{ACTIVE_CONCEPTS}" href="{CONCEPTS_URL}">概念</a>
```

And update `render_nav` Python helper:

```python
actives = {
    "home": "",
    "papers": "",
    "tutorials": "",
    "concepts": "",  # was "tags"
}
...
nav_tmpl.format(
    HOME_URL=...,
    PAPERS_URL=...,
    TUTORIALS_URL=...,
    CONCEPTS_URL=f"{prefix}concepts.html",  # was TAGS_URL
    ACTIVE_HOME=...,
    ACTIVE_PAPERS=...,
    ACTIVE_TUTORIALS=...,
    ACTIVE_CONCEPTS=actives["concepts"],  # was ACTIVE_TAGS
)
```

- [ ] **Step 3.4: Run pytest + smoke test**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -5
python3 build.py --smoke-test
```

The smoke test references `tags/diffusion.html` in its assertions. Update those:

```python
# In SMOKE_POSTS:
"tags": ["diffusion", "rl", "alignment"],  # OK, this stays (internal canonical)
# In assertions:
tag_page = (out / "concepts" / "diffusion.html").read_text(encoding="utf-8")
if "../papers/" not in tag_page or "../concepts.html" not in tag_page:
    print("FAIL: tag page paths not relative", file=sys.stderr)
    return 1
```

Note: the SMOKE_POSTS dicts use the key `"tags"` because that's the internal canonical name in the post dict shape (set up by discover_posts). Don't change the smoke posts' keys, only the file-path assertions.

- [ ] **Step 3.5: Commit**

```bash
git add build.py assets/nav-header.html
git commit -m "$(cat <<'EOF'
Phase 2: switch build.py link paths from tags/ to concepts/

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Run migration on real data + delete `tags/`

**Files:**
- 19× `papers/<slug>/index.md` (modified)
- 3× `tutorials/<slug>/index.md` (modified)
- 22× `index.md.pre-concepts` (backup, created then deleted)
- `tags/` directory (deleted)
- `tags.html` (deleted)

- [ ] **Step 4.1: Dry-run**

```bash
python3 migrate-concepts.py --convert --dry-run
```
Expect: 22 lines `DRY: would update papers/.../index.md` (some papers may not have `paper.code_url` to fold — but they all have `tags:`).

- [ ] **Step 4.2: Real migration**

```bash
python3 migrate-concepts.py --convert
```

- [ ] **Step 4.3: Validate post-migration**

```bash
python3 build.py --check
```
Expect: `OK: 22 posts validated.` (no warnings about legacy `tags:`).

- [ ] **Step 4.4: Build + verify**

```bash
python3 build.py
ls concepts/ | head -5
test -f concepts.html && echo "OK: concepts.html"
```

- [ ] **Step 4.5: Delete legacy `tags/` artifacts**

```bash
rm -rf tags/
rm -f tags.html
# Confirm build doesn't recreate them:
python3 build.py
test ! -e tags/ && echo "OK: tags/ gone"
test ! -f tags.html && echo "OK: tags.html gone"
```

- [ ] **Step 4.6: Cleanup .pre-concepts backups**

```bash
python3 migrate-concepts.py --cleanup --yes
```

- [ ] **Step 4.7: Commit**

```bash
git add papers/ tutorials/ tags/ tags.html concepts/ concepts.html 2>/dev/null || true
git rm -rf tags/ 2>/dev/null || true
git rm tags.html 2>/dev/null || true
git add papers/ tutorials/ concepts/ concepts.html
git commit -m "$(cat <<'EOF'
Phase 2: migrate frontmatter tags→concepts + delete legacy tags/

- 19 papers + 3 tutorials: tags: → concepts: rename + paper.code_url → repos
- Build output: tags.html / tags/<x>.html → concepts.html / concepts/<x>.html
- assets/nav-header.html: TAGS_URL → CONCEPTS_URL

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `build_lib/graph.py` core

**Files:**
- Create: `build_lib/graph.py`
- Create: `tests/test_graph.py`

This module exports: `parse_repo_url`, `discover_concepts`, `extract_graph`. (Concept page renderer + graph page renderer come in Task 6.)

- [ ] **Step 5.1: Write failing tests**

`tests/test_graph.py`:

```python
"""Tests for build_lib/graph.py."""
from __future__ import annotations

import pytest

from build_lib.graph import parse_repo_url, extract_graph


# --- parse_repo_url ---

def test_parse_github_url():
    out = parse_repo_url("https://github.com/foo/bar")
    assert out == {
        "host": "github.com",
        "owner": "foo",
        "name": "bar",
        "id": "repos/foo/bar",
        "url": "https://github.com/foo/bar",
    }


def test_parse_github_url_with_trailing_slash():
    out = parse_repo_url("https://github.com/foo/bar/")
    assert out["id"] == "repos/foo/bar"


def test_parse_github_url_with_subpath():
    """Take only the org/repo segments."""
    out = parse_repo_url("https://github.com/foo/bar/tree/main/x")
    assert out["id"] == "repos/foo/bar"
    assert out["owner"] == "foo"
    assert out["name"] == "bar"


def test_parse_non_github_falls_back():
    """Unknown host gets a hash-based id, but the URL is preserved."""
    out = parse_repo_url("https://gitlab.com/x/y")
    assert out["host"] == "gitlab.com"
    assert out["url"] == "https://gitlab.com/x/y"
    assert out["id"].startswith("repos/")


# --- extract_graph ---

def test_extract_graph_basic_node_counts():
    posts = [
        {"type": "paper", "slug": "l2p-2026", "title": "L2P", "date": "2026-05-22",
         "tldr": "...", "tags": ["diffusion"], "_url": "papers/l2p-2026/index.html",
         "_body_md": "", "concepts": ["diffusion"], "citations": [], "repos": []},
    ]
    concepts = {}
    g = extract_graph(posts, concepts)
    # 1 paper + 1 concept = 2 nodes
    assert len(g["nodes"]) == 2
    # 1 mentions edge
    assert any(e["kind"] == "mentions" for e in g["edges"])


def test_extract_graph_concept_node_marks_has_file_correctly():
    posts = [
        {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
         "tldr": "", "tags": ["flow-matching"], "_url": "papers/x/index.html",
         "_body_md": "", "concepts": ["flow-matching"], "citations": [], "repos": []},
    ]
    concepts = {"flow-matching": {"name": "Flow Matching", "aliases": [], "body": "user notes"}}
    g = extract_graph(posts, concepts)
    fm = [n for n in g["nodes"] if n["id"] == "concepts/flow-matching"][0]
    assert fm["has_file"] is True
    assert fm["name"] == "Flow Matching"


def test_extract_graph_concept_without_file():
    posts = [
        {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
         "tldr": "", "tags": ["lora"], "_url": "papers/x/index.html",
         "_body_md": "", "concepts": ["lora"], "citations": [], "repos": []},
    ]
    g = extract_graph(posts, concepts={})
    lora = [n for n in g["nodes"] if n["id"] == "concepts/lora"][0]
    assert lora["has_file"] is False
    # Default name = slug (titlecased? or as-is?). We use as-is.
    assert lora["name"] == "lora"


def test_extract_graph_citation_edge():
    posts = [
        {"type": "paper", "slug": "a", "title": "A", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/a/index.html",
         "_body_md": "", "concepts": [], "citations": ["b"], "repos": []},
        {"type": "paper", "slug": "b", "title": "B", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/b/index.html",
         "_body_md": "", "concepts": [], "citations": [], "repos": []},
    ]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"] if e["kind"] == "cites"]
    assert any(e["from"] == "papers/a" and e["to"] == "papers/b" for e in cite_edges)


def test_extract_graph_dedups_duplicate_citation_from_body_and_frontmatter():
    """If `citations:` and a body `[[slug]]` both reference the same target, emit one edge."""
    posts = [
        {"type": "paper", "slug": "a", "title": "A", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/a/index.html",
         "_body_md": "See [[b]] for context.", "concepts": [],
         "citations": ["b"], "repos": []},
        {"type": "paper", "slug": "b", "title": "B", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/b/index.html",
         "_body_md": "", "concepts": [], "citations": [], "repos": []},
    ]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"]
                  if e["kind"] == "cites" and e["from"] == "papers/a" and e["to"] == "papers/b"]
    assert len(cite_edges) == 1


def test_extract_graph_body_wiki_link_becomes_citation_when_no_frontmatter():
    """A [[slug]] in body alone produces a cites edge even if frontmatter citations: is empty."""
    posts = [
        {"type": "paper", "slug": "a", "title": "A", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/a/index.html",
         "_body_md": "See [[b]].", "concepts": [], "citations": [], "repos": []},
        {"type": "paper", "slug": "b", "title": "B", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/b/index.html",
         "_body_md": "", "concepts": [], "citations": [], "repos": []},
    ]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"] if e["kind"] == "cites"]
    assert any(e["from"] == "papers/a" and e["to"] == "papers/b" for e in cite_edges)


def test_extract_graph_tutorial_to_paper_is_covers():
    posts = [
        {"type": "tutorial", "slug": "t", "title": "T", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "tutorials/t/index.html",
         "_body_md": "See [[p]].", "concepts": [], "citations": [], "repos": []},
        {"type": "paper", "slug": "p", "title": "P", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/p/index.html",
         "_body_md": "", "concepts": [], "citations": [], "repos": []},
    ]
    g = extract_graph(posts, {})
    cover_edges = [e for e in g["edges"] if e["kind"] == "covers"]
    assert any(e["from"] == "tutorials/t" and e["to"] == "papers/p" for e in cover_edges)


def test_extract_graph_repo_edge():
    posts = [
        {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/x/index.html",
         "_body_md": "", "concepts": [], "citations": [],
         "repos": ["https://github.com/foo/bar"]},
    ]
    g = extract_graph(posts, {})
    impl = [e for e in g["edges"] if e["kind"] == "implements"]
    assert any(e["to"] == "repos/foo/bar" for e in impl)
    # Repo node also exists
    repo_node = [n for n in g["nodes"] if n["id"] == "repos/foo/bar"][0]
    assert repo_node["type"] == "repo"


def test_extract_graph_broken_citation_skipped():
    """If citations: refs a slug that doesn't exist, emit no edge (but no crash)."""
    posts = [
        {"type": "paper", "slug": "a", "title": "A", "date": "2026-01-01",
         "tldr": "", "tags": [], "_url": "papers/a/index.html",
         "_body_md": "", "concepts": [],
         "citations": ["ghost-2099"], "repos": []},
    ]
    g = extract_graph(posts, {})
    # Only 1 node (paper a), no edges
    cite_edges = [e for e in g["edges"] if e["kind"] == "cites"]
    assert cite_edges == []


def test_extract_graph_version_and_generated_at():
    """Returned graph has a version (int) and generated_at (placeholder string)."""
    g = extract_graph([], {})
    assert g["version"] == 1
    assert "generated_at" in g
```

- [ ] **Step 5.2: Verify fail**

`python3 -m pytest tests/test_graph.py -v` → ImportError.

- [ ] **Step 5.3: Implement `build_lib/graph.py`**

```python
"""Knowledge graph extraction for Phase 2.

Walks Phase-1-loaded posts + the concepts/ directory, produces a nodes+edges
JSON suitable for AI/RAG agents and for /graph.html visualization.

Public API:
    parse_repo_url(url)            -> dict   (host/owner/name/id/url)
    discover_concepts(root)        -> dict[slug, ConceptInfo]
    extract_graph(posts, concepts) -> dict   (the graph.json structure)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from build_lib.frontmatter import parse as parse_frontmatter
from build_lib.wiki_links import WIKI_LINK_RE


GRAPH_VERSION = 1
# Phase 2 doesn't take a real timestamp (build determinism); callers can override.
DEFAULT_GENERATED_AT = "unknown"


def parse_repo_url(url: str) -> dict[str, Any]:
    """Parse a GitHub (or other) repo URL into a structured dict.

    For github.com URLs: id = "repos/<owner>/<name>"
    For other hosts: id = "repos/<host>/<short-hash>"
    """
    u = urlparse(url.rstrip("/"))
    host = u.netloc or "unknown"
    parts = [p for p in u.path.split("/") if p]
    if host == "github.com" and len(parts) >= 2:
        owner, name = parts[0], parts[1]
        return {
            "host": host,
            "owner": owner,
            "name": name,
            "id": f"repos/{owner}/{name}",
            "url": url,
        }
    # Fallback for non-github
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return {
        "host": host,
        "owner": "",
        "name": "",
        "id": f"repos/{host}/{h}",
        "url": url,
    }


def discover_concepts(root: Path) -> dict[str, dict[str, Any]]:
    """Walk concepts/<slug>.md. Returns mapping slug -> ConceptInfo.

    ConceptInfo = {"name": str, "aliases": list[str], "parent": str | None,
                   "body_md": str (rest of file after frontmatter)}
    """
    out: dict[str, dict[str, Any]] = {}
    base = root / "concepts"
    if not base.is_dir():
        return out
    for f in sorted(base.glob("*.md")):
        slug = f.stem
        try:
            text = f.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        out[slug] = {
            "name": meta.get("name") or slug,
            "aliases": meta.get("aliases") or [],
            "parent": meta.get("parent"),
            "body_md": body,
        }
    return out


def extract_graph(posts: list[dict], concepts: dict[str, dict]) -> dict[str, Any]:
    """Construct the graph.json dict from in-memory posts + concept files.

    - posts: list of post dicts from build.discover_posts (must have _body_md, etc.)
    - concepts: dict[slug, ConceptInfo] from discover_concepts
    """
    # Build slug -> post lookup
    post_by_slug = {p["slug"]: p for p in posts}

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_node_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def _add_node(n: dict) -> None:
        if n["id"] in seen_node_ids:
            return
        seen_node_ids.add(n["id"])
        nodes.append(n)

    def _add_edge(from_id: str, to_id: str, kind: str) -> None:
        key = (from_id, to_id, kind)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"from": from_id, "to": to_id, "kind": kind})

    # 1. Add post nodes (paper / tutorial)
    for p in posts:
        kind_dir = "papers" if p["type"] == "paper" else "tutorials"
        node_id = f"{kind_dir}/{p['slug']}"
        _add_node({
            "id": node_id,
            "type": p["type"],
            "slug": p["slug"],
            "title": p["title"],
            "date": p["date"],
            "tldr": p["tldr"],
            "url": p["_url"],
        })

    # 2. Walk posts, add concept / repo nodes and edges
    for p in posts:
        kind_dir = "papers" if p["type"] == "paper" else "tutorials"
        from_id = f"{kind_dir}/{p['slug']}"

        # 2a. Concept edges (paper/tutorial -> concept)
        for concept_slug in (p.get("concepts") or []):
            concept_id = f"concepts/{concept_slug}"
            if concept_slug in concepts:
                info = concepts[concept_slug]
                name = info["name"]
                has_file = True
            else:
                name = concept_slug
                has_file = False
            _add_node({
                "id": concept_id,
                "type": "concept",
                "slug": concept_slug,
                "name": name,
                "has_file": has_file,
                "url": f"concepts/{concept_slug}.html",
            })
            _add_edge(from_id, concept_id, "mentions")

        # 2b. Citation edges via frontmatter `citations:`
        cite_targets: set[str] = set()
        for target_slug in (p.get("citations") or []):
            if target_slug in post_by_slug:
                cite_targets.add(target_slug)

        # 2c. Citation edges via body [[slug]] wiki-links
        body = p.get("_body_md") or ""
        for m in WIKI_LINK_RE.finditer(body):
            target_slug = m.group(1)
            if target_slug in post_by_slug:
                cite_targets.add(target_slug)

        # Emit cite/cover edges
        for target_slug in cite_targets:
            target = post_by_slug[target_slug]
            target_kind_dir = "papers" if target["type"] == "paper" else "tutorials"
            to_id = f"{target_kind_dir}/{target_slug}"
            edge_kind = "covers" if p["type"] == "tutorial" else "cites"
            _add_edge(from_id, to_id, edge_kind)

        # 2d. Repo edges
        for repo_url in (p.get("repos") or []):
            repo = parse_repo_url(repo_url)
            _add_node({
                "id": repo["id"],
                "type": "repo",
                "url": repo["url"],
                "host": repo["host"],
                "owner": repo["owner"],
                "name": repo["name"],
            })
            _add_edge(from_id, repo["id"], "implements")

    # 3. Add concept nodes for concepts that have a file but aren't mentioned by any post.
    for slug, info in concepts.items():
        cid = f"concepts/{slug}"
        if cid not in seen_node_ids:
            _add_node({
                "id": cid,
                "type": "concept",
                "slug": slug,
                "name": info["name"],
                "has_file": True,
                "url": f"concepts/{slug}.html",
            })

    return {
        "version": GRAPH_VERSION,
        "generated_at": DEFAULT_GENERATED_AT,
        "nodes": nodes,
        "edges": edges,
    }
```

- [ ] **Step 5.4: Verify pass**

`python3 -m pytest tests/test_graph.py -v` → all tests pass.

- [ ] **Step 5.5: Commit**

```bash
git add build_lib/graph.py tests/test_graph.py
git commit -m "$(cat <<'EOF'
Phase 2: build_lib/graph.py — extract knowledge graph from posts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Concept page + graph page renderers

**Files:**
- Modify: `build_lib/graph.py` (add render_concept_page, render_graph_page)
- Modify: `tests/test_graph.py` (add tests)
- Modify: `assets/style.css` (small additions)

- [ ] **Step 6.1: Failing tests**

Append to `tests/test_graph.py`:

```python
from build_lib.graph import render_concept_page, render_graph_page


def test_render_concept_page_includes_name_and_user_body():
    posts_mentioning = [
        {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
         "tldr": "summary", "tags": ["flow-matching"], "_url": "papers/x/index.html"},
    ]
    html = render_concept_page(
        slug="flow-matching",
        concept_meta={"name": "Flow Matching", "aliases": ["FM"], "parent": None,
                      "body_md": "Some intro paragraph."},
        posts_mentioning=posts_mentioning,
        nav_html='<nav class="site-nav"></nav>',
    )
    assert "Flow Matching" in html
    assert "Some intro paragraph" in html
    # Linked paper card appears (visiting from concepts/ — paths should be "../papers/...")
    assert "../papers/x/index.html" in html


def test_render_concept_page_with_no_user_body():
    """When concept_meta has empty body_md, page still renders the aggregation section."""
    posts_mentioning = [
        {"type": "paper", "slug": "x", "title": "X", "date": "2026-01-01",
         "tldr": "", "tags": ["lora"], "_url": "papers/x/index.html"},
    ]
    html = render_concept_page(
        slug="lora",
        concept_meta={"name": "lora", "aliases": [], "parent": None, "body_md": ""},
        posts_mentioning=posts_mentioning,
        nav_html="",
    )
    assert "lora" in html.lower()
    assert "../papers/x/index.html" in html


def test_render_graph_page_embeds_graph_json():
    graph = {"version": 1, "generated_at": "x", "nodes": [
        {"id": "papers/x", "type": "paper", "slug": "x", "title": "X",
         "date": "2026-01-01", "tldr": "", "url": "papers/x/index.html"}
    ], "edges": []}
    html = render_graph_page(graph, nav_html="")
    # cytoscape script included
    assert "cytoscape" in html.lower()
    # graph data embedded as a JSON literal
    assert "papers/x" in html
    # The script tag uses a data island, not a fetch (offline-friendly)
    assert "window.__GRAPH__" in html or '"nodes"' in html
```

- [ ] **Step 6.2: Implement renderers**

Add to `build_lib/graph.py`:

```python
import html as _html
import json

from markdown_it import MarkdownIt

# Lightweight parser for concept body (no plugin dependencies needed)
_CONCEPT_MD = MarkdownIt("commonmark", {"html": True})


def _render_concept_body(body_md: str) -> str:
    if not body_md.strip():
        return ""
    return _CONCEPT_MD.render(body_md)


def _render_post_card(post: dict, depth: int = 1) -> str:
    """Render a single post card with relative paths from concepts/ pages (depth=1)."""
    prefix = "../"
    is_tut = post["type"] == "tutorial"
    cls = "post-card--tutorial" if is_tut else "post-card--paper"
    badge_cls = "post-card__badge--tutorial" if is_tut else "post-card__badge--paper"
    badge_text = "📘 教程" if is_tut else "📄 论文"
    href = f"{prefix}{post['_url']}"
    title = _html.escape(post["title"], quote=True)
    tldr = _html.escape(post.get("tldr") or "", quote=True)
    date = _html.escape(post["date"], quote=True)
    return (
        f'<a class="post-card {cls}" href="{_html.escape(href, quote=True)}">\n'
        f'  <div class="post-card__head">\n'
        f'    <span class="post-card__badge {badge_cls}">{badge_text}</span>\n'
        f'    <span class="post-card__date">{date}</span>\n'
        f'  </div>\n'
        f'  <h3 class="post-card__title">{title}</h3>\n'
        f'  <p class="post-card__tldr">{tldr}</p>\n'
        f'</a>\n'
    )


def render_concept_page(slug: str, concept_meta: dict, posts_mentioning: list[dict],
                        nav_html: str) -> str:
    """Render a concept aggregation HTML page.

    Lives at concepts/<slug>.html (depth=1 below repo root).
    """
    name = _html.escape(concept_meta.get("name") or slug, quote=True)
    body_html = _render_concept_body(concept_meta.get("body_md") or "")
    aliases = concept_meta.get("aliases") or []
    aliases_html = ""
    if aliases:
        aliases_html = (
            '<p class="concept__aliases">aliases: '
            + ", ".join(f"<code>{_html.escape(a)}</code>" for a in aliases)
            + "</p>"
        )
    cards = "".join(_render_post_card(p, depth=1) for p in posts_mentioning)
    grid = (
        f'<div class="post-grid">\n{cards}</div>\n'
        if cards else
        '<p class="empty">还没有论文/教程提到此概念。</p>\n'
    )

    return (
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n'
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>{name} — paper-reading</title>\n'
        '  <link rel="stylesheet" href="../assets/style.css">\n'
        '</head>\n<body>\n'
        f'{nav_html}\n'
        '<main class="page page--concept">\n'
        f'  <h1>{name}</h1>\n'
        f'  {aliases_html}\n'
        f'  <div class="concept__body">\n{body_html}\n  </div>\n'
        '  <h2>提到此概念的论文 / 教程</h2>\n'
        f'  {grid}\n'
        '  <p class="back-link"><a href="../concepts.html">← 返回所有概念</a></p>\n'
        '</main>\n</body>\n</html>\n'
    )


def render_graph_page(graph: dict, nav_html: str) -> str:
    """Render /graph.html — cytoscape.js force-directed visualization."""
    graph_json = json.dumps(graph, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>知识图谱 — paper-reading</title>
  <link rel="stylesheet" href="assets/style.css">
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
  <style>
    body {{ margin: 0; }}
    .graph-shell {{ display: flex; flex-direction: column; height: 100vh; }}
    .graph-controls {{
      padding: 0.6rem 1rem; background: #f8f9fa; border-bottom: 1px solid #e0e0e0;
      display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
    }}
    .graph-controls label {{ display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.9rem; }}
    .graph-controls input[type="search"] {{
      padding: 0.3rem 0.6rem; border: 1px solid #ccc; border-radius: 4px;
      min-width: 200px;
    }}
    #cy {{ flex: 1; background: #fafbfc; }}
    .graph-tooltip {{
      position: absolute; background: rgba(33, 37, 41, 0.95); color: white;
      padding: 0.5rem 0.75rem; border-radius: 4px; font-size: 0.85rem;
      max-width: 320px; pointer-events: none; z-index: 10; display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .graph-tooltip h4 {{ margin: 0 0 0.3rem 0; font-size: 0.95rem; }}
    .graph-tooltip p {{ margin: 0; font-size: 0.8rem; line-height: 1.4; }}
  </style>
</head>
<body>
{nav_html}
<div class="graph-shell">
  <div class="graph-controls">
    <strong>知识图谱</strong>
    <label><input type="checkbox" data-type="paper" checked> 论文</label>
    <label><input type="checkbox" data-type="tutorial" checked> 教程</label>
    <label><input type="checkbox" data-type="concept" checked> 概念</label>
    <label><input type="checkbox" data-type="repo" checked> 仓库</label>
    <input type="search" id="graph-search" placeholder="搜索 slug/title…">
    <span style="margin-left:auto; color: #666; font-size: 0.85rem;">
      点节点跳转 · hover 看详情
    </span>
  </div>
  <div id="cy"></div>
  <div class="graph-tooltip" id="graph-tooltip"></div>
</div>

<script>
window.__GRAPH__ = {graph_json};
</script>
<script>
(function () {{
  const G = window.__GRAPH__;
  const colorByType = {{
    paper: '#4A90E2', tutorial: '#8B5CF6',
    concept: '#F59E0B', repo: '#6B7280',
  }};
  const shapeByType = {{
    paper: 'ellipse', tutorial: 'round-rectangle',
    concept: 'diamond', repo: 'triangle',
  }};

  // Pre-compute in-degree for node sizing
  const indeg = new Map();
  G.edges.forEach(e => indeg.set(e.to, (indeg.get(e.to) || 0) + 1));

  const elements = [
    ...G.nodes.map(n => ({{
      data: {{
        id: n.id, type: n.type, label: n.title || n.name || n.slug || n.id,
        url: n.url, raw: n,
        size: 16 + Math.min(28, (indeg.get(n.id) || 0) * 3),
      }},
    }})),
    ...G.edges.map((e, i) => ({{ data: {{ id: 'e' + i, source: e.from, target: e.to, kind: e.kind }} }})),
  ];

  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements,
    layout: {{ name: 'cose-bilkent', nodeRepulsion: 8000, idealEdgeLength: 120, animate: false }},
    style: [
      {{ selector: 'node', style: {{
        'background-color': ele => colorByType[ele.data('type')] || '#888',
        'shape': ele => shapeByType[ele.data('type')] || 'ellipse',
        'label': 'data(label)',
        'font-size': '11px',
        'text-valign': 'bottom', 'text-margin-y': 4,
        'text-wrap': 'ellipsis', 'text-max-width': 140,
        'width': 'data(size)', 'height': 'data(size)',
        'border-width': 1, 'border-color': '#333',
      }} }},
      {{ selector: 'edge', style: {{
        'width': 1,
        'line-color': '#bbb',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#bbb',
        'arrow-scale': 0.7,
      }} }},
      {{ selector: '.highlight', style: {{
        'border-width': 3, 'border-color': '#dc3545',
      }} }},
    ],
  }});

  // Click → navigate
  cy.on('tap', 'node', evt => {{
    const url = evt.target.data('url');
    if (url) window.location.href = url;
  }});

  // Hover tooltip
  const tip = document.getElementById('graph-tooltip');
  cy.on('mouseover', 'node', evt => {{
    const d = evt.target.data('raw');
    let html = '<h4>' + (d.title || d.name || d.slug || d.id) + '</h4>';
    if (d.tldr) html += '<p>' + (d.tldr.length > 240 ? d.tldr.slice(0, 240) + '…' : d.tldr) + '</p>';
    else if (d.type === 'repo') html += '<p>' + d.url + '</p>';
    else if (d.type === 'concept') {{
      const cnt = G.edges.filter(e => e.to === d.id).length;
      html += '<p>' + cnt + ' 篇提到此概念</p>';
    }}
    tip.innerHTML = html;
    tip.style.display = 'block';
  }});
  cy.on('mousemove', 'node', evt => {{
    tip.style.left = (evt.originalEvent.pageX + 12) + 'px';
    tip.style.top = (evt.originalEvent.pageY + 12) + 'px';
  }});
  cy.on('mouseout', 'node', () => {{ tip.style.display = 'none'; }});

  // Type filter
  document.querySelectorAll('.graph-controls input[type="checkbox"]').forEach(cb => {{
    cb.addEventListener('change', () => {{
      const t = cb.dataset.type;
      const sel = cy.nodes('[type = "' + t + '"]');
      if (cb.checked) sel.show(); else sel.hide();
    }});
  }});

  // Search → highlight
  const search = document.getElementById('graph-search');
  search.addEventListener('input', () => {{
    const q = search.value.trim().toLowerCase();
    cy.nodes().removeClass('highlight');
    if (!q) return;
    const matches = cy.nodes().filter(n => {{
      const d = n.data('raw');
      return (d.title || '').toLowerCase().includes(q)
        || (d.slug || '').toLowerCase().includes(q)
        || (d.name || '').toLowerCase().includes(q);
    }});
    matches.addClass('highlight');
    if (matches.length) cy.center(matches);
  }});
}})();
</script>
</body>
</html>
"""
```

- [ ] **Step 6.3: Add `.concept__body` and `.concept__aliases` styles in `assets/style.css`**

Append:

```css
/* Phase 2 — concept aggregation page */
.page--concept .concept__body { margin: 1rem 0 2rem 0; }
.page--concept .concept__body h2 { font-size: 1.2rem; }
.page--concept .concept__aliases { color: #666; font-size: 0.9rem; margin-bottom: 1rem; }
.page--concept .concept__aliases code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
.page--concept .empty { color: #888; font-style: italic; }
```

- [ ] **Step 6.4: Verify tests pass**

`python3 -m pytest tests/test_graph.py -v` → all pass.

- [ ] **Step 6.5: Commit**

```bash
git add build_lib/graph.py tests/test_graph.py assets/style.css
git commit -m "$(cat <<'EOF'
Phase 2: concept page + /graph.html renderers (cytoscape.js)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire `build.py` Pass 4-6

**Files:**
- Modify: `build.py`

Add three passes:
- Pass 4: `extract_graph_json` — produce `graph.json`
- Pass 5: `render_concept_pages` — produce `concepts/<slug>.html` for every concept (overriding the old `build_tag_page` output)
- Pass 6: `render_graph_html` — produce `/graph.html`

The existing `build_tag_page` (writing to `concepts/<x>.html`) is REPLACED by the new concept-aware version. The existing `build_tags_cloud` (writing to `concepts.html`) stays as the cloud overview.

- [ ] **Step 7.1: Add imports at top of build.py**

```python
import json
from build_lib.graph import (
    discover_concepts as graph_discover_concepts,
    extract_graph as graph_extract,
    render_concept_page as graph_render_concept_page,
    render_graph_page as graph_render_graph_page,
)
```

- [ ] **Step 7.2: Remove `build_tag_page` and its call**

The old function and its call site in `build_site` should be removed. We will produce `concepts/<slug>.html` via the new concept page renderer in Pass 5 (covers MORE concepts because it includes concepts with `concepts/<slug>.md` files even when no post mentions them).

Find in `build.py`:

```python
def build_tag_page(tag: str, tagged: list[dict], nav_tmpl: str) -> str:
    ...
```

Delete this entire function.

In `build_site`, find:

```python
    by_tag = _tag_index(posts)
    for tag in sorted(by_tag.keys()):
        page = build_tag_page(tag, by_tag[tag], nav_tmpl)
        (out_root / "concepts" / f"{tag}.html").write_text(page, encoding="utf-8")
        n_pages += 1
```

Delete this block. The concept pages will be written by Pass 5.

Note: `build_tags_cloud` still uses `_tag_index` to get counts, so keep `_tag_index` around.

- [ ] **Step 7.3: Add Pass 4-6 to main()**

After `build_posts(...)` in main(), add:

```python
    # Pass 4: knowledge graph extraction
    concepts_info = graph_discover_concepts(root)
    graph = graph_extract(posts, concepts_info)
    (root / "graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote graph.json: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges.")

    # Pass 5: render concepts/<slug>.html (aggregation pages)
    (root / "concepts").mkdir(parents=True, exist_ok=True)
    concept_slugs = sorted({n["slug"] for n in graph["nodes"] if n["type"] == "concept"})
    nav_concept = render_nav(nav_tmpl, active="concepts", depth=1)
    nav_concept = nav_concept  # depth=1 already gives ../ prefix, correct for concepts/<x>.html
    for slug in concept_slugs:
        meta = concepts_info.get(slug) or {
            "name": slug, "aliases": [], "parent": None, "body_md": "",
        }
        # posts that mention this concept
        mentioning = []
        for p in posts:
            if slug in (p.get("concepts") or []):
                mentioning.append(p)
        page = graph_render_concept_page(slug, meta, mentioning, nav_concept)
        (root / "concepts" / f"{slug}.html").write_text(page, encoding="utf-8")
    print(f"Wrote {len(concept_slugs)} concept page(s).")

    # Pass 6: render /graph.html
    nav_graph = render_nav(nav_tmpl, active="", depth=0)  # graph.html lives at root
    graph_html = graph_render_graph_page(graph, nav_graph)
    (root / "graph.html").write_text(graph_html, encoding="utf-8")
    print("Wrote graph.html.")
```

- [ ] **Step 7.4: Run + verify outputs**

```bash
python3 build.py
test -f graph.json && echo "OK: graph.json"
test -f graph.html && echo "OK: graph.html"
ls concepts/*.html | wc -l   # expect ~38 (one per current concept)
python3 -c "import json; g = json.load(open('graph.json')); print('nodes:', len(g['nodes']), 'edges:', len(g['edges']))"
```

- [ ] **Step 7.5: Run pytest**

`python3 -m pytest tests/ -v 2>&1 | tail -5` — confirm no regressions. Smoke test may need an update if it expects an older concept page format; if so, just update the smoke assertions.

- [ ] **Step 7.6: Commit**

```bash
git add build.py concepts/ graph.json graph.html
git commit -m "$(cat <<'EOF'
Phase 2: wire build.py Pass 4 (graph.json) + Pass 5 (concept pages) + Pass 6 (graph.html)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Write a sample `concepts/<slug>.md`

**Files:**
- Create: `concepts/flow-matching.md` (one example for demo / verification)

This validates the user-notes + auto-aggregation merge path. Keep it short — the point is showing the workflow.

- [ ] **Step 8.1: Create `concepts/flow-matching.md`**

```markdown
---
slug: flow-matching
name: "Flow Matching"
aliases: ["flow matching", "FM"]
---

# Flow Matching

Flow matching 是把"从噪声生成数据"建模成一个**连续时间 ODE** 的方法。

## 直觉

- DDPM 学的是 score $\nabla_x \log p_t(x)$ 的近似,反向 SDE 走轨迹。
- Flow matching 学的是**速度场** $v(x, t)$,反向 ODE 走轨迹。直观上"更短更直"。
- 训练目标:让模型预测的速度匹配 $\epsilon - x_0$ 这条线性插值的方向。

## 跟 DDPM 的关键差别

| 维度 | DDPM | Flow Matching |
|---|---|---|
| 模型预测什么 | noise / x0 / v | velocity $u(x, t)$ |
| 采样器 | SDE / DDIM 之类 | ODE (Euler / Heun) |
| 训练 loss | MSE 在 noise 空间 | MSE 在 velocity 空间 |
| 与 OT 关系 | 间接 | 直接 (rectified flow / OT-CFM) |

进一步看 [SD3](https://stability.ai/news/stable-diffusion-3-research-paper) 和 [Flux](https://github.com/black-forest-labs/flux),这两个都是 flow matching 的工业级实现。
```

- [ ] **Step 8.2: Rebuild and verify**

```bash
python3 build.py
# Verify the user notes appear on the concept page
grep "Flow matching 是把" concepts/flow-matching.html | head -1
# Verify aggregation: should still list the papers that have `flow-matching` in concepts
grep -c "post-card" concepts/flow-matching.html
```

- [ ] **Step 8.3: Commit**

```bash
git add concepts/flow-matching.md concepts/flow-matching.html graph.json
git commit -m "$(cat <<'EOF'
Phase 2: add sample concepts/flow-matching.md (user notes + auto-aggregate demo)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update skill templates

**Files:**
- Modify: `.claude/skills/reading-papers/templates/index.md`
- Modify: `.claude/skills/reading-papers/SKILL.md`
- Modify: `.claude/skills/writing-tutorial/templates/skeleton.md`
- Modify: `.claude/skills/writing-tutorial/SKILL.md`

- [ ] **Step 9.1: reading-papers template — replace `tags:` with `concepts:`, add `citations:` + `repos:`**

In `.claude/skills/reading-papers/templates/index.md`, update the frontmatter block:

```yaml
---
type: paper
slug: REPLACE-SLUG-YEAR
title: "REPLACE TITLE"
date: REPLACE-YYYY-MM-DD
tldr: |
  REPLACE multi-line summary.
concepts: [REPLACE, COMMA, SEPARATED]
citations: []      # optional: list of paper/tutorial slugs this paper cites
repos:             # optional: list of GitHub URLs related to this paper
  - ""
paper:
  arxiv_id: ""
  authors: ""
  venue: ""
  project_page: ""
  weights_url: ""
---
```

(`paper.code_url` removed; `repos` now carries that information.)

- [ ] **Step 9.2: reading-papers SKILL.md — describe new fields**

Find the section that describes frontmatter fields. Add (or update):

> - `concepts`: list of concept slugs this paper touches (was `tags` in Phase 1)
> - `citations`: list of other paper/tutorial slugs this paper extends/criticises/builds on
> - `repos`: list of GitHub URLs (replaces the old `paper.code_url`)
> - The body may also contain `[[other-slug]]` wiki-links, which are auto-extracted as citation edges by `build.py`

- [ ] **Step 9.3: writing-tutorial template**

In `.claude/skills/writing-tutorial/templates/skeleton.md`, frontmatter:

```yaml
---
type: tutorial
slug: REPLACE-DOMAIN-YEAR
title: "REPLACE: tutorial title"
date: REPLACE-YYYY-MM-DD
tldr: |
  REPLACE
concepts: [REPLACE, COMMA, SEPARATED]
citations: []       # papers this tutorial covers (renders as `covers` edge in graph)
repos: []           # GitHub repos referenced
tutorial:
  word_count: "REPLACE"
  reading_minutes: "REPLACE"
---
```

- [ ] **Step 9.4: writing-tutorial SKILL.md — describe new fields**

Same treatment as reading-papers.

- [ ] **Step 9.5: Commit**

```bash
git add .claude/skills/reading-papers/ .claude/skills/writing-tutorial/
git commit -m "$(cat <<'EOF'
Phase 2: update skill templates with concepts/citations/repos fields

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 10.1: Per-paper / Per-tutorial frontmatter sections**

Mention the new `concepts:`, `citations:`, `repos:` fields in the layout block or in a "frontmatter schema" note.

- [ ] **Step 10.2: Add a new section "Knowledge graph"**

```markdown
## Knowledge graph

After every `python3 build.py`, the following are regenerated:

- `graph.json` — full nodes + edges as a structured JSON. Designed for
  AI/RAG agents to ingest. Schema: see
  `docs/superpowers/specs/2026-05-29-md-blog-phase2-design.md` §5.
- `graph.html` — interactive force-directed visualization (cytoscape.js).
  Filter by node type, search by slug/title, click to jump.
- `concepts/<slug>.html` — per-concept aggregation page (which posts
  mention this concept). If `concepts/<slug>.md` exists, the user's notes
  render above the auto-aggregated list.

To add notes for a concept, just create `concepts/<slug>.md` with the
fields `slug:`, `name:`, optional `aliases:`, optional `parent:`, then
write user-facing markdown below the YAML block. Rebuild.

`build.py` does NOT garbage-collect stale concept pages or graph.json
entries when a concept is removed from all frontmatters; manually `rm`
the stale `concepts/<slug>.html` file if needed.
```

- [ ] **Step 10.3: Remove or update any leftover `tags.html` / `tags/` references**

Search for `tags.html`, `tags/` in CLAUDE.md and replace with `concepts.html`, `concepts/`.

- [ ] **Step 10.4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Phase 2: update CLAUDE.md with knowledge graph section + concepts/ paths

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 11.1: Clean rebuild**

```bash
python3 build.py
```
Expect:
- "Generated: N pages..."
- "Rendered 22 per-post HTML pages from markdown."
- "Wrote graph.json: ~85 nodes, ~150 edges." (rough numbers — vary with concept counts)
- "Wrote N concept page(s)."
- "Wrote graph.html."

- [ ] **Step 11.2: pytest passes**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -5
```
Expect: all tests pass (Phase 1 + Phase 2).

- [ ] **Step 11.3: graph.json schema sanity**

```bash
python3 - <<'EOF'
import json
g = json.load(open('graph.json'))
print(f"version: {g['version']}")
print(f"nodes: {len(g['nodes'])}")
print(f"edges: {len(g['edges'])}")
types = {}
for n in g['nodes']:
    types[n['type']] = types.get(n['type'], 0) + 1
print(f"node types: {types}")
kinds = {}
for e in g['edges']:
    kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
print(f"edge kinds: {kinds}")
EOF
```
Expect 4 node types (paper, tutorial, concept, repo) and at least `mentions` + `implements` edges.

- [ ] **Step 11.4: graph.html serves**

```bash
./serve.sh 8765 -bg 2>&1 | head -3
curl -s --noproxy '*' -o /dev/null -w "graph.html: %{http_code}\n" http://127.0.0.1:8765/graph.html
curl -s --noproxy '*' http://127.0.0.1:8765/graph.html | grep -c "cytoscape"
```
Expect: HTTP 200, multiple `cytoscape` references in the source.

- [ ] **Step 11.5: Browser spot-check** (described, not automated):

User should open in a real browser:
- <http://127.0.0.1:8765/graph.html> — verify nodes/edges render, click a paper node → jumps to that paper
- <http://127.0.0.1:8765/concepts/flow-matching.html> — verify user notes (intro / table) appear ABOVE the auto-generated paper list
- <http://127.0.0.1:8765/concepts.html> — concept cloud (was tags cloud) — links to `concepts/<x>.html` not `tags/<x>.html`

- [ ] **Step 11.6: Final commit (only if anything outstanding from previous tasks)**

```bash
git status
# Should be clean. Anything uncommitted from build (graph.json, concepts/*.html) is expected to already be in.
```

- [ ] **Step 11.7: Acceptance criteria — confirm each**

1. ✅ `python3 build.py` clean, regenerates graph.json + graph.html + concepts/*.html
2. ✅ 22 files have `concepts:` (no `tags:` residual)
3. ✅ `paper.code_url` migrated into `repos:` (where present)
4. ✅ `tags.html` and `tags/` removed; `concepts.html` and `concepts/` in place
5. ✅ graph.json has 4 node types + 4 edge kinds
6. ✅ /graph.html serves cytoscape viz
7. ✅ At least 1 hand-written `concepts/<slug>.md` validated
8. ✅ All tests pass
9. ✅ Skill templates updated (concepts/citations/repos in frontmatter)
10. ✅ CLAUDE.md has Knowledge-graph section

---

## Self-Review

### Spec coverage

| Spec section | Plan task |
|---|---|
| §3 Frontmatter changes (concepts/citations/repos) | Tasks 1, 2, 4 |
| §4 Concept files | Tasks 5, 6, 8 |
| §5 graph.json schema | Tasks 5, 7 |
| §6 /graph.html | Tasks 6, 7 |
| §7 Build flow Pass 4-6 | Task 7 |
| §8 Tag system migration | Tasks 3, 4 |
| §9 Skill / CLAUDE.md | Tasks 9, 10 |
| §10 Testing | Tests in each task + Task 11 |
| §11 Acceptance | Task 11.7 |

### Placeholder scan

No "TBD" / "TODO" / "implement later" in plan body. Each step contains concrete code or commands.

### Type consistency

- `extract_graph(posts, concepts)` returns dict with `nodes`/`edges` keys (Task 5) — consumed by `render_graph_page(graph, nav_html)` (Task 6) which accesses `graph` directly + as JSON literal in Task 7. ✓
- `discover_concepts(root)` returns `dict[slug -> dict]` with keys `name/aliases/parent/body_md` — consumed by `extract_graph` (in concept-info enrichment) and `render_concept_page` (Task 6, accessed as `concept_meta.get("name")` / `concept_meta.get("body_md")`). ✓
- `parse_repo_url(url)` returns dict with `host/owner/name/id/url` — consumed by `extract_graph` in Task 5. ✓

### Risks not flagged elsewhere

- The smoke test (`run_smoke_test` in build.py) does direct file assertions about generated tag pages. After Task 3, this may need adjustment. Task 3.4 notes the needed change.
- After Task 4, the legacy `tags/` directory and `tags.html` are deleted from git history of the repo. If someone has bookmarked `https://jimmysue.github.io/tags/diffusion.html`, they'll get a 404. Acceptable — Phase 2 doc explicitly chose simple migration over redirects.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-29-md-blog-phase2.md`.

User has already chosen subagent-driven execution. Next step: invoke `superpowers:subagent-driven-development`.
