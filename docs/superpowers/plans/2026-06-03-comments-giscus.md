# Comments via Giscus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GitHub-Discussions-backed comments section (via Giscus) to every per-paper / per-tutorial page, opt-out via frontmatter.

**Architecture:** A new `assets/giscus-config.json` holds Giscus's repo_id/category_id/etc. `build.py` reads the config once at start, renders a `<section class="comments">` block via a helper, and appends it to `body_html` in `build_posts()` before `assemble_post_page`. Listing/concept/graph pages remain unaffected (they use different code paths).

**Tech Stack:**
- Python 3.9 stdlib (`json`) — no new deps
- Giscus CDN (client-side `<script>` tag, no build-time JS)
- pytest

Spec reference: `docs/superpowers/specs/2026-06-03-comments-giscus-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `assets/giscus-config.json` | NEW | Single source of truth for Giscus settings (repo, repo_id, category, category_id, theme, mapping, etc.) |
| `build.py` | MODIFY | Load config; add `_giscus_block(cfg)` helper; in `build_posts()` append block to `body_html` unless frontmatter has `comments: false`; print WARN if config missing/incomplete |
| `assets/style.css` | MODIFY (append) | `.comments` / `.comments__note` styling — visual spacing + privacy note |
| `tests/test_giscus_block.py` | NEW | Unit-test the helper renders config correctly; tests for opt-out, missing/incomplete config behaviour are integration tests on build.py |
| `CLAUDE.md` | MODIFY | New "Comments (Giscus) — first-time setup" section, documenting how to re-run setup if Discussions disabled |

User-supplied (already provided in chat):
- `repo_id` = `MDEwOlJlcG9zaXRvcnkyNTI4OTcxNjU=`
- `category` = `Announcements`
- `category_id` = `DIC_kwDODxLnjc4C-anB`

These get hard-coded into `assets/giscus-config.json` in Task 1.

---

## Task 1: Giscus config file

**Files:**
- Create: `assets/giscus-config.json`

- [ ] **Step 1.1: Create the config file**

Write `assets/giscus-config.json` with these exact contents:

```json
{
  "repo": "jimmysue/jimmysue.github.io",
  "repo_id": "MDEwOlJlcG9zaXRvcnkyNTI4OTcxNjU=",
  "category": "Announcements",
  "category_id": "DIC_kwDODxLnjc4C-anB",
  "mapping": "pathname",
  "strict": "0",
  "theme": "light",
  "reactions_enabled": "1",
  "emit_metadata": "0",
  "loading": "lazy",
  "lang": "zh-CN",
  "input_position": "bottom"
}
```

- [ ] **Step 1.2: Validate JSON**

Run: `python3 -c "import json; json.load(open('assets/giscus-config.json'))"`
Expected: No output (parse succeeded).

- [ ] **Step 1.3: Commit**

```bash
git add assets/giscus-config.json
git commit -m "$(cat <<'EOF'
Giscus config: repo_id + category_id for jimmysue.github.io

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_giscus_block()` helper + unit test (TDD)

**Files:**
- Modify: `build.py` (add helper near `_added_date`, `_inject_post_meta_block`)
- Create: `tests/test_giscus_block.py`

The helper takes a config dict and returns the HTML snippet. Pure function — easy to unit-test.

- [ ] **Step 2.1: Write failing test**

`tests/test_giscus_block.py`:

```python
"""Tests for build._giscus_block() — pure render helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("build_mod", repo_root / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = {
    "repo": "owner/name",
    "repo_id": "R_kgDOSAMPLE",
    "category": "Announcements",
    "category_id": "DIC_kwDOSAMPLE",
    "mapping": "pathname",
    "strict": "0",
    "theme": "light",
    "reactions_enabled": "1",
    "emit_metadata": "0",
    "loading": "lazy",
    "lang": "zh-CN",
    "input_position": "bottom",
}


def test_giscus_block_wraps_in_section():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert '<section class="comments">' in html
    assert "</section>" in html
    assert "<h2>" in html  # has a heading


def test_giscus_block_includes_all_data_attrs():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert 'data-repo="owner/name"' in html
    assert 'data-repo-id="R_kgDOSAMPLE"' in html
    assert 'data-category="Announcements"' in html
    assert 'data-category-id="DIC_kwDOSAMPLE"' in html
    assert 'data-mapping="pathname"' in html
    assert 'data-strict="0"' in html
    assert 'data-theme="light"' in html
    assert 'data-reactions-enabled="1"' in html
    assert 'data-emit-metadata="0"' in html
    assert 'data-input-position="bottom"' in html
    assert 'data-loading="lazy"' in html
    assert 'data-lang="zh-CN"' in html


def test_giscus_block_script_src_and_attrs():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert 'src="https://giscus.app/client.js"' in html
    assert 'crossorigin="anonymous"' in html
    assert "async" in html


def test_giscus_block_includes_privacy_note():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert "GitHub Discussions" in html
    assert 'class="comments__note"' in html


def test_giscus_block_escapes_repo_field():
    """If a config value contained an HTML-special char, it must be escaped
    so the attribute string is well-formed (and not abusable)."""
    build = _load_build()
    bad_cfg = dict(CFG, repo='owner/name" data-evil="1')
    html = build._giscus_block(bad_cfg)
    # The injected " must be escaped, so `data-repo` value is a single attr
    assert 'data-evil' not in html or '&quot;' in html
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_giscus_block.py -v`
Expected: All 5 tests FAIL with `AttributeError: module 'build_mod' has no attribute '_giscus_block'`.

- [ ] **Step 2.3: Implement `_giscus_block()` in `build.py`**

Add this function near the other small helpers (e.g. just before `_inject_post_meta_block`):

```python
def _giscus_block(cfg: dict) -> str:
    """Render the Giscus comments section HTML.

    `cfg` keys: repo, repo_id, category, category_id, mapping, strict, theme,
    reactions_enabled, emit_metadata, loading, lang, input_position.
    """
    def a(key: str) -> str:
        return esc(str(cfg.get(key, "")))

    return (
        '<section class="comments">\n'
        '  <h2>讨论 / Comments</h2>\n'
        '  <p class="comments__note">评论托管在本仓库的 '
        '<a href="https://github.com/jimmysue/jimmysue.github.io/discussions">'
        'GitHub Discussions</a>, 需 GitHub 账号。</p>\n'
        '  <script src="https://giscus.app/client.js"\n'
        f'    data-repo="{a("repo")}"\n'
        f'    data-repo-id="{a("repo_id")}"\n'
        f'    data-category="{a("category")}"\n'
        f'    data-category-id="{a("category_id")}"\n'
        f'    data-mapping="{a("mapping")}"\n'
        f'    data-strict="{a("strict")}"\n'
        f'    data-reactions-enabled="{a("reactions_enabled")}"\n'
        f'    data-emit-metadata="{a("emit_metadata")}"\n'
        f'    data-input-position="{a("input_position")}"\n'
        f'    data-theme="{a("theme")}"\n'
        f'    data-lang="{a("lang")}"\n'
        f'    data-loading="{a("loading")}"\n'
        '    crossorigin="anonymous"\n'
        '    async></script>\n'
        '</section>\n'
    )
```

NOTE: `esc()` already exists in `build.py` (uses `html.escape(..., quote=True)`). Reuse it for safety — config values flow into HTML attribute positions and must be escaped.

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_giscus_block.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add build.py tests/test_giscus_block.py
git commit -m "$(cat <<'EOF'
Add _giscus_block() helper for per-post comment section HTML

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Load Giscus config + inject into per-post pages

**Files:**
- Modify: `build.py` (load config in `build_posts()` or main; inject in build_posts loop)

- [ ] **Step 3.1: Add a config-loader helper**

Add to `build.py` near the top of the helper functions (e.g., right after `slugify_tag`):

```python
def _load_giscus_config(root: Path) -> dict | None:
    """Read assets/giscus-config.json. Returns None if missing or invalid.

    Caller is responsible for printing a WARN if None is returned so that
    builds continue without comments rather than crashing.
    """
    path = root / "assets" / "giscus-config.json"
    if not path.is_file():
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = (
        "repo", "repo_id", "category", "category_id", "mapping",
        "strict", "theme", "reactions_enabled", "emit_metadata",
        "loading", "lang", "input_position",
    )
    if not isinstance(cfg, dict):
        return None
    if any(not cfg.get(k) for k in required):
        return None
    return cfg
```

Make sure `import json` is present near the top of `build.py` (it already is — used by graph.py imports).

- [ ] **Step 3.2: Inject the block in `build_posts()`**

Find `build_posts()` in `build.py`. Locate the spot where `body_html` is finalized (after `body_html = _inject_post_meta_block(body_html, p)`) and BEFORE the `assemble_post_page(...)` call. Right after the `_inject_post_meta_block` line, add:

```python
        # Append Giscus comments section (default on; opt-out per-post via `comments: false`)
        if p.get("comments", True) and giscus_cfg is not None:
            body_html = body_html + _giscus_block(giscus_cfg)
```

To make `giscus_cfg` available inside `build_posts()`, modify the function signature and load it once at the top:

Change:
```python
def build_posts(posts: list[dict], nav_tmpl: str, slug_set: set[str] | None = None) -> int:
```

To:
```python
def build_posts(posts: list[dict], nav_tmpl: str, slug_set: set[str] | None = None,
                giscus_cfg: dict | None = None) -> int:
```

And in `main()`, before calling `build_posts`, add:

```python
    giscus_cfg = _load_giscus_config(root)
    if giscus_cfg is None:
        print("WARN: assets/giscus-config.json missing or incomplete; "
              "comments section will not be rendered.", file=sys.stderr)
```

Then update both call sites of `build_posts` in `main()` (the normal-build path and the `--post <slug>` single-post path) to pass `giscus_cfg=giscus_cfg`.

- [ ] **Step 3.3: Run full build + verify a sample paper has comments**

Run: `python3 build.py`
Expected: clean build, no WARN about giscus-config.

Run:
```bash
grep -c '<section class="comments">' papers/mrt-2026/index.html
```
Expected: `1`

Run:
```bash
grep -c 'data-repo-id="MDEwOlJlcG9zaXRvcnkyNTI4OTcxNjU="' papers/mrt-2026/index.html
```
Expected: `1`

- [ ] **Step 3.4: Verify listing pages do NOT have comments**

Run:
```bash
grep -c '<section class="comments">' index.html papers.html tutorials.html concepts.html graph.html 2>&1
```
Expected: `0` for each of these files.

- [ ] **Step 3.5: Verify concept pages do NOT have comments**

Run:
```bash
grep -lr '<section class="comments">' concepts/ 2>&1 | head
```
Expected: empty (no concept page contains it).

- [ ] **Step 3.6: Run full pytest**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 3.7: Smoke test**

Run: `python3 build.py --smoke-test`
Expected: `OK` (smoke test bypasses build_posts so no comments needed; smoke test continues to pass).

- [ ] **Step 3.8: Commit**

```bash
git add build.py
git commit -m "$(cat <<'EOF'
Inject Giscus comments section in per-post pages via build.py

- New _load_giscus_config(root) reads assets/giscus-config.json with
  graceful fallback (returns None if missing/invalid; build continues
  with WARN, no comments rendered).
- build_posts() now appends _giscus_block(cfg) to body_html unless
  frontmatter has `comments: false`.
- Only per-post (paper / tutorial) pages get comments; listing,
  concept, graph pages remain unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CSS for the comments section

**Files:**
- Modify: `assets/style.css` (append)

- [ ] **Step 4.1: Append the comments rules**

Append to `assets/style.css`:

```css
/* Comments section (Giscus) */
.comments {
  margin-top: 3.5rem;
  padding-top: 2rem;
  border-top: 2px solid #eee;
}
.comments h2 {
  font-size: 1.4rem;
  margin-bottom: 0.4rem;
}
.comments__note {
  color: #666;
  font-size: 0.9rem;
  margin: 0 0 1.2rem 0;
}
.comments__note a {
  color: #0969da;
}
.giscus,
.giscus-frame {
  min-height: 200px;
}
```

- [ ] **Step 4.2: Rebuild + visual sanity check**

Run: `python3 build.py`
Then in a browser visit (server already on 8766): `http://127.0.0.1:8766/papers/mrt-2026/`
Scroll to the bottom — you should see:
- A top border separator
- "讨论 / Comments" heading
- A grey note line about hosting on GitHub Discussions
- The Giscus iframe loading (slight delay due to `lazy`)

(This is a manual check — no automated assertion. The CSS change is pure styling, low risk.)

- [ ] **Step 4.3: Commit**

```bash
git add assets/style.css
git commit -m "$(cat <<'EOF'
CSS: comments section spacing + privacy note styling

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Opt-out integration test + frontmatter docs

**Files:**
- Modify: `tests/test_giscus_block.py` (add integration test for opt-out)

The opt-out logic lives inside `build_posts()`, which is harder to unit-test in isolation (it touches the filesystem). Instead we verify behaviour by creating a tiny stub post and running the full build path.

- [ ] **Step 5.1: Write integration test**

Append to `tests/test_giscus_block.py`:

```python
def test_build_omits_comments_when_frontmatter_says_false(tmp_path, monkeypatch):
    """If a post has `comments: false`, its index.html must NOT contain the comments section."""
    build = _load_build()
    repo_root = Path(__file__).resolve().parent.parent

    # Create a fake paper dir under tmp_path
    paper_root = tmp_path / "papers" / "_optouttest-2026"
    paper_root.mkdir(parents=True)
    (paper_root / "index.md").write_text(
        "---\n"
        "type: paper\n"
        "slug: _optouttest-2026\n"
        'title: "Opt-out test"\n'
        "date: 2026-06-03\n"
        "tldr: |\n"
        "  Opt-out test.\n"
        "concepts: [optouttest]\n"
        "comments: false\n"
        "---\n"
        "\n"
        "# Opt-out test\n"
        "\n"
        "Body text.\n",
        encoding="utf-8",
    )
    # Also need assets/giscus-config.json under tmp_path so config is loaded
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "giscus-config.json").write_text(
        '{"repo": "x/y", "repo_id": "R_a", "category": "C", "category_id": "D_b",'
        ' "mapping": "pathname", "strict": "0", "theme": "light",'
        ' "reactions_enabled": "1", "emit_metadata": "0", "loading": "lazy",'
        ' "lang": "zh-CN", "input_position": "bottom"}',
        encoding="utf-8",
    )
    # nav-header so render_nav doesn't fail
    (tmp_path / "assets" / "nav-header.html").write_text(
        '<nav class="site-nav"></nav>', encoding="utf-8",
    )
    # Run discover_posts + build_posts on tmp_path
    posts, _skip = build.discover_posts(tmp_path)
    assert any(p["slug"] == "_optouttest-2026" for p in posts)
    nav_tmpl = build.load_nav_header(tmp_path)
    cfg = build._load_giscus_config(tmp_path)
    build.build_posts(posts, nav_tmpl, slug_set={p["slug"] for p in posts}, giscus_cfg=cfg)
    html = (paper_root / "index.html").read_text(encoding="utf-8")
    assert '<section class="comments">' not in html, \
        "frontmatter comments: false must suppress the comments section"


def test_build_includes_comments_by_default(tmp_path):
    """If frontmatter omits `comments`, the comments section IS rendered."""
    build = _load_build()
    paper_root = tmp_path / "papers" / "_default-2026"
    paper_root.mkdir(parents=True)
    (paper_root / "index.md").write_text(
        "---\n"
        "type: paper\n"
        "slug: _default-2026\n"
        'title: "Default test"\n'
        "date: 2026-06-03\n"
        "tldr: |\n"
        "  Default test.\n"
        "concepts: [defaulttest]\n"
        "---\n"
        "\n"
        "# Default test\n"
        "\n"
        "Body text.\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "giscus-config.json").write_text(
        '{"repo": "x/y", "repo_id": "R_a", "category": "C", "category_id": "D_b",'
        ' "mapping": "pathname", "strict": "0", "theme": "light",'
        ' "reactions_enabled": "1", "emit_metadata": "0", "loading": "lazy",'
        ' "lang": "zh-CN", "input_position": "bottom"}',
        encoding="utf-8",
    )
    (tmp_path / "assets" / "nav-header.html").write_text(
        '<nav class="site-nav"></nav>', encoding="utf-8",
    )
    posts, _skip = build.discover_posts(tmp_path)
    nav_tmpl = build.load_nav_header(tmp_path)
    cfg = build._load_giscus_config(tmp_path)
    build.build_posts(posts, nav_tmpl, slug_set={p["slug"] for p in posts}, giscus_cfg=cfg)
    html = (paper_root / "index.html").read_text(encoding="utf-8")
    assert '<section class="comments">' in html
    assert 'data-repo-id="R_a"' in html


def test_build_omits_comments_when_config_missing(tmp_path):
    """If giscus-config.json is absent, no comments rendered (no crash)."""
    build = _load_build()
    paper_root = tmp_path / "papers" / "_noconfig-2026"
    paper_root.mkdir(parents=True)
    (paper_root / "index.md").write_text(
        "---\n"
        "type: paper\n"
        "slug: _noconfig-2026\n"
        'title: "No config test"\n'
        "date: 2026-06-03\n"
        "tldr: |\n"
        "  No config.\n"
        "concepts: [noconfigtest]\n"
        "---\n"
        "\n"
        "# No config\n"
        "\n"
        "Body.\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    # NOTE: deliberately do NOT write giscus-config.json
    (tmp_path / "assets" / "nav-header.html").write_text(
        '<nav class="site-nav"></nav>', encoding="utf-8",
    )
    posts, _skip = build.discover_posts(tmp_path)
    nav_tmpl = build.load_nav_header(tmp_path)
    cfg = build._load_giscus_config(tmp_path)
    assert cfg is None
    build.build_posts(posts, nav_tmpl, slug_set={p["slug"] for p in posts}, giscus_cfg=cfg)
    html = (paper_root / "index.html").read_text(encoding="utf-8")
    assert '<section class="comments">' not in html
```

NOTE: This test imports `discover_posts` and `load_nav_header` from `build.py`. These are existing public-ish functions (no leading underscore) — verify their names in `build.py` before running. If the loader is named differently, adjust the import accordingly. Read `build.py` briefly to confirm:

```bash
grep -n "^def discover_posts\|^def load_nav_header" build.py
```

- [ ] **Step 5.2: Run the new tests**

Run: `python3 -m pytest tests/test_giscus_block.py -v`
Expected: All 8 tests PASS (5 unit + 3 integration).

- [ ] **Step 5.3: Run full pytest one more time**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -5`
Expected: All tests pass; no regressions.

- [ ] **Step 5.4: Commit**

```bash
git add tests/test_giscus_block.py
git commit -m "$(cat <<'EOF'
Integration tests for Giscus opt-out + missing-config behaviour

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CLAUDE.md setup docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 6.1: Locate insertion point**

Read the last ~30 lines of `CLAUDE.md`:

```bash
tail -30 CLAUDE.md
```

Find a sensible spot — e.g. after the "Publishing to GitHub Pages" section, before "Known gotchas" if it exists. The new section is independent and self-contained, so it can go near the end.

- [ ] **Step 6.2: Append a new section**

Append this exact section to `CLAUDE.md`:

```markdown
## Comments (Giscus) — first-time setup

The blog uses [Giscus](https://giscus.app) for per-post comments, backed by
GitHub Discussions on the same repo (`jimmysue/jimmysue.github.io`). Comments
are rendered on paper / tutorial detail pages but not on listing, concept, or
graph pages. Per-post opt-out via frontmatter `comments: false`.

If Discussions are disabled or the Giscus App is uninstalled, comments stop
working but builds still succeed (`build.py` prints a WARN and skips
injection).

To re-enable from scratch:

1. <https://github.com/jimmysue/jimmysue.github.io/settings> → **Features** →
   tick **Discussions**.
2. Install the Giscus App at <https://github.com/apps/giscus> on
   `jimmysue.github.io` (Only select repositories).
3. Either use the default `Announcements` category, or create a new
   Announcement-type category named `Comments`.
4. Visit <https://giscus.app/zh-CN>, enter the repo, choose Mapping=Pathname,
   enable reactions, theme=light, lang=zh-CN. Copy the `data-repo-id` and
   `data-category-id` from the generated `<script>` block.
5. Paste them into `assets/giscus-config.json` (the file is checked into the
   repo; just overwrite the placeholder values).
6. `python3 build.py && ./publish.sh`.

Notes:
- `data-mapping="pathname"` ties each Discussion thread to a paper's URL
  path. Don't rename a paper's slug after readers have commented — the
  thread would orphan. (Discussion stays in the repo, just unlinked from
  the new URL.)
- Comments use `data-lang="zh-CN"`; switch to `en` if the audience shifts.
- To rotate IDs (e.g., transferred repo, new category): edit
  `assets/giscus-config.json`, rebuild, push. No code change needed.
```

- [ ] **Step 6.3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
CLAUDE.md: document Giscus comments setup + opt-out

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end verification + publish

**Files:** none (operational task)

This task runs the acceptance checks from spec §10 and pushes the result.

- [ ] **Step 7.1: Clean full rebuild**

Run: `python3 build.py 2>&1 | tail -10`
Expected:
- "Generated: N pages from ... papers + ... tutorials, ..."
- "Rendered N per-post HTML pages from markdown."
- "Wrote graph.json: ...; Wrote N concept page(s); Wrote graph.html."
- No WARN about giscus-config (since config file is in place).

- [ ] **Step 7.2: Per-post pages all have comments**

Run:
```bash
for f in papers/*/index.html tutorials/*/index.html; do
  grep -L '<section class="comments">' "$f" 2>/dev/null
done
```
Expected: empty output (every per-post page contains the comments section).

- [ ] **Step 7.3: Listing / concept / graph pages don't have comments**

Run:
```bash
grep -l '<section class="comments">' index.html papers.html tutorials.html concepts.html graph.html concepts/*.html 2>/dev/null
```
Expected: empty (no listing or concept page has comments).

- [ ] **Step 7.4: Full pytest**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 7.5: Smoke test**

Run: `python3 build.py --smoke-test`
Expected: `OK`.

- [ ] **Step 7.6: Browser manual sanity (one paper)**

Server is already on `:8766`. Open `http://127.0.0.1:8766/papers/mrt-2026/` in a browser:
- Scroll to the bottom
- Should see "讨论 / Comments" heading
- Below it: the privacy note line in grey
- Then the Giscus iframe loads (may take 0.5-1s with `loading="lazy"`)
- If logged into GitHub in the browser, the comment input is visible

(Manual check — if any step fails, debug before publishing.)

- [ ] **Step 7.7: Confirm git is clean**

Run: `git status`
Expected: working tree clean (all task commits already done).

- [ ] **Step 7.8: Push**

Run: `git push origin master 2>&1 | tail -3`
Expected: `master -> master` with new commit SHAs pushed.

- [ ] **Step 7.9: Live verification**

After ~1-2 minutes (GitHub Pages cache), open
`https://jimmysue.github.io/papers/mrt-2026/` in a private browser window:
- Scroll to the bottom
- Confirm comments section renders identically to local

(If the Giscus iframe shows an error like "discussion not found", that's
expected for a brand-new mapping — Giscus creates the thread when the first
comment is posted. Post a test comment yourself to verify the round-trip.)

---

## Self-Review

### Spec coverage

| Spec § | Plan task |
|---|---|
| §3 In-scope: config file | Task 1 |
| §3 In-scope: build.py loads + injects | Task 3 |
| §3 In-scope: style.css | Task 4 |
| §3 In-scope: `comments: false` frontmatter | Task 3.2 (injection guard) + Task 5 (test) |
| §3 In-scope: user setup docs | Task 6 |
| §3 In-scope: tests | Tasks 2 + 5 |
| §3 Out-of-scope (listing / concept / graph) | Architecture: those use different render paths (build_index/build_papers/build_tutorials/build_tags_cloud + graph.render_concept_page / graph.render_graph_page), all unchanged → no listing/concept/graph file modified |
| §5 Files touched | 1:1 match — Tasks 1/2/3/4/5/6 touch the exact files in §5 |
| §6 schemas (config + frontmatter) | Task 1 (config) + Task 3.2 (frontmatter check) |
| §7 design choices (mapping/lang/etc) | Task 1 config has every choice locked in |
| §8 user setup steps | Task 6 (mirrors §8 verbatim in CLAUDE.md) |
| §9 testing | Tasks 2 (unit) + 5 (integration) + 7 (manual + curl-equiv) |
| §10 acceptance | Task 7 explicitly walks through each acceptance item |
| §11 risk: missing config | `_load_giscus_config` returns None + WARN; Task 5.3 tests this path |
| §11 risk: spam / privacy / iframe | Architectural — handled by Giscus + privacy note; no implementation work |

No gaps.

### Placeholder scan

No `TBD`, `TODO`, `implement later`, `fill in details`, or "add appropriate error handling" anywhere. Every step has either concrete code, concrete bash, or a concrete manual check.

### Type / signature consistency

- `_giscus_block(cfg: dict) -> str` — defined in Task 2, used in Task 3, tested in Tasks 2+5. ✓
- `_load_giscus_config(root: Path) -> dict | None` — defined in Task 3, used in Task 3 (main), tested in Task 5.3 (`None` path). ✓
- `build_posts(...)` gains `giscus_cfg: dict | None = None` keyword arg — Task 3 defines, Task 5 uses with that exact keyword. ✓
- Config keys (`repo`, `repo_id`, `category`, `category_id`, `mapping`, `strict`, `theme`, `reactions_enabled`, `emit_metadata`, `loading`, `lang`, `input_position`) match between Task 1 (JSON), Task 2 (renders them), Task 3 (`required` tuple), Task 5 (test config) byte-for-byte. ✓
- Frontmatter field name `comments` (bool) — same string in Task 3.2 guard and Task 5 tests. ✓

### Risks not in the plan

- `discover_posts` / `load_nav_header` API expectations in Task 5: I'm assuming these public names exist on `build.py`. Task 5.1 includes a grep step to verify. If the names differ, the test author should adjust.
- The integration tests in Task 5 require `discover_posts` to find posts under `tmp_path`. If `discover_posts` is hard-coded to scan from `cwd` rather than its `root` argument, the tests will fail spuriously. The fix would be to chdir into tmp_path during the test (`monkeypatch.chdir(tmp_path)`), but the existing function signature `discover_posts(root: Path)` suggests it takes root as arg. Worth verifying when running the tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-comments-giscus.md`.

User indicated subagent-driven execution in the prompt args. Next step: invoke `superpowers:subagent-driven-development`.
