# Homepage Sort by `added_date` Design

**Date:** 2026-06-02
**Status:** Approved
**Scope:** Small UX improvement — newly added papers/tutorials surface at top of homepage / papers.html / tutorials.html, instead of being buried by their (older) submission date.

---

## 1. 动机

Currently `index.html` / `papers.html` / `tutorials.html` sort by `frontmatter.date` (the paper's arxiv submission date) descending. When the user adds a new paper analysis of an OLDER paper (e.g. DiffusionOPD submitted 2026-05-14 added on 2026-06-02), the card lands in the middle of the grid instead of at the top, hiding the fact that the homepage updated.

Goal: **新增内容 = 视觉顶部**.

## 2. Scope

### In-scope

- Compute `added_date` per post = first-commit ISO date of the post's `index.md` via `git log`.
- Re-sort the three listing pages (index.html, papers.html, tutorials.html) by `added_date desc`.
- Show BOTH dates on each card: 📅 added_date · 论文 paper_date.
- Tests for the new helper.

### Out-of-scope

- Tag / concept aggregation pages — keep sorting by `paper.date` (those pages are "papers about X concept", paper-date is the natural ordering).
- `graph.html` — graph viz doesn't have ordered list semantics.
- A separate "recently added" highlighted section.
- "NEW" badges, animation, etc.

## 3. Architecture

### 3.1 New helper: `_added_date()` in `build.py`

```python
import subprocess

def _added_date(md_path: Path, fallback: str) -> str:
    """First-commit ISO date (YYYY-MM-DD) of the given file via git log.

    Returns `fallback` (the post's paper_date) on any error: file not in git,
    git not installed, command failure, repo with no commits.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow",
             "--format=%aI", "-1", "--", str(md_path)],
            capture_output=True, text=True, timeout=5, check=False,
        )
        out = result.stdout.strip()
        if out:
            return out[:10]  # ISO 8601 YYYY-MM-DD
    except (OSError, subprocess.SubprocessError):
        pass
    return fallback
```

### 3.2 Hook into `discover_posts()`

After populating `meta["date"]` and other fields, add:

```python
meta["_added_date"] = _added_date(md_path, fallback=meta["date"])
```

### 3.3 Re-sort listing pages

Change `_stable_desc()` to sort by `(_added_date, slug)` instead of `(date, slug)`:

```python
def _stable_desc(posts: list[dict]) -> list[dict]:
    return sorted(posts, key=lambda p: (p.get("_added_date", p["date"]), p["slug"]))[::-1]
```

Tag pages (`build_tag_page`) currently call `_stable_desc` — they should NOT, to keep paper-date ordering. Inline a paper-date sort there:

```python
def build_tag_page(...):
    tagged = sorted(tagged, key=lambda p: (p["date"], p["slug"]))[::-1]
    ...
```

(Actually `build_tag_page` was removed in Phase 2 in favor of `graph.render_concept_page`. So instead, `graph.py:render_concept_page` should sort `posts_mentioning` by `(date, slug)` desc.)

### 3.4 Card visual

In `render_card()`:

```python
added = post.get("_added_date") or post["date"]
date_html = (
    f'<span class="post-card__added">📅 {esc(added)}</span>'
    f'<span class="post-card__paper-date"> · 论文 {esc(post["date"])}</span>'
)
```

Tutorial cards keep `tutorial_meta` (word count / reading time) appended after.

CSS: `.post-card__paper-date` slightly smaller + greyer than `.post-card__added`.

## 4. Edge cases

| Case | Behavior |
|---|---|
| File never committed | `git log` empty output → fallback to `meta["date"]` |
| Git command times out (5s) | fallback to `meta["date"]` |
| `--smoke-test` runs with SMOKE_POSTS (no git history) | Each smoke post gets `_added_date` = `date` manually |
| Same `added_date` on multiple posts | tie-break by slug ascending (stable) |
| Post added today | `added_date` is today, shows at top |

## 5. Testing

- `tests/test_added_date.py` (new): mock a temporary git repo, commit a file, call `_added_date`, assert it returns the commit date.
- Test the fallback path: pass a non-git path, assert it returns the fallback string.
- Update `run_smoke_test`: add `_added_date` to each SMOKE_POSTS entry (just use the existing `date`).
- Update smoke assertions: index.html cards should now include `post-card__added` and `post-card__paper-date` classes.

## 6. Files touched

- `build.py` — new `_added_date()`, `discover_posts` populates `_added_date`, `_stable_desc` uses new key, `render_card` updates date block, `run_smoke_test` SMOKE_POSTS updated.
- `build_lib/graph.py` — `render_concept_page` sorts `posts_mentioning` by paper date (not added_date).
- `assets/style.css` — small additions for `.post-card__added` / `.post-card__paper-date`.
- `tests/test_added_date.py` — new.
- (Phase 2 docs in CLAUDE.md may want a short note; optional.)

## 7. Acceptance criteria

1. ✅ `python3 build.py` clean; homepage card for `diffusion-opd-2026` appears in **top 3** (because added 2026-06-02 is recent).
2. ✅ Each card on homepage shows both dates: `📅 2026-06-XX · 论文 2026-MM-DD`.
3. ✅ Concept pages (`concepts/<x>.html`) still sort by `paper.date` desc (regression check via diff).
4. ✅ pytest: existing 90 tests + ≥2 new for `_added_date` all pass.
5. ✅ `--smoke-test` still passes.
6. ✅ Live site updated on push.

## 8. Out-of-scope clarifications

- We are NOT adding `added_date` to frontmatter. The git log is the source of truth.
- We are NOT migrating existing 22 papers — `git log` already knows when each was added.
- We are NOT changing the graph.json schema. `_added_date` is a build-internal field.

End of spec.
