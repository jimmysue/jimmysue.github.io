# Blog Layout Redesign — Design Spec

**Date:** 2026-05-17
**Topic:** Static blog redesign: navigation header + merged feed + tag system
**Inspired by:** Hugo / Jekyll architecture (frontmatter + page templates + static generation)

## Goal

Turn the current ad-hoc `index.html` (manually-edited card list) into a proper static blog with:

1. **Navigation header** on every page — Home icon left, `论文` / `教程` / `标签` tabs right
2. **Homepage** = unified card grid of all papers + tutorials, newest first, with subtle visual distinction between the two types
3. **Tag system** — every paper / tutorial gets 3-6 tags; tag cloud page; click a tag to filter
4. **Static generation** from per-page `meta.json` frontmatter — `./build.py` rebuilds all list pages

## Non-goals

- Server-side rendering, search, RSS, comments, dark mode toggle, analytics
- Migrating away from MathJax / highlight.js (keep current rendering stack)
- Changing per-paper or per-tutorial content (only layout chrome + metadata)

## Architecture

```
paper-reading/
├── build.py                        Static generator (entry point)
├── publish.sh                      Patched: runs ./build.py first, then add/commit/push
├── assets/
│   ├── style.css                   Extended: site-nav, badge.paper, badge.tutorial, tag-cloud, tag-chip
│   └── nav-header.html             Shared HTML snippet (home icon + 3 right tabs)
├── papers/<slug>/
│   ├── index.html                  Existing (untouched content); nav header injected at top
│   └── meta.json                   NEW: { type:"paper", slug, title, date, tldr, tags }
├── tutorials/<slug>/
│   ├── index.html                  Existing; nav header injected
│   └── meta.json                   NEW: { type:"tutorial", ..., tutorial_meta:{ word_count, reading_minutes } }
├── tags/
│   └── <tag-slug>.html             GENERATED per unique tag — filtered list view
├── index.html                      GENERATED — combined grid, sorted by date desc
├── papers.html                     GENERATED — papers only
├── tutorials.html                  GENERATED — tutorials only
└── tags.html                       GENERATED — tag cloud (font-size ∝ count)
```

## meta.json schema

```json
{
  "type": "paper",
  "slug": "awm-2025",
  "title": "AWM: Advantage Weighted Matching — 把扩散模型 RL 拽回到预训练目标",
  "date": "2026-05-15",
  "tldr": "短摘要,30–80 字...",
  "tags": ["diffusion", "rl", "grpo", "flow-matching"],
  "tutorial_meta": null
}
```

Tutorials add:
```json
"tutorial_meta": { "word_count": "10.8k", "reading_minutes": "80-110" }
```

## Pages

| Path | Content |
|---|---|
| `/` | Combined paper + tutorial cards, date desc, type badge per card |
| `/papers.html` | Papers only |
| `/tutorials.html` | Tutorials only |
| `/tags.html` | Tag cloud — every unique tag with `(count)`, font-size scales 0.9rem – 2.2rem |
| `/tags/<tag-slug>.html` | All papers + tutorials tagged with that slug; back link to `/tags.html` |

## Navigation header

- Inline SVG home icon → `/`
- Right side three text tabs: `论文` (papers.html) · `教程` (tutorials.html) · `标签` (tags.html)
- `.nav-link.active` on the current page
- Sticky-top, full-width, light bg, 1px bottom border
- Mobile (≤640px): collapse to icon-left + horizontal-scroll right
- On per-paper / per-tutorial pages: a thin breadcrumb under the header showing `论文 / AWM: ...` or `教程 / 扩散 RL ...`

## Visual distinction (paper vs tutorial cards)

| | Paper | Tutorial |
|---|---|---|
| Badge text | `📄 论文` | `📘 教程` |
| Badge bg | `#e8eef5` (cool gray-blue) | `#fbeed8` (warm gold) |
| Left border | 2px solid `#3b82f6` (blue) | 2px solid `#f59e0b` (orange) |
| Meta row | date | date · word_count · reading_minutes |
| Tag chips row | up to 3 chips at bottom | up to 3 chips at bottom |

Small chips next to title: `<span class="tag-chip">diffusion</span>` linking to `/tags/diffusion.html`.

## Build pipeline

`build.py` steps (idempotent, safe to re-run):

1. Glob `papers/*/meta.json` and `tutorials/*/meta.json`
2. Sort by `date` desc
3. Render `index.html` (all), `papers.html` (paper), `tutorials.html` (tutorial)
4. Collect all unique tags → render `tags.html` (cloud) + `tags/<slug>.html` for each
5. Use Python string templates (no Jinja2) — keep zero deps

`inject-header.py` (run separately, on demand):

1. Read `assets/nav-header.html`
2. For each `papers/*/index.html` + `tutorials/*/index.html`:
   - If file contains `<!-- NAV-START -->...<!-- NAV-END -->`, replace it
   - Else: insert just inside `<body>` (before existing `<main>` or `<nav class="toc">`)
   - Adjust home/tab URLs to use `../../` prefix (relative from per-page depth)
3. Idempotent — re-running is a no-op when content matches

## Migration

`migrate.py`:

1. Parse current top-level `index.html` to extract paper-card / tutorial-card contents (title, date, tldr) — these already encode the metadata we need
2. For each `<slug>` found, also read the per-page `<h1>` and any subtitle for the canonical title
3. Assign tags by inspecting the title + tldr (small lookup dict + content-based heuristics; the running subagent has full context)
4. Write `papers/<slug>/meta.json` and `tutorials/<slug>/meta.json` for each — 14 papers + 2 tutorials

Tags are kebab-case slugs. Initial vocabulary (suggested, can grow):
- topic: `diffusion`, `flow-matching`, `llm`, `rl`, `image-gen`, `video`, `multimodal`, `try-on`, `editing`
- method: `ppo`, `grpo`, `dpo`, `ddpo`, `lora`, `distillation`, `alignment`
- system: `sd3`, `flux`, `wan2`, `hunyuanvideo`, `qwen`, `gemini`, `sora`
- nature: `production`, `tutorial`, `theory`, `benchmark`

## Subagent allocation

Four independent subagents in parallel (no shared file edits):

| ID | Task | Touches |
|---|---|---|
| A | Write `build.py` + 5 page templates (index, papers, tutorials, tags-cloud, tag-page) | `build.py`, embedded templates |
| B | Write `migrate.py`, run it, produce 16 × `meta.json` | `papers/*/meta.json`, `tutorials/*/meta.json`, `migrate.py` |
| C | Write `assets/nav-header.html` + extend `assets/style.css` | `assets/nav-header.html`, `assets/style.css` |
| D | Write `inject-header.py`, run it on all per-paper / per-tutorial HTML | `inject-header.py`, `papers/*/index.html`, `tutorials/*/index.html` |

D depends loosely on C (it injects the header that C designs), but for parallel speed: C writes the header to a known path before D starts. Alternative: C and D in one phase, then A and B in another. **Chosen: A+B+C in phase 1; D in phase 2 after C finishes.**

After subagents: main agent runs `python3 build.py`, smoke tests `localhost:88xx`, then commits + merges + pushes.

## Failure modes & guardrails

- **Sub-module gitlink trap (recently bit us):** the new `papers/*/repo-*/` rule is in `.gitignore` already. `build.py` and `inject-header.py` MUST NOT shell out to anything that could leave a stray `.git` inside `papers/`.
- **Idempotency:** `build.py` overwrites generated files; `inject-header.py` uses marker comments. Both safe to re-run.
- **Broken links:** `build.py` validates that every meta.json's `slug` resolves to a real `index.html`.
- **Tag collisions:** Tags are lowercased and kebab-cased before grouping.
- **GitHub Pages cache:** Same as before — 1–2 min after push.

## Acceptance criteria

1. `./build.py` runs without errors on the current 14 + 2 docs
2. `index.html` lists 16 cards, newest-first, with type badges
3. `papers.html`, `tutorials.html`, `tags.html` render correctly
4. `tags/<slug>.html` exists for each unique tag
5. Every `papers/*/index.html` and `tutorials/*/index.html` has the new nav header at top (marker-comment-protected)
6. Live site updated after push (verified by `last-modified` + HTML diff)
7. Mobile view (≤640px) doesn't break — nav scrolls horizontally; cards stack single-column

## Out of scope (future work)

- Search box, RSS feed, dark mode toggle, comment system
- Anchor links per card (h3 ids)
- Translation toggle (zh ↔ en)
- Replacing MathJax with KaTeX (different perf tradeoff)
