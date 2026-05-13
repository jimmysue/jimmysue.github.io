# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`paper-reading/` is a **static HTML blog**, generated and maintained by two project-local skills under `.claude/skills/`:

- **`reading-papers`** — turns one paper into one 5-section critique page at `papers/<slug>/index.html`.
- **`writing-tutorial`** — turns one **domain** (multiple papers + ≥2 GitHub repos + optional blogs) into one deep tutorial page at `tutorials/<slug>/index.html`, with spiral-structure sections.

Both are project-local (versioned with the workspace, not in `~/.claude/skills/`). They share the rendering stack (MathJax, highlight.js, TOC contract) and the shared `assets/style.css`.

The skills are the source of truth for structure. When something feels prescriptive (layout, IDs, file names, citation format), check `.claude/skills/<name>/SKILL.md` before improvising.

## Serving the blog

```bash
./serve.sh                # foreground on :8765
./serve.sh 9000           # custom port
./serve.sh 8765 -bg       # background, log to /tmp/paper-reading-server.log
```

`serve.sh` auto-falls-back to 8766–8770 if the requested port is taken. It runs `python3 -m http.server` bound to `127.0.0.1`.

**Do not open the pages via `file://`** — the relative paths to `../../assets/style.css` and the CDN-loaded MathJax/highlight.js scripts behave unpredictably outside HTTP. Always serve.

## Per-paper layout

Each paper lives under `papers/<slug>/`:

```
papers/<slug>/
  index.html              # the 图文 analysis page
  figures/                # final cropped PNGs referenced from index.html
  figures-raw/            # intermediate pdfimages/pdftoppm output (gitignore candidate)
  raw/paper.pdf           # original PDF download
  repo/                   # cloned source repo (read-only reference)
```

**The slug is always `shorttitle-year`** — title-based, not author-based (e.g., `rls-razor-2025`, `flow-opd-2026`, `d-opsd-2026`). Bare arxiv IDs (e.g. `2509.04259`) are **not** acceptable as folder names — they're opaque and make the file tree unbrowsable. See `.claude/skills/reading-papers/SKILL.md` §1b for the exact construction rules. The shared `assets/style.css` and the top-level `index.html` (the homepage card grid) belong to the workspace, not any one paper.

## Per-tutorial layout

Each tutorial (output of the `writing-tutorial` skill) lives under `tutorials/<slug>/`:

```
tutorials/<slug>/
  index.html              # the 图文 deep-dive tutorial page
  plan.md                 # outline agreed with user — Phase-3 gate; checked in
  figures/                # final cropped/drawn diagrams
  figures-raw/            # intermediates (gitignored)
  sources/
    MANIFEST.md           # every cited paper/repo/blog + rationale (checked in)
    papers/*.pdf          # anchor papers (checked in for provenance)
    repos/<org>-<name>/   # cloned reference implementations (gitignored)
  code-excerpts/          # pre-staged code snippets w/ citation header (gitignored —
                          #   regeneratable from sources/repos/ at the cited line range)
```

**Slug rule**: same as papers — `<shorttitle>-<year>` where the year is the seminal paper's year, NOT today's year. Examples: `lora-2021`, `diffusion-2020`, `flow-matching-2023`. The slug is the canonical method name when one exists.

**A tutorial is not a paper analysis.** Use `reading-papers` for "one paper → one critique". Use `writing-tutorial` for "one domain → one deep tutorial drawing from ≥2 papers + ≥2 repos." If unclear, the user's wording usually disambiguates: 精读 / 讲解 / critique / summarize → reading-papers; 写教程 / 讲透 / 系统讲解 / tutorial → writing-tutorial.

**Per-tutorial page invariants** (enforced by `writing-tutorial` skill):
- Every `<h2 id="sec-N">` has exactly 5 `<h3 id="sec-N-1..5">` sub-sections in the spiral order: 直觉 → 最小 demo → 正式化 → 代码引用 → 洞察.
- Step 2 demo code blocks tagged `class="teaching-demo"` (hand-written, non-production); ALL other code blocks must be quoted verbatim from `sources/repos/...` with `<p class="code-source">…file:Lstart-Lend — role</p>` above.
- Every display equation followed by a `<p class="math-translation">—— 翻译: …</p>` within 100 字.
- Word count 10–20k 中文字 (or English equivalent ~6–13k words) per tutorial.

## The rendering stack (non-obvious bits)

Each `papers/<slug>/index.html` loads three CDN dependencies in `<head>`:

- **MathJax 3** for LaTeX (`$...$` inline, `$$...$$` display). **Pitfall**: a literal `<` inside math gets parsed as an HTML tag — write `\lt` instead. E.g. `x_{\lt n}` not `x_{<n}`.
- **highlight.js 11.9** with `atom-one-dark` for `<pre><code class="language-python">…</code></pre>`.
- A small inline `IntersectionObserver` script for the TOC active-section highlight. It also auto-scrolls the TOC to keep the current item visible.

Smooth scroll + scroll-margin are global in `assets/style.css`:

```css
html { scroll-behavior: smooth; }
:where(h1, h2, h3, h4)[id] { scroll-margin-top: 1.5rem; }
```

## TOC contract (enforced by the skill)

The right-side floating TOC is required on every paper page:

- Every `<h2>` has `id="sec-N"` (top-level section 1–5).
- Every `<h3>` has `id="sec-N-M"`, or `id="sec-N-intro"` for an un-numbered lead-in.
- `<h4>` does **not** get an ID and is not in the TOC.
- The TOC `<nav class="toc">` lists every `<h2>`/`<h3>` in document order. `<li class="h2">` for top-level, `<li class="h3">` for sub.
- Sanity check before publishing: TOC link count must equal the count of `<h*[id^=sec-]>` in the body.
- TOC is hidden via media query at viewport ≤ 1200px (handled in `assets/style.css`); don't try to force it visible on mobile.

## Figure extraction (the part that takes the most tooling)

Figures are cropped from each paper's PDF using `poppler` + `sips` (macOS built-in). The two-stage flow:

```bash
# 1. Render whole pages at 200 DPI
pdftoppm -png -r 200 -f <page> -l <page> raw/paper.pdf figures-raw/page

# 2. Crop the figure region
sips --cropOffset <Y> <X> --cropToHeightWidth <H> <W> \
     figures-raw/page-NN.png --out figures/figN-shortname.png
```

`pdfimages -all -p` extracts embedded raster images but **misses vector figures** (most architecture diagrams). For those, render the page and crop. Letter-size pages at 200 DPI are 1700×2200 px — assume that geometry when picking offsets.

Use `pdfinfo raw/paper.pdf` for page count and `Read` on a rendered page to visually identify crop bounds.

## Code cross-reading conventions

Inline code excerpts come from `papers/<slug>/repo/`. Citation format is a `<p class="code-source">` immediately preceding the `<pre>`:

```html
<p class="code-source">repo/<file>:L<start>-L<end> — <one-line role></p>
<pre><code class="language-python">…trimmed snippet…</code></pre>
```

Keep snippets to **10–40 lines** of load-bearing code. The point is to anchor a paper claim to a concrete function, not to mirror the repo. Add a one-sentence note before/after explaining how the snippet maps to the paper's equation/algorithm.

## Adding a new paper

This is the work of the `reading-papers` skill. Direct invocation flow (the skill executes this):

1. `mkdir -p papers/<slug>/{figures,figures-raw,raw}`
2. `curl -L -o papers/<slug>/raw/paper.pdf <pdf-url>`
3. Locate code (arxiv abstract → Papers with Code → GitHub search). Either `git clone --depth 1` or fetch raw files via `https://raw.githubusercontent.com/<org>/<repo>/main/<path>`. The proxy on this machine **kills git clone via `https://github.com/`** unless `-c http.proxy=http://127.0.0.1:7897 -c https.proxy=…` is set; raw-file fetches via `curl` go through fine.
4. Read PDF with `pdftotext -layout raw/paper.pdf`.
5. Crop figures (see above).
6. Write `papers/<slug>/index.html` from the skeleton in `.claude/skills/reading-papers/SKILL.md` §5b.
7. Update the top-level `index.html`: prepend a new `<a class="paper-card">` inside `<div class="paper-grid">` (newest first).
8. Restart `./serve.sh` if not running and open the page.
9. **Publish to GitHub Pages (project rule — see next section).**

## Adding a new tutorial

This is the work of the `writing-tutorial` skill. Six phases (skill executes; user gates Phase 3):

1. **Scope** — `mkdir -p tutorials/<slug>/{figures,figures-raw,code-excerpts,sources/papers,sources/repos}`
2. **Discover** — parallel Explore subagents search arxiv / GitHub (≥1k★ filter) / blogs. Skill writes `sources/MANIFEST.md`.
3. **Plan** — main agent writes `tutorials/<slug>/plan.md` with 6–10 大节, each with 5-step briefs + drafting groups. **STOP. Ask user before continuing.**
4. **Extract** — parallel Explore subagents pre-stage `figures/secN-*.png` and `code-excerpts/secN-step4-*.txt` (verbatim from cited repo with file:Lstart-Lend header).
5. **Draft** — grouped-parallel general-purpose subagents, ONE per section, each emitting one `<section id="sec-N">` block following the spiral contract.
6. **Assemble + Verify + Publish** — main agent stitches sections, builds TOC, runs the verification checklist (in `.claude/skills/writing-tutorial/SKILL.md`), restarts `./serve.sh`, prepends a `<a class="tutorial-card">` to a 教程 section ABOVE the 论文 grid in the top-level `index.html` (creating the section if it doesn't exist yet), and asks before `./publish.sh`.

The skill is large — read `.claude/skills/writing-tutorial/SKILL.md` (~500 lines, includes rationalization table + red flags) before improvising.

## Publishing to GitHub Pages (PROJECT RULE)

**This rule is project-specific and intentionally lives here, not in the skill** (skills are generic across workspaces; deploy targets are not).

The blog is deployed to `jimmysue/jimmysue.github.io` (live at <https://jimmysue.github.io/>). After every new paper or non-trivial edit, run:

```bash
./publish.sh                    # auto: derives commit msg from changed paper slugs
./publish.sh "custom message"   # explicit message
./publish.sh --force-init       # destructive; only for full reset (won't be needed again)
```

`publish.sh` does:
1. `git add -A` (respects `.gitignore` — raw PDFs, cloned source repos, page renders, `.claude/`, dev screenshots are all excluded)
2. Commits with an auto-derived or supplied message
3. `git push origin master`

### Deploy invariants

- **Remote**: `git@github.com:jimmysue/jimmysue.github.io.git` (SSH; HTTPS form is in the README but SSH is what works on this machine).
- **Branch**: `master` (not `main` — github.io repo was created before GitHub flipped the default).
- **`.nojekyll`** at repo root tells GitHub Pages to skip Jekyll processing (must stay; otherwise the `_`-prefixed names and raw HTML could be mangled).
- **GitHub Pages cache**: ~1–2 min after push for the live site to update. `curl https://jimmysue.github.io/` immediately after push may still serve the previous content.
- **Backup of pre-replacement Jekyll site**: `~/Backups/jimmysue.github.io-backup-20260512-094255/` — full clone of the old `b0779f8` commit. Restore by `git push --force` from there if ever needed.

### When NOT to publish

- User explicitly says "draft only" / "本地预览就行" / "先不发布".
- Page renders broken locally (TOC dangling, MathJax error, figures missing).
- `git status` shows files outside `papers/<slug>/`, `tutorials/<slug>/`, `index.html`, or `assets/` (unexpected scope — verify before pushing).

### Ask before pushing

By default, **ask the user** before running `publish.sh` for any given paper, unless they've already said "and publish" / "发布到 gh page" in the same turn or earlier in the session. Pushing is visible to others and the commit history is permanent.

## Editing existing pages

- **Section title changes**: update both the `<h2>`/`<h3>` text and the matching `<li>` text in the TOC. The `id` should usually NOT change (would break external links).
- **Section reorder**: re-run the TOC count sanity check. Keep IDs in `sec-N-M` numerical order matching the visible numbering.
- **Style changes**: edit `assets/style.css` (single source of truth — every paper inherits). If the change is paper-specific, that's a smell.

## Updating the skills

Both `reading-papers` and `writing-tutorial` live at `.claude/skills/<name>/` inside this project (versioned alongside the blog). Changes that should propagate to future paper analyses / tutorials (new section requirements, new CSS, new verification checks) belong in the relevant skill, not in any one rendered page. Pattern:

1. Update `.claude/skills/<name>/SKILL.md` (instructions + HTML skeleton).
2. Update `.claude/skills/<name>/templates/*.css` if CSS changed (writing-tutorial uses `templates/style-additions.css`).
3. Append/merge CSS changes into the shared `assets/style.css` (single source of truth at render time).
4. Backfill the change into existing rendered pages if you want consistency.

The CSS in `assets/style.css` was originally copied from the skill templates. They drift if you edit only one side.

## Known gotchas seen during dev

- Headless Chrome `--screenshot --window-size=W,H` on very tall windows (>16000 px) silently captures only the bottom half. For full-page screenshots use `agent-browser screenshot --full` with a clean session, and crop afterwards if needed.
- The local proxy at `http://127.0.0.1:7897` (Clash) intercepts even `localhost` requests for `curl` — use `--noproxy '*'` for smoke tests against `./serve.sh`.
- `pdftoppm` output filename pads page numbers as `page-01.png`, `page-02.png` ... (zero-padded) when there are ≥10 pages but as `page-1.png` for ≤9-page documents. Glob both forms.
