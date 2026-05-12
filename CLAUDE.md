# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`paper-reading/` is a **static HTML blog of academic paper analyses**, generated and maintained by the `reading-papers` skill that lives at `.claude/skills/reading-papers/` (project-local skill — moved out of `~/.claude/skills/` so it stays versioned with the workspace). It is content + presentation, not application code — no build step, no package manager, no tests.

The skill is the source of truth for the structure. When something feels prescriptive (layout, IDs, file names), check `.claude/skills/reading-papers/SKILL.md` before improvising.

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

The slug is either the arxiv ID (`2604.24763`) or `firstauthor-shorttitle-year`. The shared `assets/style.css` and the top-level `index.html` (the homepage card grid) belong to the workspace, not any one paper.

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
- `git status` shows files outside `papers/<slug>/`, `index.html`, or `assets/` (unexpected scope — verify before pushing).

### Ask before pushing

By default, **ask the user** before running `publish.sh` for any given paper, unless they've already said "and publish" / "发布到 gh page" in the same turn or earlier in the session. Pushing is visible to others and the commit history is permanent.

## Editing existing pages

- **Section title changes**: update both the `<h2>`/`<h3>` text and the matching `<li>` text in the TOC. The `id` should usually NOT change (would break external links).
- **Section reorder**: re-run the TOC count sanity check. Keep IDs in `sec-N-M` numerical order matching the visible numbering.
- **Style changes**: edit `assets/style.css` (single source of truth — every paper inherits). If the change is paper-specific, that's a smell.

## Updating the skill itself

The skill lives at `.claude/skills/reading-papers/` inside this project (versioned alongside the blog). Changes that should propagate to future paper analyses (new section requirements, new CSS, new verification checks) belong in the skill, not in any one paper's HTML. Pattern:

1. Update `.claude/skills/reading-papers/SKILL.md` (instructions + HTML skeleton).
2. Update `.claude/skills/reading-papers/templates/style.css` if CSS changed.
3. Backfill the change into existing `papers/*/index.html` if you want consistency.

The CSS in `assets/style.css` was originally copied from the skill template. They drift if you edit only one side.

## Known gotchas seen during dev

- Headless Chrome `--screenshot --window-size=W,H` on very tall windows (>16000 px) silently captures only the bottom half. For full-page screenshots use `agent-browser screenshot --full` with a clean session, and crop afterwards if needed.
- The local proxy at `http://127.0.0.1:7897` (Clash) intercepts even `localhost` requests for `curl` — use `--noproxy '*'` for smoke tests against `./serve.sh`.
- `pdftoppm` output filename pads page numbers as `page-01.png`, `page-02.png` ... (zero-padded) when there are ≥10 pages but as `page-1.png` for ≤9-page documents. Glob both forms.
