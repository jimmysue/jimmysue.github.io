# `writing-tutorial` skill — design spec

**Date:** 2026-05-13
**Author:** brainstormed with user (`/superpowers:writing-skills`)
**Status:** Approved design — awaiting RED-phase baseline tests before SKILL.md drafted

---

## 1. Purpose

A project-local skill that ingests **a single topic name** ("LoRA", "扩散模型", "Flow Matching", "PPO for LLMs") and produces a single-page deep tutorial in `tutorials/<slug>/index.html` — designed to be **deep yet approachable**: a high-schooler can follow the intuition path, a researcher can verify the math, an engineer can copy the cited code straight into a working implementation.

The skill complements the existing `reading-papers` skill:
- `reading-papers`: one paper → one analysis page (5-section critique).
- `writing-tutorial`: one **domain** → one synthesized tutorial drawing from ≥2 papers + ≥2 GitHub repos + optional existing tutorials.

## 2. Non-goals

- Cross-paper literature surveys (the "survey paper" output is different).
- Multi-page book/chapter sites (out of scope — single-page only; if a topic doesn't fit, escalate back to design).
- Generating tutorials with no GitHub-cited code (one of the skill's invariants).

## 3. User-facing interface

```
User: /writing-tutorial 扩散模型
User: 帮我写一个 LoRA 的深度教程
User: write a tutorial on flow matching
```

**Output language matches input language** (Chinese in → Chinese out; English in → English out; math LaTeX always).

## 4. Output artifacts

```
tutorials/<slug>/
  index.html              # the tutorial (the deliverable)
  plan.md                 # agreed outline — Phase-3 gate
  sources/
    MANIFEST.md           # rationale for every cited paper/repo/blog
    papers/*.pdf          # discovered anchor papers
    repos/<org>-<name>/   # cloned reference implementations (gitignored)
  figures/                # final cropped/drawn diagrams used in index.html
  figures-raw/            # intermediate pdfimages/pdftoppm output (gitignored)
  code-excerpts/          # pre-staged code snippets w/ file:Lstart-Lend header
                          #   secN-step2-demo.txt, secN-step4-<n>.txt, ...
```

Slug rule: `<shorttitle>-<year>` (same as `reading-papers`), e.g. `lora-2021`, `diffusion-2020`, `flow-matching-2023`. Year is the seminal paper's year, not today.

Top-level integration:
- New `tutorials/` directory parallel to `papers/`.
- Top-level `index.html` gets a 教程 section **above** the existing 论文 grid (separate card style — to be added to `assets/style.css`).

## 5. Pedagogical contract — the spiral structure

Every `<h2>` 大节 has exactly five `<h3>` sub-steps in order:

| Step | ID | Purpose | Length budget | Hard rules |
|------|----|---------|---------------|------------|
| 1. 直觉 | `sec-N-1` | Everyday analogy, "高中生听得懂" framing, ≥1 figure | 100–200 字 | Zero undefined jargon; first-use terms defined inline |
| 2. 最小 demo | `sec-N-2` | 10–30 lines toy code exhibiting the concept | small | Tagged `class="teaching-demo"`; clearly labeled "教学示例 — 非生产代码" |
| 3. 正式化 | `sec-N-3` | The math; MathJax `$$...$$` | as needed | `\lt`/`\gt`, never literal `<`/`>`; every non-trivial step followed by a 1-sentence "翻译" back to Step-1 intuition; every symbol defined on first use |
| 4. 代码引用 | `sec-N-4` | 10–40 lines from a real repo, with line-by-line mapping back to Step-3 equations | medium | `<p class="code-source">sources/repos/<org>-<repo>/<file>:Lstart-Lend — <one-line role></p>` above; "对照" paragraph below |
| 5. 洞察 | `sec-N-5` | "为什么是这样而不是那样" — design choices, failure modes, when not to use | 1–3 bullets, ≤2 sentences each | This is the section that distinguishes a tutorial from a regurgitated paper |

A tutorial has **6–10 大节**, target total **10–20k 中文字** (or equivalent English).

## 6. Workflow — 6 phases

### Phase 1 — Scope (main agent)
- Parse topic; construct slug.
- Create `tutorials/<slug>/` workspace.

### Phase 2 — Discover (parallel subagents, fan-out)

Dispatch in one message:
- **arxiv-searcher** (Explore agent): find 2–4 anchor papers + survey papers
- **github-searcher** (Explore agent): find 2–4 high-quality repos (star filter ≥1k OR official org OR first-author academic implementation)
- **blog-searcher** (Explore agent): find 1–2 existing high-quality tutorials/blog posts as references

Each returns a ranked list with rationale. Main agent picks the final manifest and writes `sources/MANIFEST.md`:

```
# MANIFEST — <topic>

## Papers
- [Anchor] DDPM (Ho et al., 2020) — defines forward/reverse process. arxiv:2006.11239
- [Survey] Diffusion Models: A Comprehensive Survey ...

## Repos (primary first)
- [PRIMARY] huggingface/diffusers — 23k★, 官方维护 — code spine for Steps 4+
- [Comparison] lucidrains/denoising-diffusion-pytorch — 7k★ — minimal reference

## Blogs (optional)
- Lilian Weng — "What are Diffusion Models?" — concept map reference
```

**Weak discovery handling:** if primary repo has <1k★ and no academic alternative found, OR <2 anchor papers exist, proceed but emit `<aside class="provenance-warning">` in the page header explaining the gap.

### Phase 3 — Plan (main agent, **NO subagent**) — **USER GATE**

Main agent reads MANIFEST + skims primary paper(s) and writes `tutorials/<slug>/plan.md`:

```markdown
# Plan — <topic>

## Outline
1. <Section title> — <1-sentence key takeaway>
   - Step 1 (直觉): <approach + figure source>
   - Step 2 (demo): <toy code idea>
   - Step 3 (math): <equations to derive>
   - Step 4 (code): sources/repos/<repo>/<file>:Lxx-Lyy
   - Step 5 (洞察): <key insight>
   - target: ~1500 字
2. ...

## Cross-section dependencies
- §3 builds on notation introduced in §2
- §5 callbacks the demo from §1

## Drafting groups (for grouped-parallel Phase 5)
- Group A (parallel): §1, §2  — foundation
- Group B (parallel): §3, §4  — depends on A
- Group C (parallel): §5, §6, §7  — depends on B
- Group D (serial): §8 (synthesis) — depends on all
```

**STOP.** Print plan to user and ask: "Outline 是否合适？要调整章节、增删、改顺序请说。" Resume only after user confirms or revises. The plan is the single source of truth for all later phases.

### Phase 4 — Extract (parallel subagents, fan-out per section)

For each section in the plan, dispatch one Explore subagent:
- Crop figures it'll need into `figures/secN-*.png`
- Read the primary repo, extract the cited line range into `code-excerpts/secN-step4-<n>.txt` with the citation header on line 1
- Note in MANIFEST any figures that need to be *drawn fresh* (not croppable from any paper)

After Phase 4, `figures/` and `code-excerpts/` are fully populated. Drafters in Phase 5 read these instead of re-reading the source repos.

### Phase 5 — Draft (grouped parallel, general-purpose subagents)

Groups are defined in `plan.md`. Within a group, all sections drafted in parallel. Groups serial — later groups see prior groups' finished HTML.

Each drafter subagent receives:
- `plan.md` (full — for context)
- The brief for ITS section
- 1–2 sentence summaries of all previously drafted sections (within this run)
- 1-line outlines of all FOLLOWING sections (to avoid duplication)
- Paths (not contents) of pre-staged figures and code excerpts
- The spiral contract template (`templates/spiral-section.html`)
- For groups B+: full HTML of previously drafted groups (read-only)

Drafter returns exactly one `<section id="sec-N">...</section>` block conforming to the spiral contract.

### Phase 5.5 — Assemble (main agent, small context)

Main agent:
- Concatenates section blocks into the skeleton (`templates/skeleton.html`).
- Builds TOC last, after all IDs are stable.
- Runs cross-section coherence checks:
  - Terminology consistency (grep first-uses; if §3 introduces "score function" and §5 uses "梯度" interchangeably without bridge → fix)
  - Forward references resolved (§2 says "we'll see in §4" → §4 must exist)
  - Notation drift (`x_t` means the same thing throughout)

### Phase 6 — Verification + Publish

Run the **verification checklist** (next section). Any failure → fix. Then:
- Restart `./serve.sh` if not running
- Print local URL
- Ask user before running `./publish.sh` (per CLAUDE.md project rule)

## 7. Verification checklist

Run before publishing. Fail any → fix before claiming done.

```
[ ] Source coverage
    • ≥2 anchor papers cited in references section
    • ≥2 GitHub repos cited; primary is >1k★ OR official OR academic
    • Provenance warning aside present iff weak discovery

[ ] Spiral integrity
    • Every <h2 id="sec-N"> has 5 <h3 id="sec-N-1..5"> in order
    • Every Step-4 code block has <p class="code-source"> directly above
    • Every Step-2 demo block has class="teaching-demo"
    • Every equation block has a 翻译 sentence within 100 字 after

[ ] TOC sanity
    • Count(toc <a>) == Count(h2[id^=sec-], h3[id^=sec-])
    • Order matches document order

[ ] Code provenance audit
    • For each Step-4 snippet: cited file exists in sources/repos/...
    • Line range read at that path matches the snippet verbatim
    • Step-2 demos NOT cited as repo code

[ ] Accessibility (高中生层)
    • Every Step-1 paragraph free of undefined jargon (grep + scan)
    • Every <h2> section has ≥1 figure (paper crop or drawn)

[ ] Math rendering
    • grep -E '\$[^$]*<[^$]*\$' returns empty (literal < in math = bug)
    • Page loads w/o MathJax "TeX parse error" in browser console

[ ] Length budget
    • Word count within 10–20k 字 (or English equivalent)
    • Under → likely shallow, flag; over → escalate

[ ] Cross-section coherence
    • Forward references resolved
    • Terminology stable
    • Notation stable
```

## 8. File layout (skill itself)

```
.claude/skills/writing-tutorial/
  SKILL.md                          # the instructions
  templates/
    skeleton.html                   # HTML scaffold (head, MathJax, TOC stub, footer)
    spiral-section.html             # the 5-sub-step template
    style-additions.css             # CSS for .teaching-demo, .provenance-warning,
                                    #   .tutorial-card on the homepage
  references/
    spiral-contract-examples.md     # 2–3 worked examples of well-formed sections
```

## 9. Skill testing strategy (TDD per `superpowers:writing-skills`)

### RED — baseline (before writing SKILL.md)
Dispatch subagents with no skill loaded, recording verbatim what they produce on prompts like:
1. "写一篇讲透 LoRA 的教程，要让高中生看懂" (accessibility pressure)
2. "写一篇 LoRA 的深度教程，包含完整推导" (depth pressure)
3. "我赶时间，写个 LoRA 教程" (time pressure)

Expected failure modes (hypotheses):
- Code blocks paraphrased / hand-written; no GitHub citation
- Math skipped OR mid-derivation symbol drops
- Linear (math → code) rather than spiral
- No "翻译" sentences connecting math to intuition
- Discovery skipped — answers from model knowledge

### GREEN — write SKILL.md
Address each observed failure with an explicit rule. Match `reading-papers` voice and structure.

Re-run baseline scenarios with skill loaded; confirm compliance.

### REFACTOR — close loopholes
Watch for new rationalizations under the skill ("the user wanted short", "the repo is messy"); add red-flag entries and explicit counters until bulletproof.

## 10. Integration with existing project

Files to add/modify:
- **Add** `.claude/skills/writing-tutorial/SKILL.md` + `templates/` + `references/`
- **Add** `tutorials/` (the output directory — populated as tutorials are produced)
- **Modify** `index.html` (top-level) — add 教程 section above 论文 grid
- **Modify** `assets/style.css` — add `.tutorial-card`, `.teaching-demo`, `.provenance-warning`, `.spiral-step-*` classes
- **Modify** `CLAUDE.md` — add "Per-tutorial layout" + "Adding a new tutorial" sections paralleling the existing paper sections
- **Modify** `.gitignore` — exclude `tutorials/*/sources/repos/` and `tutorials/*/figures-raw/`

## 11. Open questions deferred to implementation

- Exact CSS for the spiral-step visual differentiation (decide during SKILL.md drafting; might use subtle left-border color per step)
- Whether to add a top-level `tutorials/index.html` page or just a section in the existing homepage (currently picked: homepage section; revisit if cards grow past ~10)
- How to gitignore behavior for `tutorials/<slug>/sources/papers/*.pdf` (currently: keep — small files, useful for provenance)
