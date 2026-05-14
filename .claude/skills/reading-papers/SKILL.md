---
name: reading-papers
description: Use when the user shares an academic paper (arxiv URL, local PDF, paper title, or DOI) or asks to read / 精读 / 讲解 / analyze / critique a research paper. Triggers include "读这篇论文", "帮我精读 paper", "解释 arxiv:xxxx", "summarize this paper", "critique this paper". Output is a 图文 HTML blog post — combines structured 5-section analysis, figures cropped from the PDF, and code quoted from the cloned repo. Auto-builds and serves a static blog system in the working directory.
---

# Reading Papers → 图文 Blog (with Code Cross-Read)

## Overview

Reading a paper without its code leaves blind spots. Reading without seeing the figures leaves more. This skill enforces: **locate paper + code → cross-read → extract figures → quote code → publish as a 图文 blog post → launch a local server so the user can view it**.

The output is a static HTML blog in the working directory:

```
./paper-reading/
  index.html               # Blog homepage — card per paper
  assets/
    style.css              # Shared styling (copied from this skill)
  papers/
    <slug>/
      index.html           # The 图文 analysis for one paper
      figures/             # Cropped from PDF
      raw/paper.pdf        # Original PDF
      repo/                # Cloned source repo
  serve.sh                 # Local HTTP server launcher
```

Six phases:
1. **Locate** paper + code
2. **Cross-read** paper-and-code
3. **Extract** important figures from the PDF
4. **Quote** key code excerpts
5. **Generate** the 图文 HTML page + update blog index
6. **Launch** the local server and show the URL

## Core principle

**Paper claims + code reality + visible figures = real understanding.** Every non-trivial assertion must trace to `[§paper.section]`, `repo/file.py:Lnn`, or `figures/figN.png`. Speculation about "what the authors meant" is not allowed.

If no code is found after exhausting all 6 discovery strategies, say so explicitly on the page; do not fabricate.

## When to use

- User shares an arxiv URL / PDF / paper title and asks to read, analyze, critique, summarize, or explain it
- User wants implementation-faithful, illustrated understanding — not a press-release summary
- Triggers: "读这篇论文", "帮我精读 …", "用人话讲一下 …", "解释 arxiv:xxxx", "read/critique this paper"

**Don't use for:**
- Quick one-liner about a known paper (just answer)
- Blog posts / docs without a method section
- Cross-paper literature surveys

## Output language

**Match the user's input language.** Chinese in → Chinese out. English in → English out. Math notation stays LaTeX regardless.

---

## Phase 1 — Locate paper and code

### 1a. Normalize input

Map any input to one of:
- **arxiv ID** (e.g. `2106.09685`) — preferred, gives HTML + PDF + e-print
- **Local PDF path** — read with Read tool, use `pages` parameter for chunked access
- **Title + first author** — search arxiv / Semantic Scholar first

For arxiv:
- Abstract page: `https://arxiv.org/abs/<id>` (metadata + often a code link)
- HTML version: `https://arxiv.org/html/<id>` (cleanest equation extraction)
- PDF: `https://arxiv.org/pdf/<id>` (figures)
- e-print source: `https://arxiv.org/e-print/<id>` (raw figure files, sometimes)

### 1b. Pick a slug and create the workspace

**The slug is ALWAYS `shorttitle-year` (title-based, NOT author-based).** Never use a bare arxiv ID as the folder name — arxiv IDs are opaque (you can't tell `2509.04259` from `2605.05204` at a glance), and they make the file tree unbrowsable. The slug is a human-readable handle.

Rules for the slug:
- **shorttitle**: 2–4 lowercase kebab-case words derived from the title. **Prefer the paper's nickname / method name** if there is one (e.g. `rls-razor`, `flow-opd`, `d-opsd`, `lora`) — that's what people will search for. Otherwise extract from the first content words, dropping articles and stopwords.
- **year**: 4-digit year from the arxiv submission date (not necessarily today's year). Year is mandatory — it disambiguates same-named methods proposed in different years.
- **Do NOT prefix with the first author's name.** Method names are the durable identifier; author names rot when fields move on.

Examples:
- "RL's Razor: Why Online RL Forgets Less" (Shenfeld, 2025) → `rls-razor-2025`
- "Flow-OPD: On-Policy Distillation for Flow Matching" (Fang, 2026) → `flow-opd-2026`
- "D-OPSD: On-Policy Self-Distillation..." (Jiang, 2026) → `d-opsd-2026`
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu, 2021) → `lora-2021`

```bash
SLUG=<shorttitle-year>   # e.g., rls-razor-2025
ROOT=./paper-reading
mkdir -p "$ROOT/assets" "$ROOT/papers/$SLUG/figures" "$ROOT/papers/$SLUG/raw"
```

Download the PDF:
```bash
curl -L -o "$ROOT/papers/$SLUG/raw/paper.pdf" "https://arxiv.org/pdf/<id>"
```

### 1c. Find the code (REQUIRED — try in order until found)

1. **arxiv abstract page** — `WebFetch https://arxiv.org/abs/<id>` and grep the response for `github.com`
2. **Papers with Code** — `https://paperswithcode.com/search?q=<title>` (page often links the official repo)
3. **The paper itself** — grep the PDF/HTML body for `github.com/`, `gitlab.com/`, project pages
4. **GitHub search** — `https://github.com/search?q=<paper+title>&type=repositories`, prefer the official org / first-author
5. **Author website / lab page** — first author + affiliation
6. **Ask the user** for the repo URL if all the above fail

Once found:
```bash
git clone --depth 1 <repo-url> "$ROOT/papers/$SLUG/repo/"
```

If genuinely no code exists, mark this on the page in §4 and continue text-only.

---

## Phase 2 — Cross-read paper and code

Read the paper top-to-bottom **with the cloned repo open**. For every major equation or algorithm, locate its implementation.

| Looking for | Where |
|---|---|
| Method implementation | `grep -rni "<concept>" repo/`; `model.py`, `models/`, `*_model.py` |
| Training loop | `train.py`, `trainer.py`, `engine.py`, `main.py` |
| Loss / objective | `grep -rni "loss\|criterion" repo/` |
| Hyperparameters | `config.yaml`, `configs/`, `args.py`, `*.json` |
| Data pipeline | `dataset.py`, `data/`, `dataloader.py` |

For each key equation: write LaTeX → find implementing function → note tensor shapes → **flag discrepancies**. Discrepancies are gold for the critical-summary section.

Skim experiments. Read limitations and appendix carefully — usually the most honest part.

---

## Phase 3 — Extract important figures from the PDF

Pick **3–8 important figures**: architecture diagrams, headline results plots, key ablation tables, illustrative qualitative results. Skip decorative figures.

### Strategy A — Embedded images (fastest, usually works for arxiv)

```bash
cd "$ROOT/papers/$SLUG/raw"
mkdir -p ../figures-raw
pdfimages -all -p paper.pdf ../figures-raw/img
# Produces img-<page>-<idx>.png/.jpg; inspect and pick
```

### Strategy B — Render pages and crop (for vector figures)

Requires `poppler` (macOS: `brew install poppler`):
```bash
pdftoppm -png -r 220 paper.pdf ../figures-raw/page    # page-1.png ... page-N.png
```
Then crop with `sips` (macOS built-in) or `magick convert`:
```bash
# sips: --cropToHeightWidth <H> <W> --cropOffset <Y> <X>
sips --cropToHeightWidth 900 1500 --cropOffset 300 200 \
  ../figures-raw/page-3.png --out ../figures/fig2-architecture.png
```

### Strategy C — arxiv e-print tarball (cleanest, when available)

```bash
curl -L -o "$ROOT/papers/$SLUG/raw/eprint.tar.gz" "https://arxiv.org/e-print/<id>"
tar -xzf "$ROOT/papers/$SLUG/raw/eprint.tar.gz" -C "$ROOT/papers/$SLUG/raw/eprint/"
# Look for raw .png / .pdf figure files
```

### Naming & captioning

- Save as `papers/$SLUG/figures/figN-<short-name>.png` (e.g. `fig1-architecture.png`, `tab3-ablation.png`)
- Write a one-line caption for each; you'll use it in `<figcaption>`

### Required (when present in the paper)

- Method / architecture overview diagram (usually Fig. 1 or 2)
- Headline results table or comparison plot
- One ablation table
- One illustrative qualitative result, if applicable

---

## Phase 4 — Quote key code

Pick **3–8 short code excerpts** that implement the paper's core method. For each:

1. Read the file in `repo/`
2. Trim to the load-bearing 10–40 lines
3. Embed inline in the HTML inside `<pre><code class="language-python">…</code></pre>`
4. Above the block, write a `code-source` tag with the exact path + line range, e.g. `repo/model/attention.py:L42–L78`
5. Below or beside, write 1–2 sentences linking the snippet to a paper equation/algorithm

**Required quotes when applicable:**
- Core method (the main equation / algorithm)
- Loss function
- The clever / paper-specific bit (the thing that wouldn't appear in a generic implementation)

Skip boilerplate (dataset loaders, vanilla training loop) unless it hides paper-specific subtlety.

---

## Phase 5 — Generate the 图文 HTML page and update the index

### 5a. Copy assets (once per workspace)

```bash
cp ~/.claude/skills/reading-papers/templates/style.css "$ROOT/assets/style.css"
cp ~/.claude/skills/reading-papers/templates/serve.sh "$ROOT/serve.sh"
chmod +x "$ROOT/serve.sh"
```

### 5b. Write `papers/$SLUG/index.html`

Use this exact skeleton. Fill the five `<section>`s with the analysis content. Translate section labels to the user's language. **Replace all `[[…]]` placeholders.**


```html
<!doctype html>
<html lang="[[lang-code, e.g., zh-CN or en]]">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>[[Paper Title]] — Reading Notes</title>
  <link rel="stylesheet" href="../../assets/style.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/atom-one-dark.min.css">
  <script>
    window.MathJax = { tex: { inlineMath: [['$','$'],['\\(','\\)']] } };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
  <script>document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());</script>
  <script defer src="../../assets/lightbox.js"></script>
  <script>
    // TOC active-section highlighting via IntersectionObserver
    document.addEventListener('DOMContentLoaded', () => {
      const links = document.querySelectorAll('.toc a');
      if (!links.length) return;
      const map = new Map();
      links.forEach(a => {
        const el = document.getElementById(a.getAttribute('href').slice(1));
        if (el) map.set(el, a);
      });
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          const a = map.get(e.target);
          if (!a) return;
          if (e.isIntersecting) {
            links.forEach(l => l.classList.remove('active'));
            a.classList.add('active');
            const toc = document.querySelector('.toc');
            if (toc) toc.scrollTo({ top: a.offsetTop - toc.clientHeight / 2, behavior: 'smooth' });
          }
        });
      }, { rootMargin: '-20% 0px -70% 0px' });
      map.forEach((_, el) => observer.observe(el));
    });
  </script>
</head>
<body>

<!-- Floating TOC — REQUIRED. List every <h2> and <h3> in document order.
     Each href must match the id on the corresponding heading. -->
<nav class="toc" aria-label="目录">
  <div class="toc-title">目录 / TOC</div>
  <ul>
    <li class="h2"><a href="#sec-1">1. [[出发点]]</a></li>
    <li class="h2"><a href="#sec-2">2. [[方法]]</a></li>
    <li class="h3"><a href="#sec-2-1">2.1 [[小节]]</a></li>
    <!-- … 每个 <h3> 一行, 用 li.h3 缩进 … -->
    <li class="h2"><a href="#sec-3">3. [[结论]]</a></li>
    <li class="h2"><a href="#sec-4">4. [[实现细节]]</a></li>
    <li class="h2"><a href="#sec-5">5. [[批判性总结]]</a></li>
  </ul>
</nav>

<main>
  <p><a href="../../index.html">← 返回博客首页</a></p>

  <h1>[[Paper Title]]</h1>

  <div class="meta">
    <div><strong>作者 / Authors:</strong> [[…]]</div>
    <div><strong>Venue / Year:</strong> [[…]]</div>
    <div><strong>arXiv / DOI:</strong> <a href="[[link]]">[[id]]</a></div>
    <div><strong>Code:</strong> <a href="[[repo-url]]">[[repo-url]]</a> (commit <code>[[sha]]</code>)</div>
    <div><strong>TL;DR:</strong> [[一句话, &lt;30字]]</div>
  </div>

  <section>
    <h2 id="sec-1">1. 出发点 (Motivation)</h2>
    [[要解决什么具体问题 · 之前的方案不足 · 一句话 TL;DR]]
  </section>

  <section>
    <h2 id="sec-2">2. 方法 (Method) — 高中生友好 + 数学严谨</h2>

    <h3 id="sec-2-intro">核心思想 (类比)</h3>
    [[日常类比 / 比喻 · 让没读过本领域 paper 的人 get 到直觉]]

    <figure>
      <img src="figures/fig1-architecture.png" alt="架构图">
      <figcaption>Fig. 1 — [[caption]]</figcaption>
    </figure>

    <h3 id="sec-2-1">关键数学</h3>
    [[每个核心公式: 完整 LaTeX → 逐符号解释 → 直觉 → (推荐)小数值例子]]

    <p>例如, 注意力机制:</p>
    $$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
    <ul>
      <li>$Q, K, V \in \mathbb{R}^{n \times d_k}$ …</li>
      <li>$QK^\top$ 计算每个 query 跟每个 key 的相似度 …</li>
      <li>$\sqrt{d_k}$ 防止 softmax 饱和 …</li>
    </ul>

    <h3 id="sec-2-2">算法流程</h3>
    [[伪代码 / 步骤, ≤10 步]]

    <h3 id="sec-2-3">与代码对照</h3>
    <p class="code-source">repo/model/attention.py:L42–L78 — 实现上面的 Eq. (3)</p>
    <pre><code class="language-python">[[trimmed 10–40 行]]</code></pre>
    <p>[[1–2 句话: 这个函数对应论文里的哪部分, 有没有 paper 没提的细节]]</p>
  </section>

  <section>
    <h2 id="sec-3">3. 结论 (Key Findings)</h2>
    [[主要实验结果 · 引用具体数字 · 论文 claim 的贡献 · 在什么场景下最有效]]
    <figure>
      <img src="figures/tab3-ablation.png" alt="主要结果表">
      <figcaption>Tab. 3 — [[caption]]</figcaption>
    </figure>
  </section>

  <section>
    <h2 id="sec-4">4. 实现细节 (Implementation Notes)</h2>
    <p>代码里关键、但论文未充分说明的细节 (至少 5 条):</p>
    <ul>
      <li><strong>超参数:</strong> [[lr, batch size, schedule, …]] <code>repo/configs/default.yaml:L…</code></li>
      <li><strong>初始化:</strong> [[…]] <code>repo/…:L…</code></li>
      <li><strong>数值稳定性:</strong> [[…]] <code>repo/…:L…</code></li>
      <li><strong>训练 trick:</strong> [[grad clip / EMA / warmup / …]] <code>repo/…:L…</code></li>
      <li><strong>推理 vs 训练差异:</strong> [[…]] <code>repo/…:L…</code></li>
      <li><strong>代码与论文不一致的地方:</strong> [[!重点!]]</li>
    </ul>
    <pre><code class="language-python">[[关键 snippet, 比如 loss 函数]]</code></pre>
  </section>

  <section>
    <h2 id="sec-5">5. 批判性总结 (Critical Assessment)</h2>
    <h3 id="sec-5-1">优点</h3>
    <ul><li>[[具体, 不要 "novel" / "elegant" 这种空话]]</li></ul>
    <h3 id="sec-5-2">不足 / 疑点</h3>
    <ul>
      <li>[[实验是否充分? baseline 公平? ablation 完整?]]</li>
      <li>[[claim 是否被证据支持? cherry-picking?]]</li>
      <li>[[计算 / 内存复杂度? 实际可用性?]]</li>
      <li>[[代码 vs 论文的 gap 暗示了什么?]]</li>
    </ul>
    <h3 id="sec-5-3">适用 vs 不适用</h3>
    <ul>
      <li>✅ 适用: …</li>
      <li>❌ 不适用 / 已有更好方案: …</li>
    </ul>
    <h3 id="sec-5-4">进一步阅读</h3>
    <ul><li>[[后续工作 / 对照工作 / 综述]]</li></ul>
  </section>

  <footer>
    <hr>
    <p class="meta">Read on [[YYYY-MM-DD]] · Generated with the reading-papers skill</p>
  </footer>
</main>
</body>
</html>
```

### 5b.1 TOC sidebar (REQUIRED — non-negotiable)

Every paper page **must** have a floating right-side TOC. The skeleton above already includes the `<nav class="toc">` block and the `IntersectionObserver` highlight script — fill them in by following these rules:

1. **Give every `<h2>` and `<h3>` an `id`.** Use the scheme:
   - `<h2 id="sec-N">` where `N` is the top-level section number (1–5)
   - `<h3 id="sec-N-M">` where `N.M` is the subsection number (e.g. `sec-2-4`)
   - If a section starts with an unnumbered `<h3>` (e.g. "核心思想"), use `sec-N-intro`
   - `<h4>` headings do **not** get IDs and are not in the TOC — they're too granular
2. **The TOC must list every `<h2>` and `<h3>` in document order**, with `<li class="h2">` for top-level and `<li class="h3">` for subsections (the CSS uses these classes to indent / weight).
3. **Translate the TOC link labels to the user's language.** Truncate long subsection titles to ≤16 chars so the sidebar doesn't wrap awkwardly (e.g. "2.7 推理流程 (含 CFG 双引导)" → "2.7 推理流程 (CFG)").
4. **The `IntersectionObserver` script is already in `<head>` of the skeleton** — do not modify it. It handles active-section highlighting, auto-scrolls the TOC to keep the current item visible, and smooth-scrolls when a link is clicked.
5. The TOC is hidden on viewports ≤ 1200px (handled by CSS) — that's fine, don't try to make it always visible.

**Smoke check after generation:** count of `<li class="h2">` plus `<li class="h3">` in the TOC should equal the count of `<h2 id="sec-…">` plus `<h3 id="sec-…">` in the body. If they differ, an anchor will dangle.

### 5c. Update `index.html` (blog homepage)

If `./paper-reading/index.html` doesn't exist, create it from the template at `~/.claude/skills/reading-papers/templates/index.html`.

Then **prepend** a new `<a class="paper-card">…</a>` for the current paper inside the `<div class="paper-grid">` block. The card looks like:

```html
<a class="paper-card" href="papers/<SLUG>/index.html">
  <div class="date">[[YYYY-MM-DD]]</div>
  <h3>[[Paper Title]]</h3>
  <p class="tldr">[[一句话 TL;DR, ≤60 字]]</p>
</a>
```

Cards are ordered newest first.

---

## Phase 6 — Launch and show URL

Start the local server in the background and open the browser:

```bash
cd "$ROOT"
# Start in background; pick a free port (default 8765)
./serve.sh > /tmp/paper-reading-server.log 2>&1 &
sleep 1
open "http://localhost:8765/papers/$SLUG/index.html"   # macOS
```

In the chat, give the user:
- The local URL of the new paper page
- The local URL of the blog homepage
- The on-disk path of the analysis

Then deliver a **terse 5-bullet summary** of the paper in chat (one bullet per section).

> **Project-specific publish step (if any) lives in the project's `CLAUDE.md`, not in this skill.** The skill is generic across workspaces; per-workspace deploy targets (e.g., a GitHub Pages repo) are project rules.

---

## Verification before delivery

- [ ] §2 has at least one LaTeX equation rendered via MathJax (unless paper genuinely has no method)
- [ ] §2 has at least one analogy / intuition paragraph per key equation
- [ ] §2 has at least one paper↔code mapping with `repo/…:Lnn` citation
- [ ] §3 cites **specific numbers**, not "improves performance"
- [ ] §4 has **≥5** specific implementation details, each with `repo/…:Lnn`
- [ ] §4 explicitly flags any paper-vs-code discrepancy (or states "none found")
- [ ] §5 has concrete weaknesses, not just "future work"
- [ ] HTML has **3–8 figures**, each in `<figure>` with `<figcaption>`
- [ ] HTML has **3–8 code blocks**, each preceded by `<p class="code-source">` citing the file:Lnn
- [ ] Blog `index.html` updated with the new card (newest first)
- [ ] **TOC sidebar present**: `<nav class="toc">` exists; every `<h2>` and `<h3>` has an `id="sec-…"`; TOC link count = body heading count (no dangling anchors)
- [ ] `serve.sh` is executable and the server actually started (`curl -sI http://localhost:8765/ | head -1`)
- [ ] Output language matches user input language

If any check fails — **go back and fix it** before reporting done.

---

## The "高中生能听懂" test

> 一个学过基础微积分和线代的高三学生, 读完 §2 能用自己的话说出"这个公式在做什么" — 不要求他推导, 但要能复述意图。

If they can't, add: (a) an analogy, (b) symbol-by-symbol breakdown, (c) a small concrete numeric example. **Simplify the explanation, not the formula.** Keep the math, replace the jargon.

---

## Rationalizations to refuse

| 借口 | 反驳 |
|---|---|
| "公式太复杂, 跳过/手挥一下" | 跳过 = 偷懒. 拆符号 + 找类比 + 数值例子. 实在没懂, 明说"这里 X 我没完全 follow". |
| "找不到代码, 直接写文字版" | 6 种策略全跑完才算"找不到". 跑完仍没有, 在页面顶部明确标注. |
| "代码看了, 不交叉引用太累" | 至少 3 个核心公式有 `file:Lnn` 引用. |
| "PDF 图费劲, 用文字描述就行" | 文字描述抵不上一张架构图. 至少抠出架构图 + 一张结果表. |
| "代码 snippet 太长, 用 prose 概括" | 直接引 10–40 行原代码. Prose 概括会丢细节. |
| "论文挺牛的, 跟着吹就行" | §5 必须有具体不足. 没有 = 没认真读. |
| "blog 系统太麻烦, 输出 markdown 算了" | 不行. 用户明确要求图文 HTML + 博客 + 启动. 不可降级. |
| "服务起不起来无所谓, 用户能打开文件就行" | 起服务. file:// 会破坏相对路径和某些 CDN 加载. |
| "TOC 太麻烦, 这次省了吧" | 不行. 文章 ≥ 3000 字时没有 TOC 等于在惩罚读者. h2/h3 加 id + 一个 `<nav class="toc">` 不到 50 行, 没理由跳过. |
| "用 JS 现场扫 heading 生成 TOC 算了" | 静态写出来. 现场扫的方案在 SSR / 截图工具 (无头浏览器) 里可能没跑就被截了, 不可靠. |

## Red flags — start over if ANY apply

- 输出是纯 markdown 而非 HTML
- §2 没有 MathJax 公式
- 没有从 PDF 抠图 (figures/ 是空的)
- 没有 file:Lnn 引用的代码块
- §5 只有 "future work" 没有 "weaknesses"
- 没有更新 blog 首页 index.html
- **没有 TOC sidebar**, 或 TOC 链接数 ≠ 正文 h2/h3 数 (dangling anchor)
- h2/h3 缺 `id="sec-…"` 属性
- 没有启动 serve.sh / 没有给出本地 URL
- 输出语言不匹配用户输入语言

---

## Tooling notes

- **PDF reading**: use Read tool with `pages` parameter for large PDFs. Read abstract+intro first, then method, then experiments+limitations.
- **PDF tooling**: `pdfimages`, `pdftoppm`, `pdftotext` from `poppler` (macOS: `brew install poppler`). `sips` is built-in on macOS for cropping. `magick` (ImageMagick) is a more capable alternative.
- **arxiv HTML**: `https://arxiv.org/html/<id>` — cleaner than PDF for extracting equation LaTeX.
- **WebFetch**: prefer the abstract page (lightweight) over the PDF when just hunting for code links.
- **Code search**: `grep -rni` for content; `find` for filenames.
- **Cleanup**: leave `./paper-reading/` in place — the user will return to it. Don't auto-delete the cloned repo.
- **Visual polish**: if the user wants a more bespoke design, invoke the `frontend-design` skill afterwards to redesign `assets/style.css`.

## Related skills

- `creating-posters-from-paper-and-code` — sibling, same cross-read DNA, output is a single PNG poster instead of a blog. Use only if user explicitly asks for a poster.
- `frontend-design` — invoke after the blog exists if the user wants a more distinctive visual design.
