# Markdown-Source Blog — Phase 1 Design

**Date:** 2026-05-29
**Status:** Draft (awaiting user review)
**Phase:** 1 of 2 (基础设施); Phase 2 (知识图谱) 单独 spec

---

## 1. 动机与目标

当前 `papers/<slug>/index.html` 与 `tutorials/<slug>/index.html` 是 skill 直接吐出的手写 HTML,作为 AI/RAG 索引源不友好——HTML 标签噪音多,语义结构需要从富 HTML 里反推。

**最终目标 (Phase 2 终态):** 把所有论文/教程笔记变成结构化、可被 Claude/agent 索引的知识库,支持"我读过哪些 paper 提到 flow matching"这类查询。

**Phase 1 目标 (本 spec):** 把"源是 markdown,产出是 HTML"这条管线跑通。先不碰知识图谱本身;只奠定可读、可解析的 markdown 源 + 渲染管线。

Phase 1 完成后整个 blog 仍正常工作 (浏览/发布无差别);Phase 2 在此基础上加 citations/concepts metadata 抽取与图谱可视化。

## 2. Scope

### In-scope (Phase 1)

- 定义 markdown 源文件 + frontmatter schema
- 扩展 `build.py` 把 markdown 渲染为 HTML (替代手写)
- 改造 `reading-papers` 与 `writing-tutorial` 两个 skill,输出 markdown
- 一次性把现有 22 篇 paper + 3 个 tutorial 的 `index.html` + `meta.json` 迁移成 `index.md` (含 frontmatter)
- 验证 + commit

### Out-of-scope (Phase 2 处理)

- frontmatter 的 `citations` / `concepts` / `repos` 字段填充
- 概念页 `concepts/<slug>.html` 自动生成
- 知识图谱 JSON 抽取
- 交互式 `/graph.html` 可视化页
- MCP server / 查询 API

### 不做的事 (明确排除)

- 不切换静态站生成器 (不引入 MkDocs/Hugo/Astro)
- 不改样式 (`assets/style.css` 保持不变,新管线产物在视觉上应与现状一致)
- 不改部署流程 (`./publish.sh` 沿用)
- 不保留手写 HTML 修改 index.html 的能力 (MD 是唯一入口)

## 3. 目录结构

```
papers/<slug>/
  index.md            # ← 源 (新)
  index.html          # ← build.py 产物,入库
  figures/            # 不变
  figures-raw/        # 不变 (gitignored)
  raw/                # 不变 (gitignored)
  repo/               # 不变 (gitignored)
  # meta.json         ← 删除,内容并入 index.md frontmatter

tutorials/<slug>/
  index.md
  index.html          # 产物
  plan.md             # 不变 (本就是 markdown)
  sources/            # 不变
  figures/, ...

assets/               # 不变
build.py              # 扩展
migrate.py            # 重写 (现有同名脚本是别的迁移,可覆盖或更名 migrate-md.py)
requirements.txt      # 新增
```

**HTML 与 MD 并存入库**:`publish.sh` 不需改动,GitHub Pages 直接吃 `papers/<slug>/index.html`。代价是每次重生 git diff 会动 HTML,但保证部署可复现且发布流程零改动。

## 4. Frontmatter Schema (Phase 1)

YAML frontmatter 在每个 `index.md` 文件顶部:

```yaml
---
type: paper                      # "paper" | "tutorial"
slug: l2p-2026                   # 必须等于父目录名
title: "L2P: 把 LDM 的潜在知识搬到像素空间, 8 卡训出原生 4K 扩散"
date: 2026-05-22                 # ISO 8601 (YYYY-MM-DD)
tldr: |                          # 多行字符串 (block scalar)
  南大 + 腾讯优图 (arXiv 2026/05)。直接拆掉 VAE...
tags: [diffusion, image-gen, pixel-space, transfer-learning, flow-matching]

paper:                           # paper 专属;tutorial 不写
  arxiv_id: "2605.12013"
  authors: "Zhennan Chen, Junwei Zhu, ..."
  venue: "arXiv"
  project_page: "https://nju-pcalab.github.io/projects/L2P/"
  code_url: "https://github.com/TencentYoutuResearch/T2I-L2P"
  weights_url: "https://huggingface.co/zhen-nan/L2P"

tutorial:                        # tutorial 专属
  word_count: "10.8k"
  reading_minutes: "80-110"

# Phase 2 将添加 (本 spec 暂不要求):
# citations: [paper-slug-a, paper-slug-b]
# concepts: [flow-matching, lora]
# repos: ["github.com/foo/bar"]
---
```

**必需字段**:`type`, `slug`, `title`, `date`, `tldr`, `tags`
**可选字段**:`paper.*`, `tutorial.*`

校验:
- `slug` 与父目录名不符 → build warning + skip
- 缺必需字段 → build warning + skip
- `type` 不在 `{paper, tutorial}` → build warning + skip
- `date` 非 ISO 格式 → build warning + skip

## 5. Markdown 内容约定

**核心原则**: 默认用 pure markdown;只在 markdown 表达不动时嵌入 HTML 岛。CommonMark 允许 raw HTML 透传——GitHub/Obsidian/VS Code 都原生渲染,markdown 仍"直接可读"。

### 5.1 用 pure markdown 的部分

| 内容 | 写法 |
|---|---|
| 段落、列表、标题、引用、表格 | 原生 CommonMark/GFM |
| 行内/显示数学 | `$x_t$` / `$$x_t = ...$$` (MathJax) |
| 代码块 | ` ```python ... ``` ` (highlight.js) |
| 跨论文/教程引用 | `[[slug]]` 或 `[[slug\|alias]]` (Phase 2 用于图谱边抽取) |

### 5.2 用嵌入 HTML 的部分

> **markdown 解析 gotcha**: 嵌入的块级 HTML 元素 (`<figure>`, `<p class="...">`) 必须前后留空行,否则 markdown 解析器会把它们当成段落的一部分而非独立块。模板示例都已经留好空行。

**(a) Figure + 富 caption** — 整段用 HTML 块:

```html
<figure>
  <img src="figures/fig1-teaser.png" alt="L2P 概念图: 从 latent 流形迁移到 pixel 流形">
  <figcaption>
    <strong>Fig. 1</strong> — 核心 narrative。<strong>上半</strong>:
    latent space 的流形比较平滑,从噪声到目标的轨迹"短而直"。
    <strong>下半</strong>: pixel space 流形坑坑洼洼。
  </figcaption>
</figure>
```

**(b) Math translation** — 显示公式后的"翻译"行:

```html
$$ \mathbf{x}_t = (1-\sigma)\mathbf{x}_0 + \sigma \boldsymbol{\epsilon} $$
<p class="math-translation">—— 翻译: 给定干净图, 按 $\sigma$ 比例混入噪声。</p>
```

**(c) Code citation** — 代码块前的来源标签:

````markdown
<p class="code-source">repo/diffsynth/diffusion/flow_match.py:L164-L174 — 前向 + 训练目标</p>

```python
def add_noise(self, original_samples, noise, timestep):
    ...
```
````

### 5.3 自动生成 (不写进 MD 源)

| 自动产物 | 来源 |
|---|---|
| 标题 ID (`id="sec-1"`, `id="sec-2-3"`) | build.py 扫 h2/h3 顺序;h2 自增,h3 在每个 h2 内重置 |
| 右侧悬浮 TOC | build.py 扫所有带 `id="sec-*"` 的 h2/h3 |
| Lightbox 类 | build.py 给所有 `<figure> > img` 加 `class="zoomable"` |
| MathJax / highlight.js / lightbox.js script tags | build.py 注入 `<head>` 与 `<body>` 底部 |
| 站点导航 header | build.py 从 `assets/nav-header.html` 注入 |

## 6. 生成器架构 (build.py 扩展)

### 6.1 新依赖

`requirements.txt` (新增):

```
markdown-it-py>=3.0
mdit-py-plugins>=0.4    # front_matter, dollarmath, tables, anchors
PyYAML>=6.0             # frontmatter 解析
# 以下仅迁移期间用到 (migrate-md.py),Phase 1 完成后可移除:
beautifulsoup4>=4.12
html5lib>=1.1
```

`serve.sh` 启动前提示用户运行 `pip install -r requirements.txt` (可选;若未装,build.py 启动时给清晰错误信息)。

### 6.2 构建流程 (3 遍扫描)

```
Pass 1 — discover_posts():
  扫 papers/<slug>/index.md 和 tutorials/<slug>/index.md
  解析 frontmatter + 校验
  ↓
  posts: list[dict]      # (frontmatter, body_md, file_path)
  slug_set: set[str]     # 用于 wiki-link 解析

Pass 2 — render_each_post(posts, slug_set):
  for each post:
    body_md = preprocess(post["body_md"])
        └── [[slug]] / [[slug|alias]] → <a class="wiki-link" data-slug="slug">slug|alias</a>
            (正则替换在 MD 文本上做,产物是 raw HTML,markdown-it-py 透传)
    body_html = md_parser.render(body_md)
    body_html = post_process(body_html, slug_set)
        ├── inject heading IDs (sec-N / sec-N-M)
        ├── 给 <a class="wiki-link" data-slug=...> 补 href / 标 broken
        ├── tag <figure> > img with class="zoomable"
        └── extract h2/h3 outline → 右侧 TOC HTML
    full_html = wrap_shell(head + nav + body_html + toc + scripts)
    write to <post_dir>/index.html

Pass 3 — render_site_pages(posts):
  index.html        ← 首页 card grid (已有)
  papers.html       ← 论文列表 (已有)
  tutorials.html    ← 教程列表 (已有)
  tags.html         ← 标签总览 (已有)
  tags/<tag>.html   ← 每个标签的列表 (已有)
```

Pass 3 复用现有 `build.py` 逻辑;唯一改动:`discover_posts` 不再读 `meta.json`,改读 markdown frontmatter。

### 6.3 Post-process 规则

| 规则 | 输入 HTML | 输出 HTML |
|---|---|---|
| h2 ID | `<h2>1. 出发点</h2>` (第 1 个) | `<h2 id="sec-1">1. 出发点</h2>` |
| h3 ID | `<h3>2.1 Flow Matching</h3>` (第 2 个 h2 下第 1 个 h3) | `<h3 id="sec-2-1">2.1 Flow Matching</h3>` |
| Wiki link 有效 (preprocess) | markdown source 中的 `[[asymflow-2026]]` | preprocess 阶段:正则替换为 `<a class="wiki-link" data-slug="asymflow-2026">asymflow-2026</a>` |
| Wiki link 有效 (resolve) | parser 输出中的 `<a class="wiki-link" data-slug=...>` | post-process 阶段:补 `href="../asymflow-2026/index.html"`,从 `slug_set` 验证存在 |
| Wiki link with alias | `[[asymflow-2026\|AsymFlow]]` | 同上但锚文本 = `AsymFlow` |
| Wiki link 无效 | `[[unknown-slug]]`,slug 不在 `slug_set` | `<a class="wiki-link wiki-link-broken">unknown-slug</a>` + 控制台 warning |
| Lightbox | `<figure><img src=...></figure>` | `<figure><img class="zoomable" src=...></figure>` |
| TOC | 扫所有带 `id="sec-*"` 的 h2/h3 | 生成右侧 `<nav class="toc">`,插到 body 顶部 |

### 6.4 CLI

```bash
python3 build.py                  # 全量构建 (默认)
python3 build.py --post l2p-2026  # 只构建单篇 (开发提速)
python3 build.py --check          # 只检查 frontmatter + wiki-link,不写 HTML
python3 build.py --smoke-test     # 渲染假数据到 /tmp/build-smoke (已有)
```

### 6.5 错误处理

| 情况 | 行为 |
|---|---|
| 缺 frontmatter | warning + skip |
| frontmatter 缺必需字段 | warning + skip,列出缺哪些 |
| `slug` 与父目录名不符 | warning + skip |
| `[[slug]]` 解析失败 | warning + 渲染 broken 链接,不阻塞构建 |
| markdown 解析异常 | warning + skip + traceback |

任何 skip 让 exit code 保持 0 (continue building 其他 post);全量 `--check` 模式下任何 warning 让 exit code 非 0 (CI 能 catch)。

## 7. Skill 改造

### 7.1 reading-papers

- `.claude/skills/reading-papers/SKILL.md` 更新:
  - §5b "HTML 骨架" → 替换为 "markdown 骨架"
  - 删除 TOC 契约段 (现 build.py 自动生成)
  - 图引用范例改为 `<figure>` HTML 块
  - 代码引用范例改为 `<p class="code-source">` + fenced code
  - math-translation 范例改为 `<p class="math-translation">`
  - 流程末尾加 `python3 build.py --post <slug>` (在 git add 前)
- `.claude/skills/reading-papers/templates/index.html` → 新增 `index.md` (HTML 模板可删或保留参考)

### 7.2 writing-tutorial

- `.claude/skills/writing-tutorial/SKILL.md` 同上改动 (Section invariants 仍然适用,只是从 HTML 标签变成 markdown 等价物)
- 螺旋结构的 `<h2 id="sec-N">` 契约改为:作者只写 `## N. 标题`,id 由 build.py 自动生成
- `.claude/skills/writing-tutorial/templates/skeleton.html` + `spiral-section.html` → 新增 markdown 等价物 (`skeleton.md` + `spiral-section.md`)。`style-additions.css` 不变。

### 7.3 CLAUDE.md 更新

- "Per-paper layout" 段加 `index.md`
- "Per-tutorial layout" 段加 `index.md`
- "Rendering stack" 段补充: markdown 源 + build.py 渲染
- "TOC contract" 段简化: 不再要求作者手填 id
- "Editing existing pages" 段重写: 改 markdown 不改 HTML

## 8. 迁移工具

### 8.1 策略

通用 HTML→MD 转换器 (`html2text`, `markdownify`) 会丢失我们的特殊语义 (figure/caption 关联、`p.math-translation` class、`p.code-source` class)。因此自己写**针对性**转换器 `migrate-md.py`。

### 8.2 算法

```python
def html_to_md(html_path: Path, meta_json: dict) -> str:
    soup = BeautifulSoup(html_path.read_text(), 'html5lib')

    # 1. 提取并删除右侧 <nav class="toc"> (build.py 会重新生成)
    if (toc := soup.find('nav', class_='toc')):
        toc.decompose()

    # 2. 元数据 → frontmatter (meta.json + main 内的 .meta div 抓 arxiv/authors 等)
    main = soup.find('main')
    frontmatter = build_frontmatter(meta_json, main.find(class_='meta'))

    # 3. 块级转换 (element by element)
    blocks = []
    for el in main.children:
        if el.name == 'h1':
            blocks.append(f'# {el.get_text(strip=True)}')
        elif el.name == 'h2':
            blocks.append(f'## {el.get_text(strip=True)}')   # 丢 id, build.py 重生
        elif el.name == 'h3':
            blocks.append(f'### {el.get_text(strip=True)}')
        elif el.name == 'figure':
            blocks.append(str(el))                            # 整块 HTML 原样保留
        elif el.name == 'p' and el.get('class') in (['math-translation'], ['code-source']):
            blocks.append(str(el))                            # HTML 岛保留
        elif el.name == 'pre':
            blocks.append(convert_code_block(el))             # ```lang\n...\n```
        elif el.name == 'p':
            blocks.append(inline_md(el))                      # 段落 + 内联 markdown
        elif el.name in ('ul', 'ol'):
            blocks.append(convert_list(el))
        elif el.name == 'blockquote':
            blocks.append(convert_blockquote(el))
        elif el.name == 'table':
            blocks.append(convert_table(el))                  # GFM 表格
        elif el.name == 'section':
            blocks.extend(convert_section_children(el))
        # ... 其他兜底 ...
        else:
            blocks.append(str(el))                            # 兜底:HTML 原样保留

    return frontmatter + '\n\n' + '\n\n'.join(blocks) + '\n'


def inline_md(p: Tag) -> str:
    """转 inline 标签为 markdown 等价物。"""
    # <strong> → **
    # <em> → *
    # <code> → `...`
    # <a href="..."> → [text](href)
    # 内联 math $...$ / $$...$$ → 保持原样 (它们在 HTML 里就是裸文本)
    return ...


def convert_code_block(pre: Tag) -> str:
    code = pre.find('code')
    lang_class = next((c for c in code.get('class', []) if c.startswith('language-')), '')
    lang = lang_class.removeprefix('language-')
    return f'```{lang}\n{code.get_text()}```'
```

### 8.3 迁移流程

```bash
# Step 1: 转换 (写出 .md,不删旧文件)
python3 migrate-md.py --convert

# Step 2: 重新构建,把生成 HTML 写回 index.html
python3 build.py

# Step 3: 人工对比 (随机抽 3-5 篇)
diff papers/l2p-2026/index.html papers/l2p-2026/index.html.pre-migrate

# Step 4: 确认无大问题后清理
python3 migrate-md.py --cleanup    # 删除每个 dir 的 meta.json + .pre-migrate.html 备份
```

### 8.4 转换会丢的东西 (可接受)

- 手写 TOC 的精确措辞 (有时 TOC 链接文本跟 h3 文本略有不同) → 转换后由 build.py 用 h3 原文重生
- inline HTML span 的自定义 class (当前几乎不用)
- HTML 注释 `<!-- ... -->` → 丢弃

### 8.5 安全网

- 转换前自动 `cp index.html index.html.pre-migrate`
- 任何转换异常 → skip 那篇 paper + 报错,人工修复
- `migrate-md.py --cleanup` 必须传 `--yes` 二次确认才删除备份

## 9. 测试与验证

| 层级 | 命令 | 覆盖 |
|---|---|---|
| 单元 (smoke) | `python3 build.py --smoke-test` | 假 frontmatter + 假 body → 渲染到 /tmp |
| 集成 (lint) | `python3 build.py --check` | 全 25 篇 frontmatter 合法 + 所有 `[[slug]]` 可解析 |
| 视觉 (人工) | `./serve.sh` + 随机抽 5 篇 | 渲染对照旧 HTML;math/代码/figure/TOC 无明显丢失 |

迁移完成后的额外检查:

- 全量 `python3 build.py`,git diff 看 HTML 变化是否合理 (容许 whitespace/属性顺序差;不容许内容丢失)
- 浏览首页 + 5 篇随机 paper + 1 篇 tutorial,确认 lightbox / TOC / MathJax / 高亮都工作
- 检查 `tags/<tag>.html` 列表与原来一致

## 10. 验收标准 (Phase 1 完成 = 以下全 pass)

1. ✅ `requirements.txt` 存在,`pip install -r requirements.txt` 成功
2. ✅ `python3 build.py` 全量构建无 warning 退出
3. ✅ `python3 build.py --check` 退出码 0
4. ✅ 全 25 篇 `index.md` 存在且 frontmatter 合法
5. ✅ 全 25 篇 `index.html` 渲染正确 (人工抽 5 篇视觉验证)
6. ✅ 旧 `meta.json` 全部删除
7. ✅ 两个 skill 的 SKILL.md 更新完毕,模板换成 `.md.template`
8. ✅ 用其中一个 skill 创建一篇 dummy paper (e.g. `papers/_smoke-test-2026/`,下划线前缀保证不进站点列表) 走通"skill 写 md → build.py 出 html → 浏览器看渲染"全流程,验证后删除
9. ✅ `[[slug]]` 在 5 篇 paper 里手动测试过 (随便挑两篇加上交叉引用)
10. ✅ `./publish.sh` 不需要任何改动即可工作

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| HTML→MD 转换器漏掉某种 inline pattern | 单独抽样 review 转换后 markdown;不放心的篇章人工补改;migrate-md.py 留 `--dry-run` |
| markdown-it-py 与 MathJax 冲突 (`$` 被吃掉) | 用 `mdit-py-plugins` 的 `dollarmath_plugin`,显式声明 `$...$` / `$$...$$` 为 math token,不参与 emphasis 解析 |
| build.py 重生时 HTML diff 过大,git history 噪音 | 接受这个代价;若日后嫌烦再切换 §3 备选 "HTML gitignore" 方案 |
| 现有 skill 用户 (Claude/agent) 不知道新流程 | SKILL.md 更新清楚;CLAUDE.md 也同步更新;新 skill 命名不变,只改输出 |
| 标签系统在 Phase 1 与 Phase 2 之间错位 | Phase 1 frontmatter 保留 `tags`;Phase 2 把 `tags` 升级成 `concepts` 时做兼容映射 (大多 1:1) |

## 12. 开放问题

- (无) — 所有 brainstorm 中提出的问题均已通过 AskUserQuestion 收敛。
