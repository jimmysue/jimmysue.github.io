# Markdown-Source Blog — Phase 2 Design (Knowledge Graph)

**Date:** 2026-05-29
**Status:** Approved via brainstorm
**Phase:** 2 of 2 (builds on Phase 1)

---

## 1. 动机

Phase 1 把 22 篇笔记变成 markdown + 自动渲染管线。Phase 2 在此基础上加结构化元数据 (concepts / citations / repos) + 图谱抽取 + 可视化,目标是让 Claude/agent 把整个笔记仓库当结构化知识库索引。

## 2. Scope

### In-scope (Phase 2)

- 把 `tags:` 字段重命名为 `concepts:` (一次性 frontmatter 改造)
- 新增 frontmatter 字段: `citations:` (list of paper slugs), `repos:` (list of github URLs)
- `paper.code_url` 并入 `repos:` (单值升级为多值)
- 新模块 `build_lib/graph.py`: 抽取 graph.json, 渲染概念页
- 新页面: `/concepts.html` (cloud, 取代 `tags.html`), `concepts/<slug>.html` (聚合 + 可选用户笔记)
- 新页面: `/graph.html` (cytoscape.js 交互式图谱)
- 输出 `/graph.json` (供 AI/RAG 读)
- 可选: `concepts/<slug>.md` (用户对概念的笔记;不强制创建)
- Skill 更新: reading-papers / writing-tutorial 模板加 `concepts/citations/repos` 字段
- CLAUDE.md 同步

### Out-of-scope

- 不主动补 22 篇现有 paper 的 citations 边 (用户重读时再加)
- 不为 38 个现有概念全量生成 stub `concepts/<slug>.md` (lazy, 仅当用户主动写时创建)
- 不引入 sqlite / RAG 索引服务 (graph.json + 静态文件足够,Claude 可直接读)
- 不做时间轴、聚类、热力图等高级可视化 (force-directed 基础视图先跑通)
- 不改 tag CSS class 命名 (`.tag-cloud` / `.tag-chip` 保留;只改 link path)
- 不切换 SSG (沿用 `build.py`)

## 3. Frontmatter Schema 变更

### 现有 → Phase 2

```yaml
# Phase 1:
tags: [diffusion, flow-matching]
paper:
  code_url: "https://github.com/x/y"

# Phase 2:
concepts: [diffusion, flow-matching]      # 改名,值不变
citations: [asymflow-2026, awm-2025]      # 新增 (可空)
repos:                                     # 新增 (替代 paper.code_url)
  - https://github.com/x/y
paper:
  # code_url 移除
```

### 字段约束

- `concepts: list[str]` — 必需,每项是 slug (小写 + 连字符)
- `citations: list[str]` — 可选,每项是已存在的 paper/tutorial slug
- `repos: list[str]` — 可选,每项是 GitHub URL (允许其他 host)
- `paper.code_url` — 移除 (迁入 repos)

### 校验扩展 (`build_lib/frontmatter.py`)

- `concepts` 必需且非空 (取代旧 `tags` 校验)
- `citations` 中的 slug 必须能在 `slug_set` 解析 (二次校验时);单纯 frontmatter 解析不验证存在性
- `repos` 中 URL 格式宽松 (任何字符串接受;不强校验 https://)

## 4. 概念文件 `concepts/<slug>.md`

完全**可选**。规则:

- 不存在 → 概念仍是图谱节点,聚合页 `concepts/<slug>.html` 由 build.py 完全自动生成
- 存在 → 用户笔记 + 自动聚合 paper list 拼合

文件结构:

```yaml
---
slug: flow-matching
name: "Flow Matching"
aliases: ["flow matching", "FM"]
parent: diffusion
---

# Flow Matching

(用户笔记;可空。)

## 跟 DDPM 的差别

(对比段落)
```

### 字段

- `slug` — 必需,等于文件名 stem
- `name` — 必需,显示用
- `aliases: list[str]` — 可选,搜索辅助
- `parent` — 可选,父概念 slug;Phase 2 不渲染父子层级 (Phase 3 candidate)

### 渲染

`concepts/<slug>.html` 由 build.py 生成:

```
<head> + <nav>
<main>
  <h1>{{name}}</h1>
  {{user_body_html}}        # 来自 concepts/<slug>.md, 可为空
  <h2>提到此概念的论文 / 教程</h2>
  <ul class="post-grid">
    {{each post mentioning this concept → card}}
  </ul>
</main>
```

## 5. `graph.json` Schema

```json
{
  "version": 1,
  "generated_at": "2026-05-29T22:00:00",
  "nodes": [
    {
      "id": "papers/l2p-2026",
      "type": "paper",
      "slug": "l2p-2026",
      "title": "...",
      "date": "2026-05-22",
      "tldr": "...",
      "url": "papers/l2p-2026/index.html"
    },
    {
      "id": "tutorials/rl-for-diffusion-2023",
      "type": "tutorial",
      "slug": "rl-for-diffusion-2023",
      "title": "...",
      "date": "2026-05-17",
      "url": "tutorials/rl-for-diffusion-2023/index.html"
    },
    {
      "id": "concepts/flow-matching",
      "type": "concept",
      "slug": "flow-matching",
      "name": "Flow Matching",
      "has_file": false,
      "url": "concepts/flow-matching.html"
    },
    {
      "id": "repos/TencentYoutuResearch/T2I-L2P",
      "type": "repo",
      "url": "https://github.com/TencentYoutuResearch/T2I-L2P",
      "host": "github.com",
      "owner": "TencentYoutuResearch",
      "name": "T2I-L2P"
    }
  ],
  "edges": [
    { "from": "papers/l2p-2026", "to": "concepts/flow-matching", "kind": "mentions" },
    { "from": "papers/l2p-2026", "to": "papers/asymflow-2026", "kind": "cites" },
    { "from": "papers/l2p-2026", "to": "repos/TencentYoutuResearch/T2I-L2P", "kind": "implements" },
    { "from": "tutorials/rl-for-diffusion-2023", "to": "papers/awm-2025", "kind": "covers" }
  ]
}
```

### Node ID 规则

- `papers/<slug>` (一律使用 `/` 分隔,即使 Windows 也用 forward slash)
- `tutorials/<slug>`
- `concepts/<slug>` — 即使无文件,也存在
- `repos/<owner>/<name>` — 从 GitHub URL 解析;非 github 用 `repos/<host>/<path-hash>` 兜底

### Edge `kind` 取值

- `mentions` — paper/tutorial → concept (来自 `concepts:` 字段)
- `cites` — paper → paper (来自 `citations:` 字段)
- `covers` — tutorial → paper (来自 tutorial 的 `citations:` 字段;tutorial 没有别的引用语义)
- `implements` — paper/tutorial → repo (来自 `repos:` 字段)

边不带 weight 或 relation — 全是布尔存在性 (per user instruction "直接简单形式即可")。

### 跨论文 `[[slug]]` body 引用

Phase 1 已有 `[[slug]]` wiki-link。Phase 2 graph extractor 也扫 body markdown 的 `[[slug]]` 并合并入 citations (paper→paper) / covers (tutorial→paper)。这样用户在 body 写 `[[asymflow-2026]]` 也会产生图谱边,不必都在 frontmatter 重复声明。

## 6. `/graph.html` 可视化

### 库

[cytoscape.js](https://cytoscape.org/) via CDN (jsdelivr 与现有 MathJax/highlight.js 同款管理):

```html
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.min.js"></script>
```

### 布局

`cose-bilkent` (force-directed)。节点大小 = 入度的函数 (越被引用越大)。

### 节点编码

- Paper = 蓝色圆 (#4A90E2)
- Tutorial = 紫色方 (#8B5CF6)
- Concept = 橙色菱形 (#F59E0B)
- Repo = 灰色三角 (#6B7280)

### 交互

- 点击节点 → window.location.href = node.data('url'),跳到对应页面 (或 GitHub repo)
- Hover → tooltip 显示 title + tldr (paper/tutorial) 或 name + 被多少 paper 引用 (concept)
- 顶部 filter checkbox: 显示/隐藏 paper / tutorial / concept / repo 节点
- 顶部搜索框: 按 slug/title 匹配,匹配节点高亮 + 居中

### 简化原则

- 节点 ~85 (22 doc + ~38 concept + ~25 repo),边 ~150 — 在 cose-bilkent 下交互流畅
- 不做集群、不做时间轴、不做热力图
- 反馈不好再迭代

## 7. Build 流程

新增 `build_lib/graph.py` 模块。Build pipeline 加两个 pass:

```
Pass 1: discover_posts        # (已存在) 读 paper/tutorial frontmatter
Pass 2: render_each_post      # (已存在) MD → HTML
Pass 3: render_site_pages     # (已存在,需小改): 首页/列表/标签
                              #    — `tags.html` → `concepts.html`
                              #    — link path `tags/<x>.html` → `concepts/<x>.html`
Pass 4: extract_graph         # NEW: 走 posts + concepts/ 目录 → graph.json
Pass 5: render_concept_pages  # NEW: 每个 concept slug 生成 concepts/<slug>.html
Pass 6: render_graph_page     # NEW: 写 /graph.html (静态 + cytoscape.js + 嵌入 graph.json)
```

### `build_lib/graph.py` 接口

```python
def discover_concepts(root: Path) -> dict[str, dict]:
    """扫 concepts/<slug>.md (可选目录),返回每个概念的 frontmatter + body."""
    ...

def parse_repo_url(url: str) -> dict:
    """从 GitHub/通用 URL 解析出 owner/name → 用于节点 ID."""
    ...

def extract_graph(posts: list[dict], concepts: dict[str, dict]) -> dict:
    """构造 nodes + edges,返回 graph.json 字典."""
    ...

def render_concept_page(slug: str, concept_meta: dict, concept_body_html: str,
                        posts_mentioning: list[dict], nav_html: str) -> str:
    """concept HTML 页面."""
    ...

def render_graph_page(graph: dict, nav_html: str) -> str:
    """/graph.html with embedded cytoscape script + JSON."""
    ...
```

### body `[[slug]]` 抽取

Pass 4 复用 `build_lib.wiki_links` 的 `WIKI_LINK_RE` 在 post body 上 finditer,得到一组 `[[slug]]` 引用。合并入对应 from-node 的 outgoing edges (paper→paper=cites, tutorial→paper=covers)。

## 8. 标签系统迁移

| 旧路径 | 新路径 | 说明 |
|---|---|---|
| `tags.html` | `concepts.html` | content 不变,只改文件名 |
| `tags/<x>.html` | `concepts/<x>.html` | 增强为聚合页 (有/无用户笔记自适应) |
| `tags/` 目录 | 删除 | 重建到 `concepts/` |
| frontmatter `tags:` | `concepts:` | 22 个文件 in-place 改名 |
| CSS class `.tag-cloud` / `.tag-chip` | 保留 | 不破坏视觉;仅 link path 更新 |
| `slugify_tag()` Python 函数 | 保留 (或加 alias `slugify_concept()`) | 内部细节 |

### 迁移步骤 (一次性脚本)

```bash
# 1. 重命名 22 篇 frontmatter
python3 migrate-concepts.py --convert
# 2. 重建
python3 build.py
# 3. 检查 tags/ 目录是否还有遗留
ls tags/ 2>&1 | head
# 4. 如果旧 tags/ 目录已被 build.py 重命名为 concepts/,验证 redirect 正确
# 5. 删除老的 tags/ 目录文件 (build.py 不主动清,migrate-concepts.py 包含 cleanup 选项)
python3 migrate-concepts.py --cleanup --yes
```

## 9. Skill / CLAUDE.md 更新

### reading-papers

- `templates/index.md`: `tags:` → `concepts:`; 新增可空 `citations:` 和 `repos:` 字段
- SKILL.md: 同步描述

### writing-tutorial

- `templates/skeleton.md`: 同上
- SKILL.md: 同步,且建议 tutorial 的 `citations:` 列出"覆盖的核心论文" (Phase 2 把这些当 `covers` 边)

### CLAUDE.md

- "Per-paper layout" / "Per-tutorial layout" 段: 说明新字段
- 新增 "Knowledge graph" 段: 介绍 graph.json 用途 + agent 索引建议
- "TOC contract" / "Rendering stack" 不变

## 10. 测试

- `tests/test_graph.py`: graph.py 的单测 (discover_concepts, parse_repo_url, extract_graph, body 抽取 [[slug]])
- `tests/test_frontmatter.py`: 扩展校验 `concepts:` 是必需,`tags:` 是 legacy 警告 (兼容期可读但 warn)
- `--smoke-test` 覆盖新 pass (graph extract 生成 fake JSON 校验)
- 集成: `python3 build.py --check` 应额外校验 citations 引用的 slug 都存在

## 11. 验收 (Phase 2 完成 = 全 pass)

1. ✅ `python3 build.py` 干净重建,生成 `graph.json` + `/graph.html` + 每个 concept 的 `concepts/<slug>.html`
2. ✅ 22 篇 frontmatter `tags:` → `concepts:` 完成 (无 `tags:` 残留)
3. ✅ `paper.code_url` 全部并入 `repos:`
4. ✅ `tags.html` 重定向或重命名为 `concepts.html`,旧 `tags/<x>.html` 替换为 `concepts/<x>.html`
5. ✅ `graph.json` schema 合法,包含 4 类节点 + 4 类边
6. ✅ `/graph.html` 在浏览器加载,cose-bilkent 布局,节点可点击跳转
7. ✅ 至少 1 个手写 `concepts/<slug>.md` 验证用户笔记 + 自动聚合并存
8. ✅ pytest 所有测试通过 (含 Phase 1 + Phase 2 新增)
9. ✅ skill SKILL.md + 模板更新;dummy 走通 reading-papers 写 markdown → 含 concepts/citations/repos 字段 → build.py 渲染 + 图谱新增节点
10. ✅ CLAUDE.md 加 graph 章节

## 12. 风险

| 风险 | 缓解 |
|---|---|
| 22 篇 `tags:` 一次性改名时 yaml 解析有 corner case | 写一个 `migrate-concepts.py` 脚本,逐个文件改名+ frontmatter 替换;失败可回滚 |
| cytoscape.js 在 22+ 节点 / 100+ 边规模下 cose-bilkent 是否流畅 | 实际数据小,基本无风险;若卡顿可换 `cose` 或 `breadthfirst` |
| `repos:` 多值后,有些字段 (如 weights_url) 仍单值 — schema 不一致 | 接受;weights_url 是次要字段,值少且性质不同 |
| body `[[slug]]` 抽取与 frontmatter `citations:` 字段产生重复边 | 在 extract_graph 用 set 去重 |
| Phase 1 用户已经习惯 tag 体系,改名 concepts 会迷惑 | Visual 不变 (`.tag-cloud`/`.tag-chip` CSS class 保留);只 URL 变了 |

## 13. 开放问题

(无)

---

End of spec.
