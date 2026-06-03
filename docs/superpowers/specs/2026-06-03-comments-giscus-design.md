# Comments via Giscus — Design

**Date:** 2026-06-03
**Status:** Approved (brainstorm), awaiting user review of spec
**Scope:** Small feature — add a comments section to per-paper / per-tutorial pages, powered by Giscus (GitHub Discussions).

---

## 1. 动机

The static blog at <https://jimmysue.github.io/> has no reader interaction layer. Readers can browse but not respond. Adding comments enables:

- Reader questions / corrections on paper analyses
- Discussion threads per paper (instead of email or external links)
- 👍 reactions as a lightweight quality signal
- A long-term archive of reader feedback co-located with the post

Constraints:
- **No backend** (GitHub Pages = pure static)
- **No paid SaaS** (current stack is all free CDN + GitHub Pages)
- **Audience is technical** (researchers, engineers — almost all have GitHub accounts)
- **Single maintainer** (low moderation bandwidth — relying on real GitHub identities helps)

## 2. 业界方案调研 (Survey)

| 方案 | 后端 | Auth | 数据位置 | JS | 价格 | 活跃 |
|---|---|---|---|---|---|---|
| **Giscus** | GitHub Discussions | GitHub-only | 本 repo Discussions | ~30KB | 免费 | ⭐⭐⭐⭐⭐ |
| Utterances | GitHub Issues | GitHub-only | 本 repo Issues | ~20KB | 免费 | ⭐⭐⭐ (被 Giscus 取代) |
| Disqus | 自家 SaaS | 多 (匿名 OK) | Disqus 服务器 | ~600KB | 含广告 | ⭐⭐ (隐私差) |
| Cusdis | 自托管 / 官方 SaaS | 邮件 + 匿名 | 自己 / Cusdis | ~5KB | 自托管免费 / $5月 | ⭐⭐⭐⭐ |
| Remark42 | 必自托管 (Go) | 多 OAuth | 自己服务器 | ~60KB | 自托管免费 | ⭐⭐⭐⭐ |
| Hyvor Talk | SaaS | 多 OAuth | Hyvor 服务器 | ~50KB | $5+/月 | ⭐⭐⭐⭐ |
| Webmentions | webmention.io SaaS | 跨博客 IndieWeb | 分布式 | ~5KB | 免费 | ⭐⭐ (niche) |
| Isso | 必自托管 (Python) | 邮件 | 自己服务器 | ~20KB | 自托管免费 | ⭐⭐⭐ |

**已选定: Giscus** (per brainstorm 决策 "评论者必须有 GitHub 账号 OK")

理由 (summary):
- 零后端,零成本,跟现有"GitHub Pages + 纯静态"哲学一致
- 受众 ~100% 有 GitHub 账号
- 评论数据存在**本 repo 的 GitHub Discussions** 里 (数据所有权)
- Markdown 渲染原生支持 LaTeX (`$..$`) + 代码块,跟 paper 内容契合
- 开源 ([giscus/giscus](https://github.com/giscus/giscus)),非 vendor lock-in
- 现成 reaction emoji 👍🎉🚀
- Theme 可选 light / dark / preferred_color_scheme / 自定义 CSS

## 3. Scope

### In-scope

- 在 paper / tutorial 详情页底部注入 Giscus 评论组件
- `assets/giscus-config.json` 集中保存 Giscus 配置 (repo, repo_id, category_id 等)
- `build.py` 读 config + 注入 HTML 片段
- `assets/style.css` 加 `.comments` 节的样式
- Frontmatter 字段 `comments: false` 单篇可关闭 (默认开启)
- 用户手动 setup 步骤 (装 Giscus App、开 Discussions、建 Comments 分类) 写进 `CLAUDE.md`

### Out-of-scope

- 列表 / 聚合页 (`index.html`, `papers.html`, `tutorials.html`, `concepts.html`)
- 概念页 (`concepts/<slug>.html`)
- `graph.html`
- 多语言 UI 切换 (默认中文 `data-lang="zh-CN"`)
- 自定义 Giscus theme CSS (先用官方 `light`,以后再说)
- 评论审核工具 (用 GitHub Discussions 原生 UI 即可)
- 评论数 / 热度统计聚合到 card 上 (Phase 3 candidate)

## 4. 架构

```
┌────────────────────────────────────────────────────────────┐
│ Paper / Tutorial 页 (papers/<slug>/index.html 等)           │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ <main>                                                 │ │
│ │   <h1>...</h1>                                          │ │
│ │   <aside class="post-meta">...</aside>                  │ │
│ │   <body content>                                        │ │
│ │   ...                                                   │ │
│ │   ┌──────────────────────────────────────────────────┐  │ │
│ │   │ <section class="comments">  ← 新增                │  │ │
│ │   │   <h2>讨论 / Comments</h2>                        │  │ │
│ │   │   <p class="comments__note">评论托管在...</p>     │  │ │
│ │   │   <script src="https://giscus.app/client.js"     │  │ │
│ │   │      data-repo="..." ...> </script>              │  │ │
│ │   │   <iframe loaded by Giscus → GitHub Discussions> │  │ │
│ │   │ </section>                                       │  │ │
│ │   └──────────────────────────────────────────────────┘  │ │
│ │ </main>                                                 │ │
│ └────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### 注入流

```
build.py → build_posts() loop:
  1. render_post_body(md) → body_html
  2. (NEW) if p.get("comments", True): body_html += _giscus_block(config)
  3. assemble_post_page(meta, body_html, ...) → full HTML
  4. write to <post_dir>/index.html
```

只有走 `build_posts()` 的页面 (paper + tutorial) 才挂评论。列表 / 概念 / 图谱页用其他 render 路径,不受影响。

## 5. 文件改动

| 文件 | 状态 | 改动 |
|---|---|---|
| `assets/giscus-config.json` | 新建 | Giscus 配置 (repo_id 等占位,等用户填) |
| `build.py` | 修改 | 加载 config,加 `_giscus_block()` 助手,在 `build_posts()` 注入 |
| `assets/style.css` | 修改 | 末尾追加 `.comments` 节样式 |
| `CLAUDE.md` | 修改 | 加一节"Comments setup" 描述 GitHub App / Discussions / 分类 / repo_id 获取的一次性步骤 |
| `tests/test_giscus_block.py` | 新建 | 单元测试 `_giscus_block()` + 注入逻辑 |

## 6. 数据 schemas

### 6.1 `assets/giscus-config.json` schema

```json
{
  "repo": "jimmysue/jimmysue.github.io",
  "repo_id": "R_kgDOXXXXXX",
  "category": "Comments",
  "category_id": "DIC_kwDOXXXXXX",
  "mapping": "pathname",
  "theme": "light",
  "reactions_enabled": "1",
  "loading": "lazy",
  "lang": "zh-CN",
  "input_position": "bottom"
}
```

所有 key 必需。`build.py` 在 config 缺失或字段缺失时退到"不注入评论 + 打 WARN",不要让 build 崩溃。

### 6.2 Frontmatter 新字段

```yaml
comments: false  # 单篇关闭 (可选, 默认 true)
```

只有显式写 `false` 才关闭。不写 = 自动开。

## 7. 关键设计选择

| 维度 | 选 | 为什么 |
|---|---|---|
| Mapping | `pathname` | URL 路径稳定;改 title 不影响 thread |
| Reactions | enabled (post-level 👍🎉🚀) | 轻量互动,无需评论也能表态 |
| Theme | `light` | 匹配现网视觉;后续可换 preferred_color_scheme |
| Lang | `zh-CN` | UI 中文化, 跟内容语言一致 |
| Loading | `lazy` | 不阻塞首屏 |
| Input position | `bottom` | 评论列表上方读, 输入框下方写, 习惯做法 |
| Emit metadata | `0` | 不需要回报 metadata 到 parent page |
| 默认开启 | yes (frontmatter 缺省) | 鼓励互动;单篇可 opt-out |

## 8. 用户一次性 setup 步骤

写到 `CLAUDE.md` 新章节 "Comments (Giscus) — first-time setup":

1. Repo settings → **Features → Discussions** 打钩
2. 装 Giscus App: <https://github.com/apps/giscus> → Install → 选 `jimmysue.github.io` repo
3. 进 repo Discussions → **New category** → 名 `Comments`,类型选 `Announcement` (防止陌生人乱开 thread,Giscus 自动创建)
4. 去 [giscus.app](https://giscus.app),输入 repo 名,选 Category=Comments,Mapping=Pathname,Reactions=enabled
5. 复制页面给出的 `data-repo-id` 和 `data-category-id`
6. 填入 `assets/giscus-config.json`
7. `python3 build.py && ./publish.sh`

## 9. 测试 / 验证

### 单元测试 (`tests/test_giscus_block.py`)

```python
def test_giscus_block_renders_with_config():
    cfg = {"repo": "x/y", "repo_id": "R_a", "category": "C",
           "category_id": "D_b", "mapping": "pathname",
           "theme": "light", "reactions_enabled": "1",
           "loading": "lazy", "lang": "zh-CN", "input_position": "bottom"}
    html = _giscus_block(cfg)
    assert 'data-repo="x/y"' in html
    assert 'data-repo-id="R_a"' in html
    assert 'data-category="C"' in html
    assert 'data-category-id="D_b"' in html
    assert 'class="comments"' in html
    assert 'giscus.app/client.js' in html
```

### 集成验证

- `python3 build.py` 后,sample paper (e.g. `papers/mrt-2026/index.html`) 应含:
  - `<section class="comments">`
  - `giscus.app/client.js` script tag
  - `data-repo-id=` (非空)
- `index.html`, `papers.html`, `concepts.html`, `concepts/<x>.html`, `graph.html` 都**不**含 `comments` section
- 当 frontmatter 有 `comments: false` 时,该单页不含 comments section
- 当 `assets/giscus-config.json` 缺失或 `repo_id` 为空,build 不崩溃,只 print WARN

### 手动浏览验证

- 本地 `./serve.sh` → 打开 `http://127.0.0.1:8766/papers/mrt-2026/index.html` → 滚到底部应看到 Giscus iframe 加载完成
- 登录 GitHub → 应能发评论 → 评论应出现在 repo 的 Discussions tab 对应 thread 里

## 10. 验收

1. ✅ `assets/giscus-config.json` 存在并填入真实 repo_id / category_id
2. ✅ `python3 build.py` 干净;25 个 per-post 页全部含 comments section,列表 / 概念 / 图谱页全部不含
3. ✅ 单元测试通过 (+1 个测试文件)
4. ✅ Frontmatter `comments: false` 单篇关闭生效
5. ✅ 至少 1 篇 paper 浏览器内验证 Giscus iframe 正常加载
6. ✅ CLAUDE.md 加 setup 章节
7. ✅ 部署到线上 (`publish.sh`) 后,GitHub Pages 域名上的 paper 页评论功能正常

## 11. 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| 用户 setup 步骤出错 (App 没装 / Discussions 没开) | `build.py` 在 config 缺失/不完整时 print 引导 + CLAUDE.md 写清步骤 |
| spam | GitHub Discussions 后台原生 hide/delete + report user;无匿名;低风险 |
| 隐私顾虑 (第三方 iframe) | `.comments__note` 明示 "评论托管在 GitHub Discussions";`lazy` 不主动加载 |
| URL 改了 (slug 重命名) 历史评论"丢失" | `pathname` mapping 锁住路径——只要 slug 不改不丢;改 slug 是大动作,文档约束 |
| 后期想换方案 | Giscus 只是注入的一段 HTML,删一个函数 + 一个 config 文件即可。评论数据导出走 GitHub Discussions GraphQL API |
| Giscus 服务下线 | 不太可能 (开源 + GitHub-hosted),但即使下线,评论数据仍在我的 Discussions 里,可以换方案对接同一个 API |
| GitHub 改 Discussions API | 极低概率;Giscus 维护者会跟进,我们什么都不用做 |

## 12. 开放问题

无 — 所有 brainstorm 中提出的开放点已收敛。

---

End of spec.
