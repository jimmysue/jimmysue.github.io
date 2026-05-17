#!/usr/bin/env python3
"""Static site generator for the paper-reading blog.

Reads `meta.json` from every `papers/<slug>/` and `tutorials/<slug>/` and
emits 5 page types: index.html, papers.html, tutorials.html, tags.html,
tags/<slug>.html.

Usage:
    python3 build.py                 # build the real site (CWD = repo root)
    python3 build.py --smoke-test    # render 3 fake posts to /tmp/build-smoke/

Zero third-party dependencies. Stdlib only.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# constants

REQUIRED_META_KEYS = {"type", "slug", "title", "date", "tldr", "tags"}
VALID_TYPES = {"paper", "tutorial"}
MAX_CARD_TAGS = 3

DEFAULT_NAV_HEADER = """\
<header class="site-nav">
  <div class="site-nav__inner">
    <a class="site-nav__brand" href="{HOME_URL}">paper-reading</a>
    <nav class="site-nav__links">
      <a class="site-nav__link{ACTIVE_HOME}" href="{HOME_URL}">首页</a>
      <a class="site-nav__link{ACTIVE_PAPERS}" href="{PAPERS_URL}">论文</a>
      <a class="site-nav__link{ACTIVE_TUTORIALS}" href="{TUTORIALS_URL}">教程</a>
      <a class="site-nav__link{ACTIVE_TAGS}" href="{TAGS_URL}">标签</a>
    </nav>
  </div>
</header>
"""

HEAD_TMPL = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{page_title}</title>
  <link rel="stylesheet" href="{css_path}">
</head>
<body>
"""

PAGE_TAIL = "</body>\n</html>\n"


# ---------------------------------------------------------------------------
# data loading

def slugify_tag(raw: str) -> str:
    """Canonicalise a tag: lowercase, spaces -> hyphens, strip outer whitespace."""
    s = raw.strip().lower()
    s = re.sub(r"\s+", "-", s)
    return s


def discover_posts(root: Path) -> list[dict]:
    """Walk papers/<slug>/meta.json and tutorials/<slug>/meta.json."""
    posts: list[dict] = []
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            meta_path = sub / "meta.json"
            if not meta_path.is_file():
                print(f"WARN: skipping {sub} (no meta.json)", file=sys.stderr)
                continue
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"WARN: skipping {meta_path} (invalid JSON: {e})", file=sys.stderr)
                continue
            missing = REQUIRED_META_KEYS - set(meta.keys())
            if missing:
                print(
                    f"WARN: skipping {meta_path} (missing keys: {sorted(missing)})",
                    file=sys.stderr,
                )
                continue
            if meta["type"] not in VALID_TYPES:
                print(f"WARN: skipping {meta_path} (bad type: {meta['type']!r})", file=sys.stderr)
                continue
            if meta["type"] != kind:
                # tolerated, but warn — meta says one thing, directory another
                print(
                    f"WARN: {meta_path} type={meta['type']!r} but lives under {dirname}/; using meta",
                    file=sys.stderr,
                )
            index_html = sub / "index.html"
            if not index_html.is_file():
                print(f"WARN: skipping {meta_path} (no sibling index.html)", file=sys.stderr)
                continue
            # canonicalise tags
            meta["tags"] = [slugify_tag(t) for t in meta.get("tags", [])]
            # canonicalise url
            meta["_url"] = f"{dirname}/{sub.name}/index.html"
            meta.setdefault("tutorial_meta", None)
            posts.append(meta)
    return posts


# ---------------------------------------------------------------------------
# rendering helpers

def esc(s: str) -> str:
    return html.escape(s, quote=True)


def render_head(page_title: str, css_path: str) -> str:
    return HEAD_TMPL.format(page_title=esc(page_title), css_path=css_path)


def load_nav_header(root: Path) -> str:
    """Read assets/nav-header.html if it exists, else use a built-in fallback."""
    p = root / "assets" / "nav-header.html"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return DEFAULT_NAV_HEADER


def render_nav(nav_tmpl: str, active: str, depth: int) -> str:
    """Substitute placeholders in the nav-header template.

    depth=0 → root-level page (index.html, papers.html, tags.html, …)
    depth=1 → tags/<slug>.html (links go up one level)
    """
    prefix = "" if depth == 0 else "../"
    actives = {
        "home": "",
        "papers": "",
        "tutorials": "",
        "tags": "",
    }
    if active in actives:
        actives[active] = " active"
    return nav_tmpl.format(
        HOME_URL=f"{prefix}index.html",
        PAPERS_URL=f"{prefix}papers.html",
        TUTORIALS_URL=f"{prefix}tutorials.html",
        TAGS_URL=f"{prefix}tags.html",
        ACTIVE_HOME=actives["home"],
        ACTIVE_PAPERS=actives["papers"],
        ACTIVE_TUTORIALS=actives["tutorials"],
        ACTIVE_TAGS=actives["tags"],
    )


def render_card(post: dict, depth: int) -> str:
    """Render one post card.

    `depth` controls relative-path prefix to other pages:
      0 → on a root page (links: papers/<slug>/index.html, tags/<tag>.html)
      1 → on tags/<slug>.html (links: ../papers/<slug>/index.html, ../tags/<tag>.html)
    """
    prefix = "" if depth == 0 else "../"
    is_tut = post["type"] == "tutorial"
    cls = "post-card--tutorial" if is_tut else "post-card--paper"
    badge_cls = "post-card__badge--tutorial" if is_tut else "post-card__badge--paper"
    badge_text = "📘 教程" if is_tut else "📄 论文"

    # date + tutorial-meta extras
    date_html = esc(post["date"])
    if is_tut and post.get("tutorial_meta"):
        tm = post["tutorial_meta"]
        wc = tm.get("word_count")
        rm = tm.get("reading_minutes")
        extras: list[str] = []
        if wc:
            extras.append(f"{esc(str(wc))} 字")
        if rm:
            # normalise hyphen-minus to en-dash for ranges; preserve as-is otherwise
            rm_str = str(rm).replace("-", "–")
            extras.append(f"{esc(rm_str)} 分钟")
        if extras:
            date_html = f"{esc(post['date'])} · " + " · ".join(extras)

    # tag chips (cap at MAX_CARD_TAGS)
    tags = post.get("tags", [])
    shown = tags[:MAX_CARD_TAGS]
    tag_html_parts = [
        f'<a class="tag-chip" href="{prefix}tags/{esc(t)}.html">{esc(t)}</a>'
        for t in shown
    ]
    if len(tags) > MAX_CARD_TAGS:
        more = len(tags) - MAX_CARD_TAGS
        tag_html_parts.append(f'<span class="tag-chip tag-chip--more">+{more} more</span>')
    tag_block = "\n      ".join(tag_html_parts)
    if not tag_block:
        tag_block = ""

    href = f"{prefix}{post['_url']}"
    return (
        f'<a class="post-card {cls}" href="{esc(href)}">\n'
        f'  <div class="post-card__head">\n'
        f'    <span class="post-card__badge {badge_cls}">{badge_text}</span>\n'
        f'    <span class="post-card__date">{date_html}</span>\n'
        f'  </div>\n'
        f'  <h3 class="post-card__title">{esc(post["title"])}</h3>\n'
        f'  <p class="post-card__tldr">{esc(post["tldr"])}</p>\n'
        f'  <div class="post-card__tags">\n      {tag_block}\n  </div>\n'
        f'</a>\n'
    )


def render_grid(posts: list[dict], depth: int) -> str:
    cards = "".join(render_card(p, depth) for p in posts)
    return f'<div class="post-grid">\n{cards}</div>\n'


# ---------------------------------------------------------------------------
# page assemblers

def sort_posts(posts: list[dict]) -> list[dict]:
    # Sort by date desc, ties broken by slug asc for determinism.
    return sorted(posts, key=lambda p: (p["date"], p["slug"]), reverse=False)[::-1] \
        if False else sorted(posts, key=lambda p: (p["date"], p["slug"]))[::-1]


def _stable_desc(posts: list[dict]) -> list[dict]:
    # date desc, slug asc as tiebreaker → idempotent.
    return sorted(posts, key=lambda p: (p["date"], p["slug"]))[::-1]


def build_index(posts: list[dict], nav_tmpl: str) -> str:
    posts = _stable_desc(posts)
    body = render_head("paper-reading — 论文 & 教程", "assets/style.css")
    body += render_nav(nav_tmpl, active="home", depth=0)
    body += '<main class="page page--listing">\n'
    body += "<h1>paper-reading</h1>\n"
    body += '<p class="page__intro">论文精读与系统化教程。</p>\n'
    body += render_grid(posts, depth=0)
    body += "</main>\n"
    body += PAGE_TAIL
    return body


def build_papers(posts: list[dict], nav_tmpl: str) -> str:
    only = _stable_desc([p for p in posts if p["type"] == "paper"])
    body = render_head("论文 — paper-reading", "assets/style.css")
    body += render_nav(nav_tmpl, active="papers", depth=0)
    body += '<main class="page page--listing">\n'
    body += "<h1>论文</h1>\n"
    body += f'<p class="page__intro">共 {len(only)} 篇论文精读。</p>\n'
    body += render_grid(only, depth=0)
    body += "</main>\n"
    body += PAGE_TAIL
    return body


def build_tutorials(posts: list[dict], nav_tmpl: str) -> str:
    only = _stable_desc([p for p in posts if p["type"] == "tutorial"])
    body = render_head("教程 — paper-reading", "assets/style.css")
    body += render_nav(nav_tmpl, active="tutorials", depth=0)
    body += '<main class="page page--listing">\n'
    body += "<h1>教程</h1>\n"
    body += f'<p class="page__intro">共 {len(only)} 篇深度教程。</p>\n'
    body += render_grid(only, depth=0)
    body += "</main>\n"
    body += PAGE_TAIL
    return body


def _tag_index(posts: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in posts:
        for t in p.get("tags", []):
            out.setdefault(t, []).append(p)
    return out


def build_tags_cloud(posts: list[dict], nav_tmpl: str) -> str:
    by_tag = _tag_index(posts)
    counts = {t: len(ps) for t, ps in by_tag.items()}
    body = render_head("标签 — paper-reading", "assets/style.css")
    body += render_nav(nav_tmpl, active="tags", depth=0)
    body += '<main class="page page--tags">\n'
    body += "<h1>标签</h1>\n"
    body += f'<p class="page__intro">共 {len(counts)} 个标签。</p>\n'

    if counts:
        min_c = min(counts.values())
        max_c = max(counts.values())
        spread = max(1, max_c - min_c)
        body += '<div class="tag-cloud">\n'
        for tag in sorted(counts.keys()):
            c = counts[tag]
            size = 0.9 + 1.3 * (c - min_c) / spread
            size_str = f"{size:.2f}"
            body += (
                f'  <a class="tag-cloud__item" href="tags/{esc(tag)}.html" '
                f'style="font-size: {size_str}rem;">\n'
                f'    {esc(tag)} <span class="tag-cloud__count">{c}</span>\n'
                f'  </a>\n'
            )
        body += "</div>\n"
    else:
        body += '<p class="empty">还没有标签。</p>\n'

    body += "</main>\n"
    body += PAGE_TAIL
    return body


def build_tag_page(tag: str, tagged: list[dict], nav_tmpl: str) -> str:
    tagged = _stable_desc(tagged)
    n_paper = sum(1 for p in tagged if p["type"] == "paper")
    n_tut = sum(1 for p in tagged if p["type"] == "tutorial")
    body = render_head(f"标签: {tag} — paper-reading", "../assets/style.css")
    body += render_nav(nav_tmpl, active="tags", depth=1)
    body += '<main class="page page--tag">\n'
    body += f"<h1>标签: {esc(tag)}</h1>\n"
    body += (
        f'<p class="page__intro">共 {len(tagged)} 篇 '
        f'({n_paper} 论文 / {n_tut} 教程)</p>\n'
    )
    body += '<p class="back-link"><a href="../tags.html">← 返回所有标签</a></p>\n'
    body += render_grid(tagged, depth=1)
    body += "</main>\n"
    body += PAGE_TAIL
    return body


# ---------------------------------------------------------------------------
# top-level build

def build_site(posts: list[dict], out_root: Path, nav_tmpl: str) -> dict:
    """Write all generated pages. Returns a small summary dict."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tags").mkdir(parents=True, exist_ok=True)

    n_pages = 0

    (out_root / "index.html").write_text(build_index(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1
    (out_root / "papers.html").write_text(build_papers(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1
    (out_root / "tutorials.html").write_text(build_tutorials(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1
    (out_root / "tags.html").write_text(build_tags_cloud(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1

    by_tag = _tag_index(posts)
    for tag in sorted(by_tag.keys()):
        page = build_tag_page(tag, by_tag[tag], nav_tmpl)
        (out_root / "tags" / f"{tag}.html").write_text(page, encoding="utf-8")
        n_pages += 1

    n_paper = sum(1 for p in posts if p["type"] == "paper")
    n_tut = sum(1 for p in posts if p["type"] == "tutorial")
    return {
        "pages": n_pages,
        "papers": n_paper,
        "tutorials": n_tut,
        "tags": len(by_tag),
    }


# ---------------------------------------------------------------------------
# smoke test

SMOKE_POSTS = [
    {
        "type": "paper",
        "slug": "awm-2025",
        "title": "AWM: Active Weight Manipulation for diffusion alignment",
        "date": "2026-05-15",
        "tldr": "A clean reparameterisation that turns DPO-style preference data into per-step weight edits.",
        "tags": ["diffusion", "rl", "alignment"],
        "tutorial_meta": None,
        "_url": "papers/awm-2025/index.html",
    },
    {
        "type": "paper",
        "slug": "flow-opd-2026",
        "title": "Flow-OPD: optimal pre-conditioning for flow matching",
        "date": "2026-03-02",
        "tldr": "Pre-conditioning the velocity field via OT couplings cuts NFE by ~3× at iso-FID.",
        "tags": ["flow-matching", "diffusion", "optimal-transport", "sampling"],
        "tutorial_meta": None,
        "_url": "papers/flow-opd-2026/index.html",
    },
    {
        "type": "tutorial",
        "slug": "rlhf-evolution-2024",
        "title": "RLHF 演化史: PPO → DPO → GRPO → 扩散 RL",
        "date": "2026-04-21",
        "tldr": "一条主线串起 RLHF 五年史:从在线 PPO 到离线 DPO,再到 GRPO 与扩散模型 RL 的统一框架。",
        "tags": ["rl", "rlhf", "ppo", "dpo", "diffusion"],
        "tutorial_meta": {"word_count": "12.4k", "reading_minutes": "80-110"},
        "_url": "tutorials/rlhf-evolution-2024/index.html",
    },
]


def run_smoke_test() -> int:
    out = Path("/tmp/build-smoke")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # we don't have a real assets/nav-header.html in /tmp — use the default
    nav_tmpl = DEFAULT_NAV_HEADER
    summary = build_site(SMOKE_POSTS, out, nav_tmpl)

    # verify every expected file exists
    expected = ["index.html", "papers.html", "tutorials.html", "tags.html"]
    tags = sorted({t for p in SMOKE_POSTS for t in p["tags"]})
    expected += [f"tags/{t}.html" for t in tags]
    missing = [name for name in expected if not (out / name).is_file()]
    if missing:
        print(f"FAIL: missing files: {missing}", file=sys.stderr)
        return 1

    # quick spot checks on rendered content
    idx = (out / "index.html").read_text(encoding="utf-8")
    if "post-card--paper" not in idx or "post-card--tutorial" not in idx:
        print("FAIL: index.html missing card classes", file=sys.stderr)
        return 1
    if "📄 论文" not in idx or "📘 教程" not in idx:
        print("FAIL: badges missing", file=sys.stderr)
        return 1
    if "12.4k 字" not in idx or "80–110 分钟" not in idx:
        print("FAIL: tutorial meta not rendered with en-dash", file=sys.stderr)
        return 1

    cloud = (out / "tags.html").read_text(encoding="utf-8")
    if 'class="tag-cloud"' not in cloud or "tag-cloud__count" not in cloud:
        print("FAIL: tag cloud markup off", file=sys.stderr)
        return 1
    if "font-size:" not in cloud:
        print("FAIL: tag cloud font-size missing", file=sys.stderr)
        return 1

    tag_page = (out / "tags" / "diffusion.html").read_text(encoding="utf-8")
    if "../papers/" not in tag_page or "../tags.html" not in tag_page:
        print("FAIL: tag page paths not relative", file=sys.stderr)
        return 1
    if "../assets/style.css" not in tag_page:
        print("FAIL: tag page css path wrong", file=sys.stderr)
        return 1

    # idempotency: rebuild and diff
    first = {name: (out / name).read_bytes() for name in expected}
    build_site(SMOKE_POSTS, out, nav_tmpl)
    second = {name: (out / name).read_bytes() for name in expected}
    if first != second:
        print("FAIL: build not idempotent", file=sys.stderr)
        return 1

    print("OK")
    print(
        f"Generated: {summary['pages']} pages from "
        f"{summary['papers']} papers + {summary['tutorials']} tutorials, "
        f"with {summary['tags']} unique tags"
    )
    return 0


# ---------------------------------------------------------------------------
# entry point

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the paper-reading static site.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root containing papers/, tutorials/, assets/ (default: CWD)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an in-memory smoke test against /tmp/build-smoke/ and exit.",
    )
    args = parser.parse_args(argv)

    if args.smoke_test:
        return run_smoke_test()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: root {root} is not a directory", file=sys.stderr)
        return 2

    posts = discover_posts(root)
    if not posts:
        print("WARN: no posts discovered — generated pages will be empty", file=sys.stderr)

    nav_tmpl = load_nav_header(root)
    summary = build_site(posts, root, nav_tmpl)
    print(
        f"Generated: {summary['pages']} pages from "
        f"{summary['papers']} papers + {summary['tutorials']} tutorials, "
        f"with {summary['tags']} unique tags"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
