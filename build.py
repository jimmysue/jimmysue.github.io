#!/usr/bin/env python3
"""Static site generator for the paper-reading blog.

Reads frontmatter from every `papers/<slug>/index.md` and
`tutorials/<slug>/index.md` and emits 5 page types: index.html, papers.html,
tutorials.html, tags.html, tags/<slug>.html.  Also renders each post's
index.md to index.html via build_lib.

Usage:
    python3 build.py                 # build the real site (CWD = repo root)
    python3 build.py --smoke-test    # render 3 fake posts to /tmp/build-smoke/
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

from build_lib.frontmatter import parse as parse_frontmatter
from build_lib.frontmatter import validate as validate_frontmatter
from build_lib.graph import (
    discover_concepts as graph_discover_concepts,
    extract_graph as graph_extract,
    render_concept_page as graph_render_concept_page,
    render_graph_page as graph_render_graph_page,
)
from build_lib.markdown import render_post_body
from build_lib.post_assembly import assemble_post_page

# ---------------------------------------------------------------------------
# constants

MAX_CARD_TAGS = 3

DEFAULT_NAV_HEADER = """\
<header class="site-nav">
  <div class="site-nav__inner">
    <a class="site-nav__brand" href="{HOME_URL}">paper-reading</a>
    <nav class="site-nav__links">
      <a class="site-nav__link{ACTIVE_HOME}" href="{HOME_URL}">首页</a>
      <a class="site-nav__link{ACTIVE_PAPERS}" href="{PAPERS_URL}">论文</a>
      <a class="site-nav__link{ACTIVE_TUTORIALS}" href="{TUTORIALS_URL}">教程</a>
      <a class="site-nav__link{ACTIVE_CONCEPTS}" href="{CONCEPTS_URL}">概念</a>
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


def discover_posts(root: Path) -> tuple[list[dict], int]:
    """Walk papers/<slug>/index.md and tutorials/<slug>/index.md.

    Returns (posts, skip_count) where skip_count is the number of dirs
    skipped due to missing index.md, parse errors, or validation errors.
    """
    posts: list[dict] = []
    skip_count = 0
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            md_path = sub / "index.md"
            if not md_path.is_file():
                print(f"WARN: skipping {sub} (no index.md)", file=sys.stderr)
                skip_count += 1
                continue
            try:
                text = md_path.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except Exception as e:
                print(f"WARN: skipping {md_path} (frontmatter parse error: {e})", file=sys.stderr)
                skip_count += 1
                continue
            errs = validate_frontmatter(meta, expected_slug=sub.name, expected_dir=dirname)
            if errs:
                print(f"WARN: skipping {md_path}:", file=sys.stderr)
                for err in errs:
                    print(f"  - {err}", file=sys.stderr)
                skip_count += 1
                continue
            # Phase 2: `concepts` is the canonical field; legacy `tags` accepted as fallback during transition.
            raw = meta.get("concepts") or meta.get("tags") or []
            meta["tags"] = [slugify_tag(t) for t in raw]  # keep internal key "tags" for now to minimize churn
            # `date` may have been parsed as datetime.date — normalise to ISO string
            if not isinstance(meta.get("date"), str):
                meta["date"] = str(meta["date"])
            # URL
            meta["_url"] = f"{dirname}/{sub.name}/index.html"
            meta["_md_path"] = md_path
            meta["_body_md"] = body
            # tutorial_meta — map from frontmatter `tutorial:` block (if any)
            meta["tutorial_meta"] = meta.get("tutorial") or None
            posts.append(meta)
    return posts, skip_count


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
    depth=2 → papers/<slug>/index.html or tutorials/<slug>/index.html
    """
    prefix = "../" * depth
    actives = {
        "home": "",
        "papers": "",
        "tutorials": "",
        "concepts": "",
    }
    if active in actives:
        actives[active] = " active"
    return nav_tmpl.format(
        HOME_URL=f"{prefix}index.html",
        PAPERS_URL=f"{prefix}papers.html",
        TUTORIALS_URL=f"{prefix}tutorials.html",
        CONCEPTS_URL=f"{prefix}concepts.html",
        ACTIVE_HOME=actives["home"],
        ACTIVE_PAPERS=actives["papers"],
        ACTIVE_TUTORIALS=actives["tutorials"],
        ACTIVE_CONCEPTS=actives["concepts"],
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
        f'<a class="tag-chip" href="{prefix}concepts/{esc(t)}.html">{esc(t)}</a>'
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
    body = render_head("概念 — paper-reading", "assets/style.css")
    body += render_nav(nav_tmpl, active="concepts", depth=0)
    body += '<main class="page page--tags">\n'
    body += "<h1>概念</h1>\n"
    body += f'<p class="page__intro">共 {len(counts)} 个概念。</p>\n'

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
                f'  <a class="tag-cloud__item" href="concepts/{esc(tag)}.html" '
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




# ---------------------------------------------------------------------------
# per-post HTML rendering


def build_posts(posts: list[dict], nav_tmpl: str, slug_set: set[str] | None = None) -> int:
    """Render each post's index.md to index.html. Returns len(posts) on success;
    raises on any I/O error (no partial-progress reporting).

    slug_set: full set of known slugs for wiki-link resolution.  When None,
    defaults to the slugs found in *posts* (correct for full-site builds).
    Pass the site-wide set explicitly when rendering a single-post subset so
    that cross-references to other papers resolve correctly.
    """
    if slug_set is None:
        slug_set = {p["slug"] for p in posts}
    total_warnings: list[str] = []
    for p in posts:
        md_path: Path = p["_md_path"]
        body_md: str = p["_body_md"]
        slug: str = p["slug"]
        # papers/<slug>/ → asset prefix "../../"
        current_dir = f"{md_path.parent.parent.name}/{slug}"
        body_html, toc_html, warnings = render_post_body(
            body_md, slug_set, current_post_dir=current_dir,
        )
        total_warnings.extend(warnings)
        nav_html = render_nav(nav_tmpl, active="papers" if p["type"] == "paper" else "tutorials",
                              depth=2)
        html_out = assemble_post_page(
            meta=p, body_html=body_html, toc_html=toc_html,
            nav_html=nav_html, asset_prefix="../../",
        )
        out_path = md_path.parent / "index.html"
        out_path.write_text(html_out, encoding="utf-8")

    # Dedup warnings (multiple [[broken-slug]] in same file produce duplicates)
    unique_warnings = list(dict.fromkeys(total_warnings))
    if unique_warnings:
        print(f"WARN: {len(unique_warnings)} unique build warning(s):", file=sys.stderr)
        for w in unique_warnings:
            print(f"  - {w}", file=sys.stderr)
    return len(posts)


# ---------------------------------------------------------------------------
# top-level build

def build_site(posts: list[dict], out_root: Path, nav_tmpl: str) -> dict:
    """Write all generated pages. Returns a small summary dict."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "concepts").mkdir(parents=True, exist_ok=True)

    n_pages = 0

    (out_root / "index.html").write_text(build_index(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1
    (out_root / "papers.html").write_text(build_papers(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1
    (out_root / "tutorials.html").write_text(build_tutorials(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1
    (out_root / "concepts.html").write_text(build_tags_cloud(posts, nav_tmpl), encoding="utf-8")
    n_pages += 1

    by_tag = _tag_index(posts)

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
    # Note: per-concept pages (concepts/<slug>.html) are written in Pass 5 of main(),
    # not in build_site(), so they are not present in smoke-test output.
    expected = ["index.html", "papers.html", "tutorials.html", "concepts.html"]
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

    cloud = (out / "concepts.html").read_text(encoding="utf-8")
    if 'class="tag-cloud"' not in cloud or "tag-cloud__count" not in cloud:
        print("FAIL: tag cloud markup off", file=sys.stderr)
        return 1
    if "font-size:" not in cloud:
        print("FAIL: tag cloud font-size missing", file=sys.stderr)
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
    parser.add_argument(
        "--post",
        metavar="SLUG",
        default=None,
        help="Render only a single post (papers/<slug> or tutorials/<slug>).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate frontmatter + wiki-links across all posts; "
             "exit non-zero on any warning. Don't write HTML.",
    )
    args = parser.parse_args(argv)

    if args.smoke_test:
        return run_smoke_test()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: root {root} is not a directory", file=sys.stderr)
        return 2

    posts, skip_count = discover_posts(root)
    if not posts:
        print("WARN: no posts discovered — generated pages will be empty", file=sys.stderr)

    if args.post:
        all_slug_set = {p["slug"] for p in posts}
        posts = [p for p in posts if p["slug"] == args.post]
        if not posts:
            print(f"ERROR: no post with slug {args.post!r}", file=sys.stderr)
            return 2
        # Build only this post, skip site-level pages.
        # Pass the site-wide slug_set so wiki-links to other posts resolve.
        nav_tmpl = load_nav_header(root)
        n = build_posts(posts, nav_tmpl, slug_set=all_slug_set)
        print(f"Rendered {n} post(s).")
        return 0

    if args.check:
        slug_set = {p["slug"] for p in posts}
        all_warnings: list[str] = []
        for p in posts:
            _, _, warns = render_post_body(
                p["_body_md"], slug_set,
                current_post_dir=f"{Path(p['_url']).parent}",
            )
            all_warnings.extend(warns)
        # Dedup wiki-link warnings (multiple [[broken-slug]] refs produce duplicates)
        all_warnings = list(dict.fromkeys(all_warnings))
        n_warn = len(all_warnings)
        if all_warnings:
            for w in all_warnings:
                print(f"WARN: {w}", file=sys.stderr)
        if n_warn or skip_count:
            print(
                f"FAIL: {skip_count} post(s) skipped due to frontmatter errors, "
                f"{n_warn} unique wiki-link warning(s).",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {len(posts)} posts validated.")
        return 0

    nav_tmpl = load_nav_header(root)
    summary = build_site(posts, root, nav_tmpl)
    print(
        f"Generated: {summary['pages']} pages from "
        f"{summary['papers']} papers + {summary['tutorials']} tutorials, "
        f"with {summary['tags']} unique tags"
    )
    n_posts = build_posts(posts, nav_tmpl)
    print(f"Rendered {n_posts} per-post HTML pages from markdown.")

    # Pass 4: knowledge graph extraction
    concepts_info = graph_discover_concepts(root)
    graph = graph_extract(posts, concepts_info)
    (root / "graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote graph.json: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges.")

    # Pass 5: render concepts/<slug>.html (aggregation pages)
    (root / "concepts").mkdir(parents=True, exist_ok=True)
    concept_slugs = sorted({n["slug"] for n in graph["nodes"] if n["type"] == "concept"})
    nav_concept = render_nav(nav_tmpl, active="concepts", depth=1)
    for slug in concept_slugs:
        meta = concepts_info.get(slug) or {
            "name": slug, "aliases": [], "parent": None, "body_md": "",
        }
        mentioning = [p for p in posts if slug in (p.get("concepts") or [])]
        page = graph_render_concept_page(slug, meta, mentioning, nav_concept)
        (root / "concepts" / f"{slug}.html").write_text(page, encoding="utf-8")
    print(f"Wrote {len(concept_slugs)} concept page(s).")

    # Pass 6: render /graph.html
    nav_graph = render_nav(nav_tmpl, active="", depth=0)
    graph_html = graph_render_graph_page(graph, nav_graph)
    (root / "graph.html").write_text(graph_html, encoding="utf-8")
    print("Wrote graph.html.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
