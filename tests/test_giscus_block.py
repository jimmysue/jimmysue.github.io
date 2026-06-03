"""Tests for build._giscus_block() — pure render helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("build_mod", repo_root / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = {
    "repo": "owner/name",
    "repo_id": "R_kgDOSAMPLE",
    "category": "Announcements",
    "category_id": "DIC_kwDOSAMPLE",
    "mapping": "pathname",
    "strict": "0",
    "theme": "light",
    "reactions_enabled": "1",
    "emit_metadata": "0",
    "loading": "lazy",
    "lang": "zh-CN",
    "input_position": "bottom",
}


def test_giscus_block_wraps_in_section():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert '<section class="comments">' in html
    assert "</section>" in html
    assert "<h2>" in html


def test_giscus_block_includes_all_data_attrs():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert 'data-repo="owner/name"' in html
    assert 'data-repo-id="R_kgDOSAMPLE"' in html
    assert 'data-category="Announcements"' in html
    assert 'data-category-id="DIC_kwDOSAMPLE"' in html
    assert 'data-mapping="pathname"' in html
    assert 'data-strict="0"' in html
    assert 'data-theme="light"' in html
    assert 'data-reactions-enabled="1"' in html
    assert 'data-emit-metadata="0"' in html
    assert 'data-input-position="bottom"' in html
    assert 'data-loading="lazy"' in html
    assert 'data-lang="zh-CN"' in html


def test_giscus_block_script_src_and_attrs():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert 'src="https://giscus.app/client.js"' in html
    assert 'crossorigin="anonymous"' in html
    assert "async" in html


def test_giscus_block_includes_privacy_note():
    build = _load_build()
    html = build._giscus_block(CFG)
    assert "GitHub Discussions" in html
    assert 'class="comments__note"' in html


def test_giscus_block_escapes_repo_field():
    """If a config value contained an HTML-special char, it must be escaped
    so the attribute string is well-formed."""
    build = _load_build()
    bad_cfg = dict(CFG, repo='owner/name" data-evil="1')
    html = build._giscus_block(bad_cfg)
    # The injected " must be escaped, so `data-repo` value is a single attr
    assert 'data-evil' not in html or '&quot;' in html


def test_build_omits_comments_when_frontmatter_says_false(tmp_path):
    """If a post has `comments: false`, its index.html must NOT contain the comments section."""
    build = _load_build()

    paper_root = tmp_path / "papers" / "_optouttest-2026"
    paper_root.mkdir(parents=True)
    (paper_root / "index.md").write_text(
        "---\n"
        "type: paper\n"
        "slug: _optouttest-2026\n"
        'title: "Opt-out test"\n'
        "date: 2026-06-03\n"
        "tldr: |\n"
        "  Opt-out test.\n"
        "concepts: [optouttest]\n"
        "comments: false\n"
        "---\n"
        "\n"
        "# Opt-out test\n"
        "\n"
        "Body text.\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "giscus-config.json").write_text(
        '{"repo": "x/y", "repo_id": "R_a", "category": "C", "category_id": "D_b",'
        ' "mapping": "pathname", "strict": "0", "theme": "light",'
        ' "reactions_enabled": "1", "emit_metadata": "0", "loading": "lazy",'
        ' "lang": "zh-CN", "input_position": "bottom"}',
        encoding="utf-8",
    )
    (tmp_path / "assets" / "nav-header.html").write_text(
        '<nav class="site-nav"></nav>', encoding="utf-8",
    )
    posts, _skip = build.discover_posts(tmp_path)
    assert any(p["slug"] == "_optouttest-2026" for p in posts), \
        f"discover_posts didn't find the test paper; found: {[p['slug'] for p in posts]}"
    nav_tmpl = build.load_nav_header(tmp_path)
    cfg = build._load_giscus_config(tmp_path)
    build.build_posts(posts, nav_tmpl, slug_set={p["slug"] for p in posts}, giscus_cfg=cfg)
    html = (paper_root / "index.html").read_text(encoding="utf-8")
    assert '<section class="comments">' not in html, \
        "frontmatter comments: false must suppress the comments section"


def test_build_includes_comments_by_default(tmp_path):
    """If frontmatter omits `comments`, the comments section IS rendered."""
    build = _load_build()
    paper_root = tmp_path / "papers" / "_default-2026"
    paper_root.mkdir(parents=True)
    (paper_root / "index.md").write_text(
        "---\n"
        "type: paper\n"
        "slug: _default-2026\n"
        'title: "Default test"\n'
        "date: 2026-06-03\n"
        "tldr: |\n"
        "  Default test.\n"
        "concepts: [defaulttest]\n"
        "---\n"
        "\n"
        "# Default test\n"
        "\n"
        "Body text.\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "giscus-config.json").write_text(
        '{"repo": "x/y", "repo_id": "R_a", "category": "C", "category_id": "D_b",'
        ' "mapping": "pathname", "strict": "0", "theme": "light",'
        ' "reactions_enabled": "1", "emit_metadata": "0", "loading": "lazy",'
        ' "lang": "zh-CN", "input_position": "bottom"}',
        encoding="utf-8",
    )
    (tmp_path / "assets" / "nav-header.html").write_text(
        '<nav class="site-nav"></nav>', encoding="utf-8",
    )
    posts, _skip = build.discover_posts(tmp_path)
    nav_tmpl = build.load_nav_header(tmp_path)
    cfg = build._load_giscus_config(tmp_path)
    build.build_posts(posts, nav_tmpl, slug_set={p["slug"] for p in posts}, giscus_cfg=cfg)
    html = (paper_root / "index.html").read_text(encoding="utf-8")
    assert '<section class="comments">' in html
    assert 'data-repo-id="R_a"' in html


def test_build_omits_comments_when_config_missing(tmp_path):
    """If giscus-config.json is absent, no comments rendered (no crash)."""
    build = _load_build()
    paper_root = tmp_path / "papers" / "_noconfig-2026"
    paper_root.mkdir(parents=True)
    (paper_root / "index.md").write_text(
        "---\n"
        "type: paper\n"
        "slug: _noconfig-2026\n"
        'title: "No config test"\n'
        "date: 2026-06-03\n"
        "tldr: |\n"
        "  No config.\n"
        "concepts: [noconfigtest]\n"
        "---\n"
        "\n"
        "# No config\n"
        "\n"
        "Body.\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    # NOTE: deliberately do NOT write giscus-config.json
    (tmp_path / "assets" / "nav-header.html").write_text(
        '<nav class="site-nav"></nav>', encoding="utf-8",
    )
    posts, _skip = build.discover_posts(tmp_path)
    nav_tmpl = build.load_nav_header(tmp_path)
    cfg = build._load_giscus_config(tmp_path)
    assert cfg is None
    build.build_posts(posts, nav_tmpl, slug_set={p["slug"] for p in posts}, giscus_cfg=cfg)
    html = (paper_root / "index.html").read_text(encoding="utf-8")
    assert '<section class="comments">' not in html
