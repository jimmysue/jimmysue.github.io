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
