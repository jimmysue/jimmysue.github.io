"""Tests for build_lib/post_assembly.py."""
from __future__ import annotations

from build_lib.post_assembly import assemble_post_page


def test_assembled_page_has_doctype():
    out = assemble_post_page(
        meta={"title": "Test", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<h1>X</h1>",
        toc_html="",
        nav_html='<nav class="site-nav"></nav>',
        asset_prefix="../../",
    )
    assert out.startswith("<!doctype html>") or out.startswith("<!DOCTYPE html>")
    assert "<html" in out and "</html>" in out


def test_assembled_page_includes_title():
    out = assemble_post_page(
        meta={"title": "My Paper Title", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<h1>X</h1>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert "<title>My Paper Title" in out


def test_assembled_page_loads_mathjax_highlightjs_lightbox():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert "mathjax" in out.lower()
    assert "highlight" in out.lower()
    assert "lightbox" in out.lower()


def test_assembled_page_uses_asset_prefix():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert '"../../assets/style.css"' in out
    assert '"../../assets/lightbox.js"' in out


def test_assembled_page_includes_toc_when_provided():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html='<nav class="toc"><ul><li><a href="#sec-1">A</a></li></ul></nav>',
        nav_html="",
        asset_prefix="../../",
    )
    assert 'class="toc"' in out


def test_assembled_page_includes_nav():
    out = assemble_post_page(
        meta={"title": "T", "type": "paper", "slug": "x", "date": "2026-01-01",
              "tldr": "y", "tags": []},
        body_html="<p>x</p>",
        toc_html="",
        nav_html='<nav class="site-nav">HEADER</nav>',
        asset_prefix="../../",
    )
    assert 'class="site-nav"' in out
    assert "HEADER" in out


def test_title_html_special_chars_are_escaped():
    """Title containing <, &, " must be HTML-escaped in the <title> tag."""
    out = assemble_post_page(
        meta={"title": '<b>Bold</b> & "Quoted"'},
        body_html="<p>x</p>",
        toc_html="",
        nav_html="",
        asset_prefix="../../",
    )
    assert "<title>&lt;b&gt;Bold&lt;/b&gt; &amp; &quot;Quoted&quot;</title>" in out
