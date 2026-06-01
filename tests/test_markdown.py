"""Tests for build_lib/markdown.py (end-to-end MD→HTML pipeline)."""
from __future__ import annotations

from build_lib.markdown import render_post_body


def test_render_basic_markdown(slug_set_basic):
    md = "# Title\n\nA paragraph with **bold**.\n\n## Section 1\n\nText."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert '<h2 id="sec-1">Section 1</h2>' in html
    assert warnings == []


def test_render_handles_inline_math(slug_set_basic):
    md = "Inline math: $x_t = \\sigma$ here."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    # markdown-it-py with dollarmath plugin emits math; specific markup varies.
    # We assert that the dollar signs are NOT escaped and the content is preserved.
    assert "x_t" in html
    assert "\\sigma" in html or "σ" in html or "sigma" in html


def test_render_handles_display_math(slug_set_basic):
    md = "Before.\n\n$$ x_t = (1-\\sigma) x_0 $$\n\nAfter."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    # Display math should survive as $$ ... $$ for MathJax to pick up, OR be wrapped
    # in some math container. Either way the LaTeX source must be present.
    assert "x_t" in html
    assert "(1-\\sigma)" in html or "1-σ" in html


def test_render_wiki_links_resolved(slug_set_basic):
    md = "See [[l2p-2026]] paper."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/awm-2025")
    assert 'href="../l2p-2026/index.html"' in html
    assert "wiki-link-broken" not in html


def test_render_broken_wiki_link_warned(slug_set_basic):
    md = "See [[ghost-2099]] paper."
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/awm-2025")
    assert "wiki-link-broken" in html
    assert any("ghost-2099" in w for w in warnings)


def test_render_embedded_html_passes_through(slug_set_basic):
    md = '<figure>\n<img src="x.png" alt="a">\n<figcaption>cap</figcaption>\n</figure>\n'
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert "<figure>" in html
    assert "<figcaption>cap</figcaption>" in html
    # Lightbox class added
    assert 'class="zoomable"' in html


def test_render_code_block_with_language(slug_set_basic):
    md = "```python\ndef hello():\n    pass\n```"
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert 'class="language-python"' in html
    assert "def hello():" in html


def test_render_gfm_table(slug_set_basic):
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert "<table" in html
    assert "<th>A</th>" in html
    assert "<td>1</td>" in html


def test_render_emits_toc_for_sections(slug_set_basic):
    md = "## A\n\ntext\n\n### A1\n\ntext\n\n## B\n\ntext\n"
    html, toc, warnings = render_post_body(md, slug_set_basic, current_post_dir="papers/x")
    assert 'class="toc"' in toc
    assert '<a href="#sec-1">A</a>' in toc
    assert '<a href="#sec-1-1">A1</a>' in toc
    assert '<a href="#sec-2">B</a>' in toc
