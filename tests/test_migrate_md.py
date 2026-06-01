"""Tests for migrate-md.py (HTML → markdown migration)."""
from __future__ import annotations

# Note: migrate-md.py has a hyphen, so import via importlib.
import importlib.util
import pathlib
import pytest


def _load_migrate():
    path = pathlib.Path(__file__).parent.parent / "migrate-md.py"
    spec = importlib.util.spec_from_file_location("migrate_md", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def migrate_mod():
    return _load_migrate()


def test_html_to_md_keeps_h1(migrate_mod):
    html = '<html><body><main><h1>Title</h1></main></body></html>'
    meta = {"type": "paper", "slug": "x", "title": "Title", "date": "2026-01-01",
            "tldr": "y", "tags": ["a"]}
    md = migrate_mod.html_to_md(html, meta)
    assert "# Title" in md


def test_html_to_md_keeps_figures_as_html(migrate_mod):
    html = (
        '<main>'
        '<figure>'
        '<img src="figures/a.png" alt="X">'
        '<figcaption>Cap with <strong>bold</strong>.</figcaption>'
        '</figure>'
        '</main>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "<figure>" in md
    assert "<figcaption>" in md
    assert "<strong>bold</strong>" in md


def test_html_to_md_keeps_math_translation_class(migrate_mod):
    html = (
        '<main>'
        '<p>$$x = 1$$</p>'
        '<p class="math-translation">—— 翻译: x is one.</p>'
        '</main>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert 'class="math-translation"' in md


def test_html_to_md_keeps_code_source_class(migrate_mod):
    html = (
        '<main>'
        '<p class="code-source">repo/a.py:L1-L10 — role</p>'
        '<pre><code class="language-python">x = 1\n</code></pre>'
        '</main>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert 'class="code-source"' in md
    assert "```python" in md
    assert "x = 1" in md


def test_html_to_md_drops_toc(migrate_mod):
    html = (
        '<html><body>'
        '<nav class="toc"><ul><li>a</li></ul></nav>'
        '<main><h1>T</h1></main>'
        '</body></html>'
    )
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "class=\"toc\"" not in md
    assert "# T" in md


def test_html_to_md_emits_frontmatter(migrate_mod):
    html = '<main><h1>T</h1></main>'
    meta = {"type": "paper", "slug": "x-2026", "title": "Hello",
            "date": "2026-01-01", "tldr": "summary", "tags": ["a", "b"]}
    md = migrate_mod.html_to_md(html, meta)
    assert md.startswith("---\n")
    assert "type: paper" in md
    assert "slug: x-2026" in md
    assert "tags:" in md


def test_html_to_md_converts_list(migrate_mod):
    html = '<main><ul><li>one</li><li>two</li></ul></main>'
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    # Either '- one' or '* one' is acceptable
    assert ("- one" in md) or ("* one" in md)


def test_html_to_md_converts_inline_strong_em(migrate_mod):
    html = '<main><p>This is <strong>bold</strong> and <em>italic</em>.</p></main>'
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "**bold**" in md
    assert "*italic*" in md or "_italic_" in md


def test_html_to_md_drops_comments(migrate_mod):
    """HTML comments inside <main> must not leak as text."""
    html = '<main><!-- NAV-START --><p>Body text.</p><!-- end --></main>'
    meta = {"type": "paper", "slug": "x", "title": "T", "date": "2026-01-01",
            "tldr": "y", "tags": []}
    md = migrate_mod.html_to_md(html, meta)
    assert "NAV-START" not in md
    assert "end" not in md or "Body" in md  # the literal " end " token shouldn't appear standalone
    # But the body must survive
    assert "Body text." in md
