"""Tests for build_lib/headings.py."""
from __future__ import annotations

from build_lib.headings import inject_ids, build_toc_html


def test_inject_ids_single_h2():
    html = "<h2>First</h2>"
    out = inject_ids(html)
    assert '<h2 id="sec-1">First</h2>' in out


def test_inject_ids_h2_increments():
    html = "<h2>A</h2><p>x</p><h2>B</h2>"
    out = inject_ids(html)
    assert 'id="sec-1">A</h2>' in out
    assert 'id="sec-2">B</h2>' in out


def test_inject_ids_h3_resets_per_h2():
    html = "<h2>A</h2><h3>a1</h3><h3>a2</h3><h2>B</h2><h3>b1</h3>"
    out = inject_ids(html)
    assert 'id="sec-1">A</h2>' in out
    assert 'id="sec-1-1">a1</h3>' in out
    assert 'id="sec-1-2">a2</h3>' in out
    assert 'id="sec-2">B</h2>' in out
    assert 'id="sec-2-1">b1</h3>' in out


def test_inject_ids_h3_before_first_h2_skipped():
    """An h3 with no enclosing h2 has no parent number — skip ID injection."""
    html = "<h3>orphan</h3><h2>A</h2><h3>child</h3>"
    out = inject_ids(html)
    # Orphan h3 stays unchanged
    assert "<h3>orphan</h3>" in out
    # h2 gets sec-1, child h3 gets sec-1-1
    assert 'id="sec-1">A</h2>' in out
    assert 'id="sec-1-1">child</h3>' in out


def test_inject_ids_preserves_existing_attributes():
    """If author writes <h2 class="foo">X</h2>, the class should be kept."""
    html = '<h2 class="custom">X</h2>'
    out = inject_ids(html)
    assert 'id="sec-1"' in out
    assert 'class="custom"' in out
    assert ">X</h2>" in out
    # Exactly one id= attribute (catches duplicate-attribute bug)
    assert out.count('id=') == 1


def test_inject_ids_skips_h1_h4():
    html = "<h1>Title</h1><h2>A</h2><h4>note</h4>"
    out = inject_ids(html)
    # h1 unchanged
    assert "<h1>Title</h1>" in out
    # h4 unchanged
    assert "<h4>note</h4>" in out
    # h2 gets id
    assert 'id="sec-1"' in out


# --- TOC ---

def test_inject_ids_skips_heading_with_existing_id():
    """If h2 has id=, don't inject another. Output remains valid HTML."""
    html = '<h2 id="custom">X</h2><h2>Y</h2>'
    out = inject_ids(html)
    # h2 with custom id: untouched
    assert '<h2 id="custom">X</h2>' in out
    # Next h2 still gets sec-2 (counter incremented past the custom one)
    assert 'id="sec-2"' in out
    # Critically: no duplicate id attributes
    assert out.count('id="') == 2  # one custom, one sec-2


def test_inject_ids_is_idempotent():
    """Calling inject_ids twice on the same input produces the same output."""
    html = "<h2>A</h2><h3>a1</h3><h2>B</h2>"
    once = inject_ids(html)
    twice = inject_ids(once)
    assert once == twice


# --- TOC ---

def test_build_toc_html_basic():
    html = (
        '<h2 id="sec-1">Intro</h2>'
        '<h3 id="sec-1-1">Setup</h3>'
        '<h2 id="sec-2">Method</h2>'
    )
    toc = build_toc_html(html)
    assert 'class="toc"' in toc
    assert '<a href="#sec-1">Intro</a>' in toc
    assert '<a href="#sec-1-1">Setup</a>' in toc
    assert '<a href="#sec-2">Method</a>' in toc
    # h2 entries should have class h2, h3 entries class h3 (spec §TOC contract)
    assert 'class="h2"' in toc
    assert 'class="h3"' in toc


def test_build_toc_html_strips_inline_markup():
    """If h2 contains <strong>, the TOC link text is plain."""
    html = '<h2 id="sec-1">Intro <strong>important</strong></h2>'
    toc = build_toc_html(html)
    assert "Intro important" in toc
    # No nested <strong> in the TOC link
    assert "<strong>" not in toc


def test_build_toc_html_empty_when_no_headings():
    toc = build_toc_html("<p>no headings here</p>")
    assert toc == "" or 'class="toc"' not in toc
