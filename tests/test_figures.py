"""Tests for build_lib/figures.py."""
from __future__ import annotations

from build_lib.figures import tag_for_lightbox


def test_tag_adds_zoomable_class():
    html = '<figure><img src="x.png" alt="a"><figcaption>cap</figcaption></figure>'
    out = tag_for_lightbox(html)
    assert 'class="zoomable"' in out
    assert '<img class="zoomable" src="x.png" alt="a">' in out


def test_tag_preserves_existing_img_class():
    html = '<figure><img class="big" src="x.png"></figure>'
    out = tag_for_lightbox(html)
    assert "big" in out
    assert "zoomable" in out


def test_tag_skips_img_outside_figure():
    html = '<p><img src="inline.png" alt="x"></p>'
    out = tag_for_lightbox(html)
    assert "zoomable" not in out
    assert '<img src="inline.png" alt="x">' in out


def test_tag_multiple_figures():
    html = (
        '<figure><img src="a.png"><figcaption>A</figcaption></figure>'
        '<p>between</p>'
        '<figure><img src="b.png"></figure>'
    )
    out = tag_for_lightbox(html)
    # both <img> inside figures should get zoomable
    assert out.count('class="zoomable"') == 2


def test_tag_already_zoomable_unchanged():
    """If a <figure> > img already has class='zoomable', don't duplicate."""
    html = '<figure><img class="zoomable" src="x.png"></figure>'
    out = tag_for_lightbox(html)
    assert out.count("zoomable") == 1
