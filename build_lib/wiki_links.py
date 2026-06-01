"""Wiki-link [[slug]] preprocess (MD text) and resolve (HTML).

Two-step design:
  1. preprocess(md_text)  — regex replaces [[slug]] / [[slug|alias]] with
                           raw HTML; markdown-it-py passes it through.
  2. resolve(html, slug_set, current_post_dir)
                          — adds href= to known slugs, marks unknowns broken.
"""
from __future__ import annotations

import re


# slug = ASCII letters/digits/hyphens, no whitespace, no | or ]
WIKI_LINK_RE = re.compile(r"\[\[([A-Za-z0-9][A-Za-z0-9\-_]*)(?:\|([^\]]+))?\]\]")

# Find <a class="wiki-link" ... data-slug="x">text</a> for resolve()
WIKI_ANCHOR_RE = re.compile(
    r'<a class="wiki-link" data-slug="([^"]+)">([^<]*)</a>'
)


def preprocess(md_text: str) -> str:
    """Replace [[slug]] and [[slug|alias]] in markdown text with raw HTML anchors.

    The output anchor keeps `data-slug` so the post-process pass can
    resolve href= later.
    """
    def _sub(m: re.Match) -> str:
        slug = m.group(1)
        alias = m.group(2) if m.group(2) else slug
        return f'<a class="wiki-link" data-slug="{slug}">{alias}</a>'
    return WIKI_LINK_RE.sub(_sub, md_text)


def resolve(html: str, slug_set: set[str], current_post_dir: str) -> tuple[str, list[str]]:
    """Resolve wiki-link anchors. Returns (rewritten_html, warnings).

    `current_post_dir` is like "papers/awm-2025" — used to compute relative
    href (the target is always "../<target-slug>/index.html").

    Unknown slugs get class="wiki-link wiki-link-broken" and no href.
    """
    warnings: list[str] = []

    def _sub(m: re.Match) -> str:
        slug = m.group(1)
        text = m.group(2)
        if slug in slug_set:
            href = f"../{slug}/index.html"
            return (
                f'<a class="wiki-link" data-slug="{slug}" href="{href}">{text}</a>'
            )
        else:
            warnings.append(f"{current_post_dir}: broken wiki-link [[{slug}]]")
            return (
                f'<a class="wiki-link wiki-link-broken" data-slug="{slug}">{text}</a>'
            )

    return WIKI_ANCHOR_RE.sub(_sub, html), warnings
