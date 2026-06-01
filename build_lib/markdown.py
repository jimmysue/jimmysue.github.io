"""End-to-end markdown → HTML body pipeline.

Flow:
  1. preprocess: [[slug]] → raw HTML anchors
  2. parse: markdown-it-py with plugins (front_matter, dollarmath, tables)
  3. post-process: inject sec-N IDs → resolve wiki-link href → tag figures
  4. extract TOC HTML separately

Returns: (body_html, toc_html, warnings)
"""
from __future__ import annotations

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

from build_lib.figures import tag_for_lightbox
from build_lib.headings import build_toc_html, inject_ids
from build_lib.wiki_links import preprocess as preprocess_wiki, resolve as resolve_wiki


def _make_parser() -> MarkdownIt:
    md = (
        MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False})
        .enable("table")
        .enable("strikethrough")
        # front_matter_plugin: suppresses any un-stripped frontmatter (prevents YAML
        # from rendering as body text with a leading <hr>). Belt-and-suspenders — caller
        # is expected to strip via build_lib.frontmatter.parse first.
        .use(front_matter_plugin)
        .use(
            dollarmath_plugin,
            allow_labels=False,
            allow_space=True,
            allow_digits=False,   # False prevents $5 / $10 currency from being parsed as math
            double_inline=False,
        )
    )
    return md


_PARSER = _make_parser()


def render_post_body(
    md_text: str,
    slug_set: set[str],
    current_post_dir: str,
) -> tuple[str, str, list[str]]:
    """Render markdown body (frontmatter already stripped) to HTML.

    Returns:
        body_html — the main content HTML
        toc_html  — the <nav class="toc"> block (may be "")
        warnings  — list of human-readable warnings (e.g. broken wiki links)
    """
    # 1. Preprocess wiki-links on raw text
    md_with_wiki_html = preprocess_wiki(md_text)

    # 2. Parse markdown
    body_html = _PARSER.render(md_with_wiki_html)

    # 3a. Inject heading IDs
    body_html = inject_ids(body_html)

    # 3b. Resolve wiki-link hrefs
    body_html, warnings = resolve_wiki(body_html, slug_set, current_post_dir)

    # 3c. Tag figures for lightbox
    body_html = tag_for_lightbox(body_html)

    # 4. Build TOC from injected IDs
    toc_html = build_toc_html(body_html)

    return body_html, toc_html, warnings
