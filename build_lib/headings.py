"""Heading ID injection (sec-N / sec-N-M) + right-rail TOC HTML."""
from __future__ import annotations

import re


# Match <h2> or <h3> open tag with optional existing attributes.
# Capture: 1=tag name (h2 or h3), 2=existing attrs (may be empty), 3=text content
H2_OR_H3_RE = re.compile(
    r"<(h[23])([^>]*)>(.*?)</\1>",
    re.DOTALL,
)

# For TOC: match injected <h2 id="sec-N">text</h2> and <h3 id="sec-N-M">text</h3>
TOC_HEADING_RE = re.compile(
    r'<(h[23])\s+id="(sec-[\d\-]+)"[^>]*>(.*?)</\1>',
    re.DOTALL,
)

# Strip inline HTML for TOC link text
INLINE_TAG_RE = re.compile(r"<[^>]+>")


def inject_ids(html: str) -> str:
    """Walk h2/h3 in document order, assign sec-N / sec-N-M IDs."""
    h2_count = 0
    h3_count = 0

    def _sub(m: re.Match) -> str:
        nonlocal h2_count, h3_count
        tag = m.group(1)
        attrs = m.group(2) or ""
        text = m.group(3)
        attrs_str = attrs.strip()

        # If this heading already has an id=, leave it alone (idempotency).
        # Still update counters so subsequent headings stay consistent.
        if re.search(r'\bid\s*=', attrs_str):
            if tag == "h2":
                h2_count += 1
                h3_count = 0
            return m.group(0)

        if tag == "h2":
            h2_count += 1
            h3_count = 0
            heading_id = f"sec-{h2_count}"
        else:  # h3
            if h2_count == 0:
                # Orphan h3 before any h2 — leave unchanged
                return m.group(0)
            h3_count += 1
            heading_id = f"sec-{h2_count}-{h3_count}"
        # Preserve any existing attributes (class, etc.), prepend id=
        if attrs_str:
            new_open = f'<{tag} id="{heading_id}" {attrs_str}>'
        else:
            new_open = f'<{tag} id="{heading_id}">'
        return f"{new_open}{text}</{tag}>"

    return H2_OR_H3_RE.sub(_sub, html)


def build_toc_html(html: str) -> str:
    """Build the right-rail <nav class='toc'> block from already-injected IDs.

    Returns "" if no headings found.
    """
    items: list[tuple[str, str, str]] = []
    for m in TOC_HEADING_RE.finditer(html):
        tag, heading_id, text = m.group(1), m.group(2), m.group(3)
        clean_text = INLINE_TAG_RE.sub("", text).strip()
        items.append((tag, heading_id, clean_text))

    if not items:
        return ""

    lines = ['<nav class="toc" aria-label="目录">']
    lines.append('  <div class="toc-title">目录 / TOC</div>')
    lines.append('  <ul>')
    for tag, heading_id, text in items:
        lines.append(
            f'    <li class="{tag}"><a href="#{heading_id}">{text}</a></li>'
        )
    lines.append("  </ul>")
    lines.append("</nav>")
    return "\n".join(lines) + "\n"
