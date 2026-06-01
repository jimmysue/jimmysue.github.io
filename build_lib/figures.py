"""Add class='zoomable' to <img> inside <figure>, for the lightbox JS."""
from __future__ import annotations

import re


# Match <figure ...>...</figure> non-greedy, then transform <img> tags inside.
_FIGURE_RE = re.compile(r"<figure([^>]*)>(.*?)</figure>", re.DOTALL)


def tag_for_lightbox(html: str) -> str:
    """Add class='zoomable' to every <img> inside <figure>.

    Idempotent: if class='zoomable' already present, leave it alone.
    """
    def _fig_sub(m: re.Match) -> str:
        fig_attrs = m.group(1)
        inner = m.group(2)
        new_inner = _retag_imgs(inner)
        return f"<figure{fig_attrs}>{new_inner}</figure>"

    return _FIGURE_RE.sub(_fig_sub, html)


_IMG_OPEN_RE = re.compile(r"<img\b([^>]*?)/?>", re.DOTALL)
_CLASS_ATTR_RE = re.compile(r'''\bclass\s*=\s*(["'])([^"']*)\1''')


def _retag_imgs(fragment: str) -> str:
    def _sub(m: re.Match) -> str:
        attrs = m.group(1)
        cm = _CLASS_ATTR_RE.search(attrs)
        if cm:
            classes = cm.group(2).split()
            if "zoomable" in classes:
                return m.group(0)  # already has it
            classes.append("zoomable")
            # Replace the matched class="..." (or class='...') with normalised double-quoted form
            new_attrs = _CLASS_ATTR_RE.sub(f'class="{" ".join(classes)}"', attrs, count=1)
        else:
            new_attrs = ' class="zoomable"' + attrs
        return f"<img{new_attrs}>"

    return _IMG_OPEN_RE.sub(_sub, fragment)
