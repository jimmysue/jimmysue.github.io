#!/usr/bin/env python3
"""
migrate.py — generate `meta.json` for every paper/tutorial card found in the
top-level `index.html`.

Run from the workspace root (the directory containing `index.html`,
`papers/`, and `tutorials/`).

Writes:
  papers/<slug>/meta.json
  tutorials/<slug>/meta.json

Schema (per file):
  {
    "type": "paper" | "tutorial",
    "slug": "<slug>",
    "title": "<canonical <h1> from the per-doc index.html>",
    "date": "YYYY-MM-DD",
    "tldr": "<tldr text from the card>",
    "tags": [],
    "tutorial_meta": null | { "word_count": "...", "reading_minutes": "..." }
  }

Cards with no matching folder are reported but NOT written. Folders with
no card are skipped (no meta.json) with a warning.

Stdlib only. No external deps.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


# ----------------------------------------------------------------------
# HTML helpers
# ----------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_tags(s: str) -> str:
    """Strip HTML tags and collapse whitespace; unescape entities."""
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = WS_RE.sub(" ", s).strip()
    return s


# ----------------------------------------------------------------------
# Card parser — pull out each <a class="paper-card"> / <a class="tutorial-card">
# ----------------------------------------------------------------------

class CardParser(HTMLParser):
    """
    Captures every <a class="paper-card"> / <a class="tutorial-card"> with its
    inner HTML. We don't try to parse the inner blocks here — strip_tags +
    regex is enough since the structure is simple and stable.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.cards: list[dict] = []
        self._depth = 0          # depth inside the current card <a>
        self._current: dict | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")
        if tag == "a" and self._current is None and cls in ("paper-card", "tutorial-card"):
            self._current = {
                "class": cls,
                "href": attrs_d.get("href", ""),
            }
            self._depth = 1
            self._buf = []
            return

        if self._current is not None:
            if tag == "a":
                self._depth += 1
            # Re-emit the start tag so we can post-process inner HTML.
            attr_str = "".join(f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs)
            self._buf.append(f"<{tag}{attr_str}>")

    def handle_startendtag(self, tag, attrs):
        if self._current is not None:
            attr_str = "".join(f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs)
            self._buf.append(f"<{tag}{attr_str}/>")

    def handle_endtag(self, tag):
        if self._current is None:
            return
        if tag == "a":
            self._depth -= 1
            if self._depth == 0:
                self._current["inner"] = "".join(self._buf)
                self.cards.append(self._current)
                self._current = None
                self._buf = []
                return
        self._buf.append(f"</{tag}>")

    def handle_data(self, data):
        if self._current is not None:
            self._buf.append(data)

    def handle_entityref(self, name):
        if self._current is not None:
            self._buf.append(f"&{name};")

    def handle_charref(self, name):
        if self._current is not None:
            self._buf.append(f"&#{name};")


# ----------------------------------------------------------------------
# Field extraction from one card's inner HTML
# ----------------------------------------------------------------------

DATE_RE = re.compile(r'<div\s+class="date">(.*?)</div>', re.DOTALL)
H3_RE = re.compile(r'<h3[^>]*>(.*?)</h3>', re.DOTALL)
TLDR_RE = re.compile(r'<p\s+class="tldr">(.*?)</p>', re.DOTALL)


def parse_card_fields(inner: str, type_: str) -> dict:
    date_raw = ""
    word_count = None
    reading_minutes = None
    title = ""
    tldr = ""

    m = DATE_RE.search(inner)
    if m:
        date_full = strip_tags(m.group(1))
        # tutorial-card date can be "2026-05-17 · 10.8k 字 · 80–110 分钟"
        parts = [p.strip() for p in re.split(r"·|•", date_full) if p.strip()]
        if parts:
            date_raw = parts[0]
        if type_ == "tutorial" and len(parts) >= 2:
            # part[1] like "10.8k 字"  →  "10.8k"
            wc_m = re.match(r"([\w.]+)", parts[1])
            if wc_m:
                word_count = wc_m.group(1)
        if type_ == "tutorial" and len(parts) >= 3:
            # part[2] like "80–110 分钟"  →  "80-110" (normalise en-dash to '-')
            rm_m = re.match(r"([\d–—\-]+)", parts[2])
            if rm_m:
                reading_minutes = rm_m.group(1).replace("–", "-").replace("—", "-")

    m = H3_RE.search(inner)
    if m:
        title = strip_tags(m.group(1))

    m = TLDR_RE.search(inner)
    if m:
        tldr = strip_tags(m.group(1))

    return {
        "date": date_raw,
        "title": title,
        "tldr": tldr,
        "word_count": word_count,
        "reading_minutes": reading_minutes,
    }


# ----------------------------------------------------------------------
# Read canonical <h1> from per-doc index.html
# ----------------------------------------------------------------------

H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.DOTALL | re.IGNORECASE)


def read_h1(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = H1_RE.search(text)
    if not m:
        return None
    return strip_tags(m.group(1))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def slug_from_href(href: str) -> tuple[str, str] | None:
    """Return (type, slug) from an href like 'papers/awm-2025/index.html'."""
    parts = href.strip("/").split("/")
    if len(parts) < 2:
        return None
    if parts[0] == "papers":
        return ("paper", parts[1])
    if parts[0] == "tutorials":
        return ("tutorial", parts[1])
    return None


def main() -> int:
    root = Path.cwd()
    index_html = root / "index.html"
    if not index_html.exists():
        print(f"ERROR: index.html not found in {root}", file=sys.stderr)
        return 1

    parser = CardParser()
    parser.feed(index_html.read_text(encoding="utf-8"))

    seen_slugs: dict[str, set[str]] = {"paper": set(), "tutorial": set()}
    written = 0

    for card in parser.cards:
        ts = slug_from_href(card["href"])
        if ts is None:
            print(f"WARN: skipping card with unrecognized href: {card['href']}", file=sys.stderr)
            continue
        type_, slug = ts

        fields = parse_card_fields(card["inner"], type_)

        # Try to read canonical title from per-doc index.html (more complete
        # than the card <h3> which may be truncated).
        sub = root / (f"{type_}s") / slug / "index.html"
        if not sub.exists():
            print(f"WARN: card references missing file: {sub.relative_to(root)} — skipping", file=sys.stderr)
            continue

        canonical_h1 = read_h1(sub)
        title = canonical_h1 if canonical_h1 else fields["title"]

        if type_ == "tutorial":
            tutorial_meta = {
                "word_count": fields["word_count"] or "",
                "reading_minutes": fields["reading_minutes"] or "",
            }
        else:
            tutorial_meta = None

        meta = {
            "type": type_,
            "slug": slug,
            "title": title,
            "date": fields["date"],
            "tldr": fields["tldr"],
            "tags": [],
            "tutorial_meta": tutorial_meta,
        }

        out_path = sub.parent / "meta.json"
        out_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        seen_slugs[type_].add(slug)
        written += 1
        print(f"wrote {out_path.relative_to(root)}")

    # Warn about folders that exist but have no card.
    for sub_kind, subdir in (("paper", "papers"), ("tutorial", "tutorials")):
        d = root / subdir
        if not d.is_dir():
            continue
        for entry in sorted(d.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name not in seen_slugs[sub_kind]:
                print(
                    f"WARN: {subdir}/{entry.name}/ has no matching card in index.html — no meta.json written",
                    file=sys.stderr,
                )

    print(f"\nDone. Wrote {written} meta.json file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
