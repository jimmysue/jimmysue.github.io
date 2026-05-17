#!/usr/bin/env python3
"""
inject-header.py — Idempotently inject a shared site nav header into every
paper and tutorial detail page.

For each `papers/<slug>/index.html` and `tutorials/<slug>/index.html`:
  1. Load assets/nav-header.html and substitute placeholders for depth-2
     (../../) detail pages. Active flags are all empty (no tab is active
     on a detail page).
  2. If <!-- NAV-START --> ... <!-- NAV-END --> already present, replace
     the block; otherwise insert just after the opening <body> tag.
  3. Strip the legacy "返回博客首页" back-link line if present.
  4. Write the file back as UTF-8.

Re-running is a no-op (REPLACE-only) — proving idempotency.

stdlib only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAV_SNIPPET_PATH = ROOT / "assets" / "nav-header.html"

# Markers in the nav snippet (must surround the block end-to-end).
NAV_START = "<!-- NAV-START -->"
NAV_END = "<!-- NAV-END -->"

# Regex to find the existing nav block (DOTALL: spans lines).
NAV_BLOCK_RE = re.compile(
    re.escape(NAV_START) + r".*?" + re.escape(NAV_END),
    re.DOTALL,
)

# Match the opening <body ...> tag (allow attributes).
BODY_OPEN_RE = re.compile(r"(<body\b[^>]*>)", re.IGNORECASE)

# Legacy back-link line. Conservative: anchor href must end with
# `../../index.html`, AND anchor text must contain "返回", "←", or "back"
# (case-insensitive for the latter).
BACK_LINK_RE = re.compile(
    r"^[ \t]*<p>\s*<a\s+href=\"\.\./\.\./index\.html\"\s*>"
    r"(?P<text>[^<]*)</a>\s*</p>\s*\r?\n?",
    re.MULTILINE,
)


def substitute_for_depth_2(snippet: str) -> str:
    """Substitute the placeholders for a depth-2 detail page."""
    replacements = {
        "{HOME_URL}": "../../index.html",
        "{PAPERS_URL}": "../../papers.html",
        "{TUTORIALS_URL}": "../../tutorials.html",
        "{TAGS_URL}": "../../tags.html",
        "{ACTIVE_PAPERS}": "",
        "{ACTIVE_TUTORIALS}": "",
        "{ACTIVE_TAGS}": "",
    }
    out = snippet
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def is_target_file(path: Path) -> bool:
    """Filter: only `papers/<slug>/index.html` and `tutorials/<slug>/index.html`.

    Reject anything inside raw/figures/repo/figures-raw subdirs, and the
    top-level index.html.
    """
    if path.name != "index.html":
        return False
    parts = path.relative_to(ROOT).parts
    # Must be exactly: papers/<slug>/index.html or tutorials/<slug>/index.html
    if len(parts) != 3:
        return False
    if parts[0] not in ("papers", "tutorials"):
        return False
    # Guardrails — should be excluded by the len(parts)==3 check, but
    # belt-and-suspenders:
    forbidden = {"raw", "figures", "repo", "figures-raw"}
    if any(part in forbidden for part in parts):
        return False
    return True


def remove_back_link(html: str) -> tuple[str, bool]:
    """Strip the legacy '← 返回博客首页' line. Returns (new_html, removed?)."""
    def matcher(m: re.Match) -> str:
        text = m.group("text")
        text_lower = text.lower()
        if ("返回" in text) or ("←" in text) or ("back" in text_lower):
            return ""  # drop the line
        return m.group(0)  # keep — anchor text didn't match the heuristic

    new_html, n = BACK_LINK_RE.subn(matcher, html)
    return new_html, (new_html != html)


def inject(html: str, nav_block: str) -> tuple[str, str]:
    """Inject (or replace) the nav block.

    Returns (new_html, action) where action is 'new' | 'replaced' | 'noop'.
    """
    if NAV_START in html and NAV_END in html:
        new_html, n = NAV_BLOCK_RE.subn(nav_block, html, count=1)
        if new_html == html:
            return html, "noop"
        return new_html, "replaced"

    # Insert after the first <body ...> tag.
    m = BODY_OPEN_RE.search(html)
    if not m:
        # No <body> — bail out as noop and let the caller log.
        return html, "noop"

    insert_at = m.end()
    # Insert on a new line right after <body>.
    new_html = html[:insert_at] + "\n" + nav_block + html[insert_at:]
    return new_html, "new"


def process_file(path: Path, nav_block: str) -> dict:
    """Process a single file. Returns a dict with counters/diagnostics."""
    result = {
        "path": path,
        "action": "noop",
        "back_link_removed": False,
        "error": None,
    }
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        result["error"] = f"read failed: {e}"
        return result

    # 1) Inject / replace nav block.
    new_html, action = inject(original, nav_block)
    result["action"] = action

    # 2) Strip legacy back-link.
    new_html, removed = remove_back_link(new_html)
    result["back_link_removed"] = removed

    if new_html == original:
        # Nothing changed — leave file untouched.
        return result

    try:
        path.write_text(new_html, encoding="utf-8")
    except OSError as e:
        result["error"] = f"write failed: {e}"
    return result


def main() -> int:
    if not NAV_SNIPPET_PATH.exists():
        print(f"ERROR: nav snippet not found at {NAV_SNIPPET_PATH}", file=sys.stderr)
        return 2

    snippet_raw = NAV_SNIPPET_PATH.read_text(encoding="utf-8").rstrip("\n")
    nav_block = substitute_for_depth_2(snippet_raw)

    # Sanity-check the snippet has both markers.
    if NAV_START not in nav_block or NAV_END not in nav_block:
        print(
            f"ERROR: nav snippet missing {NAV_START} or {NAV_END} markers",
            file=sys.stderr,
        )
        return 2

    candidates = sorted(
        list((ROOT / "papers").glob("*/index.html"))
        + list((ROOT / "tutorials").glob("*/index.html"))
    )

    counters = {"new": 0, "replaced": 0, "noop": 0, "back_link": 0, "error": 0}
    files_seen = 0

    for path in candidates:
        if not is_target_file(path):
            continue
        files_seen += 1
        res = process_file(path, nav_block)
        rel = path.relative_to(ROOT)
        if res["error"]:
            counters["error"] += 1
            print(f"ERROR: {rel} — {res['error']}", file=sys.stderr)
            continue
        action = res["action"]
        if action == "new":
            counters["new"] += 1
            label = "INJECT"
            tag = "(new)"
        elif action == "replaced":
            counters["replaced"] += 1
            label = "REPLACE"
            tag = ""
        else:
            counters["noop"] += 1
            label = "NOOP"
            tag = ""
        if res["back_link_removed"]:
            counters["back_link"] += 1
            tag = (tag + " [back-link removed]").strip()
        line = f"{label}: {rel}"
        if tag:
            line += f" {tag}"
        print(line)

    print(
        f"Processed {files_seen} files: "
        f"{counters['new']} new, "
        f"{counters['replaced']} replaced, "
        f"{counters['noop']} noop, "
        f"{counters['back_link']} back-links removed"
        + (f", {counters['error']} errors" if counters['error'] else "")
    )
    return 0 if counters["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
