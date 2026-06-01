#!/usr/bin/env python3
"""One-shot HTML → markdown migrator for paper-reading blog.

Reads each papers/<slug>/index.html + meta.json, emits papers/<slug>/index.md.
Same for tutorials/<slug>/. Backs up the original index.html → index.html.pre-migrate.

Usage:
    python3 migrate-md.py --convert         # write .md, keep .html.pre-migrate backups
    python3 migrate-md.py --cleanup --yes   # delete meta.json + .pre-migrate backups
    python3 migrate-md.py --dry-run         # show what would be written, don't write
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag


# Tags whose entire HTML we keep as a raw island (the markdown is HTML).
PRESERVE_AS_HTML = {
    ("figure", None),
    ("p", "math-translation"),
    ("p", "code-source"),
    ("table", None),  # markdown tables don't handle multi-line cells; keep HTML
}


def _matches_preserve(el: Tag) -> bool:
    name = el.name
    classes = el.get("class") or []
    for tag, cls in PRESERVE_AS_HTML:
        if name == tag and (cls is None or cls in classes):
            return True
    return False


def html_to_md(html: str, meta: dict[str, Any]) -> str:
    soup = BeautifulSoup(html, "html5lib")

    # Drop scripts and the right-rail TOC
    for s in soup.find_all("script"):
        s.decompose()
    for nav in soup.find_all("nav", class_="toc"):
        nav.decompose()
    # Drop site nav (build.py reinjects)
    for nav in soup.find_all("nav", class_="site-nav"):
        nav.decompose()

    main = soup.find("main")
    if main is None:
        # Some pages may not have <main>; fall back to <body>
        main = soup.find("body") or soup
    # Drop any .meta sidebar div inside main (we already have meta.json)
    for d in main.find_all("div", class_="meta"):
        d.decompose()

    blocks: list[str] = []
    for el in list(main.children):
        if isinstance(el, NavigableString):
            txt = str(el).strip()
            if txt:
                blocks.append(txt)
            continue
        if not isinstance(el, Tag):
            continue
        blocks.append(_convert_block(el))

    body_md = "\n\n".join(b for b in blocks if b)
    fm = _build_frontmatter(meta)
    return fm + "\n\n" + body_md + "\n"


def _convert_block(el: Tag) -> str:
    """Convert a top-level block element to markdown (or preserve as HTML)."""
    if _matches_preserve(el):
        # Lightly normalise: ensure surrounding whitespace
        return str(el)

    name = el.name

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = _inline_md(el)
        return f"{'#' * level} {text}"

    if name == "p":
        return _inline_md(el)

    if name == "ul":
        return _convert_list(el, ordered=False)
    if name == "ol":
        return _convert_list(el, ordered=True)

    if name == "blockquote":
        inner = "\n\n".join(_inline_md(child) for child in el.find_all(["p"], recursive=False))
        if not inner:
            inner = _inline_md(el)
        return "\n".join(f"> {line}" for line in inner.splitlines())

    if name == "pre":
        return _convert_code_block(el)

    if name == "section":
        # <section> wraps groups of headings — recurse into children
        return "\n\n".join(_convert_block(c) for c in el.children
                           if isinstance(c, Tag))

    if name == "hr":
        return "---"

    # Fallback: preserve as HTML
    return str(el)


def _convert_list(el: Tag, ordered: bool) -> str:
    lines = []
    for i, li in enumerate(el.find_all("li", recursive=False), start=1):
        prefix = f"{i}." if ordered else "-"
        # Children of li might be paragraphs/sub-lists
        text = _inline_md(li).strip()
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def _convert_code_block(pre: Tag) -> str:
    code = pre.find("code")
    if code is None:
        return str(pre)
    classes = code.get("class") or []
    lang = ""
    for c in classes:
        if c.startswith("language-"):
            lang = c[len("language-"):]
            break
    text = code.get_text()
    # Strip trailing newline duplication
    text = text.rstrip("\n")
    return f"```{lang}\n{text}\n```"


_INLINE_MAP = {
    "strong": ("**", "**"),
    "b": ("**", "**"),
    "em": ("*", "*"),
    "i": ("*", "*"),
    "code": ("`", "`"),
}


def _inline_md(el: Tag | NavigableString) -> str:
    """Convert inline HTML to markdown text."""
    if isinstance(el, NavigableString):
        return str(el)
    if not isinstance(el, Tag):
        return ""
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in _INLINE_MAP:
                open_, close_ = _INLINE_MAP[child.name]
                parts.append(f"{open_}{_inline_md(child)}{close_}")
            elif child.name == "a":
                href = child.get("href", "")
                text = _inline_md(child)
                parts.append(f"[{text}]({href})")
            elif child.name == "br":
                parts.append("\n")
            elif child.name == "img":
                src = child.get("src", "")
                alt = child.get("alt", "")
                parts.append(f"![{alt}]({src})")
            else:
                # Fallback: keep nested HTML
                parts.append(str(child))
    return "".join(parts).strip()


def _build_frontmatter(meta: dict[str, Any]) -> str:
    """Render frontmatter YAML from a meta.json dict."""
    out = {
        "type": meta["type"],
        "slug": meta["slug"],
        "title": meta["title"],
        "date": meta["date"],
        "tldr": meta["tldr"],
        "tags": meta.get("tags", []),
    }
    if meta.get("tutorial_meta"):
        out["tutorial"] = meta["tutorial_meta"]
    yaml_text = yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=1000)
    return f"---\n{yaml_text}---"


# ---------------------------------------------------------------------------
# CLI

def migrate_all(repo_root: Path, dry_run: bool = False) -> int:
    n = 0
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = repo_root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            html_path = sub / "index.html"
            meta_path = sub / "meta.json"
            md_out = sub / "index.md"
            if not html_path.is_file():
                continue
            if not meta_path.is_file():
                print(f"SKIP: {sub} (no meta.json)", file=sys.stderr)
                continue
            html_text = html_path.read_text(encoding="utf-8")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            md = html_to_md(html_text, meta)
            if dry_run:
                print(f"DRY: would write {md_out} ({len(md)} bytes)")
                n += 1
                continue
            # Backup the original HTML
            backup = sub / "index.html.pre-migrate"
            if not backup.exists():
                shutil.copy2(html_path, backup)
            md_out.write_text(md, encoding="utf-8")
            print(f"WROTE: {md_out}")
            n += 1
    return n


def cleanup_all(repo_root: Path) -> int:
    n = 0
    for kind, dirname in (("paper", "papers"), ("tutorial", "tutorials")):
        base = repo_root / dirname
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            for fname in ("meta.json", "index.html.pre-migrate"):
                f = sub / fname
                if f.is_file():
                    f.unlink()
                    print(f"DELETED: {f}")
                    n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate paper-reading HTML → markdown.")
    ap.add_argument("--root", default=".", help="Repo root (default: CWD)")
    ap.add_argument("--convert", action="store_true", help="Write index.md files.")
    ap.add_argument("--cleanup", action="store_true",
                    help="Delete meta.json and .pre-migrate backups.")
    ap.add_argument("--dry-run", action="store_true", help="Don't write, just report.")
    ap.add_argument("--yes", action="store_true",
                    help="Confirm destructive --cleanup.")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.convert:
        n = migrate_all(root, dry_run=args.dry_run)
        print(f"Migrated {n} posts ({'dry-run' if args.dry_run else 'written'}).")
        return 0
    if args.cleanup:
        if not args.yes:
            print("Refusing to --cleanup without --yes (destructive).", file=sys.stderr)
            return 2
        n = cleanup_all(root)
        print(f"Deleted {n} files.")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
