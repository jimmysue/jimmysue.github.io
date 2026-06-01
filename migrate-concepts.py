#!/usr/bin/env python3
"""One-shot migration: tags→concepts + paper.code_url→repos in markdown frontmatter.

Usage:
    python3 migrate-concepts.py --convert   # write changes, backup .pre-concepts
    python3 migrate-concepts.py --cleanup --yes  # delete .pre-concepts backups
    python3 migrate-concepts.py --dry-run   # just report
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


FRONTMATTER_RE = re.compile(r"\A(---\s*\n)(.*?)(\n---\s*\n)", re.DOTALL)


def transform(text: str) -> str:
    """Apply the rename + code_url fold transform to a single markdown file's text."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return text
    head, yaml_block, tail = m.group(1), m.group(2), m.group(3)
    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text

    changed = False

    if "tags" in data:
        if "concepts" not in data:
            data["concepts"] = data["tags"]
        del data["tags"]
        changed = True

    paper = data.get("paper")
    if isinstance(paper, dict) and "code_url" in paper:
        code_url = paper.pop("code_url")
        if code_url:
            repos = data.get("repos") or []
            if not isinstance(repos, list):
                repos = []
            if code_url not in repos:
                repos.append(code_url)
            data["repos"] = repos
        changed = True

    if not changed:
        return text

    new_yaml = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)
    return f"{head}{new_yaml.rstrip()}{tail}{text[m.end():]}"


def migrate_all(root: Path, dry_run: bool) -> int:
    n = 0
    for kind in ("papers", "tutorials"):
        base = root / kind
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            md = sub / "index.md"
            if not md.is_file():
                continue
            src = md.read_text(encoding="utf-8")
            out = transform(src)
            if out == src:
                continue
            if dry_run:
                print(f"DRY: would update {md}")
                n += 1
                continue
            backup = sub / "index.md.pre-concepts"
            if not backup.exists():
                shutil.copy2(md, backup)
            md.write_text(out, encoding="utf-8")
            print(f"WROTE: {md}")
            n += 1
    return n


def cleanup_all(root: Path) -> int:
    n = 0
    for kind in ("papers", "tutorials"):
        base = root / kind
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            backup = sub / "index.md.pre-concepts"
            if backup.is_file():
                backup.unlink()
                print(f"DELETED: {backup}")
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    if args.convert:
        n = migrate_all(root, args.dry_run)
        print(f"Migrated {n} files.")
        return 0
    if args.cleanup:
        if not args.yes:
            print("Refusing to --cleanup without --yes (destructive).", file=sys.stderr)
            return 2
        n = cleanup_all(root)
        print(f"Deleted {n} backups.")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
