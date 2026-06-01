"""Knowledge graph extraction for Phase 2.

Walks Phase-1-loaded posts + the concepts/ directory, produces a nodes+edges
JSON suitable for AI/RAG agents and for /graph.html visualization.

Public API:
    parse_repo_url(url)            -> dict   (host/owner/name/id/url)
    discover_concepts(root)        -> dict[slug, ConceptInfo]
    extract_graph(posts, concepts) -> dict   (the graph.json structure)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from build_lib.frontmatter import parse as parse_frontmatter
from build_lib.wiki_links import WIKI_LINK_RE


GRAPH_VERSION = 1
DEFAULT_GENERATED_AT = "unknown"


def parse_repo_url(url: str) -> dict[str, Any]:
    """Parse a GitHub (or other) repo URL into a structured dict."""
    u = urlparse(url.rstrip("/"))
    host = u.netloc or "unknown"
    parts = [p for p in u.path.split("/") if p]
    if host == "github.com" and len(parts) >= 2:
        owner, name = parts[0], parts[1]
        return {
            "host": host,
            "owner": owner,
            "name": name,
            "id": f"repos/{owner}/{name}",
            "url": url,
        }
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return {
        "host": host,
        "owner": "",
        "name": "",
        "id": f"repos/{host}/{h}",
        "url": url,
    }


def discover_concepts(root: Path) -> dict[str, dict[str, Any]]:
    """Walk concepts/<slug>.md. Returns mapping slug -> ConceptInfo."""
    out: dict[str, dict[str, Any]] = {}
    base = root / "concepts"
    if not base.is_dir():
        return out
    for f in sorted(base.glob("*.md")):
        slug = f.stem
        try:
            text = f.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        out[slug] = {
            "name": meta.get("name") or slug,
            "aliases": meta.get("aliases") or [],
            "parent": meta.get("parent"),
            "body_md": body,
        }
    return out


def extract_graph(posts: list[dict], concepts: dict[str, dict]) -> dict[str, Any]:
    """Construct the graph.json dict from in-memory posts + concept files."""
    post_by_slug = {p["slug"]: p for p in posts}

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_node_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def _add_node(n: dict) -> None:
        if n["id"] in seen_node_ids:
            return
        seen_node_ids.add(n["id"])
        nodes.append(n)

    def _add_edge(from_id: str, to_id: str, kind: str) -> None:
        key = (from_id, to_id, kind)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"from": from_id, "to": to_id, "kind": kind})

    # 1. Post nodes
    for p in posts:
        kind_dir = "papers" if p["type"] == "paper" else "tutorials"
        node_id = f"{kind_dir}/{p['slug']}"
        _add_node({
            "id": node_id,
            "type": p["type"],
            "slug": p["slug"],
            "title": p["title"],
            "date": p["date"],
            "tldr": p["tldr"],
            "url": p["_url"],
        })

    # 2. Concept / repo nodes + edges
    for p in posts:
        kind_dir = "papers" if p["type"] == "paper" else "tutorials"
        from_id = f"{kind_dir}/{p['slug']}"

        for concept_slug in (p.get("concepts") or []):
            concept_id = f"concepts/{concept_slug}"
            if concept_slug in concepts:
                info = concepts[concept_slug]
                name = info["name"]
                has_file = True
            else:
                name = concept_slug
                has_file = False
            _add_node({
                "id": concept_id,
                "type": "concept",
                "slug": concept_slug,
                "name": name,
                "has_file": has_file,
                "url": f"concepts/{concept_slug}.html",
            })
            _add_edge(from_id, concept_id, "mentions")

        cite_targets: set[str] = set()
        for target_slug in (p.get("citations") or []):
            if target_slug in post_by_slug:
                cite_targets.add(target_slug)

        body = p.get("_body_md") or ""
        for m in WIKI_LINK_RE.finditer(body):
            target_slug = m.group(1)
            if target_slug in post_by_slug:
                cite_targets.add(target_slug)

        for target_slug in cite_targets:
            target = post_by_slug[target_slug]
            target_kind_dir = "papers" if target["type"] == "paper" else "tutorials"
            to_id = f"{target_kind_dir}/{target_slug}"
            edge_kind = "covers" if p["type"] == "tutorial" else "cites"
            _add_edge(from_id, to_id, edge_kind)

        for repo_url in (p.get("repos") or []):
            if not repo_url:
                continue
            repo = parse_repo_url(repo_url)
            _add_node({
                "id": repo["id"],
                "type": "repo",
                "url": repo["url"],
                "host": repo["host"],
                "owner": repo["owner"],
                "name": repo["name"],
            })
            _add_edge(from_id, repo["id"], "implements")

    # 3. Standalone concept nodes (file exists but no post mentions it)
    for slug, info in concepts.items():
        cid = f"concepts/{slug}"
        if cid not in seen_node_ids:
            _add_node({
                "id": cid,
                "type": "concept",
                "slug": slug,
                "name": info["name"],
                "has_file": True,
                "url": f"concepts/{slug}.html",
            })

    return {
        "version": GRAPH_VERSION,
        "generated_at": DEFAULT_GENERATED_AT,
        "nodes": nodes,
        "edges": edges,
    }
