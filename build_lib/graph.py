"""Knowledge graph extraction for Phase 2.

Walks Phase-1-loaded posts + the concepts/ directory, produces a nodes+edges
JSON suitable for AI/RAG agents and for /graph.html visualization.

Public API:
    parse_repo_url(url)            -> dict   (host/owner/name/id/url)
    discover_concepts(root)        -> dict[slug, ConceptInfo]
    extract_graph(posts, concepts) -> dict   (the graph.json structure)
    render_concept_page(...)       -> str    (HTML for concepts/<slug>.html)
    render_graph_page(graph, ...)  -> str    (HTML for /graph.html)
"""
from __future__ import annotations

import hashlib
import html as _html
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from markdown_it import MarkdownIt

from build_lib.frontmatter import parse as parse_frontmatter
from build_lib.wiki_links import WIKI_LINK_RE

_CONCEPT_MD = MarkdownIt("commonmark", {"html": True})


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


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _render_concept_body(body_md: str) -> str:
    if not body_md.strip():
        return ""
    return _CONCEPT_MD.render(body_md)


def _render_post_card(post: dict, depth: int = 1) -> str:
    """Render a single post card with relative paths from concepts/ pages (depth=1)."""
    prefix = "../"
    is_tut = post["type"] == "tutorial"
    cls = "post-card--tutorial" if is_tut else "post-card--paper"
    badge_cls = "post-card__badge--tutorial" if is_tut else "post-card__badge--paper"
    badge_text = "📘 教程" if is_tut else "📄 论文"
    href = f"{prefix}{post['_url']}"
    title = _html.escape(post["title"], quote=True)
    tldr = _html.escape(post.get("tldr") or "", quote=True)
    date = _html.escape(post["date"], quote=True)
    return (
        f'<a class="post-card {cls}" href="{_html.escape(href, quote=True)}">\n'
        f'  <div class="post-card__head">\n'
        f'    <span class="post-card__badge {badge_cls}">{badge_text}</span>\n'
        f'    <span class="post-card__date">{date}</span>\n'
        f'  </div>\n'
        f'  <h3 class="post-card__title">{title}</h3>\n'
        f'  <p class="post-card__tldr">{tldr}</p>\n'
        f'</a>\n'
    )


def render_concept_page(slug: str, concept_meta: dict, posts_mentioning: list[dict],
                        nav_html: str) -> str:
    """Render a concept aggregation HTML page (concepts/<slug>.html, depth=1)."""
    name = _html.escape(concept_meta.get("name") or slug, quote=True)
    body_html = _render_concept_body(concept_meta.get("body_md") or "")
    aliases = concept_meta.get("aliases") or []
    aliases_html = ""
    if aliases:
        aliases_html = (
            '<p class="concept__aliases">aliases: '
            + ", ".join(f"<code>{_html.escape(a)}</code>" for a in aliases)
            + "</p>"
        )
    cards = "".join(_render_post_card(p, depth=1) for p in posts_mentioning)
    grid = (
        f'<div class="post-grid">\n{cards}</div>\n'
        if cards else
        '<p class="empty">还没有论文/教程提到此概念。</p>\n'
    )

    return (
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n'
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>{name} — paper-reading</title>\n'
        '  <link rel="stylesheet" href="../assets/style.css">\n'
        '</head>\n<body>\n'
        f'{nav_html}\n'
        '<main class="page page--concept">\n'
        f'  <h1>{name}</h1>\n'
        f'  {aliases_html}\n'
        f'  <div class="concept__body">\n{body_html}\n  </div>\n'
        '  <h2>提到此概念的论文 / 教程</h2>\n'
        f'  {grid}\n'
        '  <p class="back-link"><a href="../concepts.html">← 返回所有概念</a></p>\n'
        '</main>\n</body>\n</html>\n'
    )


def render_graph_page(graph: dict, nav_html: str) -> str:
    """Render /graph.html — cytoscape.js force-directed visualization."""
    graph_json = json.dumps(graph, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>知识图谱 — paper-reading</title>
  <link rel="stylesheet" href="assets/style.css">
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
  <style>
    body {{ margin: 0; }}
    .graph-shell {{ display: flex; flex-direction: column; height: 100vh; }}
    .graph-controls {{
      padding: 0.6rem 1rem; background: #f8f9fa; border-bottom: 1px solid #e0e0e0;
      display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
    }}
    .graph-controls label {{ display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.9rem; }}
    .graph-controls input[type="search"] {{
      padding: 0.3rem 0.6rem; border: 1px solid #ccc; border-radius: 4px;
      min-width: 200px;
    }}
    #cy {{ flex: 1; background: #fafbfc; }}
    .graph-tooltip {{
      position: absolute; background: rgba(33, 37, 41, 0.95); color: white;
      padding: 0.5rem 0.75rem; border-radius: 4px; font-size: 0.85rem;
      max-width: 320px; pointer-events: none; z-index: 10; display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .graph-tooltip h4 {{ margin: 0 0 0.3rem 0; font-size: 0.95rem; }}
    .graph-tooltip p {{ margin: 0; font-size: 0.8rem; line-height: 1.4; }}
  </style>
</head>
<body>
{nav_html}
<div class="graph-shell">
  <div class="graph-controls">
    <strong>知识图谱</strong>
    <label><input type="checkbox" data-type="paper" checked> 论文</label>
    <label><input type="checkbox" data-type="tutorial" checked> 教程</label>
    <label><input type="checkbox" data-type="concept" checked> 概念</label>
    <label><input type="checkbox" data-type="repo" checked> 仓库</label>
    <input type="search" id="graph-search" placeholder="搜索 slug/title…">
    <span style="margin-left:auto; color: #666; font-size: 0.85rem;">
      点节点跳转 · hover 看详情
    </span>
  </div>
  <div id="cy"></div>
  <div class="graph-tooltip" id="graph-tooltip"></div>
</div>

<script>
window.__GRAPH__ = {graph_json};
</script>
<script>
(function () {{
  const G = window.__GRAPH__;
  const colorByType = {{
    paper: '#4A90E2', tutorial: '#8B5CF6',
    concept: '#F59E0B', repo: '#6B7280',
  }};
  const shapeByType = {{
    paper: 'ellipse', tutorial: 'round-rectangle',
    concept: 'diamond', repo: 'triangle',
  }};

  const indeg = new Map();
  G.edges.forEach(e => indeg.set(e.to, (indeg.get(e.to) || 0) + 1));

  const elements = [
    ...G.nodes.map(n => ({{
      data: {{
        id: n.id, type: n.type, label: n.title || n.name || n.slug || n.id,
        url: n.url, raw: n,
        size: 16 + Math.min(28, (indeg.get(n.id) || 0) * 3),
      }},
    }})),
    ...G.edges.map((e, i) => ({{ data: {{ id: 'e' + i, source: e.from, target: e.to, kind: e.kind }} }})),
  ];

  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements,
    layout: {{ name: 'cose-bilkent', nodeRepulsion: 8000, idealEdgeLength: 120, animate: false }},
    style: [
      {{ selector: 'node', style: {{
        'background-color': ele => colorByType[ele.data('type')] || '#888',
        'shape': ele => shapeByType[ele.data('type')] || 'ellipse',
        'label': 'data(label)',
        'font-size': '11px',
        'text-valign': 'bottom', 'text-margin-y': 4,
        'text-wrap': 'ellipsis', 'text-max-width': 140,
        'width': 'data(size)', 'height': 'data(size)',
        'border-width': 1, 'border-color': '#333',
      }} }},
      {{ selector: 'edge', style: {{
        'width': 1,
        'line-color': '#bbb',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#bbb',
        'arrow-scale': 0.7,
      }} }},
      {{ selector: '.highlight', style: {{
        'border-width': 3, 'border-color': '#dc3545',
      }} }},
    ],
  }});

  cy.on('tap', 'node', evt => {{
    const url = evt.target.data('url');
    if (url) window.location.href = url;
  }});

  const tip = document.getElementById('graph-tooltip');
  cy.on('mouseover', 'node', evt => {{
    const d = evt.target.data('raw');
    let html = '<h4>' + (d.title || d.name || d.slug || d.id) + '</h4>';
    if (d.tldr) html += '<p>' + (d.tldr.length > 240 ? d.tldr.slice(0, 240) + '…' : d.tldr) + '</p>';
    else if (d.type === 'repo') html += '<p>' + d.url + '</p>';
    else if (d.type === 'concept') {{
      const cnt = G.edges.filter(e => e.to === d.id).length;
      html += '<p>' + cnt + ' 篇提到此概念</p>';
    }}
    tip.innerHTML = html;
    tip.style.display = 'block';
  }});
  cy.on('mousemove', 'node', evt => {{
    tip.style.left = (evt.originalEvent.pageX + 12) + 'px';
    tip.style.top = (evt.originalEvent.pageY + 12) + 'px';
  }});
  cy.on('mouseout', 'node', () => {{ tip.style.display = 'none'; }});

  document.querySelectorAll('.graph-controls input[type="checkbox"]').forEach(cb => {{
    cb.addEventListener('change', () => {{
      const t = cb.dataset.type;
      const sel = cy.nodes('[type = "' + t + '"]');
      if (cb.checked) sel.show(); else sel.hide();
    }});
  }});

  const search = document.getElementById('graph-search');
  search.addEventListener('input', () => {{
    const q = search.value.trim().toLowerCase();
    cy.nodes().removeClass('highlight');
    if (!q) return;
    const matches = cy.nodes().filter(n => {{
      const d = n.data('raw');
      return (d.title || '').toLowerCase().includes(q)
        || (d.slug || '').toLowerCase().includes(q)
        || (d.name || '').toLowerCase().includes(q);
    }});
    matches.addClass('highlight');
    if (matches.length) cy.center(matches);
  }});
}})();
</script>
</body>
</html>
"""
