"""Tests for build_lib/graph.py."""
from __future__ import annotations

from build_lib.graph import parse_repo_url, extract_graph


# --- parse_repo_url ---

def test_parse_github_url():
    out = parse_repo_url("https://github.com/foo/bar")
    assert out["host"] == "github.com"
    assert out["owner"] == "foo"
    assert out["name"] == "bar"
    assert out["id"] == "repos/foo/bar"
    assert out["url"] == "https://github.com/foo/bar"


def test_parse_github_url_with_trailing_slash():
    out = parse_repo_url("https://github.com/foo/bar/")
    assert out["id"] == "repos/foo/bar"


def test_parse_github_url_with_subpath():
    out = parse_repo_url("https://github.com/foo/bar/tree/main/x")
    assert out["id"] == "repos/foo/bar"
    assert out["owner"] == "foo"
    assert out["name"] == "bar"


def test_parse_non_github_falls_back():
    out = parse_repo_url("https://gitlab.com/x/y")
    assert out["host"] == "gitlab.com"
    assert out["url"] == "https://gitlab.com/x/y"
    assert out["id"].startswith("repos/")


# --- extract_graph ---

def _make_post(slug, type_="paper", concepts=None, citations=None, repos=None, body=""):
    return {
        "type": type_,
        "slug": slug,
        "title": slug.upper(),
        "date": "2026-01-01",
        "tldr": "",
        "tags": concepts or [],
        "_url": f"{type_}s/{slug}/index.html",
        "_body_md": body,
        "concepts": concepts or [],
        "citations": citations or [],
        "repos": repos or [],
    }


def test_extract_graph_basic_node_counts():
    posts = [_make_post("l2p-2026", concepts=["diffusion"])]
    g = extract_graph(posts, {})
    assert len(g["nodes"]) == 2  # 1 paper + 1 concept
    assert any(e["kind"] == "mentions" for e in g["edges"])


def test_extract_graph_concept_node_marks_has_file_correctly():
    posts = [_make_post("x", concepts=["flow-matching"])]
    concepts = {"flow-matching": {"name": "Flow Matching", "aliases": [], "parent": None, "body_md": "user notes"}}
    g = extract_graph(posts, concepts)
    fm = [n for n in g["nodes"] if n["id"] == "concepts/flow-matching"][0]
    assert fm["has_file"] is True
    assert fm["name"] == "Flow Matching"


def test_extract_graph_concept_without_file():
    posts = [_make_post("x", concepts=["lora"])]
    g = extract_graph(posts, {})
    lora = [n for n in g["nodes"] if n["id"] == "concepts/lora"][0]
    assert lora["has_file"] is False
    assert lora["name"] == "lora"


def test_extract_graph_citation_edge():
    posts = [
        _make_post("a", citations=["b"]),
        _make_post("b"),
    ]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"] if e["kind"] == "cites"]
    assert any(e["from"] == "papers/a" and e["to"] == "papers/b" for e in cite_edges)


def test_extract_graph_dedups_duplicate_citation_from_body_and_frontmatter():
    posts = [
        _make_post("a", citations=["b"], body="See [[b]] for context."),
        _make_post("b"),
    ]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"]
                  if e["kind"] == "cites" and e["from"] == "papers/a" and e["to"] == "papers/b"]
    assert len(cite_edges) == 1


def test_extract_graph_body_wiki_link_becomes_citation_when_no_frontmatter():
    posts = [
        _make_post("a", body="See [[b]]."),
        _make_post("b"),
    ]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"] if e["kind"] == "cites"]
    assert any(e["from"] == "papers/a" and e["to"] == "papers/b" for e in cite_edges)


def test_extract_graph_tutorial_to_paper_is_covers():
    posts = [
        _make_post("t", type_="tutorial", body="See [[p]]."),
        _make_post("p"),
    ]
    g = extract_graph(posts, {})
    cover_edges = [e for e in g["edges"] if e["kind"] == "covers"]
    assert any(e["from"] == "tutorials/t" and e["to"] == "papers/p" for e in cover_edges)


def test_extract_graph_repo_edge():
    posts = [_make_post("x", repos=["https://github.com/foo/bar"])]
    g = extract_graph(posts, {})
    impl = [e for e in g["edges"] if e["kind"] == "implements"]
    assert any(e["to"] == "repos/foo/bar" for e in impl)
    repo_node = [n for n in g["nodes"] if n["id"] == "repos/foo/bar"][0]
    assert repo_node["type"] == "repo"


def test_extract_graph_broken_citation_skipped():
    posts = [_make_post("a", citations=["ghost-2099"])]
    g = extract_graph(posts, {})
    cite_edges = [e for e in g["edges"] if e["kind"] == "cites"]
    assert cite_edges == []


def test_extract_graph_version_and_generated_at():
    g = extract_graph([], {})
    assert g["version"] == 1
    assert "generated_at" in g
