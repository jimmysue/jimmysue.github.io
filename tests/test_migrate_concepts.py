"""Tests for migrate-concepts.py."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load():
    path = pathlib.Path(__file__).parent.parent / "migrate-concepts.py"
    spec = importlib.util.spec_from_file_location("migrate_concepts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mig():
    return _load()


def test_rename_tags_to_concepts(mig):
    src = """---
type: paper
slug: x
title: Test
date: 2026-01-01
tldr: y
tags:
- a
- b
---

body
"""
    out = mig.transform(src)
    assert "tags:" not in out
    assert "concepts:" in out
    assert "- a" in out and "- b" in out
    assert "\nbody\n" in out


def test_fold_code_url_into_repos(mig):
    src = """---
type: paper
slug: x
title: Test
date: 2026-01-01
tldr: y
tags: [a]
paper:
  arxiv_id: '123'
  code_url: 'https://github.com/foo/bar'
  weights_url: 'https://huggingface.co/x'
---

body
"""
    out = mig.transform(src)
    assert "code_url" not in out
    assert "repos:" in out
    assert "https://github.com/foo/bar" in out
    assert "weights_url" in out


def test_idempotent_on_already_migrated(mig):
    src = """---
type: paper
slug: x
title: Test
date: 2026-01-01
tldr: y
concepts: [a]
repos:
- https://github.com/foo/bar
---

body
"""
    out = mig.transform(src)
    assert "concepts:" in out
    assert "tags:" not in out
    assert "repos:" in out


def test_preserves_other_frontmatter_keys(mig):
    src = """---
type: tutorial
slug: x
title: T
date: 2026-01-01
tldr: y
tags: [a]
tutorial:
  word_count: '5k'
  reading_minutes: '30'
---

body
"""
    out = mig.transform(src)
    assert "tutorial:" in out
    assert "word_count" in out
    assert "5k" in out


def test_no_frontmatter_returns_unchanged(mig):
    src = "# Just markdown\n\nNo frontmatter.\n"
    out = mig.transform(src)
    assert out == src


def test_missing_tags_field_no_op_for_rename(mig):
    src = """---
type: paper
slug: x
title: T
date: 2026-01-01
tldr: y
concepts: [a]
paper:
  code_url: 'https://github.com/foo/bar'
---

body
"""
    out = mig.transform(src)
    assert "code_url" not in out
    assert "repos:" in out
    assert "https://github.com/foo/bar" in out
