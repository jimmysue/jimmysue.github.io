"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_frontmatter_paper() -> str:
    return """---
type: paper
slug: l2p-2026
title: "L2P: example"
date: 2026-05-22
tldr: |
  A multi-line
  summary.
concepts: [diffusion, flow-matching]
paper:
  arxiv_id: "2605.12013"
  authors: "Author A, Author B"
---

# L2P

Body text.
"""


@pytest.fixture
def slug_set_basic() -> set[str]:
    return {"l2p-2026", "asymflow-2026", "awm-2025"}
