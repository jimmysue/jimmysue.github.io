"""Tests for build.py's _added_date() git-log helper."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import importlib.util


def _load_build():
    """build.py uses hyphen-free name, import normally is fine but it's at repo root."""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("build_mod", repo_root / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_added_date_returns_fallback_for_non_git_path(tmp_path):
    build = _load_build()
    f = tmp_path / "not-in-git.md"
    f.write_text("hello")
    out = build._added_date(f, fallback="2099-01-01")
    assert out == "2099-01-01"


def test_added_date_returns_first_commit_date(tmp_path):
    build = _load_build()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    f = repo / "x.md"
    f.write_text("hi")
    subprocess.run(["git", "add", "x.md"], cwd=repo, check=True)
    env = {"GIT_AUTHOR_DATE": "2024-03-15T10:00:00", "GIT_COMMITTER_DATE": "2024-03-15T10:00:00"}
    env = {**os.environ, **env}
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    # Important: run helper from inside the repo so git finds the commit
    cwd = Path.cwd()
    try:
        os.chdir(repo)
        out = build._added_date(Path("x.md"), fallback="2099-01-01")
    finally:
        os.chdir(cwd)
    assert out == "2024-03-15"


def test_added_date_handles_missing_file(tmp_path):
    build = _load_build()
    out = build._added_date(tmp_path / "ghost.md", fallback="2099-01-01")
    assert out == "2099-01-01"
