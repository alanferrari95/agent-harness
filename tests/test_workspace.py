from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from agent_harness.workspace import Workspace, WorkspaceError, sanitized_env


def test_copy_mode_never_touches_the_source(task):
    before = (task.repo / "calc.py").read_text()
    with Workspace.create(task.repo) as ws:
        (ws.root / "calc.py").write_text("changed")
        assert ws.root != task.repo.resolve()
    assert (task.repo / "calc.py").read_text() == before


def test_workspace_is_removed_on_close(task):
    ws = Workspace.create(task.repo)
    root = ws.root
    ws.close()
    assert not root.exists()


def test_resolve_refuses_escapes_and_git(task):
    with Workspace.create(task.repo) as ws:
        with pytest.raises(PermissionError):
            ws.resolve("../outside.txt")
        with pytest.raises(PermissionError):
            ws.resolve(".git/config")
        assert ws.resolve("tests/test_calc.py").is_file()


def test_unknown_mode_fails(task):
    with pytest.raises(WorkspaceError):
        Workspace.create(task.repo, mode="docker")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_worktree_mode(task):
    repo = task.repo
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run(
        [*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], check=True
    )
    with Workspace.create(repo, mode="git-worktree") as ws:
        (ws.root / "calc.py").write_text("changed")
        root = ws.root
    assert not root.exists()
    status = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True)
    assert status.stdout == ""
    worktrees = subprocess.run([*git, "worktree", "list"], capture_output=True, text=True)
    assert len(worktrees.stdout.strip().splitlines()) == 1


def test_secrets_are_not_passed_to_test_processes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("SOME_TOKEN", "x")
    env = sanitized_env()
    assert not {"ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SOME_TOKEN"} & set(env)
    assert env.get("PATH") == os.environ.get("PATH")
