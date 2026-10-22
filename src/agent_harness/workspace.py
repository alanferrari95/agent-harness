"""Isolated workspace: the harness never touches the original repository.

Two modes:
- "copy": a temporary copy of the source tree (default; works for any folder).
- "git-worktree": a detached `git worktree` of the source repository's HEAD.

Every path the model sends goes through `resolve`, which refuses anything that escapes
the workspace root or reaches into `.git`. This is an application-level guard, not an
OS sandbox (see "What it does not do" in the README).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache"}
# Test subprocesses never see the harness's credentials.
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


class WorkspaceError(Exception):
    pass


def _force_remove(func, path, _exc):
    # Git marks object files read-only; Windows refuses to delete them otherwise.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force_remove)
    else:
        shutil.rmtree(path, onerror=_force_remove)


def sanitized_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def expand_command(command: list[str]) -> list[str]:
    return [sys.executable if part == "{python}" else part for part in command]


class Workspace:
    def __init__(self, root: Path, source: Path, mode: str, holder: Path):
        self.root = root.resolve()
        self.source = source.resolve()
        self.mode = mode
        self._holder = holder

    @classmethod
    def create(cls, source: Path, mode: str = "copy") -> Workspace:
        source = Path(source).resolve()
        if not source.is_dir():
            raise WorkspaceError(f"source repository not found: {source}")
        holder = Path(tempfile.mkdtemp(prefix="agent-harness-"))
        root = holder / "repo"
        if mode == "copy":
            shutil.copytree(source, root, ignore=shutil.ignore_patterns(*IGNORED_DIRS))
        elif mode == "git-worktree":
            result = subprocess.run(
                ["git", "-C", str(source), "worktree", "add", "--detach", str(root), "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                remove_tree(holder)
                raise WorkspaceError(f"git worktree add failed: {result.stderr.strip()}")
        else:
            remove_tree(holder)
            raise WorkspaceError(f"unknown workspace mode '{mode}' (use copy or git-worktree)")
        return cls(root, source, mode, holder)

    def resolve(self, path: str) -> Path:
        """Map a model-supplied relative path to a real path inside the workspace."""
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"path escapes the workspace: {path}")
        if ".git" in candidate.relative_to(self.root).parts:
            raise PermissionError(f"path inside .git is not accessible: {path}")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def list_files(self) -> list[str]:
        files = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                files.append(self.relative(Path(dirpath) / name))
        return sorted(files)

    def run(self, command: list[str], timeout: int = 120) -> tuple[int, str]:
        try:
            result = subprocess.run(
                expand_command(command),
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=sanitized_env(),
            )
        except subprocess.TimeoutExpired:
            return 124, f"command timed out after {timeout}s"
        return result.returncode, (result.stdout + result.stderr).strip()

    def close(self) -> None:
        if self.mode == "git-worktree":
            subprocess.run(
                ["git", "-C", str(self.source), "worktree", "remove", "--force", str(self.root)],
                capture_output=True,
            )
        remove_tree(self._holder)

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
