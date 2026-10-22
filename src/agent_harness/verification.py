"""External verification: the harness decides whether the task is done.

The model's "finish" is a claim, not evidence. After the loop ends, the harness copies
the workspace, restores the protected files (the acceptance tests) from the pristine
fixture, and runs the test command itself. Editing the tests to make them pass does not
help: the run is judged against the original tests, and the tampering is recorded.
"""

from __future__ import annotations

import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .tasks import Task
from .workspace import IGNORED_DIRS, Workspace, remove_tree


def _protected_files(root: Path, patterns: list[str]) -> set[str]:
    found = set()
    for path in root.rglob("*"):
        if path.is_file() and not IGNORED_DIRS.intersection(path.parts):
            rel = path.relative_to(root).as_posix()
            if any(fnmatch(rel, pattern) for pattern in patterns):
                found.add(rel)
    return found


def verify(workspace: Workspace, task: Task) -> dict[str, Any]:
    pristine = task.repo.resolve()
    original = _protected_files(pristine, task.protected)
    current = _protected_files(workspace.root, task.protected)
    tampered = sorted(
        rel
        for rel in original | current
        if rel not in current
        or rel not in original
        or (workspace.root / rel).read_bytes() != (pristine / rel).read_bytes()
    )

    holder = Path(tempfile.mkdtemp(prefix="agent-harness-verify-"))
    check_root = holder / "repo"
    try:
        shutil.copytree(workspace.root, check_root, ignore=shutil.ignore_patterns(*IGNORED_DIRS))
        for rel in current - original:
            (check_root / rel).unlink()
        for rel in original:
            target = check_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pristine / rel, target)
        checker = Workspace(check_root, pristine, "copy", holder)
        code, output = checker.run(task.test_command)
    finally:
        remove_tree(holder)
    return {
        "tests_passed": code == 0,
        "exit_code": code,
        "tampered_files": tampered,
        "output_tail": output[-1500:],
    }
