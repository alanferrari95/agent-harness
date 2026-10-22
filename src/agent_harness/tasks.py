"""Eval tasks: a fixture repository with failing tests, plus the goal in plain words.

Layout of evals/tasks/<id>/:
    task.yaml          goal, test command, protected paths, tags
    repo/              the fixture project the agent works on (never modified)
    fake_policy.yaml   what the rule-based fake model intends to do (fake model only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TEST_COMMAND = ["{python}", "-m", "unittest", "-q"]


@dataclass
class Task:
    id: str
    goal: str
    repo: Path
    test_command: list[str] = field(default_factory=lambda: list(DEFAULT_TEST_COMMAND))
    protected: list[str] = field(default_factory=lambda: ["tests/*"])
    tags: list[str] = field(default_factory=list)
    fake_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def agents_md(self) -> str:
        path = self.repo / "AGENTS.md"
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def load_task(directory: str | Path) -> Task:
    directory = Path(directory)
    raw = yaml.safe_load((directory / "task.yaml").read_text(encoding="utf-8"))
    policy_path = directory / "fake_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    return Task(
        id=raw["id"],
        goal=raw["goal"].strip(),
        repo=directory / "repo",
        test_command=raw.get("test_command", list(DEFAULT_TEST_COMMAND)),
        protected=raw.get("protected", ["tests/*"]),
        tags=raw.get("tags", []),
        fake_policy=policy or {},
    )


def load_tasks(root: str | Path) -> list[Task]:
    root = Path(root)
    return [load_task(d) for d in sorted(root.iterdir()) if (d / "task.yaml").is_file()]
