from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_harness.config import HarnessConfig, LoopConfig, ToolsConfig
from agent_harness.models.base import ModelRequest, ModelResponse, ToolCall
from agent_harness.permissions import PermissionPolicy
from agent_harness.tasks import Task

REPO_ROOT = Path(__file__).resolve().parents[1]

CALC = '''
def add(a, b):
    return a - b


def double(x):
    return x * 2
'''

CALC_TEST = '''
import unittest

from calc import add, double


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_double(self):
        self.assertEqual(double(4), 8)
'''


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8", newline="\n")


@pytest.fixture
def task(tmp_path: Path) -> Task:
    """A tiny project whose `add` is broken; tests live in tests/."""
    repo = tmp_path / "fixture" / "repo"
    write_files(
        repo,
        {"calc.py": CALC, "tests/__init__.py": "", "tests/test_calc.py": CALC_TEST},
    )
    return Task(id="calc", goal="Fix add.", repo=repo)


def make_config(**overrides) -> HarnessConfig:
    loop = overrides.pop("loop", {})
    tools = overrides.pop("tools", {})
    rules = overrides.pop(
        "rules",
        [
            {"tool": "search_code", "decision": "allow"},
            {"tool": "read_file", "decision": "allow"},
            {"tool": "edit_file", "decision": "allow"},
            {"tool": "run_tests", "decision": "allow"},
        ],
    )
    return HarnessConfig(
        version="test",
        loop=LoopConfig(**{"max_steps": 10, **loop}),
        tools=ToolsConfig(**{"actionable_errors": True, **tools}),
        permissions=PermissionPolicy.from_config(rules),
        **overrides,
    )


class ScriptedModel:
    """Returns a fixed sequence of responses; records every request it receives."""

    def __init__(self, calls: list[ToolCall | None | Callable], cost: float = 0.0):
        self.name = "scripted"
        self.calls = list(calls)
        self.cost = cost
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        call = self.calls.pop(0) if self.calls else ToolCall("finish", {})
        if callable(call):
            return call(request)
        return ModelResponse(call, "", 100, 10, self.cost)


FIX_ADD = ToolCall(
    "edit_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}
)
