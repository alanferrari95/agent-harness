"""Context assembly: what the model sees on each call, rebuilt from scratch every step.

Order matters for prompt caching, so stable parts go first and volatile parts last:
task, project map and AGENTS.md (stable), then harness state, earlier steps and the last
observation (change every step). The system prompt and tool list never change in a run.

Compaction (threshold-based, deterministic):
- When the rendered history passes `threshold_tokens`, the oldest steps are folded into
  a structured summary and only the last `keep_recent` steps stay verbatim.
- The summary has fixed sections and is merged incrementally: new facts are added to
  the previous summary instead of rewriting it, so early decisions survive.
- It folds whole step records, so a tool call is never separated from its result.
- It is rule-based, not an LLM call: free, reproducible, and testable in CI. The cost is
  that it keeps facts (paths, results, errors), not reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .models.base import estimate_tokens
from .state import RunState, StepRecord

SYSTEM_PROMPT = """You are a coding agent working inside a harness.
Each turn, call exactly one tool. You only see what the harness shows you below; you do \
not remember earlier turns.
Find the relevant code with search_code and read_file, make minimal edits with \
edit_file, and check them with run_tests. When the task is done, call finish.
The harness runs the tests itself before accepting the result. Do not edit tests to make \
them pass; fix the code under test."""


@dataclass
class CompactionConfig:
    enabled: bool = False
    threshold_tokens: int = 1500
    keep_recent: int = 3


@dataclass
class Summary:
    """Fixed sections, merged incrementally."""

    steps_folded: int = 0
    files_read: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    test_runs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def absorb(self, record: StepRecord) -> None:
        self.steps_folded += 1
        first_line = record.observation.splitlines()[0] if record.observation else ""
        path = record.args.get("path", "")
        if record.tool == "read_file" and not first_line.startswith("ERROR"):
            if path not in self.files_read:
                self.files_read.append(path)
        elif record.tool == "edit_file" and first_line.startswith("OK"):
            self.edits.append(f"step {record.step}: {path}")
        elif record.tool == "run_tests":
            self.test_runs.append(f"step {record.step}: {first_line}")
        if first_line.startswith(("ERROR", "DENIED")):
            self.errors.append(f"step {record.step} {record.tool}: {first_line[:160]}")

    def render(self) -> str:
        def section(title: str, items: list[str]) -> str:
            return f"{title}: " + ("; ".join(items) if items else "none")

        return "\n".join(
            [
                f"({self.steps_folded} earlier steps folded into this summary)",
                section("Files read", self.files_read),
                section("Edits applied", self.edits),
                section("Test runs", self.test_runs),
                section("Errors and denials", self.errors[-5:]),
            ]
        )


class ContextBuilder:
    def __init__(
        self, goal: str, project_map: str, agents_md: str, compaction: CompactionConfig
    ):
        self.goal = goal
        self.project_map = project_map
        self.agents_md = agents_md
        self.compaction = compaction
        self.summary = Summary()
        self.folded = 0  # number of history records already folded into the summary
        self.compactions = 0

    @staticmethod
    def _render_step(record: StepRecord) -> str:
        return f"- step {record.step}: {record.call_repr()}\n{record.observation}"

    def _history_text(self, history: list[StepRecord]) -> str:
        return "\n".join(self._render_step(r) for r in history[self.folded : -1])

    def maybe_compact(self, history: list[StepRecord]) -> bool:
        """Fold old steps when the verbatim history passes the threshold."""
        cfg = self.compaction
        if not cfg.enabled or estimate_tokens(self._history_text(history)) < cfg.threshold_tokens:
            return False
        cut = max(self.folded, len(history) - max(cfg.keep_recent, 1))
        if cut <= self.folded:
            return False
        for record in history[self.folded : cut]:
            self.summary.absorb(record)
        self.folded = cut
        self.compactions += 1
        return True

    def build(self, state: RunState, history: list[StepRecord]) -> str:
        parts = [f"# Task\n{self.goal}", f"# Project files\n{self.project_map}"]
        if self.agents_md:
            parts.append(f"# Repository instructions (AGENTS.md)\n{self.agents_md}")
        parts.append(f"# Harness state\n{json.dumps(state.to_dict(), sort_keys=True)}")
        if self.summary.steps_folded:
            parts.append(f"# Earlier steps (compacted)\n{self.summary.render()}")
        earlier = self._history_text(history)
        if earlier:
            parts.append(f"# Recent steps\n{earlier}")
        if history:
            last = history[-1]
            parts.append(f"# Last observation\nstep {last.step}: {last.call_repr()}\n{last.observation}")
        else:
            parts.append("# Last observation\n(nothing yet: this is the first step)")
        return "\n\n".join(parts)


def project_map(files: list[str], limit: int = 200) -> str:
    shown = files[:limit]
    extra = f"\n... and {len(files) - limit} more files" if len(files) > limit else ""
    return "\n".join(shown) + extra
