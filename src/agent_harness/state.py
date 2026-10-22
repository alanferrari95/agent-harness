"""State that lives outside the model.

The model remembers nothing between calls. The harness keeps a compact record of what
is still true (files read, files changed, the last test result, open errors) and sends
it back on every step. It holds what is current, not the whole history: `last_error`
disappears when the next test run passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    step: int
    tool: str
    args: dict[str, Any]
    observation: str

    def call_repr(self) -> str:
        shown = {k: (v if len(str(v)) <= 120 else str(v)[:120] + "...") for k, v in self.args.items()}
        return f"{self.tool}({shown})"


@dataclass
class RunState:
    files_read: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    last_test: str | None = None
    last_test_step: int | None = None
    last_edit_step: int | None = None
    last_error: str | None = None
    denied_calls: int = 0

    def update(self, record: StepRecord) -> None:
        obs = record.observation
        path = record.args.get("path")
        if obs.startswith("DENIED"):
            self.denied_calls += 1
            return
        if record.tool == "read_file" and not obs.startswith("ERROR") and path:
            if path not in self.files_read:
                self.files_read.append(path)
        elif record.tool == "edit_file" and obs.startswith("OK") and path:
            if path not in self.files_edited:
                self.files_edited.append(path)
            self.last_edit_step = record.step
        elif record.tool == "run_tests":
            self.last_test_step = record.step
            passed = obs.startswith("PASSED")
            self.last_test = "PASSED" if passed else "FAILED"
            self.last_error = None if passed else obs[:300]

    @property
    def edits_unverified(self) -> bool:
        """True when files changed after the last test run (or tests never ran)."""
        if self.last_edit_step is None:
            return False
        return self.last_test_step is None or self.last_test_step < self.last_edit_step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
