"""A rule-based fake model, for tests and CI. No API key, fully deterministic.

It follows a per-task script of intended actions (evals/tasks/<id>/fake_policy.yaml),
but like a real model it decides from what the harness shows it: it reads only the
rendered prompt, mostly the "# Last observation" section. It never looks at harness
internals, so a harness feature can only help it through the text it produces.

Scripted flaws, each exercising one harness feature:
- `typo: true`        first edit attempt has a whitespace slip in old_text. It recovers
                      only if the error quotes the closest line (actionable errors).
- `with_line_numbers` first edit attempt pastes read_file's "  12| " prefixes into
                      old_text. It recovers only if the error says so.
- `on_ambiguous`      if the edit is rejected as ambiguous ("appears N times"), retry
                      with this longer old_text. Needs an error that says why it failed.
- `tool: finish` mid-script: claims success too early. Verify-on-stop rejects the claim;
                      the model then runs the tests and continues on failure.
- `repeat: N`         issues the same call N times (a model going in circles).
- editing a protected test file: only a deny rule stops it before the tests are bent.
- `tool: none`        replies with text and no tool call (`text` is the reply).

Generic habits: after a passing test run it calls finish; after a failing one it moves
on with its script. It also has a declared context window: a prompt larger than that
raises ContextOverflow, as a real API would reject it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base import ContextOverflow, ModelRequest, ModelResponse, ToolCall, estimate_tokens

LAST_OBSERVATION = "# Last observation\n"
CLOSEST_LINE = re.compile(r'Closest line \(line \d+\): (".*?")\. Copy', re.DOTALL)
LINE_PREFIX = re.compile(r"^\s*\d+\| ", re.MULTILINE)
NO_CALL = "none"


@dataclass
class _Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    typo: bool = False
    with_line_numbers: bool = False
    on_ambiguous: dict[str, Any] | None = None
    repeat: int = 1
    text: str = ""


def _with_typo(text: str) -> str:
    """The kind of slip models make: one doubled space."""
    index = text.find(" ", len(text) - len(text.lstrip()))
    if index == -1:
        return text + " "
    return text[:index] + " " + text[index:]


def _with_line_numbers(text: str) -> str:
    return "\n".join(f"{n:4}| {line}" for n, line in enumerate(text.split("\n"), start=1))


class FakeModel:
    def __init__(self, policy: dict[str, Any], context_window_tokens: int = 6000):
        self.name = "fake-rules"
        self.window = context_window_tokens
        self.steps = [_Step(**raw) for raw in policy.get("steps", [])]
        self.cursor = 0
        self.emitted_in_step = 0
        self.last_call: ToolCall | None = None
        self.last_step: _Step | None = None

    def complete(self, request: ModelRequest) -> ModelResponse:
        tools_text = json.dumps([spec.__dict__ for spec in request.tools])
        prompt_tokens = estimate_tokens(request.system + request.prompt + tools_text)
        if prompt_tokens > self.window:
            raise ContextOverflow(
                f"prompt is ~{prompt_tokens} tokens; the context window is {self.window}"
            )
        observation = request.prompt.rsplit(LAST_OBSERVATION, 1)[-1]
        observation = observation.split("\n", 1)[1] if "\n" in observation else ""
        call = self._react(observation) or self._next_scripted()
        self.last_call = call
        if call.name == NO_CALL:
            text = self.last_step.text if self.last_step else ""
            return ModelResponse(None, text, prompt_tokens, estimate_tokens(text) + 5, 0.0, True)
        output_tokens = estimate_tokens(json.dumps({"name": call.name, "args": call.args})) + 5
        return ModelResponse(call, "", prompt_tokens, output_tokens, 0.0, True)

    # ------------------------------------------------------------ decision rules

    def _react(self, observation: str) -> ToolCall | None:
        last, step = self.last_call, self.last_step
        if last is None:
            return None
        if last.name == "edit_file" and observation.startswith("ERROR") and step:
            old_text = last.args.get("old_text", "")
            hint = CLOSEST_LINE.search(observation)
            if hint and "\n" not in old_text.strip("\n"):
                hinted = json.loads(hint.group(1))
                if hinted != old_text:
                    return ToolCall("edit_file", {**step.args, "old_text": hinted})
            if "line-number prefixes" in observation:
                return ToolCall("edit_file", {**last.args, "old_text": LINE_PREFIX.sub("", old_text)})
            if "appears" in observation and step.on_ambiguous:
                retry = {**step.args, **step.on_ambiguous}
                if retry != last.args:
                    return ToolCall("edit_file", retry)
        if last.name == "finish" and observation.startswith("HARNESS: finish rejected"):
            return ToolCall("run_tests", {})
        if last.name == "run_tests" and observation.startswith("PASSED"):
            return ToolCall("finish", {"summary": "Tests pass."})
        return None

    def _next_scripted(self) -> ToolCall:
        while self.cursor < len(self.steps):
            step = self.steps[self.cursor]
            first_attempt = self.emitted_in_step == 0
            self.emitted_in_step += 1
            if self.emitted_in_step >= step.repeat:
                self.cursor += 1
                self.emitted_in_step = 0
            self.last_step = step
            args = dict(step.args)
            if first_attempt and "old_text" in args:
                if step.typo:
                    args["old_text"] = _with_typo(args["old_text"])
                if step.with_line_numbers:
                    args["old_text"] = _with_line_numbers(args["old_text"])
            return ToolCall(step.tool, args)
        self.last_step = None
        return ToolCall("finish", {"summary": "Script complete."})
