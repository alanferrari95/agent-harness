"""A rule-based fake model, for tests and CI. No API key, fully deterministic.

It follows a per-task script of intended actions (evals/tasks/<id>/fake_policy.yaml),
but like a real model it decides from what the harness shows it: it reads only the
rendered prompt, mostly the "# Last observation" section. It never looks at harness
internals, so a harness feature can only help it through the text it produces.

Scripted flaws, each exercising one harness feature:
- `typo: true`        first edit attempt has a whitespace slip in old_text. It recovers
                      only if the error quotes the closest line (actionable errors).
- `on_ambiguous`      if the edit is rejected as ambiguous ("appears N times"), retry
                      with this longer old_text. Needs an error that says why it failed.
- `tool: finish` mid-script: claims success too early. Verify-on-stop rejects the claim;
                      the model then runs the tests and continues on failure.
- `repeat: N`         issues the same call N times (a model going in circles).
- editing a protected test file: only a deny rule stops it before the tests are bent.

It also has a declared context window: a prompt larger than that raises ContextOverflow,
as a real API would reject it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base import ContextOverflow, ModelRequest, ModelResponse, ToolCall, estimate_tokens

LAST_OBSERVATION = "# Last observation\n"
CLOSEST_LINE = re.compile(r'Closest line \(line \d+\): (".*?")\. Copy', re.DOTALL)


@dataclass
class _Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    typo: bool = False
    on_ambiguous: dict[str, Any] | None = None
    repeat: int = 1


def _with_typo(text: str) -> str:
    """The kind of slip models make: one doubled space."""
    index = text.find(" ", len(text) - len(text.lstrip()))
    if index == -1:
        return text + " "
    return text[:index] + " " + text[index:]


class FakeModel:
    def __init__(self, policy: dict[str, Any], context_window_tokens: int = 6000):
        self.name = "fake-rules"
        self.window = context_window_tokens
        self.steps = [_Step(**raw) for raw in policy.get("steps", [])]
        self.cursor = 0
        self.emitted_in_step = 0
        self.last_call: ToolCall | None = None
        self.last_step: _Step | None = None
        self.reacting = False  # last call was a reaction, not a scripted step

    def complete(self, request: ModelRequest) -> ModelResponse:
        tools_text = json.dumps([spec.__dict__ for spec in request.tools])
        prompt_tokens = estimate_tokens(request.system + request.prompt + tools_text)
        if prompt_tokens > self.window:
            raise ContextOverflow(
                f"prompt is ~{prompt_tokens} tokens; the context window is {self.window}"
            )
        observation = request.prompt.rsplit(LAST_OBSERVATION, 1)[-1]
        observation = observation.split("\n", 1)[1] if "\n" in observation else ""
        call = self._decide(observation)
        self.last_call = call
        output_tokens = estimate_tokens(json.dumps({"name": call.name, "args": call.args})) + 5
        return ModelResponse(
            tool_call=call,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tokens_estimated=True,
        )

    # ------------------------------------------------------------ decision rules

    def _decide(self, observation: str) -> ToolCall:
        reaction = self._react(observation)
        if reaction is not None:
            self.reacting = True
            return reaction
        self.reacting = False
        return self._next_scripted()

    def _react(self, observation: str) -> ToolCall | None:
        last, step = self.last_call, self.last_step
        if last is None or step is None:
            return None
        if last.name == "edit_file" and observation.startswith("ERROR"):
            hint = CLOSEST_LINE.search(observation)
            if hint and "\n" not in last.args["old_text"].strip("\n"):
                hinted = json.loads(hint.group(1))
                if hinted != last.args["old_text"]:
                    return ToolCall("edit_file", {**step.args, "old_text": hinted})
            if "appears" in observation and step.on_ambiguous:
                retry = {**step.args, **step.on_ambiguous}
                if retry != last.args:
                    return ToolCall("edit_file", retry)
        if last.name == "finish" and observation.startswith("HARNESS: finish rejected"):
            return ToolCall("run_tests", {})
        if self.reacting and last.name == "run_tests" and observation.startswith("PASSED"):
            return ToolCall("finish", {"summary": "Tests pass."})
        return None

    def _next_scripted(self) -> ToolCall:
        while self.cursor < len(self.steps):
            step = self.steps[self.cursor]
            if self.emitted_in_step < step.repeat:
                first_attempt = self.emitted_in_step == 0
                self.emitted_in_step += 1
                if self.emitted_in_step >= step.repeat:
                    self.cursor += 1
                    self.emitted_in_step = 0
                self.last_step = step
                args = dict(step.args)
                if step.typo and first_attempt and "old_text" in args:
                    args["old_text"] = _with_typo(args["old_text"])
                return ToolCall(step.tool, args)
            self.cursor += 1
            self.emitted_in_step = 0
        self.last_step = None
        return ToolCall("finish", {"summary": "Script complete."})
