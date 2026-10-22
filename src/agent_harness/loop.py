"""The agent loop: a linear while, with cheap budgets and a stuck detector.

    while not done:
        context  = build(goal, map, state, history)
        decision = model(context, tools)
        if decision is finish: (verify-on-stop) stop
        observation = execute(decision)   # permissions, schema check, tool
        record(observation); update(state)
    verify(acceptance tests)              # outside the model

Stop reasons: finished, budget_steps, budget_cost, budget_time, stuck, malformed,
context_overflow, model_error. Whatever the reason, the harness verifies the workspace
and returns a result: a run never ends in an exception.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .context import SYSTEM_PROMPT, ContextBuilder, project_map
from .models.base import ContextOverflow, Model, ModelError, ModelRequest
from .permissions import Approver, approve_all
from .state import RunState, StepRecord
from .tasks import Task
from .tools import FINISH, TOOLS, ToolContext, execute
from .tracing import Tracer
from .verification import verify
from .workspace import Workspace

FINISH_REJECTED = (
    "HARNESS: finish rejected. Files changed after the last test run (or tests never ran). "
    "Call run_tests, read the result, and fix any failure before calling finish."
)
NO_TOOL_CALL = "HARNESS: your reply had no tool call. Call exactly one of: {tools}."


@dataclass
class RunResult:
    run_id: str
    task: str
    version: str
    model: str
    success: bool
    stop_reason: str
    steps: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    tokens_estimated: bool = False
    verification: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Harness:
    def __init__(
        self,
        config: HarnessConfig,
        model: Model,
        approver: Approver = approve_all,
    ):
        self.config = config
        self.model = model
        self.approver = approver

    def _tools(self) -> dict[str, Any]:
        exposed = self.config.permissions.exposed_tools() | {FINISH}
        return {name: tool for name, tool in TOOLS.items() if name in exposed}

    def run(self, task: Task, trace_path: Path | None = None) -> RunResult:
        cfg = self.config
        run_id = uuid.uuid4().hex[:12]
        tracer = Tracer(trace_path, run_id)
        started = time.monotonic()
        tools = self._tools()
        specs = [tool.spec for tool in tools.values()]
        state = RunState()
        history: list[StepRecord] = []
        signatures: list[str] = []
        totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        estimated = False
        malformed = 0
        stop_reason = "budget_steps"

        workspace = Workspace.create(task.repo, cfg.workspace_mode)
        tool_ctx = ToolContext(
            workspace, task.test_command, cfg.tools.actionable_errors, cfg.tools.test_timeout
        )
        context = ContextBuilder(
            task.goal, project_map(workspace.list_files()), task.agents_md, cfg.compaction
        )
        tracer.emit(
            "run_start",
            task=task.id,
            version=cfg.version,
            config=cfg.source_path,
            model=self.model.name,
            tools=[spec.name for spec in specs],
            workspace_mode=cfg.workspace_mode,
        )

        def record(step: int, tool: str, args: dict[str, Any], observation: str, **extra: Any):
            entry = StepRecord(step, tool, args, observation)
            history.append(entry)
            state.update(entry)
            tracer.emit("step", step=step, tool=tool, args=args, observation=observation, **extra)

        try:
            for step in range(1, cfg.loop.max_steps + 1):
                elapsed = time.monotonic() - started
                if elapsed >= cfg.loop.max_seconds:
                    stop_reason = "budget_time"
                    break
                if totals["cost_usd"] >= cfg.loop.max_cost_usd:
                    stop_reason = "budget_cost"
                    break

                if context.maybe_compact(history):
                    tracer.emit("compaction", step=step, folded_steps=context.folded)
                prompt = context.build(state, history)
                try:
                    response = self.model.complete(ModelRequest(SYSTEM_PROMPT, prompt, specs))
                except ContextOverflow as exc:
                    tracer.emit("model_error", step=step, error=str(exc), prompt_chars=len(prompt))
                    stop_reason = "context_overflow"
                    break
                except ModelError as exc:
                    tracer.emit("model_error", step=step, error=str(exc))
                    stop_reason = "model_error"
                    break

                totals["input_tokens"] += response.input_tokens
                totals["output_tokens"] += response.output_tokens
                totals["cost_usd"] += response.cost_usd
                estimated = estimated or response.tokens_estimated
                usage = {
                    "prompt_chars": len(prompt),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": round(response.cost_usd, 6),
                }

                call = response.tool_call
                if call is None:
                    malformed += 1
                    observation = NO_TOOL_CALL.format(tools=", ".join(tools))
                    record(step, "(none)", {}, observation, model_text=response.text, **usage)
                    if malformed >= cfg.loop.max_malformed:
                        stop_reason = "malformed"
                        break
                    continue
                malformed = 0

                signature = call.name + json.dumps(call.args, sort_keys=True, default=str)
                signatures.append(signature)
                threshold = cfg.loop.stuck_threshold
                if threshold and signatures[-threshold:] == [signature] * threshold:
                    observation = f"HARNESS: the same call {threshold} times in a row. Stopping."
                    record(step, call.name, call.args, observation, **usage)
                    stop_reason = "stuck"
                    break

                if call.name == FINISH:
                    if cfg.loop.verify_on_stop and state.edits_unverified:
                        record(step, FINISH, call.args, FINISH_REJECTED, **usage)
                        continue
                    record(step, FINISH, call.args, "HARNESS: finish accepted; verifying.", **usage)
                    stop_reason = "finished"
                    break

                observation = self._dispatch(call.name, call.args, tools, tool_ctx, tracer, step)
                record(step, call.name, call.args, observation, **usage)

            verification = verify(workspace, task)
        finally:
            workspace.close()

        result = RunResult(
            run_id=run_id,
            task=task.id,
            version=cfg.version,
            model=self.model.name,
            success=verification["tests_passed"],
            stop_reason=stop_reason,
            steps=len(history),
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            cost_usd=round(totals["cost_usd"], 6),
            seconds=round(time.monotonic() - started, 2),
            tokens_estimated=estimated,
            verification=verification,
            state=state.to_dict(),
        )
        tracer.emit("run_end", **{k: v for k, v in result.to_dict().items() if k != "run_id"})
        return result

    def _dispatch(self, name, args, tools, tool_ctx, tracer, step) -> str:
        if name not in tools:
            return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(tools)}."
        verdict = self.config.permissions.check(name, args)
        approved = None
        if verdict.decision == "ask":
            approved = self.approver(name, args)
        tracer.emit(
            "permission",
            step=step,
            tool=name,
            decision=verdict.decision,
            rule=verdict.rule_index,
            approved=approved,
        )
        if verdict.decision == "deny":
            reason = f" ({verdict.reason})" if verdict.reason else ""
            return (
                f"DENIED by permission policy: {name} on {args.get('path', '(no path)')}"
                f"{reason}. Change your approach instead of retrying."
            )
        if verdict.decision == "ask" and not approved:
            return "DENIED: the user did not approve this action."
        return execute(tools[name], args, tool_ctx)
