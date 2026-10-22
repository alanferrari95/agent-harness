"""JSONL traces: one JSON object per line, one file per run.

Event types: run_start, step, permission, compaction, run_end. Each step records the
full prompt the model saw, the call it made, the observation and the usage. Every line
carries the run id and a timestamp, so traces from many runs can be concatenated and
analyzed like any table (pandas.read_json(path, lines=True)).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

MAX_TRACE_TEXT = 4000


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TRACE_TEXT:
        return value[:MAX_TRACE_TEXT] + "...[clipped]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    return value


class Tracer:
    def __init__(self, path: Path | None, run_id: str):
        self.path = path
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event}
        # The prompt is kept whole: a trace must show exactly what the model saw.
        record.update({k: v if k == "prompt" else _clip(v) for k, v in fields.items()})
        self.events.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_trace(path: Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def summarize_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Condense one run's trace: outcome, tool usage, errors, and where it turned."""
    start = next((e for e in events if e["event"] == "run_start"), {})
    end = next((e for e in events if e["event"] == "run_end"), {})
    steps = [e for e in events if e["event"] == "step"]
    tools = Counter(e["tool"] for e in steps)
    errors = [e for e in steps if str(e.get("observation", "")).startswith(("ERROR", "DENIED"))]
    test_runs = [
        f"step {e['step']}: {str(e.get('observation', '')).splitlines()[0]}"
        for e in steps
        if e["tool"] == "run_tests" and e.get("observation")
    ]
    return {
        "run_id": start.get("run_id"),
        "task": start.get("task"),
        "version": start.get("version"),
        "model": start.get("model"),
        "success": end.get("success"),
        "stop_reason": end.get("stop_reason"),
        "steps": len(steps),
        "tool_calls": dict(sorted(tools.items())),
        "errors_and_denials": len(errors),
        "test_runs": test_runs,
        "compactions": sum(1 for e in events if e["event"] == "compaction"),
        "input_tokens": end.get("input_tokens"),
        "output_tokens": end.get("output_tokens"),
        "cost_usd": end.get("cost_usd"),
        "seconds": end.get("seconds"),
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"run {summary['run_id']}  task={summary['task']}  version={summary['version']}  "
        f"model={summary['model']}",
        f"  success={summary['success']}  stop={summary['stop_reason']}  steps={summary['steps']}  "
        f"tokens={summary['input_tokens']}/{summary['output_tokens']}  "
        f"cost=${summary['cost_usd']}  time={summary['seconds']}s",
        f"  tools: {summary['tool_calls']}",
        f"  errors/denials: {summary['errors_and_denials']}  compactions: {summary['compactions']}",
    ]
    lines += [f"  tests {run}" for run in summary["test_runs"]]
    return "\n".join(lines)
