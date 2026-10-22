"""Eval suite: run every task under one harness version and record the metrics.

Metrics per version and model: success rate (judged by the harness's own test run),
steps, input/output tokens, cost and wall time. Each run leaves a JSONL trace, and the
suite leaves a summary.json under results/<version>/<model>/. The comparison table and
the CI gate read those files; no number in the README is typed by hand.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .config import HarnessConfig
from .loop import Harness, RunResult
from .models.factory import make_model
from .permissions import Approver, approve_all
from .tasks import Task


def model_slug(model_id: str) -> str:
    # Directory-safe on every OS: "ollama/qwen3:8b" -> "ollama_qwen3-8b"
    return model_id.replace("/", "_").replace(":", "-")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(results: list[RunResult], config: HarnessConfig, model_id: str) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    for r in results:
        entry = per_task.setdefault(r.task, {"runs": 0, "successes": 0, "steps": [], "stops": []})
        entry["runs"] += 1
        entry["successes"] += int(r.success)
        entry["steps"].append(r.steps)
        entry["stops"].append(r.stop_reason)
    tasks = {
        task: {
            "runs": e["runs"],
            "successes": e["successes"],
            "mean_steps": round(mean(e["steps"]), 2),
            "stop_reasons": dict(Counter(e["stops"])),
        }
        for task, e in sorted(per_task.items())
    }
    runs = len(results)
    successes = sum(r.success for r in results)
    return {
        "version": config.version,
        "description": config.description,
        "changes_from": config.changes_from,
        "config": Path(config.source_path).as_posix() if config.source_path else None,
        "model": model_id,
        "tokens_estimated": any(r.tokens_estimated for r in results),
        "measured_at": dt.date.today().isoformat(),
        "commit": _git_commit(),
        "n_tasks": len(tasks),
        "runs": runs,
        "successes": successes,
        "success_rate": round(successes / runs, 4) if runs else 0.0,
        "mean_steps": round(mean(r.steps for r in results), 2) if runs else 0.0,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "cost_usd": round(sum(r.cost_usd for r in results), 6),
        "seconds": round(sum(r.seconds for r in results), 2),
        "stop_reasons": dict(Counter(r.stop_reason for r in results)),
        "tasks": tasks,
    }


def run_suite(
    config: HarnessConfig,
    model_id: str,
    tasks: list[Task],
    out_dir: Path | None = None,
    traces_dir: Path | None = None,
    repeats: int = 1,
    approver: Approver = approve_all,
    log=print,
) -> dict[str, Any]:
    results: list[RunResult] = []
    for task in tasks:
        for repeat in range(repeats):
            model = make_model(model_id, task, config.model)
            trace = None
            if traces_dir:
                trace = traces_dir / config.version / model_slug(model_id) / f"{task.id}-{repeat}.jsonl"
            result = Harness(config, model, approver).run(task, trace)
            results.append(result)
            mark = "PASS" if result.success else "FAIL"
            log(
                f"  {config.version} {model_id} {task.id:<22} {mark}  "
                f"steps={result.steps:<3} stop={result.stop_reason}"
            )
    summary = summarize(results, config, model_id)
    if out_dir:
        target = out_dir / config.version / model_slug(model_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (target / "runs.jsonl").open("w", encoding="utf-8") as handle:
            for r in results:
                row = r.to_dict()
                row["verification"].pop("output_tail", None)
                handle.write(json.dumps(row) + "\n")
    return summary


# ------------------------------------------------------------------ comparison table


def load_summaries(results_dir: Path) -> list[dict[str, Any]]:
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in results_dir.glob("*/*/summary.json")]

    def key(s: dict[str, Any]) -> tuple:
        number = s["version"].lstrip("v")
        return (int(number) if number.isdigit() else 0, s["version"], s["model"])

    return sorted(summaries, key=key)


def comparison_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Version | Change (one per version) | Model | Success | Mean steps "
        "| Tokens in / out | Cost (USD) | Total time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        approx = "~" if s["tokens_estimated"] else ""
        rate = f"{s['successes']}/{s['runs']} ({s['success_rate']:.0%})"
        change = s["description"].splitlines()[0] if s["description"] else ""
        lines.append(
            f"| {s['version']} | {change} | `{s['model']}` | {rate} | {s['mean_steps']} "
            f"| {approx}{s['input_tokens']:,} / {approx}{s['output_tokens']:,} "
            f"| {s['cost_usd']:.4f} | {s['seconds']} |"
        )
    return "\n".join(lines)


def task_matrix(summaries: list[dict[str, Any]]) -> str:
    columns = [s for s in summaries]
    tasks = sorted({t for s in columns for t in s["tasks"]})
    header = "| Task | " + " | ".join(f"{s['version']} ({s['model']})" for s in columns) + " |"
    lines = [header, "|---|" + "---|" * len(columns)]
    for task in tasks:
        cells = []
        for s in columns:
            t = s["tasks"].get(task)
            if t is None:
                cells.append("")
                continue
            ok = "pass" if t["successes"] == t["runs"] else f"{t['successes']}/{t['runs']}"
            if t["successes"] == 0:
                ok = "FAIL"
            stop = ", ".join(t["stop_reasons"])
            cells.append(f"{ok} · {t['mean_steps']:g} steps · {stop}")
        lines.append(f"| `{task}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "_No eval results yet. Run `agent-harness eval --all`._"
    models = sorted({s["model"] for s in summaries})
    dates = sorted({s["measured_at"] for s in summaries})
    notes = [
        f"Generated by `agent-harness report` from `results/*/*/summary.json` "
        f"(measured {dates[0]}{' to ' + dates[-1] if len(dates) > 1 else ''}; "
        f"models: {', '.join(f'`{m}`' for m in models)}).",
    ]
    if any(s["tokens_estimated"] for s in summaries):
        notes.append(
            "`~` marks token counts estimated as characters / 4 (fake model). "
            "The fake model costs nothing, so its cost column is 0."
        )
    return "\n\n".join(
        [comparison_table(summaries), " ".join(notes), "Per task:", task_matrix(summaries)]
    )


START, END = "<!-- results:start -->", "<!-- results:end -->"


def inject(document: str, block: str) -> str:
    if START not in document or END not in document:
        raise ValueError(f"markers {START} / {END} not found")
    head, rest = document.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


# ------------------------------------------------------------------ promotion gate


def assert_no_regression(current: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Raise if success_rate is missing or below the reference version's.

    Same contract as invoice-mlops's promotion gate: missing key fails, strictly lower
    fails, equal passes.
    """
    for name, summary in (("current", current), ("baseline", baseline)):
        if "success_rate" not in summary:
            raise RuntimeError(f"{name} summary missing key 'success_rate'")
    rate, reference = float(current["success_rate"]), float(baseline["success_rate"])
    if rate < reference:
        lost = [
            task
            for task, t in baseline.get("tasks", {}).items()
            if t["successes"] > current.get("tasks", {}).get(task, {}).get("successes", 0)
        ]
        raise RuntimeError(
            f"success_rate={rate} below baseline={reference} "
            f"({baseline.get('version')}, {baseline.get('model')}); regressed tasks: {lost}"
        )
    return f"success_rate={rate} >= baseline={reference}"
