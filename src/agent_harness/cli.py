"""Command line: run one task, run the eval suite, build the report, gate a version.

    agent-harness run --task evals/tasks/like_button
    agent-harness eval --all
    agent-harness report
    agent-harness gate --current out/v6/fake/summary.json --baseline evals/baseline.json
    agent-harness traces runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .evals import (
    assert_no_regression,
    inject,
    load_summaries,
    model_slug,
    render_report,
    run_suite,
)
from .loop import Harness
from .models.base import ModelError
from .models.factory import make_model
from .permissions import approve_all, approve_on_console
from .tasks import load_task, load_tasks
from .tracing import format_summary, read_trace, summarize_trace

DEFAULT_CONFIG = "configs/v6.yaml"


def _all_configs(directory: Path) -> list[Path]:
    def number(path: Path) -> int:
        digits = path.stem.lstrip("v")
        return int(digits) if digits.isdigit() else 0

    return sorted(directory.glob("v*.yaml"), key=number)


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    task = load_task(args.task)
    model_id = args.model or config.model.id
    approver = approve_on_console if args.approve == "console" else approve_all
    model = make_model(model_id, task, config.model)
    trace = Path(args.traces) / config.version / model_slug(model_id) / f"{task.id}.jsonl"
    result = Harness(config, model, approver).run(task, trace)
    print(format_summary(summarize_trace(read_trace(trace))))
    print(f"trace: {trace}")
    print(f"verification: tests_passed={result.verification['tests_passed']} "
          f"tampered={result.verification['tampered_files']}")
    return 0 if result.success else 1


def cmd_eval(args: argparse.Namespace) -> int:
    configs = _all_configs(Path(args.configs_dir)) if args.all else [Path(args.config)]
    tasks = load_tasks(args.tasks)
    if args.only:
        tasks = [t for t in tasks if t.id in set(args.only)]
    for path in configs:
        config = load_config(path)
        model_id = args.model or config.model.id
        print(f"== {config.version}: {config.description.splitlines()[0] if config.description else ''}")
        summary = run_suite(
            config,
            model_id,
            tasks,
            out_dir=Path(args.out),
            traces_dir=Path(args.traces),
            repeats=args.repeats,
        )
        print(
            f"   success {summary['successes']}/{summary['runs']} "
            f"({summary['success_rate']:.0%})  mean steps {summary['mean_steps']}  "
            f"tokens {summary['input_tokens']}/{summary['output_tokens']}  "
            f"cost ${summary['cost_usd']:.4f}  {summary['seconds']}s"
        )
        if args.write_baseline:
            Path(args.write_baseline).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"   baseline written to {args.write_baseline}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = render_report(load_summaries(Path(args.results)))
    Path(args.output).write_text("# Eval results\n\n" + report + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if args.readme:
        readme = Path(args.readme)
        readme.write_text(inject(readme.read_text(encoding="utf-8"), report), encoding="utf-8")
        print(f"updated {args.readme}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    try:
        print("gate passed:", assert_no_regression(current, baseline))
    except RuntimeError as exc:
        print(f"gate FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    path = Path(args.path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        print(format_summary(summarize_trace(read_trace(file))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-harness", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent on one task")
    run.add_argument("--task", required=True, help="task directory, e.g. evals/tasks/like_button")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--model", help="fake, claude-* or gemini-* (default: the config's)")
    run.add_argument("--approve", choices=["auto", "console"], default="auto")
    run.add_argument("--traces", default="runs")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="run the eval suite for one or all versions")
    ev.add_argument("--config", default=DEFAULT_CONFIG)
    ev.add_argument("--all", action="store_true", help="every configs/v*.yaml, in order")
    ev.add_argument("--configs-dir", default="configs")
    ev.add_argument("--model", help="fake, claude-* or gemini-* (default: the config's)")
    ev.add_argument("--tasks", default="evals/tasks")
    ev.add_argument("--only", nargs="*", help="task ids to run")
    ev.add_argument("--repeats", type=int, default=1)
    ev.add_argument("--out", default="results")
    ev.add_argument("--traces", default="runs")
    ev.add_argument("--write-baseline", help="also write the summary to this path")
    ev.set_defaults(func=cmd_eval)

    rep = sub.add_parser("report", help="build the comparison table from results/")
    rep.add_argument("--results", default="results")
    rep.add_argument("--output", default="docs/results.md")
    rep.add_argument("--readme", help="also inject the table into this README")
    rep.set_defaults(func=cmd_report)

    gate = sub.add_parser("gate", help="fail if success rate drops below the baseline")
    gate.add_argument("--current", required=True)
    gate.add_argument("--baseline", default="evals/baseline.json")
    gate.set_defaults(func=cmd_gate)

    tr = sub.add_parser("traces", help="summarize JSONL traces (a file or a directory)")
    tr.add_argument("path")
    tr.set_defaults(func=cmd_traces)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
