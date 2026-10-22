"""Summarize JSONL traces: one block per run, plus totals across runs.

    python scripts/summarize_traces.py runs/
    python scripts/summarize_traces.py runs/v6/fake/like_button-0.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from agent_harness.tracing import format_summary, read_trace, summarize_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="a .jsonl trace or a directory of traces")
    args = parser.parse_args()
    path = Path(args.path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    summaries = [summarize_trace(read_trace(f)) for f in files]
    for summary in summaries:
        print(format_summary(summary))
    if len(summaries) > 1:
        stops = Counter(s["stop_reason"] for s in summaries)
        tools = Counter()
        for s in summaries:
            tools.update(s["tool_calls"])
        passed = sum(bool(s["success"]) for s in summaries)
        print(f"\n{len(summaries)} runs, {passed} passed; stop reasons: {dict(stops)}")
        print(f"tool calls: {dict(tools.most_common())}")


if __name__ == "__main__":
    main()
