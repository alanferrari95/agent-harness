"""Typed tools with actionable errors.

Each tool is a name, a description, a JSON Schema and a function. The harness validates
the arguments against the schema before running anything, and every failure comes back
to the model as text it can act on, never as an exception that kills the loop.

`actionable_errors` is a config switch so the eval suite can measure what the hints are
worth: with it off, every failure is a bare "ERROR: <tool> failed."
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models.base import ToolSpec
from .workspace import Workspace

MAX_OUTPUT_CHARS = 4000
MAX_SEARCH_HITS = 50
LINE_PREFIX = re.compile(r"^\s*\d+\| ", re.MULTILINE)
FINISH = "finish"


@dataclass
class ToolContext:
    workspace: Workspace
    test_command: list[str]
    actionable_errors: bool = True
    test_timeout: int = 120


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolFailure(Exception):
    """Raised inside a tool. `hint` is shown only when actionable errors are on."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


# ---------------------------------------------------------------- tool implementations


def search_code(ctx: ToolContext, query: str) -> str:
    ws = ctx.workspace
    needle = query.lower()
    hits = []
    for rel in ws.list_files():
        try:
            lines = (ws.root / rel).read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, start=1):
            if needle in line.lower():
                hits.append(f"{rel}:{number}: {line.strip()}")
    if not hits:
        raise ToolFailure(
            f"no matches for '{query}'",
            "Try a shorter or different term (a function name, a CSS class, a test name).",
        )
    shown = hits[:MAX_SEARCH_HITS]
    more = len(hits) - len(shown)
    footer = f"\n... {more} more matches; refine the query." if more else ""
    return "\n".join(shown) + footer


def _missing_file(ctx: ToolContext, path: str) -> ToolFailure:
    candidates = difflib.get_close_matches(path, ctx.workspace.list_files(), n=3, cutoff=0.4)
    hint = "Use search_code to find the right path."
    if candidates:
        hint = f"Did you mean: {', '.join(candidates)}?"
    return ToolFailure(f"file '{path}' does not exist", hint)


def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = 200) -> str:
    target = ctx.workspace.resolve(path)
    if not target.is_file():
        raise _missing_file(ctx, path)
    lines = target.read_text(encoding="utf-8").splitlines()
    start = max(offset, 1)
    chunk = lines[start - 1 : start - 1 + limit]
    body = "\n".join(f"{n:4}| {line}" for n, line in enumerate(chunk, start=start))
    end = start + len(chunk) - 1
    header = f"{path} (lines {start}-{end} of {len(lines)})"
    if end < len(lines):
        header += f"; call read_file with offset={end + 1} for more"
    return f"{header}\n{body}"


def _closest_line_hint(content: str, old_text: str, path: str) -> str:
    first = next((ln for ln in old_text.splitlines() if ln.strip()), "")
    lines = content.splitlines()
    stripped = [ln.strip() for ln in lines]
    match = difflib.get_close_matches(first.strip(), stripped, n=1, cutoff=0.5)
    if not match:
        return f"No similar line found; call read_file on {path} and copy the text exactly."
    index = stripped.index(match[0])
    exact = json.dumps(lines[index])
    return (
        f"Closest line (line {index + 1}): {exact}. "
        "Copy the exact text from there, including whitespace."
    )


def edit_file(ctx: ToolContext, path: str, old_text: str, new_text: str) -> str:
    target = ctx.workspace.resolve(path)
    if not target.is_file():
        raise _missing_file(ctx, path)
    if old_text == new_text:
        raise ToolFailure("old_text and new_text are identical", "Nothing would change.")
    if LINE_PREFIX.search(old_text):
        raise ToolFailure(
            "old_text contains line-number prefixes like '  12| '",
            "Those come from read_file output; remove them and send only the file text.",
        )
    content = target.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        raise ToolFailure(
            f"old_text not found in {path}", _closest_line_hint(content, old_text, path)
        )
    if count > 1:
        starts = [m.start() for m in re.finditer(re.escape(old_text), content)]
        line_numbers = ", ".join(str(content.count("\n", 0, s) + 1) for s in starts)
        raise ToolFailure(
            f"old_text appears {count} times in {path} (lines {line_numbers})",
            "Include more surrounding lines so the match is unique.",
        )
    start_line = content.count("\n", 0, content.index(old_text)) + 1
    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    end_line = start_line + new_text.count("\n")
    return f"OK: edited {path} (now lines {start_line}-{end_line})."


def run_tests(ctx: ToolContext) -> str:
    code, output = ctx.workspace.run(ctx.test_command, timeout=ctx.test_timeout)
    status = "PASSED" if code == 0 else f"FAILED (exit code {code})"
    return f"{status}\n{_truncate(output, 2000)}"


# ---------------------------------------------------------------- catalog

TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            "search_code",
            "Search all text files in the project for a case-insensitive substring. "
            "Returns path:line: text for each match (up to 50).",
            _schema({"query": {"type": "string", "description": "Text to look for."}}, ["query"]),
            search_code,
        ),
        Tool(
            "read_file",
            "Read a file with line numbers ('  12| text'). Use offset/limit for long files. "
            "Line-number prefixes are not part of the file.",
            _schema(
                {
                    "path": {"type": "string", "description": "Path relative to the project."},
                    "offset": {"type": "integer", "description": "First line, 1-based."},
                    "limit": {"type": "integer", "description": "Maximum lines to return."},
                },
                ["path"],
            ),
            read_file,
        ),
        Tool(
            "edit_file",
            "Replace old_text with new_text in a file. old_text must appear exactly once; "
            "copy it from read_file output without the line-number prefixes. "
            "Never edit by line number.",
            _schema(
                {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Exact text to replace."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                ["path", "old_text", "new_text"],
            ),
            edit_file,
        ),
        Tool(
            "run_tests",
            "Run the project's test suite. Returns PASSED or FAILED with the output tail.",
            _schema({}, []),
            run_tests,
        ),
        Tool(
            FINISH,
            "Call when the task is done. The harness will verify the result itself.",
            _schema({"summary": {"type": "string", "description": "What you changed."}}, []),
            lambda ctx, summary="": summary,
        ),
    ]
}

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def validate_args(tool: Tool, args: Any) -> str | None:
    """Return an actionable message if args do not match the tool's schema."""
    schema = tool.parameters
    if not isinstance(args, dict):
        return f"arguments for {tool.name} must be a JSON object"
    expected = ", ".join(schema["properties"]) or "none"
    missing = [name for name in schema["required"] if name not in args]
    if missing:
        return f"missing required argument(s) {missing} for {tool.name}; expected: {expected}"
    unknown = [name for name in args if name not in schema["properties"]]
    if unknown:
        return f"unknown argument(s) {unknown} for {tool.name}; expected: {expected}"
    for name, value in args.items():
        kind = schema["properties"][name].get("type")
        python_type = _JSON_TYPES.get(kind)
        is_bool = isinstance(value, bool) and kind != "boolean"
        if python_type and (not isinstance(value, python_type) or is_bool):
            return f"argument '{name}' of {tool.name} must be a {kind}"
    return None


def execute(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> str:
    """Run a tool. Every failure becomes an observation, never an exception."""
    problem = validate_args(tool, args)
    if problem:
        return f"ERROR: {problem}." if ctx.actionable_errors else f"ERROR: {tool.name} failed."
    try:
        return tool.func(ctx, **args)
    except ToolFailure as failure:
        if not ctx.actionable_errors:
            return f"ERROR: {tool.name} failed."
        return f"ERROR: {failure.message}. {failure.hint}".rstrip()
    except PermissionError as exc:
        # The workspace guard is a floor: always explained, whatever the config says.
        return f"DENIED: {exc}."
    except Exception as exc:  # noqa: BLE001 - any tool bug goes back to the model as data
        return f"ERROR: {type(exc).__name__}: {exc}"
