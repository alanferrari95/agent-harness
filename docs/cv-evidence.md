# CV bullets → files and evidence

The two bullets this repository backs, claim by claim. "Evidence" points to a test that
fails if the claim stops being true, a row of the generated eval table, or a trace.

**Bullet 1.** *Implemented the seven subsystems of a coding agent: typed tools with
actionable errors, a loop with step/cost budgets and stuck detection, external state,
context compaction and an isolated workspace.*

| Claim | Files | Evidence |
|---|---|---|
| Seven subsystems | README table "Subsystems → files" | All seven in the minimal form of the paper's Table 1: loop, LLM integration, tools, memory and context, safety; orchestration's minimal form is none (single agent, Rec. 12, as in Aider and Mini-SWE-Agent); extensibility's is structural typing (`Model` protocol), plus YAML configs and rules as data |
| Typed tools | [`tools.py`](../src/agent_harness/tools.py) | JSON Schema per tool, validated before execution: `test_schema_validation_messages`, `test_every_tool_schema_is_strict` |
| Actionable errors | [`tools.py`](../src/agent_harness/tools.py) | `test_edit_not_found_quotes_closest_line`, `test_edit_ambiguous_reports_count_and_lines`, `test_edit_rejects_line_number_prefixes`, `test_missing_file_suggests_paths`; eval v1 → v2 fixes 3 tasks (30% → 60%) |
| Loop with step/cost budgets | [`loop.py`](../src/agent_harness/loop.py) | `test_step_budget`, `test_cost_budget`, `test_time_budget`, `test_malformed_replies_are_capped` |
| Stuck detection | [`loop.py`](../src/agent_harness/loop.py) | `test_stuck_detection_stops_on_third_identical_call`; eval v4 → v5: `stuck_search` ends at 3 steps instead of 15, mean steps 7.0 → 5.8 |
| External state | [`state.py`](../src/agent_harness/state.py) | `test_state_lives_outside_the_model` (the state JSON is in the next prompt) |
| Context compaction | [`context.py`](../src/agent_harness/context.py) | `test_compaction_folds_whole_steps_and_keeps_the_tail`, `test_summary_is_merged_incrementally`; eval v5 → v6: `long_context` goes from `context_overflow` to pass |
| Isolated workspace | [`workspace.py`](../src/agent_harness/workspace.py) | `test_copy_mode_never_touches_the_source`, `test_git_worktree_mode`, `test_resolve_refuses_escapes_and_git`, `test_the_original_repository_is_untouched` |

**Bullet 2.** *External verification (the harness runs the tests instead of trusting the
model), permission rules, JSONL traces and an eval suite to compare harness versions one
change at a time.*

| Claim | Files | Evidence |
|---|---|---|
| External verification | [`verification.py`](../src/agent_harness/verification.py), [`loop.py`](../src/agent_harness/loop.py) | `test_finish_is_a_claim_not_evidence`, `test_tampered_tests_do_not_fool_verification`, `test_verify_on_stop_rejects_untested_edits`; eval v2 → v3 fixes `like_button` |
| Permission rules | [`permissions.py`](../src/agent_harness/permissions.py), [`configs/`](../configs/) | `test_first_matching_rule_wins`, `test_undeclared_tools_do_not_exist`, `test_deny_rule_blocks_edits_to_tests`, `test_ask_rule_consults_the_approver`; eval v3 → v4 fixes `cheat_tests` |
| JSONL traces | [`tracing.py`](../src/agent_harness/tracing.py), [`scripts/summarize_traces.py`](../scripts/summarize_traces.py) | [`examples/trace_like_button_v1.jsonl`](examples/trace_like_button_v1.jsonl) (finish accepted, tests fail) vs [`examples/trace_like_button_v6.jsonl`](examples/trace_like_button_v6.jsonl) (finish rejected, tests run, fixed); every step stores the exact prompt: `test_trace_records_the_exact_prompt` |
| Eval suite | [`evals/tasks/`](../evals/tasks/), [`evals.py`](../src/agent_harness/evals.py) | 10 tasks with failing tests (`test_every_task_fixture_starts_red`); [`results.md`](results.md) generated from `results/*/*/summary.json` |
| One change at a time | [`configs/v1.yaml`](../configs/v1.yaml) … [`v6.yaml`](../configs/v6.yaml) | `test_each_version_changes_one_thing` fails if a version changes more than one knob |
| Regression gate in CI | [`ci.yml`](../.github/workflows/ci.yml), [`evals/baseline.json`](../evals/baseline.json) | `test_gate_*` (same contract as invoice-mlops's `assert_min_f1`); the `eval gate (fake model)` job |
| No framework | [`pyproject.toml`](../pyproject.toml) | Runtime dependency: `pyyaml` only; `anthropic` is an optional extra; Gemini over plain REST |
