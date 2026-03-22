#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


EXIT_CODE_RE = re.compile(r"Process exited with code (\d+)")
PATCH_FILE_RE = re.compile(r"^\*\*\* (Add File|Update File|Delete File): (.+)$", re.MULTILINE)
PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)

KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "apply_patch": re.compile(r"\bapply_patch\b", re.IGNORECASE),
    "exec_command": re.compile(r"\bexec_command\b", re.IGNORECASE),
    "write_stdin": re.compile(r"\bwrite_stdin\b", re.IGNORECASE),
    "update_plan": re.compile(r"\bupdate_plan\b", re.IGNORECASE),
    "blinded": re.compile(r"\bblinded\b", re.IGNORECASE),
    "unblinded": re.compile(r"\bunblinded\b", re.IGNORECASE),
    "asimov": re.compile(r"\bAsimov\b", re.IGNORECASE),
    "significance": re.compile(r"\bsignificance\b", re.IGNORECASE),
    "background_pdf_choice": re.compile(r"\bbackground_pdf_choice\b", re.IGNORECASE),
    "roofit": re.compile(r"\bRooFit\b", re.IGNORECASE),
    "conditional_pass": re.compile(r"\bconditional_pass\b", re.IGNORECASE),
    "capped_noncompliant": re.compile(r"\bcapped_noncompliant\b", re.IGNORECASE),
    "blocked": re.compile(r"\bblocked\b", re.IGNORECASE),
    "policy_rejection": re.compile(r"blocked by policy|Rejected\(", re.IGNORECASE),
}

FAILURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "policy_rejection": re.compile(r"blocked by policy|Rejected\(", re.IGNORECASE),
    "traceback": re.compile(r"\bTraceback\b"),
    "runtime_error": re.compile(r"\bRuntimeError\b"),
    "roofit_duplicate_argument": re.compile(r"RooArgSet::checkForDup"),
    "conditional_pass": re.compile(r"\bconditional_pass\b", re.IGNORECASE),
    "capped_noncompliant": re.compile(r"\bcapped_noncompliant\b", re.IGNORECASE),
    "warning": re.compile(r"\bwarning\b", re.IGNORECASE),
    "blocked": re.compile(r"\bblocked\b", re.IGNORECASE),
}

SEARCH_GUIDE = [
    {
        "label": "Turn spine",
        "pattern": r'"user_message"|"update_plan"|"task_complete"|"phase":"final_answer"',
    },
    {
        "label": "Tool activity",
        "pattern": r'"apply_patch"|exec_command|write_stdin',
    },
    {
        "label": "Failures and recoveries",
        "pattern": r"Process exited with code|Traceback|RuntimeError|ERROR|blocked|warning|Rejected",
    },
    {
        "label": "Physics outcomes",
        "pattern": r"blinded|unblinded|Asimov|significance|background_pdf_choice|conditional_pass|capped_noncompliant",
    },
]


@dataclass
class TurnSummary:
    turn_id: str
    turn_index: int
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    user_messages: list[dict[str, Any]] = field(default_factory=list)
    commentary_messages: list[dict[str, Any]] = field(default_factory=list)
    final_answers: list[dict[str, Any]] = field(default_factory=list)
    task_complete: dict[str, Any] | None = None
    tool_counts: Counter[str] = field(default_factory=Counter)
    custom_tool_counts: Counter[str] = field(default_factory=Counter)
    patch_files: list[dict[str, str]] = field(default_factory=list)
    nonzero_exit_codes: list[int] = field(default_factory=list)
    failure_markers: Counter[str] = field(default_factory=Counter)


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def extract_command_head(command: str | None) -> str | None:
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return None
    head = Path(tokens[0]).name
    if head in {"bash", "zsh", "sh"} and len(tokens) >= 3 and tokens[1] == "-lc":
        inner = tokens[2]
        return extract_command_head(inner)
    return head


def patch_operations(patch_text: str) -> list[dict[str, str]]:
    operations: list[dict[str, str]] = []
    for action, path in PATCH_FILE_RE.findall(patch_text):
        operations.append({"action": action.lower().replace(" ", "_"), "path": path})
    for moved_path in PATCH_MOVE_RE.findall(patch_text):
        operations.append({"action": "move_to", "path": moved_path})
    return operations


def relativize_path(path: str, workspace_root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(workspace_root))
    except Exception:
        return str(path_obj)


def trim_text(text: str, *, max_len: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def summarize_user_message(message: str) -> str:
    needle = "## My request for Codex:"
    if needle in message:
        message = message.split(needle, 1)[1].strip()
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    sentence_candidates = re.split(r"(?<=[.!?])\s+", " ".join(lines))
    summary = sentence_candidates[0]
    if len(summary) < 120 and len(sentence_candidates) > 1:
        summary = f"{summary} {sentence_candidates[1]}"
    return trim_text(summary, max_len=180)


def percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(100.0 * count / total):.1f}%"


def extract_text_from_message_payload(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "output_text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    return ""


def detect_failure_markers(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for label, pattern in FAILURE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            counts[label] += len(matches)
    return counts


def collect_summary(records: list[dict[str, Any]], trace_path: Path) -> dict[str, Any]:
    top_level_counts: Counter[str] = Counter()
    response_item_counts: Counter[str] = Counter()
    event_msg_counts: Counter[str] = Counter()
    message_phase_counts: Counter[str] = Counter()
    function_call_counts: Counter[str] = Counter()
    custom_tool_counts: Counter[str] = Counter()
    exec_command_heads: Counter[str] = Counter()
    exit_code_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    patch_file_actions: Counter[str] = Counter()
    patch_files: list[dict[str, str]] = []

    session_meta: dict[str, Any] | None = None
    plan_updates: list[dict[str, Any]] = []
    final_token_event: dict[str, Any] | None = None

    turns: dict[str, TurnSummary] = {}
    turn_order: list[str] = []
    current_turn_id: str | None = None

    full_text_chunks: list[str] = []

    def ensure_turn(turn_id: str) -> TurnSummary:
        if turn_id not in turns:
            turns[turn_id] = TurnSummary(turn_id=turn_id, turn_index=len(turn_order) + 1)
            turn_order.append(turn_id)
        return turns[turn_id]

    for record in records:
        line_no = record["_line_no"]
        timestamp = record.get("timestamp")
        top_level_counts[record.get("type", "unknown")] += 1

        if record["type"] == "session_meta":
            session_meta = record.get("payload", {})

        if record["type"] == "turn_context":
            payload = record.get("payload", {})
            current_turn_id = payload["turn_id"]
            turn = ensure_turn(current_turn_id)
            turn.start_timestamp = turn.start_timestamp or timestamp
            turn.line_start = turn.line_start or line_no

        payload = record.get("payload", {})
        record_turn_id = None
        if record["type"] == "event_msg" and payload.get("turn_id") in turns:
            record_turn_id = payload["turn_id"]
        elif current_turn_id is not None:
            record_turn_id = current_turn_id

        if record_turn_id is not None:
            turn = ensure_turn(record_turn_id)
            turn.end_timestamp = timestamp
            turn.line_end = line_no
            if turn.line_start is None:
                turn.line_start = line_no
            if turn.start_timestamp is None:
                turn.start_timestamp = timestamp

        if record["type"] == "response_item":
            payload_type = payload.get("type", "unknown")
            response_item_counts[payload_type] += 1
            if payload_type == "message":
                phase = payload.get("phase") or "(none)"
                message_phase_counts[phase] += 1
                message_text = extract_text_from_message_payload(payload)
                if message_text:
                    full_text_chunks.append(message_text)
                if record_turn_id is not None:
                    turn = ensure_turn(record_turn_id)
                    message_record = {"line_no": line_no, "phase": phase, "text": message_text}
                    if phase == "commentary":
                        turn.commentary_messages.append(message_record)
                    elif phase == "final_answer":
                        turn.final_answers.append(message_record)
            elif payload_type == "function_call":
                name = payload.get("name", "unknown")
                function_call_counts[name] += 1
                arguments = maybe_json_loads(payload.get("arguments"))
                if record_turn_id is not None:
                    ensure_turn(record_turn_id).tool_counts[name] += 1
                if name == "update_plan":
                    if isinstance(arguments, str):
                        arguments = maybe_json_loads(arguments)
                    if isinstance(arguments, dict):
                        plan_updates.append({"line_no": line_no, "payload": arguments})
                if name == "exec_command" and isinstance(arguments, dict):
                    head = extract_command_head(arguments.get("cmd"))
                    if head:
                        exec_command_heads[head] += 1
            elif payload_type == "function_call_output":
                output_text = payload.get("output") or ""
                full_text_chunks.append(output_text)
                for code in EXIT_CODE_RE.findall(output_text):
                    exit_code_counts[code] += 1
                    if record_turn_id is not None and code != "0":
                        ensure_turn(record_turn_id).nonzero_exit_codes.append(int(code))
                marker_counts = detect_failure_markers(output_text)
                failure_counts.update(marker_counts)
                if record_turn_id is not None:
                    ensure_turn(record_turn_id).failure_markers.update(marker_counts)
            elif payload_type == "custom_tool_call":
                name = payload.get("name", "unknown")
                custom_tool_counts[name] += 1
                if record_turn_id is not None:
                    ensure_turn(record_turn_id).custom_tool_counts[name] += 1
                patch_input = payload.get("input") or payload.get("arguments") or ""
                if name == "apply_patch":
                    operations = patch_operations(patch_input)
                    patch_files.extend(operations)
                    for operation in operations:
                        patch_file_actions[operation["action"]] += 1
                    if record_turn_id is not None:
                        ensure_turn(record_turn_id).patch_files.extend(operations)
                    full_text_chunks.append(patch_input)
            elif payload_type == "custom_tool_call_output":
                output_text = payload.get("output") or ""
                full_text_chunks.append(output_text)
                marker_counts = detect_failure_markers(output_text)
                failure_counts.update(marker_counts)
                if record_turn_id is not None:
                    ensure_turn(record_turn_id).failure_markers.update(marker_counts)
            elif payload_type == "reasoning":
                encrypted = payload.get("encrypted_content")
                if encrypted:
                    full_text_chunks.append("reasoning_encrypted")

        elif record["type"] == "event_msg":
            event_type = payload.get("type", "unknown")
            event_msg_counts[event_type] += 1
            if event_type == "user_message":
                message = payload.get("message", "")
                full_text_chunks.append(message)
                if record_turn_id is not None:
                    ensure_turn(record_turn_id).user_messages.append(
                        {
                            "line_no": line_no,
                            "summary": summarize_user_message(message),
                            "message": message,
                        }
                    )
            elif event_type == "agent_message":
                message = payload.get("message", "")
                full_text_chunks.append(message)
            elif event_type == "task_complete":
                last_message = payload.get("last_agent_message", "")
                full_text_chunks.append(last_message)
                final_marker_counts = detect_failure_markers(last_message)
                failure_counts.update(final_marker_counts)
                if record_turn_id is not None:
                    turn = ensure_turn(record_turn_id)
                    turn.task_complete = {
                        "line_no": line_no,
                        "message": last_message,
                    }
                    turn.failure_markers.update(final_marker_counts)
            elif event_type == "token_count":
                final_token_event = payload

    full_text = "\n".join(full_text_chunks)
    for keyword, pattern in KEYWORD_PATTERNS.items():
        matches = pattern.findall(full_text)
        if matches:
            keyword_counts[keyword] = len(matches)

    start_time = parse_timestamp(records[0].get("timestamp")) if records else None
    end_time = parse_timestamp(records[-1].get("timestamp")) if records else None
    wall_seconds = None
    if start_time is not None and end_time is not None:
        wall_seconds = (end_time - start_time).total_seconds()

    unique_patch_paths = sorted({operation["path"] for operation in patch_files})
    workspace_root = trace_path.resolve().parent
    touched_directories = Counter()
    relative_patch_paths: list[str] = []
    for path in unique_patch_paths:
        relative_str = relativize_path(path, workspace_root)
        relative_patch_paths.append(relative_str)
        parts = Path(relative_str).parts
        touched_directories[parts[0] if len(parts) > 1 else "."] += 1

    turn_payloads: list[dict[str, Any]] = []
    for turn_id in turn_order:
        turn = turns[turn_id]
        start_dt = parse_timestamp(turn.start_timestamp)
        end_dt = parse_timestamp(turn.end_timestamp)
        duration_seconds = None
        if start_dt is not None and end_dt is not None:
            duration_seconds = (end_dt - start_dt).total_seconds()
        turn_payloads.append(
            {
                "turn_id": turn.turn_id,
                "turn_index": turn.turn_index,
                "line_range": {"start": turn.line_start, "end": turn.line_end},
                "start_timestamp": turn.start_timestamp,
                "end_timestamp": turn.end_timestamp,
                "duration_seconds": duration_seconds,
                "user_requests": [
                    {
                        "line_no": item["line_no"],
                        "summary": item["summary"],
                    }
                    for item in turn.user_messages
                ],
                "commentary_count": len(turn.commentary_messages),
                "final_answer_count": len(turn.final_answers),
                "tool_counts": dict(turn.tool_counts),
                "custom_tool_counts": dict(turn.custom_tool_counts),
                "patch_files": turn.patch_files,
                "patch_files_relative": [
                    {**item, "path": relativize_path(item["path"], workspace_root)}
                    for item in turn.patch_files
                ],
                "nonzero_exit_codes": turn.nonzero_exit_codes,
                "failure_markers": dict(turn.failure_markers),
                "task_complete": turn.task_complete,
            }
        )

    summary = {
        "trace_path": str(trace_path),
        "record_count": len(records),
        "line_count": records[-1]["_line_no"] if records else 0,
        "time_span": {
            "start": records[0].get("timestamp") if records else None,
            "end": records[-1].get("timestamp") if records else None,
            "wall_clock_seconds": wall_seconds,
        },
        "session_meta": {
            "id": (session_meta or {}).get("id"),
            "originator": (session_meta or {}).get("originator"),
            "source": (session_meta or {}).get("source"),
            "cli_version": (session_meta or {}).get("cli_version"),
            "model_provider": (session_meta or {}).get("model_provider"),
        },
        "top_level_counts": dict(top_level_counts),
        "response_item_counts": dict(response_item_counts),
        "event_msg_counts": dict(event_msg_counts),
        "message_phase_counts": dict(message_phase_counts),
        "function_call_counts": dict(function_call_counts),
        "custom_tool_counts": dict(custom_tool_counts),
        "exec_command_head_counts": dict(exec_command_heads.most_common()),
        "exit_code_counts": dict(exit_code_counts),
        "failure_marker_counts": dict(failure_counts),
        "keyword_counts": dict(keyword_counts),
        "plan_updates": plan_updates,
        "turns": turn_payloads,
        "edit_summary": {
            "patch_calls": custom_tool_counts.get("apply_patch", 0),
            "file_operation_counts": dict(patch_file_actions),
            "unique_files_touched": unique_patch_paths,
            "unique_files_touched_relative": relative_patch_paths,
            "touched_directory_counts": dict(touched_directories),
        },
        "token_usage": (final_token_event or {}).get("info", {}).get("total_token_usage"),
        "rate_limits": final_token_event.get("rate_limits") if final_token_event else None,
        "search_guide": SEARCH_GUIDE,
    }
    return summary


def top_examples(summary: dict[str, Any], key: str, limit: int = 5) -> list[tuple[str, int]]:
    counts = Counter(summary.get(key, {}))
    return counts.most_common(limit)


def load_figure_manifest(figure_dir: Path | None) -> list[dict[str, Any]]:
    if figure_dir is None:
        return []
    manifest_path = figure_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        figures = payload.get("figures", [])
        normalized: list[dict[str, Any]] = []
        for figure in figures:
            path = Path(figure["path"]).resolve()
            normalized.append(
                {
                    "title": figure.get("title", path.name),
                    "caption": figure.get("caption", ""),
                    "path": str(path),
                }
            )
        return normalized
    figures = []
    for path in sorted(figure_dir.glob("*.svg")):
        figures.append({"title": path.stem, "caption": "", "path": str(path.resolve())})
    return figures


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes, remaining = divmod(int(round(seconds)), 60)
    if minutes == 0:
        return f"{remaining}s"
    return f"{minutes}m {remaining:02d}s"


def build_abstract(summary: dict[str, Any]) -> str:
    turns = summary["turns"]
    top_level = summary["top_level_counts"]
    response_items = summary["response_item_counts"]
    tools = summary["function_call_counts"]
    edits = summary["edit_summary"]
    return (
        f"This report analyzes a Codex rollout trace containing {summary['record_count']} JSONL records "
        f"across {len(turns)} turns and {format_duration(summary['time_span']['wall_clock_seconds'])} of wall-clock activity. "
        f"The trace is dominated by `response_item` records ({top_level.get('response_item', 0)} / {summary['record_count']}) "
        f"and shows a tool-heavy execution profile led by `{max(tools, key=tools.get)}` calls. "
        f"The session exhibits a repair-and-run pattern: reconnaissance and contract checking, four patch applications touching "
        f"{len(edits['unique_files_touched'])} files, then blinded and unblinded physics runs with explicit treatment of warnings and policy gates."
    )


def turn_label(turn: dict[str, Any]) -> str:
    request_summaries = [item["summary"] for item in turn["user_requests"] if item["summary"]]
    if request_summaries:
        return request_summaries[0]
    if turn.get("task_complete", {}).get("message"):
        return trim_text(turn["task_complete"]["message"], max_len=120)
    return f"Turn {turn['turn_index']}"


def build_markdown(summary: dict[str, Any], figure_manifest: list[dict[str, Any]] | None = None) -> str:
    top_level = summary["top_level_counts"]
    response_items = summary["response_item_counts"]
    event_msgs = summary["event_msg_counts"]
    tools = summary["function_call_counts"]
    custom_tools = summary["custom_tool_counts"]
    edits = summary["edit_summary"]
    token_usage = summary.get("token_usage") or {}
    top_commands = top_examples(summary, "exec_command_head_counts", limit=8)
    top_failures = top_examples(summary, "failure_marker_counts", limit=8)
    top_keywords = top_examples(summary, "keyword_counts", limit=10)

    lines: list[str] = []
    lines.append("# Paper-Style Examination of a Codex Rollout Trace")
    lines.append("")
    lines.append("## Abstract")
    lines.append(build_abstract(summary))
    lines.append("")
    if figure_manifest:
        lines.append("## Visual Summary")
        lines.append(
            "The figures below are generated directly from the rollout trace and are meant to complement the textual analysis with a compact, paper-style visual narrative."
        )
        lines.append("")
        for figure in figure_manifest:
            lines.append(f"![{figure['title']}]({figure['path']})")
            if figure.get("caption"):
                lines.append(f"*{figure['caption']}*")
            lines.append("")
    lines.append("## Corpus and Runtime Metadata")
    lines.append(f"- Trace file: `{summary['trace_path']}`")
    lines.append(f"- Records: `{summary['record_count']}` across `{summary['line_count']}` lines")
    lines.append(
        f"- Wall-clock span: `{summary['time_span']['start']}` to `{summary['time_span']['end']}` "
        f"({format_duration(summary['time_span']['wall_clock_seconds'])})"
    )
    lines.append(f"- Session id: `{summary['session_meta'].get('id')}`")
    lines.append(
        f"- Origin: `{summary['session_meta'].get('originator')}` via `{summary['session_meta'].get('source')}`, "
        f"CLI `{summary['session_meta'].get('cli_version')}`"
    )
    if token_usage:
        lines.append(
            f"- Final cumulative token usage: input `{token_usage.get('input_tokens')}`, cached input `{token_usage.get('cached_input_tokens')}`, "
            f"output `{token_usage.get('output_tokens')}`, reasoning output `{token_usage.get('reasoning_output_tokens')}`, "
            f"total `{token_usage.get('total_tokens')}`"
        )
    lines.append("")
    lines.append("## Structural Profile")
    lines.append(
        f"- Top-level mix: `session_meta` {top_level.get('session_meta', 0)}, `turn_context` {top_level.get('turn_context', 0)}, "
        f"`event_msg` {top_level.get('event_msg', 0)}, `response_item` {top_level.get('response_item', 0)}"
    )
    lines.append(
        f"- `response_item` subtypes: {', '.join(f'`{name}` {count}' for name, count in Counter(response_items).most_common())}"
    )
    lines.append(
        f"- `event_msg` subtypes: {', '.join(f'`{name}` {count}' for name, count in Counter(event_msgs).most_common())}"
    )
    lines.append(
        f"- Message phases: {', '.join(f'`{name}` {count}' for name, count in Counter(summary['message_phase_counts']).most_common())}"
    )
    lines.append(
        f"- Tool calls: {', '.join(f'`{name}` {count}' for name, count in Counter(tools).most_common())}"
    )
    if custom_tools:
        lines.append(
            f"- Custom tool calls: {', '.join(f'`{name}` {count}' for name, count in Counter(custom_tools).most_common())}"
        )
    if top_commands:
        lines.append(
            f"- Most common exec-command heads: {', '.join(f'`{name}` {count}' for name, count in top_commands)}"
        )
    lines.append("")
    lines.append("## Turn-by-Turn Narrative")
    for turn in summary["turns"]:
        label = turn_label(turn)
        lines.append(f"### Turn {turn['turn_index']}: {label}")
        lines.append(
            f"- Span: lines `{turn['line_range']['start']}`-`{turn['line_range']['end']}`, duration `{format_duration(turn['duration_seconds'])}`"
        )
        if turn["user_requests"]:
            lines.append(
                f"- User request(s): {', '.join(f'line {item['line_no']}: {item['summary']}' for item in turn['user_requests'])}"
            )
        if turn["tool_counts"]:
            lines.append(
                f"- Tool mix: {', '.join(f'`{name}` {count}' for name, count in Counter(turn['tool_counts']).most_common())}"
            )
        if turn["custom_tool_counts"]:
            lines.append(
                f"- Edit actions: {', '.join(f'`{name}` {count}' for name, count in Counter(turn['custom_tool_counts']).most_common())}"
            )
        if turn["patch_files"]:
            lines.append(
                f"- Files touched in this turn: {', '.join(sorted({item['path'] for item in turn.get('patch_files_relative', turn['patch_files'])}))}"
            )
        if turn["nonzero_exit_codes"]:
            lines.append(f"- Non-zero command exits: `{turn['nonzero_exit_codes']}`")
        if turn["failure_markers"]:
            lines.append(
                f"- Failure markers: {', '.join(f'`{name}` {count}' for name, count in Counter(turn['failure_markers']).most_common())}"
            )
        if turn.get("task_complete", {}).get("message"):
            lines.append(f"- Task completion summary: {trim_text(turn['task_complete']['message'], max_len=550)}")
        lines.append("")
    lines.append("## Editing Footprint")
    lines.append(
        f"- Patch calls: `{edits['patch_calls']}` with file operations {', '.join(f'`{name}` {count}' for name, count in Counter(edits['file_operation_counts']).most_common())}"
    )
    lines.append(
        f"- Unique files touched: `{len(edits['unique_files_touched'])}`"
    )
    lines.append(
        f"- Directory concentration: {', '.join(f'`{name}` {count}' for name, count in Counter(edits['touched_directory_counts']).most_common())}"
    )
    if edits["unique_files_touched"]:
        lines.append("- File list:")
        for path in edits.get("unique_files_touched_relative", edits["unique_files_touched"]):
            lines.append(f"  - `{path}`")
    lines.append("")
    if summary["plan_updates"]:
        lines.append("## Planning Signal")
        for item in summary["plan_updates"]:
            payload = item["payload"]
            explanation = payload.get("explanation", "")
            if explanation:
                lines.append(f"- Plan update at line `{item['line_no']}`: {trim_text(explanation, max_len=260)}")
            plan_steps = payload.get("plan") or []
            if plan_steps:
                rendered_steps = ", ".join(
                    f"{step['status']}:{step['step']}" for step in plan_steps
                )
                lines.append(
                    f"- Steps: {rendered_steps}"
                )
        lines.append("")
    lines.append("## Failure, Recovery, and Governance Signals")
    lines.append(
        f"- Exit-code distribution: {', '.join(f'`{code}` {count}' for code, count in Counter(summary['exit_code_counts']).most_common())}"
    )
    if top_failures:
        lines.append(
            f"- Most frequent failure or governance markers: {', '.join(f'`{name}` {count}' for name, count in top_failures)}"
        )
    lines.append(
        "- Qualitative reading: the trace contains both hard operational interruptions and soft scientific cautions. "
        "A destructive shell command was rejected by policy, but the agent continued through alternative actions. "
        "Later outputs preserve repeated RooFit duplicate-argument warnings and mark downstream readiness as conditional rather than silently upgrading the result."
    )
    lines.append("")
    lines.append("## Behavioral Interpretation")
    lines.append(
        "- The session shows cross-turn persistence. Turn 1 establishes the repaired environment and blinded outputs, Turn 2 answers a narrow read-only follow-up from generated artifacts, and Turn 3 re-enters the pipeline in unblinded mode rather than starting from scratch."
    )
    lines.append(
        "- The run is tool-centric rather than chat-centric. `response_item` messages are outnumbered by tool calls, and `exec_command` plus `write_stdin` dominate the action budget."
    )
    lines.append(
        "- The agent behaves in a governance-aware way. It emits a plan, preserves `blocked` and `conditional_pass` states in summaries, and keeps warning-bearing scientific claims separated from clean pass states."
    )
    lines.append(
        "- The trace also exhibits direct code intervention. Four `apply_patch` calls modify environment wrappers, analysis runtime defaults, statistical fitting behavior, and reporting surfaces before the physics outputs are accepted."
    )
    lines.append("")
    lines.append("## Outcome Claims Captured in the Trace")
    for turn in summary["turns"]:
        message = turn.get("task_complete", {}).get("message")
        if message:
            lines.append(f"- Turn {turn['turn_index']} claim: {trim_text(message, max_len=600)}")
    lines.append("")
    lines.append("## Search and Index Terms")
    if top_keywords:
        lines.append(
            f"- High-signal keywords present in the trace: {', '.join(f'`{name}` {count}' for name, count in top_keywords)}"
        )
    lines.append("- Suggested `rg` patterns:")
    for item in summary["search_guide"]:
        lines.append(f"  - {item['label']}: `{item['pattern']}`")
    lines.append("")
    lines.append("## Reproducibility")
    if figure_manifest:
        lines.append(
            f"- Regenerate visual assets with: `python3 scripts/render_rollout_visuals.py {summary['trace_path']} --output-dir reports/rollout_figures --data-out reports/rollout_visual_context.json`"
        )
        lines.append(
            f"- Rebuild the report with embedded figures using: `python3 scripts/analyze_rollout.py {summary['trace_path']} --json-out reports/rollout_examination_summary.json --md-out reports/rollout_examination_report.md --figure-dir reports/rollout_figures`"
        )
    else:
        lines.append(
            f"- Regenerate this report with: `python3 scripts/analyze_rollout.py {summary['trace_path']} --json-out reports/rollout_examination_summary.json --md-out reports/rollout_examination_report.md`"
        )
    lines.append(
        "- Threats to validity: the analyzer is schema-driven and extracts stable fields, but any scientific claim in this report is still second-order because it is derived from the session trace rather than recomputed directly from the analysis artifacts."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Codex rollout JSONL trace.")
    parser.add_argument("trace", type=Path, help="Path to rollout JSONL trace")
    parser.add_argument("--json-out", type=Path, help="Where to write the structured summary JSON")
    parser.add_argument("--md-out", type=Path, help="Where to write the Markdown report")
    parser.add_argument("--figure-dir", type=Path, help="Directory containing rendered SVG figures and an optional manifest.json")
    args = parser.parse_args()

    records = load_jsonl(args.trace)
    summary = collect_summary(records, args.trace)
    figure_manifest = load_figure_manifest(args.figure_dir)
    markdown = build_markdown(summary, figure_manifest=figure_manifest)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown)
    if not args.json_out and not args.md_out:
        print(markdown)


if __name__ == "__main__":
    main()
