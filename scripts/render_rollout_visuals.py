#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.analyze_rollout import (
    EXIT_CODE_RE,
    collect_summary,
    detect_failure_markers,
    format_duration,
    load_jsonl,
    parse_timestamp,
    patch_operations,
    relativize_path,
    trim_text,
    turn_label,
)


PALETTE = {
    "bg": "#fbf7ef",
    "panel": "#f2ede4",
    "panel_stroke": "#dfd6c7",
    "text": "#1f2937",
    "muted": "#667085",
    "grid": "#d7d1c5",
    "tool_call": "#3f83d6",
    "tool_output": "#38a3a5",
    "assistant_msg": "#f4a261",
    "reasoning": "#8f95a2",
    "user_msg": "#c084fc",
    "audit": "#b8d5f2",
    "blinded": "#bfe6d4",
    "query": "#f7e2ae",
    "unblinded": "#f6d1c8",
    "patch": "#3f83d6",
    "milestone": "#2a9d8f",
    "warning": "#dd6b20",
    "exit": "#c53030",
    "turn1": "#4f7dd4",
    "turn2": "#e9a03b",
    "turn3": "#d96b5f",
}

ACTIVITY_COLORS = {
    "tool_call": PALETTE["tool_call"],
    "tool_output": PALETTE["tool_output"],
    "assistant_msg": PALETTE["assistant_msg"],
    "reasoning": PALETTE["reasoning"],
    "user_msg": PALETTE["user_msg"],
}


def fmt_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return f"{seconds / 60.0:.1f} min"


def fmt_tokens(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def fmt_time(dt: datetime | None) -> str:
    if dt is None:
        return "n/a"
    return dt.strftime("%H:%M")


def estimate_text_width(text: str, font_size: int) -> float:
    return len(text) * font_size * 0.58


class SvgCanvas:
    def __init__(self, width: int, height: int, background: str = PALETTE["bg"]) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            """
<style>
  text { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; fill: #1f2937; }
  .title { font-size: 34px; font-weight: 700; }
  .subtitle { font-size: 16px; fill: #667085; }
  .section { font-size: 22px; font-weight: 650; }
  .card-title { font-size: 17px; fill: #667085; }
  .card-value { font-size: 46px; font-weight: 700; }
  .card-note { font-size: 13px; fill: #667085; }
  .axis { font-size: 12px; fill: #667085; }
  .small { font-size: 12px; fill: #667085; }
  .label { font-size: 14px; font-weight: 600; }
  .caption { font-size: 13px; fill: #667085; }
</style>
""",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}" />',
        ]

    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        rx: float = 18,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x="{x:.2f}"',
            f'y="{y:.2f}"',
            f'width="{width:.2f}"',
            f'height="{height:.2f}"',
            f'fill="{fill}"',
            f'rx="{rx:.2f}"',
        ]
        if stroke:
            attrs.append(f'stroke="{stroke}"')
            attrs.append(f'stroke-width="{stroke_width:.2f}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<rect {' '.join(attrs)} />")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1.0,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x1="{x1:.2f}"',
            f'y1="{y1:.2f}"',
            f'x2="{x2:.2f}"',
            f'y2="{y2:.2f}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
            'stroke-linecap="round"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<line {' '.join(attrs)} />")

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
    ) -> None:
        attrs = [
            f'cx="{x:.2f}"',
            f'cy="{y:.2f}"',
            f'r="{radius:.2f}"',
            f'fill="{fill}"',
        ]
        if stroke:
            attrs.append(f'stroke="{stroke}"')
            attrs.append(f'stroke-width="{stroke_width:.2f}"')
        self.add(f"<circle {' '.join(attrs)} />")

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        opacity: float | None = None,
    ) -> None:
        point_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        attrs = [f'points="{point_attr}"', f'fill="{fill}"']
        if stroke:
            attrs.append(f'stroke="{stroke}"')
            attrs.append(f'stroke-width="{stroke_width:.2f}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<polygon {' '.join(attrs)} />")

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str = "none",
        stroke: str,
        stroke_width: float = 2.0,
        opacity: float | None = None,
    ) -> None:
        point_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        attrs = [
            f'points="{point_attr}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
            'stroke-linecap="round"',
            'stroke-linejoin="round"',
        ]
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<polyline {' '.join(attrs)} />")

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        cls: str | None = None,
        fill: str | None = None,
        font_size: int | None = None,
        font_weight: int | None = None,
        anchor: str = "start",
        baseline: str = "middle",
    ) -> None:
        attrs = [
            f'x="{x:.2f}"',
            f'y="{y:.2f}"',
            f'text-anchor="{anchor}"',
            f'dominant-baseline="{baseline}"',
        ]
        if cls:
            attrs.append(f'class="{cls}"')
        if fill:
            attrs.append(f'fill="{fill}"')
        if font_size:
            attrs.append(f'font-size="{font_size}"')
        if font_weight:
            attrs.append(f'font-weight="{font_weight}"')
        self.add(f"<text {' '.join(attrs)}>{html.escape(value)}</text>")

    def text_block(
        self,
        x: float,
        y: float,
        lines: list[str],
        *,
        fill: str = PALETTE["text"],
        font_size: int = 14,
        font_weight: int = 500,
        anchor: str = "start",
        line_height: int = 18,
    ) -> None:
        attrs = [
            f'x="{x:.2f}"',
            f'y="{y:.2f}"',
            f'text-anchor="{anchor}"',
            'dominant-baseline="hanging"',
            f'fill="{fill}"',
            f'font-size="{font_size}"',
            f'font-weight="{font_weight}"',
        ]
        tspans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else line_height
            tspans.append(f'<tspan x="{x:.2f}" dy="{dy}">{html.escape(line)}</tspan>')
        self.add(f"<text {' '.join(attrs)}>{''.join(tspans)}</text>")

    def panel(self, x: float, y: float, width: float, height: float) -> None:
        self.rect(
            x,
            y,
            width,
            height,
            fill=PALETTE["panel"],
            stroke=PALETTE["panel_stroke"],
            stroke_width=1.0,
            rx=24,
        )

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def derive_record_turn_ids(records: list[dict[str, Any]]) -> list[str | None]:
    known_turn_ids: set[str] = set()
    current_turn_id: str | None = None
    result: list[str | None] = []
    for record in records:
        payload = record.get("payload", {})
        if record["type"] == "turn_context":
            current_turn_id = payload.get("turn_id")
            if current_turn_id:
                known_turn_ids.add(current_turn_id)
        record_turn_id = None
        if record["type"] == "event_msg" and payload.get("turn_id") in known_turn_ids:
            record_turn_id = payload.get("turn_id")
        else:
            record_turn_id = current_turn_id
        result.append(record_turn_id)
    return result


def describe_patch(files: list[str]) -> str:
    file_set = set(files)
    if {"scripts/_repo_env.sh", "scripts/check_repo_env.sh", "scripts/run_in_repo_env.sh"} & file_set:
        return "Add repo env wrappers"
    if "analysis/stats/fit.py" in file_set:
        return "Switch blinded fit to Asimov"
    if "scripts/generate_stage2_contracts.py" in file_set:
        return "Add stage-2 contract generator"
    if file_set == {"analysis/report/make_report.py"} or "analysis/report/make_report.py" in file_set:
        return "Fix report wording"
    if not files:
        return "Patch"
    if len(files) == 1:
        return f"Patch {Path(files[0]).name}"
    return trim_text("Patch " + ", ".join(Path(path).name for path in files[:3]), max_len=44)


def describe_milestone(message: str, turn_index: int) -> str:
    lowered = message.lower()
    if "blinded full-statistics" in lowered:
        return "Blinded run complete"
    if "selected background functions" in lowered or "background functions were" in lowered:
        return "Background query answered"
    if "observed significance" in lowered:
        return "Observed significance complete"
    return f"Turn {turn_index} complete"


def build_visual_context(records: list[dict[str, Any]], trace_path: Path) -> dict[str, Any]:
    summary = collect_summary(records, trace_path)
    workspace_root = trace_path.resolve().parent
    record_turn_ids = derive_record_turn_ids(records)
    turn_map = {turn["turn_id"]: turn for turn in summary["turns"]}

    start_dt = parse_timestamp(records[0].get("timestamp")) if records else None
    end_dt = parse_timestamp(records[-1].get("timestamp")) if records else None

    token_points: list[dict[str, Any]] = []
    patch_events: list[dict[str, Any]] = []
    exit_events: list[dict[str, Any]] = []
    milestone_events: list[dict[str, Any]] = []
    user_events: list[dict[str, Any]] = []
    minute_buckets: defaultdict[int, Counter[str]] = defaultdict(Counter)

    flagged_outputs = 0
    total_outputs = 0

    for record, record_turn_id in zip(records, record_turn_ids):
        timestamp = parse_timestamp(record.get("timestamp"))
        if timestamp is None or start_dt is None:
            continue
        payload = record.get("payload", {})
        minute_bucket = int((timestamp - start_dt).total_seconds() // 60)

        category = None
        if record["type"] == "response_item":
            payload_type = payload.get("type")
            if payload_type in {"function_call", "custom_tool_call"}:
                category = "tool_call"
            elif payload_type in {"function_call_output", "custom_tool_call_output"}:
                category = "tool_output"
            elif payload_type == "message":
                category = "assistant_msg"
            elif payload_type == "reasoning":
                category = "reasoning"
        elif record["type"] == "event_msg":
            event_type = payload.get("type")
            if event_type in {"agent_message", "task_complete"}:
                category = "assistant_msg"
            elif event_type == "user_message":
                category = "user_msg"

        if category:
            minute_buckets[minute_bucket][category] += 1

        if record["type"] == "event_msg" and payload.get("type") == "token_count":
            usage = (payload.get("info") or {}).get("total_token_usage") or {}
            total_tokens = usage.get("total_tokens")
            if total_tokens is not None:
                token_points.append({"time": timestamp, "total_tokens": total_tokens})

        if record["type"] == "response_item" and payload.get("type") == "custom_tool_call" and payload.get("name") == "apply_patch":
            patch_input = payload.get("input") or ""
            operations = patch_operations(patch_input)
            files = sorted(
                {
                    relativize_path(operation["path"], workspace_root)
                    for operation in operations
                    if operation["action"] != "move_to"
                }
            )
            turn_index = turn_map.get(record_turn_id, {}).get("turn_index", 0)
            patch_events.append(
                {
                    "time": timestamp,
                    "turn_index": turn_index,
                    "files": files,
                    "label": describe_patch(files),
                }
            )

        if record["type"] == "response_item" and payload.get("type") == "function_call_output":
            output = payload.get("output") or ""
            total_outputs += 1
            nonzero_codes = [int(code) for code in EXIT_CODE_RE.findall(output) if code != "0"]
            if detect_failure_markers(output) or nonzero_codes:
                flagged_outputs += 1
            for code in nonzero_codes:
                exit_events.append(
                    {
                        "time": timestamp,
                        "code": code,
                        "turn_index": turn_map.get(record_turn_id, {}).get("turn_index", 0),
                    }
                )

        if record["type"] == "event_msg" and payload.get("type") == "task_complete":
            turn = turn_map.get(payload.get("turn_id"), {})
            turn_index = turn.get("turn_index", 0)
            message = payload.get("last_agent_message", "")
            milestone_events.append(
                {
                    "time": timestamp,
                    "turn_index": turn_index,
                    "label": describe_milestone(message, turn_index),
                }
            )

        if record["type"] == "event_msg" and payload.get("type") == "user_message":
            turn = turn_map.get(payload.get("turn_id"), {})
            message = payload.get("message", "")
            user_events.append(
                {
                    "time": timestamp,
                    "turn_index": turn.get("turn_index", 0),
                    "label": trim_text(message.split("## My request for Codex:")[-1].strip().splitlines()[0] if "## My request for Codex:" in message else message, max_len=60),
                }
            )

    combined_tool_counts = Counter(summary["function_call_counts"])
    combined_tool_counts.update(summary["custom_tool_counts"])

    turn_profiles: list[dict[str, Any]] = []
    for turn in summary["turns"]:
        turn_profiles.append(
            {
                "turn_index": turn["turn_index"],
                "label": turn_label(turn),
                "duration_seconds": turn["duration_seconds"],
                "tool_calls": sum(turn["tool_counts"].values()) + sum(turn.get("custom_tool_counts", {}).values()),
                "patch_calls": sum(turn.get("custom_tool_counts", {}).values()),
                "commentary_count": turn.get("commentary_count", 0),
            }
        )

    turns = summary["turns"]
    phases: list[dict[str, Any]] = []
    if start_dt and end_dt and turns:
        first_turn_end = parse_timestamp(turns[0]["end_timestamp"])
        last_patch_time = max((event["time"] for event in patch_events), default=start_dt)
        if last_patch_time > start_dt:
            phases.append({"label": "Audit & repair", "start": start_dt, "end": last_patch_time, "fill": PALETTE["audit"]})
        if first_turn_end and first_turn_end > last_patch_time:
            phases.append({"label": "Blinded run", "start": last_patch_time, "end": first_turn_end, "fill": PALETTE["blinded"]})
        if len(turns) >= 2:
            second_start = parse_timestamp(turns[1]["start_timestamp"])
            second_end = parse_timestamp(turns[1]["end_timestamp"])
            if second_start and second_end:
                phases.append({"label": "Query follow-up", "start": second_start, "end": second_end, "fill": PALETTE["query"]})
        if len(turns) >= 3:
            third_start = parse_timestamp(turns[2]["start_timestamp"])
            third_end = parse_timestamp(turns[2]["end_timestamp"])
            if third_start and third_end:
                phases.append({"label": "Unblinded rerun", "start": third_start, "end": third_end, "fill": PALETTE["unblinded"]})

    gaps: list[dict[str, Any]] = []
    for previous, current in zip(turns, turns[1:]):
        previous_end = parse_timestamp(previous["end_timestamp"])
        current_start = parse_timestamp(current["start_timestamp"])
        if previous_end and current_start and current_start > previous_end:
            gaps.append(
                {
                    "start": previous_end,
                    "end": current_start,
                    "seconds": (current_start - previous_end).total_seconds(),
                }
            )

    return {
        "summary": summary,
        "start": start_dt,
        "end": end_dt,
        "tool_counts": dict(combined_tool_counts),
        "tool_call_total": sum(combined_tool_counts.values()),
        "files_touched": len(summary["edit_summary"]["unique_files_touched"]),
        "total_tokens": (summary.get("token_usage") or {}).get("total_tokens"),
        "token_points": token_points,
        "turn_profiles": turn_profiles,
        "phases": phases,
        "gaps": gaps,
        "patch_events": patch_events,
        "exit_events": exit_events,
        "milestone_events": milestone_events,
        "user_events": user_events,
        "minute_buckets": dict(sorted(minute_buckets.items())),
        "flagged_outputs": flagged_outputs,
        "total_outputs": total_outputs,
    }


def time_to_x(value: datetime, start: datetime, end: datetime, left: float, width: float) -> float:
    span = max((end - start).total_seconds(), 1.0)
    offset = (value - start).total_seconds()
    return left + width * offset / span


def nice_upper_bound(value: float, steps: int = 5) -> float:
    if value <= 0:
        return 1.0
    raw_step = value / steps
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        factor = 1
    elif normalized <= 2:
        factor = 2
    elif normalized <= 5:
        factor = 5
    else:
        factor = 10
    return factor * magnitude * steps


def draw_metric_card(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    value: str,
    note: str = "",
) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 28, title, cls="card-title", baseline="hanging")
    canvas.text(x + 24, y + height / 2 + 10, value, cls="card-value")
    if note:
        canvas.text(x + 24, y + height - 22, note, cls="card-note", baseline="baseline")


def draw_bar_chart(canvas: SvgCanvas, x: float, y: float, width: float, height: float, data: list[tuple[str, int]]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 30, "Tool call breakdown", cls="section", baseline="hanging")
    canvas.text(x + 24, y + 60, "Combined function and custom tool invocations across the full trace.", cls="caption", baseline="hanging")

    chart_x = x + 56
    chart_y = y + 92
    chart_w = width - 96
    chart_h = height - 130
    max_value = max((value for _, value in data), default=1)
    upper = nice_upper_bound(max_value)
    for step in range(6):
        value = upper * step / 5
        py = chart_y + chart_h - chart_h * step / 5
        canvas.line(chart_x, py, chart_x + chart_w, py, stroke=PALETTE["grid"], stroke_width=1)
        canvas.text(chart_x - 10, py, str(int(value)), cls="axis", anchor="end")

    bar_gap = 20
    bar_width = (chart_w - bar_gap * (len(data) - 1)) / max(len(data), 1)
    colors = [PALETTE["tool_call"], PALETTE["tool_output"], PALETTE["warning"], PALETTE["muted"]]
    for index, (label, value) in enumerate(data):
        bx = chart_x + index * (bar_width + bar_gap)
        bar_height = chart_h * value / upper if upper else 0
        by = chart_y + chart_h - bar_height
        canvas.rect(bx, by, bar_width, bar_height, fill=colors[index % len(colors)], rx=12)
        canvas.text(bx + bar_width / 2, by - 10, str(value), font_size=14, font_weight=700, anchor="middle", baseline="baseline")
        canvas.text_block(bx + bar_width / 2, chart_y + chart_h + 12, [label], fill=PALETTE["text"], font_size=13, font_weight=600, anchor="middle")


def draw_line_chart(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 30, "Token burn over time", cls="section", baseline="hanging")
    token_points = context["token_points"]
    subtitle = f"{len(token_points)} cumulative token snapshots extracted from `token_count` events."
    canvas.text(x + 24, y + 60, subtitle, cls="caption", baseline="hanging")

    chart_x = x + 56
    chart_y = y + 92
    chart_w = width - 88
    chart_h = height - 130
    start = context["start"]
    end = context["end"]
    max_tokens = max((point["total_tokens"] for point in token_points), default=1)
    upper = nice_upper_bound(max_tokens)

    for phase in context["phases"]:
        px1 = time_to_x(phase["start"], start, end, chart_x, chart_w)
        px2 = time_to_x(phase["end"], start, end, chart_x, chart_w)
        canvas.rect(px1, chart_y, px2 - px1, chart_h, fill=phase["fill"], rx=0, opacity=0.22)

    for step in range(6):
        value = upper * step / 5
        py = chart_y + chart_h - chart_h * step / 5
        canvas.line(chart_x, py, chart_x + chart_w, py, stroke=PALETTE["grid"], stroke_width=1)
        canvas.text(chart_x - 12, py, fmt_tokens(int(value)), cls="axis", anchor="end")

    ticks = 7
    for step in range(ticks):
        fraction = step / (ticks - 1)
        tick_time = start + (end - start) * fraction
        px = chart_x + chart_w * fraction
        canvas.line(px, chart_y + chart_h, px, chart_y + chart_h + 6, stroke=PALETTE["muted"], stroke_width=1)
        canvas.text(px, chart_y + chart_h + 20, fmt_time(tick_time), cls="axis", anchor="middle", baseline="hanging")

    if token_points:
        line_points = [
            (
                time_to_x(point["time"], start, end, chart_x, chart_w),
                chart_y + chart_h - chart_h * point["total_tokens"] / upper,
            )
            for point in token_points
        ]
        area_points = [(line_points[0][0], chart_y + chart_h)] + line_points + [(line_points[-1][0], chart_y + chart_h)]
        canvas.polygon(area_points, fill=PALETTE["tool_call"], opacity=0.14)
        canvas.polyline(line_points, stroke=PALETTE["tool_call"], stroke_width=3)
        marker_stride = max(1, len(line_points) // 12)
        for index, (px, py) in enumerate(line_points):
            if index % marker_stride == 0 or index == len(line_points) - 1:
                canvas.circle(px, py, 4.5, fill=PALETTE["bg"], stroke=PALETTE["tool_call"], stroke_width=2)
        last_x, last_y = line_points[-1]
        canvas.text(last_x - 6, last_y - 12, fmt_tokens(token_points[-1]["total_tokens"]), font_size=13, font_weight=700, anchor="end", baseline="baseline")


def draw_turn_profile(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 30, "Turn profile", cls="section", baseline="hanging")
    canvas.text(x + 24, y + 60, "Duration-scaled bars summarize how each turn uses tools and edits.", cls="caption", baseline="hanging")

    profiles = context["turn_profiles"]
    max_duration = max((profile["duration_seconds"] or 0 for profile in profiles), default=1)
    colors = {1: PALETTE["turn1"], 2: PALETTE["turn2"], 3: PALETTE["turn3"]}

    bar_left = x + 360
    bar_width = width - 520
    row_top = y + 98
    row_gap = 74

    for index, profile in enumerate(profiles):
        py = row_top + index * row_gap
        color = colors.get(profile["turn_index"], PALETTE["tool_call"])
        label = f"Turn {profile['turn_index']}"
        canvas.text(x + 28, py + 10, label, font_size=15, font_weight=700, baseline="baseline")
        canvas.text_block(x + 28, py + 20, [trim_text(profile["label"], max_len=46)], fill=PALETTE["muted"], font_size=13, font_weight=500)

        canvas.rect(bar_left, py, bar_width, 24, fill="#e7e1d5", rx=12)
        duration = profile["duration_seconds"] or 0
        filled = bar_width * duration / max_duration if max_duration else 0
        canvas.rect(bar_left, py, filled, 24, fill=color, rx=12)
        canvas.text(bar_left + filled + 12, py + 12, fmt_minutes(duration), font_size=13, font_weight=700, baseline="middle")

        details = f"{profile['tool_calls']} tool ops"
        if profile["patch_calls"]:
            details += f" | {profile['patch_calls']} patch calls"
        details += f" | {profile['commentary_count']} commentary messages"
        canvas.text(bar_left, py + 44, details, cls="small", baseline="baseline")


def render_dashboard(context: dict[str, Any], output_path: Path) -> None:
    canvas = SvgCanvas(1600, 1120)
    canvas.text(60, 52, "Codex Rollout Dashboard", cls="title", baseline="hanging")
    canvas.text(
        60,
        94,
        "A reproducible, trace-derived summary of duration, token growth, tool use, and turn-level workload.",
        cls="subtitle",
        baseline="hanging",
    )

    tool_counts = context["tool_counts"]
    duration_value = fmt_minutes(context["summary"]["time_span"]["wall_clock_seconds"])
    token_value = fmt_tokens(context["total_tokens"])
    tool_value = str(context["tool_call_total"])
    file_value = str(context["files_touched"])

    card_y = 130
    card_w = 350
    card_h = 134
    card_gap = 20
    draw_metric_card(canvas, 60, card_y, card_w, card_h, title="Duration", value=duration_value, note="Wall-clock trace span")
    draw_metric_card(canvas, 60 + (card_w + card_gap), card_y, card_w, card_h, title="Total tokens", value=token_value, note="Final cumulative token usage")
    draw_metric_card(canvas, 60 + 2 * (card_w + card_gap), card_y, card_w, card_h, title="Tool calls", value=tool_value, note="Function + custom tools")
    draw_metric_card(canvas, 60 + 3 * (card_w + card_gap), card_y, card_w, card_h, title="Files touched", value=file_value, note="15 files across 4 patches")

    tool_order = ["exec_command", "write_stdin", "apply_patch", "update_plan"]
    bar_data = [(name, tool_counts.get(name, 0)) for name in tool_order]
    draw_bar_chart(canvas, 60, 300, 700, 360, bar_data)
    draw_line_chart(canvas, 800, 300, 740, 360, context)
    draw_turn_profile(canvas, 60, 700, 1480, 340, context)

    canvas.save(output_path)


def draw_phase_ribbon(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Phase timeline", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Heuristic phase segmentation: audit/repair ends at the last patch; the remainder of turn 1 is treated as the blinded run.",
        cls="caption",
        baseline="hanging",
    )
    bar_x = x + 40
    bar_y = y + 86
    bar_w = width - 80
    bar_h = 34
    start = context["start"]
    end = context["end"]

    canvas.rect(bar_x, bar_y, bar_w, bar_h, fill="#e7e1d5", rx=17)
    for phase in context["phases"]:
        px1 = time_to_x(phase["start"], start, end, bar_x, bar_w)
        px2 = time_to_x(phase["end"], start, end, bar_x, bar_w)
        canvas.rect(px1, bar_y, max(px2 - px1, 2), bar_h, fill=phase["fill"], rx=17)
        label = f"{phase['label']} ({fmt_minutes((phase['end'] - phase['start']).total_seconds())})"
        mid_x = (px1 + px2) / 2
        if px2 - px1 > estimate_text_width(label, 13) + 12:
            canvas.text(mid_x, bar_y + bar_h / 2, label, font_size=13, font_weight=700, anchor="middle")
        else:
            canvas.text(mid_x, bar_y + bar_h + 18, trim_text(phase["label"], max_len=18), cls="axis", anchor="middle", baseline="hanging")

    for gap in context["gaps"]:
        gx1 = time_to_x(gap["start"], start, end, bar_x, bar_w)
        gx2 = time_to_x(gap["end"], start, end, bar_x, bar_w)
        gy = bar_y + bar_h + 34
        canvas.line(gx1, gy, gx2, gy, stroke=PALETTE["muted"], stroke_width=1.5, dash="4 4")
        canvas.line(gx1, gy - 7, gx1, gy + 7, stroke=PALETTE["muted"], stroke_width=1.5)
        canvas.line(gx2, gy - 7, gx2, gy + 7, stroke=PALETTE["muted"], stroke_width=1.5)
        canvas.text((gx1 + gx2) / 2, gy + 18, f"{gap['seconds'] / 60.0:.1f} min gap", cls="axis", anchor="middle", baseline="hanging")

    for tick in range(7):
        fraction = tick / 6
        tick_time = start + (end - start) * fraction
        px = bar_x + bar_w * fraction
        canvas.line(px, bar_y + bar_h, px, bar_y + bar_h + 8, stroke=PALETTE["muted"], stroke_width=1)
        canvas.text(px, bar_y + bar_h + 24, fmt_time(tick_time), cls="axis", anchor="middle", baseline="hanging")


def draw_activity_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Per-minute activity density", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Stacked minute buckets separate tool calls, tool outputs, assistant messages, reasoning records, and user prompts.",
        cls="caption",
        baseline="hanging",
    )

    chart_x = x + 48
    chart_y = y + 88
    chart_w = width - 90
    chart_h = height - 130
    start = context["start"]
    end = context["end"]

    max_bucket_total = max((sum(bucket.values()) for bucket in context["minute_buckets"].values()), default=1)
    upper = nice_upper_bound(max_bucket_total)
    for phase in context["phases"]:
        px1 = time_to_x(phase["start"], start, end, chart_x, chart_w)
        px2 = time_to_x(phase["end"], start, end, chart_x, chart_w)
        canvas.rect(px1, chart_y, px2 - px1, chart_h, fill=phase["fill"], rx=0, opacity=0.18)

    for step in range(6):
        value = upper * step / 5
        py = chart_y + chart_h - chart_h * step / 5
        canvas.line(chart_x, py, chart_x + chart_w, py, stroke=PALETTE["grid"], stroke_width=1)
        canvas.text(chart_x - 10, py, str(int(value)), cls="axis", anchor="end")

    bucket_indices = sorted(context["minute_buckets"])
    bucket_count = max(bucket_indices[-1] + 1 if bucket_indices else 1, 1)
    bar_gap = 2
    bar_width = max((chart_w - bar_gap * (bucket_count - 1)) / bucket_count, 2)
    category_order = ["tool_call", "tool_output", "assistant_msg", "reasoning", "user_msg"]

    for bucket_index in range(bucket_count):
        bucket = context["minute_buckets"].get(bucket_index, Counter())
        bx = chart_x + bucket_index * (bar_width + bar_gap)
        current_top = chart_y + chart_h
        for category in category_order:
            count = bucket.get(category, 0)
            if count <= 0:
                continue
            block_h = chart_h * count / upper
            by = current_top - block_h
            canvas.rect(bx, by, bar_width, block_h, fill=ACTIVITY_COLORS[category], rx=3)
            current_top = by

    tick_minutes = min(5, max(2, bucket_count // 8))
    for minute in range(0, bucket_count + 1, tick_minutes):
        fraction = minute / max(bucket_count - 1, 1)
        px = chart_x + chart_w * min(fraction, 1.0)
        tick_time = start + (end - start) * min(fraction, 1.0)
        canvas.line(px, chart_y + chart_h, px, chart_y + chart_h + 6, stroke=PALETTE["muted"], stroke_width=1)
        canvas.text(px, chart_y + chart_h + 18, fmt_time(tick_time), cls="axis", anchor="middle", baseline="hanging")

    legend_x = x + width - 312
    legend_y = y + 30
    for index, category in enumerate(category_order):
        lx = legend_x + (index % 2) * 148
        ly = legend_y + (index // 2) * 22
        canvas.rect(lx, ly, 14, 14, fill=ACTIVITY_COLORS[category], rx=4)
        canvas.text(lx + 22, ly + 7, category.replace("_", " "), cls="small")


def draw_interventions_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Interventions and milestones", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Patches, non-zero exits, and turn completions are anchored on a common time axis to show recovery structure.",
        cls="caption",
        baseline="hanging",
    )

    lane_x = x + 48
    lane_y = y + 94
    lane_w = width - 96
    lane_h = height - 140
    baseline_y = lane_y + lane_h / 2 + 8
    start = context["start"]
    end = context["end"]

    for phase in context["phases"]:
        px1 = time_to_x(phase["start"], start, end, lane_x, lane_w)
        px2 = time_to_x(phase["end"], start, end, lane_x, lane_w)
        canvas.rect(px1, lane_y, px2 - px1, lane_h, fill=phase["fill"], rx=0, opacity=0.16)

    canvas.line(lane_x, baseline_y, lane_x + lane_w, baseline_y, stroke=PALETTE["muted"], stroke_width=2)

    legend_items = [
        ("Patch", PALETTE["patch"]),
        ("Turn complete", PALETTE["milestone"]),
        ("Non-zero exit", PALETTE["exit"]),
    ]
    legend_x = x + width - 320
    legend_y = y + 28
    for index, (label, color) in enumerate(legend_items):
        lx = legend_x + index * 102
        canvas.rect(lx, legend_y, 14, 14, fill=color, rx=4)
        canvas.text(lx + 22, legend_y + 7, label, cls="small")

    upper_box_y = lane_y + 18
    lower_box_y = baseline_y + 26

    for index, event in enumerate(context["patch_events"]):
        px = time_to_x(event["time"], start, end, lane_x, lane_w)
        box_y = upper_box_y if index % 2 == 0 else lower_box_y
        box_h = 52
        label_lines = [event["label"], fmt_time(event["time"])]
        box_w = max(160, max(estimate_text_width(line, 14) for line in label_lines) + 28)
        box_x = min(max(px - box_w / 2, lane_x), lane_x + lane_w - box_w)
        fill = PALETTE["audit"] if index < 2 else PALETTE["blinded"]
        canvas.line(px, baseline_y, px, box_y + (0 if box_y > baseline_y else box_h), stroke=PALETTE["patch"], stroke_width=2)
        canvas.circle(px, baseline_y, 5, fill=PALETTE["patch"], stroke=PALETTE["bg"], stroke_width=2)
        canvas.rect(box_x, box_y if box_y < baseline_y else box_y, box_w, box_h, fill=fill, stroke=PALETTE["patch"], stroke_width=1.2, rx=14)
        canvas.text_block(box_x + 14, box_y + 10, label_lines, fill=PALETTE["text"], font_size=14, font_weight=600)

    milestone_offsets = {1: -82, 2: -112, 3: -82}
    for event in context["milestone_events"]:
        px = time_to_x(event["time"], start, end, lane_x, lane_w)
        box_y = baseline_y + milestone_offsets.get(event["turn_index"], -90)
        box_w = max(170, estimate_text_width(event["label"], 14) + 26)
        box_x = min(max(px - box_w / 2, lane_x), lane_x + lane_w - box_w)
        canvas.line(px, baseline_y, px, box_y + 34, stroke=PALETTE["milestone"], stroke_width=2)
        canvas.circle(px, baseline_y, 6, fill=PALETTE["milestone"], stroke=PALETTE["bg"], stroke_width=2)
        canvas.rect(box_x, box_y, box_w, 34, fill="#e7f3ee", stroke=PALETTE["milestone"], stroke_width=1.2, rx=14)
        canvas.text(box_x + box_w / 2, box_y + 17, event["label"], font_size=13, font_weight=700, anchor="middle")

    for index, event in enumerate(context["exit_events"]):
        px = time_to_x(event["time"], start, end, lane_x, lane_w)
        offset = 18 + index * 16
        triangle = [(px, baseline_y - 10), (px - 8, baseline_y - 26), (px + 8, baseline_y - 26)]
        canvas.polygon(triangle, fill=PALETTE["exit"])
        canvas.text(px, baseline_y + offset, f"exit {event['code']}", cls="axis", anchor="middle", baseline="hanging")

    tick_count = 7
    for tick in range(tick_count):
        fraction = tick / (tick_count - 1)
        tick_time = start + (end - start) * fraction
        px = lane_x + lane_w * fraction
        canvas.line(px, baseline_y + 8, px, baseline_y + 18, stroke=PALETTE["muted"], stroke_width=1)
        canvas.text(px, baseline_y + 30, fmt_time(tick_time), cls="axis", anchor="middle", baseline="hanging")


def render_timeline(context: dict[str, Any], output_path: Path) -> None:
    canvas = SvgCanvas(1600, 1280)
    canvas.text(60, 52, "Temporal Structure of the Rollout", cls="title", baseline="hanging")
    canvas.text(
        60,
        94,
        "Timeline-oriented view of phases, activity density, interventions, and completion milestones extracted from the same JSONL trace.",
        cls="subtitle",
        baseline="hanging",
    )

    draw_phase_ribbon(canvas, 60, 130, 1480, 190, context)
    draw_activity_panel(canvas, 60, 350, 1480, 380, context)
    draw_interventions_panel(canvas, 60, 760, 1480, 420, context)
    canvas.save(output_path)


def render_manifest(output_dir: Path, figure_paths: list[Path]) -> None:
    captions = {
        "rollout_dashboard.svg": {
            "title": "Figure 1. Rollout Dashboard",
            "caption": "Summary dashboard showing wall-clock duration, cumulative token footprint, combined tool-call breakdown, cumulative token burn, and duration-scaled turn profiles.",
        },
        "rollout_timeline.svg": {
            "title": "Figure 2. Temporal Structure and Interventions",
            "caption": "Timeline view showing derived phases, per-minute activity density, patch interventions, non-zero shell exits, and turn-completion milestones on a shared time axis.",
        },
    }
    payload = {
        "figures": [
            {
                "path": str(path.resolve()),
                "title": captions[path.name]["title"],
                "caption": captions[path.name]["caption"],
            }
            for path in figure_paths
            if path.name in captions
        ]
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dependency-free SVG visuals for a Codex rollout trace.")
    parser.add_argument("trace", type=Path, help="Path to rollout JSONL trace")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rollout_figures"), help="Directory for SVG outputs")
    parser.add_argument("--data-out", type=Path, help="Optional JSON dump of the plotting context")
    args = parser.parse_args()

    records = load_jsonl(args.trace)
    context = build_visual_context(records, args.trace)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = args.output_dir / "rollout_dashboard.svg"
    timeline_path = args.output_dir / "rollout_timeline.svg"

    render_dashboard(context, dashboard_path)
    render_timeline(context, timeline_path)
    render_manifest(args.output_dir, [dashboard_path, timeline_path])

    if args.data_out:
        serializable = dict(context)
        serializable["start"] = context["start"].isoformat() if context["start"] else None
        serializable["end"] = context["end"].isoformat() if context["end"] else None
        serializable["token_points"] = [
            {"time": point["time"].isoformat(), "total_tokens": point["total_tokens"]}
            for point in context["token_points"]
        ]
        serializable["phases"] = [
            {**phase, "start": phase["start"].isoformat(), "end": phase["end"].isoformat()}
            for phase in context["phases"]
        ]
        for key in ["patch_events", "exit_events", "milestone_events", "user_events", "gaps"]:
            serializable[key] = [
                {
                    **item,
                    **{
                        subkey: item[subkey].isoformat()
                        for subkey in ["time", "start", "end"]
                        if subkey in item and isinstance(item[subkey], datetime)
                    },
                }
                for item in context[key]
            ]
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        args.data_out.write_text(json.dumps(serializable, indent=2) + "\n")


if __name__ == "__main__":
    main()
