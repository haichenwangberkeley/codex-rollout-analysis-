#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta
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


def fmt_elapsed_time(start: datetime | None, value: datetime | None) -> str:
    if start is None or value is None:
        return "n/a"
    seconds = max((value - start).total_seconds(), 0.0)
    if seconds < 60:
        return "0m"
    minutes = seconds / 60.0
    if minutes < 10:
        return f"{minutes:.1f}m"
    return f"{int(round(minutes))}m"


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
    title_lines = wrap_for_pixels(title, width - 48, 17, max_len=80)[:2]
    note_lines = wrap_for_pixels(note, width - 48, 13, max_len=120)[:3] if note else []
    canvas.text_block(x + 24, y + 24, title_lines, fill=PALETTE["muted"], font_size=17, font_weight=600, line_height=18)
    canvas.text(x + 24, y + height / 2 + 10, value, cls="card-value")
    if note_lines:
        note_y = y + height - 18 - (len(note_lines) - 1) * 15
        canvas.text_block(x + 24, note_y, note_lines, fill=PALETTE["muted"], font_size=13, font_weight=500, line_height=15)


def draw_bar_chart(canvas: SvgCanvas, x: float, y: float, width: float, height: float, data: list[tuple[str, int]]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 30, "Tool call breakdown", cls="section", baseline="hanging")
    canvas.text_block(
        x + 24,
        y + 60,
        wrap_for_pixels("Combined function and custom tool invocations across the full trace.", width - 48, 13, max_len=120)[:2],
        fill=PALETTE["muted"],
        font_size=13,
        font_weight=500,
        line_height=15,
    )

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
        canvas.text_block(
            bx + bar_width / 2,
            chart_y + chart_h + 12,
            wrap_for_pixels(label, bar_width, 13, max_len=60)[:2],
            fill=PALETTE["text"],
            font_size=13,
            font_weight=600,
            anchor="middle",
            line_height=15,
        )


def draw_line_chart(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 30, "Token burn over time", cls="section", baseline="hanging")
    token_points = context["token_points"]
    subtitle = f"{len(token_points)} cumulative token snapshots extracted from `token_count` events."
    canvas.text_block(
        x + 24,
        y + 60,
        wrap_for_pixels(subtitle, width - 48, 13, max_len=120)[:2],
        fill=PALETTE["muted"],
        font_size=13,
        font_weight=500,
        line_height=15,
    )

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
        canvas.text(px, chart_y + chart_h + 20, fmt_elapsed_time(start, tick_time), cls="axis", anchor="middle", baseline="hanging")

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
        canvas.text(px, bar_y + bar_h + 24, fmt_elapsed_time(start, tick_time), cls="axis", anchor="middle", baseline="hanging")


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
        canvas.text(px, chart_y + chart_h + 18, fmt_elapsed_time(start, tick_time), cls="axis", anchor="middle", baseline="hanging")

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
        canvas.text(px, baseline_y + 30, fmt_elapsed_time(start, tick_time), cls="axis", anchor="middle", baseline="hanging")


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


ACTION_CATEGORY_COLORS = {
    "Task Intake": "#d96b5f",
    "Documentation Review": "#4f7dd4",
    "Input Discovery": "#8f95a2",
    "Spec Alignment": "#f4a261",
    "Pipeline Execution": "#2a9d8f",
    "Fit / Significance": "#3f83d6",
    "Reporting": "#38a3a5",
    "Governance / Handoff": "#c084fc",
    "Verification": "#e9a03b",
    "Finalization": "#bfe6d4",
}

SKILL_STATUS_COLORS = {
    "explicitly_opened": PALETTE["tool_call"],
    "explicitly_opened_and_artifact_backed": PALETTE["milestone"],
    "artifact_backed_without_explicit_open": PALETTE["warning"],
}


def wrap_text(text: str, width: int) -> list[str]:
    cleaned = trim_text(text, max_len=500)
    return textwrap.wrap(cleaned, width=width) or [cleaned]


def wrap_for_pixels(text: str, pixel_width: float, font_size: int, max_len: int = 500) -> list[str]:
    if pixel_width <= 24:
        return [trim_text(text, max_len=min(max_len, 24))]
    chars = max(8, int(pixel_width / max(font_size * 0.58, 1)))
    cleaned = trim_text(text, max_len=max_len)
    return textwrap.wrap(cleaned, width=chars) or [cleaned]


def skill_slug_from_path(record: dict[str, Any]) -> str | None:
    skill_path = record.get("skill_path") or ""
    if not skill_path:
        return None
    path = Path(skill_path)
    parent = path.parent.name if path.name == "SKILL.md" else path.stem
    if not parent:
        return None
    parent = re.sub(r"^local-", "", parent)
    return parent.replace("-", "_")


def display_skill_name(record: dict[str, Any]) -> str:
    raw = (record.get("skill_name") or "").strip()
    if raw and raw != "SKILL":
        return raw
    slug = skill_slug_from_path(record)
    if slug:
        return slug
    return "unknown_skill"


def humanize_skill_label(record: dict[str, Any]) -> str:
    label = record.get("doc_title")
    if label and label != "SKILL":
        return trim_text(label, max_len=48)
    slug = skill_slug_from_path(record)
    if slug:
        words = [part for part in slug.split("_") if part]
        pretty = " ".join(word.upper() if word.isupper() else word.capitalize() for word in words)
        return trim_text(pretty, max_len=48)
    raw = record.get("skill_name") or "unknown skill"
    return trim_text(raw, max_len=48)


def summarize_text(text: str, max_len: int = 120) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned or cleaned == "SKILL":
        return ""
    checkpoint_match = re.match(
        r"Support checkpoint `([^`]+)` by producing or validating:\s*(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if checkpoint_match:
        checkpoint_id = checkpoint_match.group(1).replace("_", " ")
        artifacts = checkpoint_match.group(2)
        artifact_count = len([item for item in artifacts.split(",") if item.strip()])
        if artifact_count > 0:
            return f"Produce or validate {artifact_count} artifacts for the {checkpoint_id} checkpoint."
        return f"Produce or validate artifacts for the {checkpoint_id} checkpoint."
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    first = parts[0] if parts else cleaned
    if len(first) <= max_len:
        return first
    words = first.split()
    shortened: list[str] = []
    for word in words:
        candidate = " ".join(shortened + [word]).strip()
        if len(candidate) > max_len:
            break
        shortened.append(word)
    return " ".join(shortened) if shortened else first[:max_len]


def build_line_time_index(raw_actions: list[dict[str, Any]]) -> dict[int, list[datetime]]:
    line_times: defaultdict[int, list[datetime]] = defaultdict(list)
    for action in raw_actions:
        when = action.get("_dt")
        if when is None:
            continue
        lines = set(action.get("evidence_lines", []))
        line_no = action.get("line_no")
        if isinstance(line_no, int):
            lines.add(line_no)
        for line in lines:
            if isinstance(line, int):
                line_times[line].append(when)
    for values in line_times.values():
        values.sort()
    return dict(line_times)


def times_for_lines(line_times: dict[int, list[datetime]], lines: set[int]) -> list[datetime]:
    result: list[datetime] = []
    for line in lines:
        result.extend(line_times.get(line, []))
    result.sort()
    return result


def build_time_ticks(start: datetime | None, end: datetime | None, tick_count: int = 7) -> list[datetime]:
    if start is None or end is None:
        return []
    if start == end:
        return [start]
    return [start + (end - start) * (index / max(tick_count - 1, 1)) for index in range(tick_count)]


def draw_time_ticks(
    canvas: SvgCanvas,
    left: float,
    baseline_y: float,
    width: float,
    *,
    start: datetime | None,
    end: datetime | None,
    tick_count: int = 7,
) -> None:
    if start is None or end is None:
        return
    ticks = build_time_ticks(start, end, tick_count=tick_count)
    for tick_time in ticks:
        px = time_to_x(tick_time, start, end, left, width)
        canvas.line(px, baseline_y, px, baseline_y + 8, stroke=PALETTE["muted"], stroke_width=1)
        canvas.text(px, baseline_y + 20, fmt_elapsed_time(start, tick_time), cls="axis", anchor="middle", baseline="hanging")


def shade_narrative_segments(
    canvas: SvgCanvas,
    segments: list[dict[str, Any]],
    *,
    start: datetime | None,
    end: datetime | None,
    left: float,
    top: float,
    width: float,
    height: float,
    opacity: float,
) -> None:
    if start is None or end is None:
        return
    for segment in segments:
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        if seg_start is None or seg_end is None:
            continue
        px1 = time_to_x(seg_start, start, end, left, width)
        px2 = time_to_x(seg_end, start, end, left, width)
        fill = ACTION_CATEGORY_COLORS.get(segment.get("category"), PALETTE["tool_call"])
        canvas.rect(px1, top, max(px2 - px1, 2), height, fill=fill, rx=0, opacity=opacity)


def assign_tracks(
    events: list[dict[str, Any]],
    *,
    start: datetime | None,
    end: datetime | None,
    left: float,
    width: float,
    max_tracks: int | None,
    min_gap: float = 14,
) -> list[dict[str, Any]]:
    if start is None or end is None:
        return []
    track_right: list[float] = []
    laid_out: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item["time"], item["label"])):
        x = time_to_x(event["time"], start, end, left, width)
        box_width = event.get("box_width", 120.0)
        left_edge = min(max(x - box_width / 2, left), left + width - box_width)
        right_edge = left_edge + box_width
        track = None
        for index in range(len(track_right)):
            if left_edge >= track_right[index] + min_gap:
                track = index
                break
        if track is None:
            if max_tracks is None or len(track_right) < max_tracks:
                track = len(track_right)
                track_right.append(-10_000.0)
            else:
                track = min(range(max_tracks), key=lambda index: track_right[index])
        track_right[track] = right_edge
        laid_out.append({**event, "x": x, "track": track, "box_width": box_width, "box_left": left_edge, "box_right": right_edge})
    return laid_out


def compact_action_label(label: str) -> str:
    compact = label
    compact = compact.replace("Check analysis dependencies", "Check deps")
    compact = compact.replace("Inspect ROOT inputs and branches", "Inspect ROOT inputs")
    compact = compact.replace("Validate report and checkpoint artifacts", "Validate report/checkpoint artifacts")
    compact = compact.replace("Write checkpoint refresh log", "Write checkpoint log")
    compact = compact.replace("Inspect significance outputs", "Inspect significance")
    compact = compact.replace("Summarize completed run", "Summarize run")
    return compact


def fit_action_box_text(event: dict[str, Any], box_width: float) -> tuple[list[str], int]:
    label = event.get("label", "")
    candidates = [label]
    compact = compact_action_label(label)
    if compact != label:
        candidates.append(compact)
    count = event.get("count", 1)
    redundant_count = bool(re.search(r"\bx\d+\b", label)) or " for " in label
    for candidate in candidates:
        for font_size in (12, 11, 10):
            lines = wrap_for_pixels(candidate, max(box_width - 24, 80), font_size, max_len=90)[:2]
            if count > 2 and not redundant_count and len(lines) < 2:
                lines.append(f"{count} actions")
            max_line_width = max(estimate_text_width(line, font_size) for line in lines) if lines else 0
            if len(lines) <= 2 and max_line_width <= box_width - 24:
                return lines, font_size
    fallback = wrap_for_pixels(compact, max(box_width - 24, 80), 10, max_len=70)[:2]
    return fallback, 10


def boxes_overlap(a: dict[str, Any], b: dict[str, Any], padding: float = 2.0) -> bool:
    return not (
        a["x2"] + padding <= b["x1"]
        or b["x2"] + padding <= a["x1"]
        or a["y2"] + padding <= b["y1"]
        or b["y2"] + padding <= a["y1"]
    )


def count_box_overlaps(boxes: list[dict[str, Any]]) -> int:
    count = 0
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            if boxes_overlap(first, second):
                count += 1
    return count


def plan_important_action_layout(
    events: list[dict[str, Any]],
    *,
    start: datetime | None,
    end: datetime | None,
    panel_x: float,
    panel_y: float,
    panel_width: float,
) -> dict[str, Any]:
    lane_x = panel_x + 48
    lane_y = panel_y + 112
    lane_w = panel_width - 96
    top_offset = 28
    bottom_offset = 24
    attempts: list[dict[str, Any]] = []

    for attempt in range(8):
        track_pitch = 64 + attempt * 6
        min_gap = 14 + attempt * 2
        enriched: list[dict[str, Any]] = []
        for event in events:
            lines, font_size = fit_action_box_text(event, event.get("box_width", 240))
            line_height = 14 if font_size >= 11 else 13
            box_h = 26 + line_height * len(lines)
            enriched.append({**event, "lines": lines, "font_size": font_size, "line_height": line_height, "box_h": box_h})

        above_seed = enriched[::2]
        below_seed = enriched[1::2]
        above = assign_tracks(above_seed, start=start, end=end, left=lane_x, width=lane_w, max_tracks=None, min_gap=min_gap)
        below = assign_tracks(below_seed, start=start, end=end, left=lane_x, width=lane_w, max_tracks=None, min_gap=min_gap)

        top_needed = 40.0
        bottom_needed = 40.0
        placed_above: list[dict[str, Any]] = []
        placed_below: list[dict[str, Any]] = []
        boxes: list[dict[str, Any]] = []

        for event in above:
            box_y = lane_y + top_offset + max(0, event["track"]) * track_pitch
            placed = {
                **event,
                "box_x": event["box_left"],
                "box_y": box_y,
                "x1": event["box_left"],
                "y1": box_y,
                "x2": event["box_left"] + event["box_width"],
                "y2": box_y + event["box_h"],
                "side": "above",
            }
            top_needed = max(top_needed, placed["y2"] - lane_y)
            placed_above.append(placed)
            boxes.append(placed)

        baseline_y = lane_y + top_needed + 16
        for event in below:
            box_y = baseline_y + bottom_offset + max(0, event["track"]) * track_pitch
            placed = {
                **event,
                "box_x": event["box_left"],
                "box_y": box_y,
                "x1": event["box_left"],
                "y1": box_y,
                "x2": event["box_left"] + event["box_width"],
                "y2": box_y + event["box_h"],
                "side": "below",
            }
            bottom_needed = max(bottom_needed, placed["y2"] - baseline_y)
            placed_below.append(placed)
            boxes.append(placed)

        lane_h = top_needed + 16 + bottom_needed + 28
        panel_height = 112 + lane_h + 52
        overlaps = count_box_overlaps(boxes)
        layout = {
            "lane_x": lane_x,
            "lane_y": lane_y,
            "lane_w": lane_w,
            "lane_h": lane_h,
            "baseline_y": baseline_y,
            "above": placed_above,
            "below": placed_below,
            "boxes": boxes,
            "track_pitch": track_pitch,
            "min_gap": min_gap,
            "panel_height": panel_height,
            "overlap_count": overlaps,
        }
        attempts.append(layout)
        if overlaps == 0:
            return layout

    return attempts[-1]


def infer_action_files(action: dict[str, Any]) -> list[str]:
    files = list(action.get("files") or [])
    if files:
        return files
    source = " ".join(
        part for part in [action.get("command"), action.get("description")] if isinstance(part, str) and part
    )
    if not source:
        return []
    normalized = source.replace('"', "").replace("'", "")
    matches = re.findall(
        r"(?<![\w/.-])((?:[\w.$()/-]+)\.(?:md|py|json|ya?ml|txt|cfg|toml|csv|svg|pdf|jsonl)|Makefile|README(?:\.md)?)",
        normalized,
        flags=re.IGNORECASE,
    )
    if "reports/final_analysis_report_" in normalized and ".md" in normalized:
        matches.append("reports/final_analysis_report_<run>.md")
    matches.extend(
        re.findall(
            r"((?:\$OUTDIR|outputs|reports|analysis|tests|input-data)(?:/[A-Za-z0-9_.()$-]+)*)",
            normalized,
        )
    )
    if ".current_run_outdir" in normalized:
        matches.append(".current_run_outdir")
    cleaned: list[str] = []
    for match in matches:
        if match not in cleaned:
            cleaned.append(match)
    return cleaned


def infer_discovery_targets(action: dict[str, Any]) -> list[str]:
    source = " ".join(
        part for part in [action.get("command"), action.get("description")] if isinstance(part, str) and part
    )
    if not source:
        return []
    normalized = source.replace('"', "").replace("'", "")
    targets: list[str] = []

    if "rg --files" in normalized:
        targets.append("repo files")

    path_matches = re.findall(
        r"((?:\./)?(?:analysis|tests|reports|outputs|skills(?:_legacy)?|docs|input-data)(?:/[A-Za-z0-9_.()$-]+)*)",
        normalized,
    )
    for match in path_matches:
        cleaned = match[2:] if match.startswith("./") else match
        if cleaned not in targets:
            targets.append(cleaned)

    if re.search(r"\bls\s+-[^\n]*\btests\b", normalized) and "tests" not in targets:
        targets.append("tests")
    if re.search(r"\bls\s+-[^\n]*\breports\b", normalized) and "reports" not in targets:
        targets.append("reports")
    if re.search(r"\bls\s+-[^\n]*\boutputs\b", normalized) and "outputs" not in targets:
        targets.append("outputs")
    if re.search(r"\bls\s+-1\b|\bls\s+-la\b", normalized) and not any(target in {"repo files", "."} for target in targets):
        if "README.md" in normalized or "analysis" in normalized or "tests" in normalized:
            pass
        elif "." not in targets:
            targets.append(".")

    file_targets = infer_action_files(action)
    for target in file_targets:
        if target not in targets:
            targets.append(target)
    return targets


def summarize_named_items(prefix: str, items: list[str], *, summary_noun: str, singular_fallback: str) -> str:
    if not items:
        return singular_fallback
    names = short_file_labels(items, max_items=min(len(items), 4))
    if len(items) == 1:
        return f"{prefix} {names[0]}"
    if len(items) == 2:
        return f"{prefix} {names[0]} and {names[1]}"
    if len(items) <= 4:
        return f"{prefix} {', '.join(names)}"
    lead = short_file_labels(items, max_items=2)
    if prefix == "Read":
        return f"Read files for {lead[0]} and {lead[1]}"
    if prefix == "Inspect":
        return f"Inspect targets for {lead[0]} and {lead[1]}"
    return f"{prefix} {summary_noun} for {lead[0]} and {lead[1]}"


def summarize_inline_python_label(action: dict[str, Any]) -> str:
    source = " ".join(
        part for part in [action.get("command"), action.get("description")] if isinstance(part, str) and part
    ).lower()
    if not source:
        return "Run Python check"
    if "importlib.import_module" in source or "find_spec" in source:
        return "Check analysis dependencies"
    if "uproot.open" in source or "path('input-data')" in source or "photon_pt" in source:
        return "Inspect ROOT inputs and branches"
    if "skill_refresh_log.jsonl" in source:
        return "Write checkpoint refresh log"
    if "samples.registry.json" in source and "kind')=='signal" in source:
        return "Inspect signal sample mix"
    if "samples.registry.json" in source and "regions" in source and "yield" in source:
        return "Compare data and MC yields"
    if "samples.registry.json" in source:
        return "Inspect sample registry"
    if "significance.json" in source and "asimov" in source:
        return "Inspect significance outputs"
    if "execution_contract.json" in source or "final_report_review.json" in source or "skill_checkpoint_status.json" in source:
        return "Validate report and checkpoint artifacts"
    if "run_manifest.json" in source:
        return "Inspect run manifest"
    if "datetime" in source and "stat().st_mtime" in source:
        return "Check artifact timestamps"
    return "Run Python check"


def summarize_inline_python_cluster(cluster: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for item in cluster:
        label = summarize_inline_python_label(item)
        if label not in labels:
            labels.append(label)
    if len(labels) == 1:
        if len(cluster) > 1:
            return f"{labels[0]} x{len(cluster)}"
        return labels[0]
    if len(labels) <= 3:
        return "; ".join(labels)
    return f"Python checks for {labels[0]} and {labels[1]}"


def format_action_timeline_label(action: dict[str, Any]) -> str:
    kind = action.get("action_kind") or "action"
    label = action.get("label") or kind.replace("_", " ")
    files = infer_action_files(action)
    if kind == "read_file":
        if files:
            return summarize_named_items("Read", files, summary_noun="files", singular_fallback="Read files")
        return "Read files"
    if kind == "discover_files":
        targets = infer_discovery_targets(action)
        if targets:
            return summarize_named_items("Inspect", targets, summary_noun="targets", singular_fallback="Inspect discovered files")
        return "Inspect discovered files"
    if kind == "inline_python":
        return summarize_inline_python_label(action)
    if kind == "echo":
        return "Echo quick check"
    if kind == "input_discovery":
        return "Inspect input data"
    return label


def build_important_action_events(parsed_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included_kinds = {
        "user_request",
        "read_trigger",
        "read_analysis_json",
        "read_file",
        "discover_files",
        "input_discovery",
        "spec_mismatch",
        "apply_patch",
        "plan",
        "run_pipeline",
        "run_fit",
        "generate_report",
        "governance_consolidation",
        "checkpoint_repair",
        "inspect_handoff_artifacts",
        "artifact_validation",
        "run_tests",
        "git_status",
        "inline_python",
        "echo",
        "final_summary",
    }
    clusterable_kinds = {"read_file", "discover_files", "input_discovery", "inline_python", "echo"}
    cluster_window_seconds = 90.0
    cluster_line_gap = 25

    filtered = [action for action in parsed_actions if action.get("action_kind") in included_kinds]
    clusters: list[list[dict[str, Any]]] = []
    for action in filtered:
        if not clusters:
            clusters.append([action])
            continue
        previous = clusters[-1][-1]
        same_kind = action.get("action_kind") == previous.get("action_kind")
        same_label = action.get("label") == previous.get("label")
        same_category = action.get("category") == previous.get("category")
        time_close = (
            action.get("_dt") is not None
            and previous.get("_dt") is not None
            and abs((action["_dt"] - previous["_dt"]).total_seconds()) <= cluster_window_seconds
        )
        can_cluster = same_kind and same_category and time_close and (
            action.get("action_kind") in clusterable_kinds
            or (same_label and abs((action["_dt"] - previous["_dt"]).total_seconds()) <= 5)
        )
        if can_cluster:
            current_line = action.get("line_no")
            previous_line = previous.get("line_no")
            if isinstance(current_line, int) and isinstance(previous_line, int):
                allowed_gap = cluster_line_gap if action.get("action_kind") in clusterable_kinds else 3
                can_cluster = abs(current_line - previous_line) <= allowed_gap
        if can_cluster:
            clusters[-1].append(action)
        else:
            clusters.append([action])

    events: list[dict[str, Any]] = []
    for cluster in clusters:
        first = cluster[0]
        label = format_action_timeline_label(first)
        if len(cluster) > 1:
            if first.get("action_kind") == "read_file":
                merged_files: list[str] = []
                for item in cluster:
                    merged_files.extend(infer_action_files(item))
                merged_files = sorted(dict.fromkeys(merged_files))
                if merged_files:
                    label = summarize_named_items("Read", merged_files, summary_noun="files", singular_fallback=f"Read files x{len(cluster)}")
                else:
                    label = f"Read files x{len(cluster)}"
            elif first.get("action_kind") == "discover_files":
                merged_targets: list[str] = []
                for item in cluster:
                    merged_targets.extend(infer_discovery_targets(item))
                merged_targets = sorted(dict.fromkeys(merged_targets))
                if merged_targets:
                    label = summarize_named_items("Inspect", merged_targets, summary_noun="targets", singular_fallback=f"Inspect files x{len(cluster)}")
                else:
                    label = f"Inspect files x{len(cluster)}"
            elif first.get("action_kind") == "inline_python":
                label = summarize_inline_python_cluster(cluster)
            elif first.get("action_kind") == "echo":
                label = f"Echo checks x{len(cluster)}"
            elif first.get("action_kind") == "input_discovery":
                label = f"Inspect input data x{len(cluster)}"
        events.append(
            {
                "time": first.get("_dt"),
                "label": label,
                "category": first.get("category") or "Uncategorized",
                "action_kind": first.get("action_kind") or "unknown",
                "count": len(cluster),
                "description": trim_text(" | ".join(item.get("description", "") for item in cluster if item.get("description")), max_len=260),
                "box_width": min(max(estimate_text_width(label, 13) + 34, 124), 240),
            }
        )
    return [event for event in events if event.get("time") is not None]


def short_file_labels(paths: list[str], max_items: int = 2) -> list[str]:
    if not paths:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in {".", "./"}:
            short = "repo root"
        else:
            parts = Path(path).parts
            short = Path(path).name
            if not short:
                short = path.rstrip("/") or "repo root"
            if short in seen and len(parts) >= 2:
                short = "/".join(parts[-2:])
        parts = Path(path).parts
        if short in seen:
            continue
        seen.add(short)
        labels.append(short)
        if len(labels) >= max_items:
            break
    return labels


def format_named_targets(prefix: str, items: list[str], *, singular_fallback: str) -> str:
    if not items:
        return singular_fallback
    names = short_file_labels(items, max_items=min(len(items), 4))
    if len(items) == 1:
        return f"{prefix} {names[0]}"
    if len(items) == 2:
        return f"{prefix} {names[0]} and {names[1]}"
    if len(items) <= 4:
        return f"{prefix} {', '.join(names)}"
    if prefix == "Read":
        return f"Read files for {names[0]} and {names[1]}"
    if prefix == "Inspect":
        return f"Inspect targets for {names[0]} and {names[1]}"
    return f"{prefix} {summary_noun} for {names[0]} and {names[1]}"


def build_codex_context(reconstruction_path: Path) -> dict[str, Any]:
    data = json.loads(reconstruction_path.read_text())
    goal = data.get("goal_context", {})
    skill_records = data.get("skill_reconstruction", [])
    action_flow = data.get("action_flow", {})
    raw_actions = action_flow.get("raw_actions", [])
    steps = action_flow.get("steps", [])
    categories = action_flow.get("high_level_categories", [])
    checkpoints = data.get("governance_artifacts", {}).get("checkpoint_status", {}).get("checkpoints", [])

    parsed_actions: list[dict[str, Any]] = []
    for action in raw_actions:
        entry = dict(action)
        entry["_dt"] = parse_timestamp(action.get("timestamp"))
        parsed_actions.append(entry)
    parsed_actions = [action for action in parsed_actions if action.get("_dt") is not None]
    parsed_actions.sort(key=lambda item: item["_dt"])

    stage_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    stage_order: list[str] = []
    for record in skill_records:
        stage = record.get("workflow_stage") or "unknown"
        if stage not in stage_groups:
            stage_order.append(stage)
        stage_groups[stage].append(record)

    checkpoint_times = [
        parse_timestamp(item.get("timestamp_utc"))
        for item in checkpoints
        if item.get("timestamp_utc")
    ]
    checkpoint_times = [item for item in checkpoint_times if item is not None]
    action_times = [action["_dt"] for action in parsed_actions]
    start = min(action_times) if action_times else (min(checkpoint_times) if checkpoint_times else None)
    end = max(action_times) if action_times else (max(checkpoint_times) if checkpoint_times else None)
    if start and end and end <= start:
        end = start + timedelta(minutes=1)

    minute_buckets: defaultdict[int, Counter[str]] = defaultdict(Counter)
    if start is not None:
        for action in parsed_actions:
            minute_index = int((action["_dt"] - start).total_seconds() // 60)
            minute_buckets[minute_index][action.get("category") or "Uncategorized"] += 1

    line_times = build_line_time_index(parsed_actions)

    unique_evidence_lines = set()
    for record in skill_records:
        unique_evidence_lines.update(record.get("evidence_lines", []))
    for step in steps:
        unique_evidence_lines.update(step.get("evidence_lines", []))

    narrative_steps: list[dict[str, Any]] = []
    for step in steps:
        step_lines = set(step.get("evidence_lines", []))
        line_start = step.get("line_start")
        line_end = step.get("line_end")
        if isinstance(line_start, int) and isinstance(line_end, int) and line_end >= line_start:
            step_lines.update(range(line_start, line_end + 1))
        matched_times = times_for_lines(line_times, step_lines)
        narrative_steps.append(
            {
                **step,
                "start": matched_times[0] if matched_times else None,
                "last_seen": matched_times[-1] if matched_times else None,
                "color": ACTION_CATEGORY_COLORS.get(step.get("category"), PALETTE["tool_call"]),
            }
        )

    last_known_start = start
    for step in narrative_steps:
        if step["start"] is None:
            step["start"] = last_known_start
        else:
            last_known_start = step["start"]
    if end is not None:
        for index, step in enumerate(narrative_steps):
            next_start = next(
                (candidate["start"] for candidate in narrative_steps[index + 1 :] if candidate.get("start") is not None),
                end,
            )
            step["end"] = next_start if next_start is not None else end
            if step["end"] is None:
                step["end"] = step["start"]
    else:
        for step in narrative_steps:
            step["end"] = step["start"]

    important_actions = build_important_action_events(parsed_actions)

    skill_events: list[dict[str, Any]] = []
    skills_without_time = 0
    for record in skill_records:
        evidence_times = times_for_lines(line_times, set(record.get("evidence_lines", [])))
        checkpoint_event_times = [
            parse_timestamp(value)
            for value in record.get("checkpoint_timestamps_utc", [])
            if value
        ]
        checkpoint_event_times = [value for value in checkpoint_event_times if value is not None]
        event_times = evidence_times + checkpoint_event_times
        if not event_times:
            skills_without_time += 1
            continue
        label = humanize_skill_label(record)
        skill_events.append(
            {
                "time": min(event_times),
                "label": label,
                "status": record.get("consideration_status", "unknown"),
                "stage": record.get("workflow_stage") or "unknown",
                "source": "evidence_lines" if evidence_times else "checkpoint",
                "box_width": min(max(estimate_text_width(label, 12) + 28, 120), 220),
            }
        )

    observed_categories = [
        category
        for category in ACTION_CATEGORY_COLORS
        if any(category in bucket for bucket in minute_buckets.values())
    ]

    category_counts = {item["category"]: item["n_actions"] for item in categories}
    max_category_actions = max(category_counts.values(), default=1)
    max_minute_activity = max((sum(bucket.values()) for bucket in minute_buckets.values()), default=1)

    skill_status_counts = Counter(record.get("consideration_status", "unknown") for record in skill_records)
    action_kind_counts = Counter(action.get("action_kind", "unknown") for action in raw_actions)
    trace_context: dict[str, Any] | None = None
    rollout_path = data.get("rollout_path")
    if rollout_path:
        trace_path = Path(rollout_path)
        if trace_path.exists():
            trace_records = load_jsonl(trace_path)
            trace_context = build_visual_context(trace_records, trace_path)

    return {
        "source": data,
        "reconstruction_path": reconstruction_path,
        "rollout_file": data.get("rollout_file"),
        "rollout_path": data.get("rollout_path"),
        "goal_context": goal,
        "skill_records": skill_records,
        "raw_actions": raw_actions,
        "steps": steps,
        "narrative_steps": narrative_steps,
        "important_actions": important_actions,
        "skill_events": skill_events,
        "skills_without_time": skills_without_time,
        "categories": categories,
        "activity_categories": observed_categories,
        "minute_buckets": dict(sorted(minute_buckets.items())),
        "checkpoints": checkpoints,
        "stage_groups": dict(stage_groups),
        "stage_order": stage_order,
        "start": start,
        "end": end,
        "metrics": {
            "skills_total": len(skill_records),
            "actions_total": len(raw_actions),
            "steps_total": len(steps),
            "categories_total": len(categories),
            "checkpoints_total": len(checkpoints),
            "evidence_lines_total": len(unique_evidence_lines),
        },
        "max_category_actions": max_category_actions,
        "max_minute_activity": max_minute_activity,
        "skill_status_counts": dict(skill_status_counts),
        "action_kind_counts": dict(action_kind_counts),
        "trace_context": trace_context,
    }


def draw_goal_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    goal = context["goal_context"]
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Goal Context", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Directly extracted from the reconstruction output: user request, trigger requirements, and assistant restatement.",
        cls="caption",
        baseline="hanging",
    )

    left = x + 28
    top = y + 90
    canvas.text(left, top, "User goal", font_size=15, font_weight=700, baseline="hanging")
    canvas.text_block(left, top + 24, wrap_text(goal.get("user_goal") or "n/a", 78), font_size=14, font_weight=500, line_height=18)

    trigger_top = top + 96
    canvas.text(left, trigger_top, "Trigger requirements", font_size=15, font_weight=700, baseline="hanging")
    trigger_lines = goal.get("trigger_excerpt") or []
    if trigger_lines:
        wrapped = []
        for item in trigger_lines[:5]:
            wrapped.extend(wrap_text("- " + item, 78))
        canvas.text_block(left, trigger_top + 24, wrapped[:10], font_size=14, font_weight=500, line_height=18)
    else:
        canvas.text(left, trigger_top + 28, "No trigger excerpt found", cls="small", baseline="hanging")

    right = x + width * 0.58
    canvas.text(right, top, "Assistant restatement", font_size=15, font_weight=700, baseline="hanging")
    canvas.text_block(right, top + 24, wrap_text(goal.get("assistant_goal") or "n/a", 46), font_size=14, font_weight=500, line_height=18)

    evidence = sorted(
        set(goal.get("user_goal_lines", []) + goal.get("trigger_excerpt_lines", []) + goal.get("assistant_goal_lines", []))
    )
    evidence_text = "Goal evidence lines: " + (", ".join(str(item) for item in evidence) if evidence else "n/a")
    canvas.text_block(right, trigger_top, wrap_text(evidence_text, 46), fill=PALETTE["muted"], font_size=13, font_weight=500, line_height=17)


def draw_category_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Action Categories", cls="section", baseline="hanging")
    canvas.text_block(
        x + 24,
        y + 54,
        wrap_for_pixels("High-level categories derived by reconstruct_rollout_evidence from the action flow.", width - 48, 13, max_len=120)[:2],
        fill=PALETTE["muted"],
        font_size=13,
        font_weight=500,
        line_height=15,
    )

    categories = sorted(context["categories"], key=lambda item: item["n_actions"], reverse=True)
    if not categories:
        canvas.text(x + 24, y + 94, "No action categories available.", cls="small", baseline="hanging")
        return

    bar_left = x + 210
    bar_width = width - 280
    row_top = y + 94
    row_gap = 36
    upper = nice_upper_bound(context["max_category_actions"])

    for index, entry in enumerate(categories[:10]):
        py = row_top + index * row_gap
        label = entry["category"]
        color = ACTION_CATEGORY_COLORS.get(label, PALETTE["tool_call"])
        canvas.text_block(x + 24, py + 1, wrap_for_pixels(label, 170, 13, max_len=48)[:2], font_size=13, font_weight=600, line_height=14)
        canvas.rect(bar_left, py, bar_width, 18, fill="#e7e1d5", rx=9)
        filled = bar_width * entry["n_actions"] / upper if upper else 0
        canvas.rect(bar_left, py, filled, 18, fill=color, rx=9)
        canvas.text(bar_left + filled + 8, py + 9, str(entry["n_actions"]), font_size=12, font_weight=700, baseline="middle")
        canvas.text(x + width - 24, py + 9, f"{entry['n_steps']} steps", cls="small", anchor="end")


def draw_checkpoint_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Checkpoint Timeline", cls="section", baseline="hanging")
    canvas.text_block(
        x + 24,
        y + 54,
        wrap_for_pixels("Checkpoint timestamps and triggers taken from the reconstructed handoff artifacts.", width - 48, 13, max_len=120)[:2],
        fill=PALETTE["muted"],
        font_size=13,
        font_weight=500,
        line_height=15,
    )
    checkpoints = context["checkpoints"]
    if not checkpoints:
        canvas.text(x + 24, y + 96, "No checkpoint timeline found.", cls="small", baseline="hanging")
        return

    lane_x = x + 54
    lane_y = y + height / 2 + 12
    lane_w = width - 108
    canvas.line(lane_x, lane_y, lane_x + lane_w, lane_y, stroke=PALETTE["muted"], stroke_width=2)

    if context["start"] and context["end"] and context["start"] != context["end"]:
        for checkpoint in checkpoints:
            when = parse_timestamp(checkpoint.get("timestamp_utc"))
            if when is None:
                continue
            px = time_to_x(when, context["start"], context["end"], lane_x, lane_w)
            color = PALETTE["milestone"] if checkpoint.get("status") == "pass" else PALETTE["exit"]
            canvas.circle(px, lane_y, 7, fill=color, stroke=PALETTE["bg"], stroke_width=2)
            label_y = lane_y - 54 if checkpoints.index(checkpoint) % 2 == 0 else lane_y + 18
            canvas.text_block(
                px,
                label_y,
                [
                    checkpoint.get("checkpoint_id", ""),
                    checkpoint.get("trigger", ""),
                    checkpoint.get("timestamp_utc", "")[11:16],
                ],
                font_size=12,
                font_weight=600,
                anchor="middle",
                line_height=15,
            )
    else:
        step = lane_w / max(len(checkpoints) - 1, 1)
        for index, checkpoint in enumerate(checkpoints):
            px = lane_x + index * step
            color = PALETTE["milestone"] if checkpoint.get("status") == "pass" else PALETTE["exit"]
            canvas.circle(px, lane_y, 7, fill=color, stroke=PALETTE["bg"], stroke_width=2)
            canvas.text_block(
                px,
                lane_y + 18,
                [checkpoint.get("checkpoint_id", ""), checkpoint.get("trigger", "")],
                font_size=12,
                font_weight=600,
                anchor="middle",
                line_height=15,
            )


def draw_skill_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Skill Reconstruction", cls="section", baseline="hanging")
    canvas.text_block(
        x + 24,
        y + 54,
        wrap_for_pixels("Each card shows the skill title, canonical skill name, and a short purpose summary reconstructed from rollout evidence.", width - 48, 13, max_len=180)[:2],
        fill=PALETTE["muted"],
        font_size=13,
        font_weight=500,
        line_height=15,
    )

    records = context["skill_records"]
    if not records:
        canvas.text(x + 24, y + 96, "No skill reconstruction records found.", cls="small", baseline="hanging")
        return

    columns = 2 if width >= 1200 else 1
    column_gap = 24
    card_w = (width - 48 - column_gap * (columns - 1)) / columns
    card_h = 96
    row_gap = 10
    top = y + 92

    for index, record in enumerate(records):
        col = index % columns
        row = index // columns
        px = x + 24 + col * (card_w + column_gap)
        py = top + row * (card_h + row_gap)
        if py + card_h > y + height - 16:
            break
        stage = record.get("workflow_stage") or "unknown"
        color = ACTION_CATEGORY_COLORS.get("Governance / Handoff", PALETTE["tool_call"])
        canvas.rect(px, py, card_w, card_h, fill="#f8f3ea", stroke=PALETTE["panel_stroke"], stroke_width=1.0, rx=16)
        canvas.rect(px, py, 10, card_h, fill=color, rx=16)
        title = humanize_skill_label(record)
        skill_name = display_skill_name(record)
        purpose = summarize_text(record.get("purpose") or record.get("goal") or record.get("trigger") or "", max_len=130)
        if not purpose:
            purpose = "No additional detail was recovered from rollout evidence; this card is derived from the skill path."
        meta = skill_name if stage == "unknown" else f"{skill_name} | {stage}"
        canvas.text_block(px + 20, py + 10, wrap_for_pixels(title, card_w - 90, 14, max_len=70)[:2], font_size=14, font_weight=700, line_height=15)
        canvas.text_block(px + 20, py + 40, wrap_for_pixels(meta, card_w - 90, 12, max_len=80)[:1], fill=PALETTE["muted"], font_size=12, font_weight=600, line_height=13)
        canvas.text_block(px + 20, py + 56, wrap_for_pixels(purpose, card_w - 34, 12, max_len=130)[:3], font_size=12, font_weight=500, line_height=13)
        canvas.text(px + card_w - 18, py + 16, f"{len(record.get('evidence_lines', []))} lines", cls="small", anchor="end", baseline="baseline")


def draw_workflow_narrative_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Workflow Narrative", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Step-by-step flow reconstructed from action evidence. Colors mark the high-level action category.",
        cls="caption",
        baseline="hanging",
    )

    steps = context["steps"]
    if not steps:
        canvas.text(x + 24, y + 96, "No workflow steps found.", cls="small", baseline="hanging")
        return

    card_x = x + 38
    card_w = width - 76
    top = y + 90
    card_h = 58
    gap = 10

    for index, step in enumerate(steps):
        py = top + index * (card_h + gap)
        if py + card_h > y + height - 20:
            break
        color = ACTION_CATEGORY_COLORS.get(step["category"], PALETTE["tool_call"])
        canvas.rect(card_x, py, card_w, card_h, fill="#f8f3ea", stroke=PALETTE["panel_stroke"], stroke_width=1.0, rx=18)
        canvas.rect(card_x, py, 12, card_h, fill=color, rx=18)
        canvas.circle(card_x + 28, py + card_h / 2, 14, fill=color, stroke=PALETTE["bg"], stroke_width=2)
        canvas.text(card_x + 28, py + card_h / 2, step["step_id"].replace("S", ""), font_size=13, font_weight=700, anchor="middle")
        canvas.text(card_x + 52, py + 16, step["label"], font_size=14, font_weight=700, baseline="baseline")
        canvas.text(card_x + 52, py + 33, step["category"], cls="small", baseline="baseline")
        canvas.text(card_x + 52, py + 49, trim_text(step["description"], max_len=95), font_size=12, font_weight=500, baseline="baseline")
        canvas.text(card_x + card_w - 14, py + 16, "lines " + ",".join(str(item) for item in step.get("evidence_lines", [])[:3]), cls="small", anchor="end", baseline="baseline")
        if index < len(steps) - 1 and py + card_h + gap + card_h <= y + height - 20:
            canvas.line(card_x + 28, py + card_h, card_x + 28, py + card_h + gap, stroke=color, stroke_width=2)


def draw_stage_skill_panel(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Skills By Workflow Stage", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Stages come from the reconstructed skill records and checkpoint-derived workflow stage labels.",
        cls="caption",
        baseline="hanging",
    )

    stage_groups = context["stage_groups"]
    stage_order = context["stage_order"]
    if not stage_order:
        canvas.text(x + 24, y + 96, "No stage-grouped skills found.", cls="small", baseline="hanging")
        return

    box_x = x + 24
    box_w = width - 48
    top = y + 90
    stage_h = 76
    gap = 12

    for index, stage in enumerate(stage_order):
        py = top + index * (stage_h + gap)
        if py + stage_h > y + height - 20:
            break
        skills = stage_groups.get(stage, [])
        color = list(ACTION_CATEGORY_COLORS.values())[index % len(ACTION_CATEGORY_COLORS)]
        canvas.rect(box_x, py, box_w, stage_h, fill="#f8f3ea", stroke=PALETTE["panel_stroke"], stroke_width=1.0, rx=18)
        canvas.rect(box_x, py, 14, stage_h, fill=color, rx=18)
        canvas.text(box_x + 24, py + 18, stage, font_size=14, font_weight=700, baseline="baseline")
        pill_x = box_x + 24
        pill_y = py + 34
        for skill in skills:
            label = skill["skill_name"]
            pill_w = estimate_text_width(label, 12) + 24
            if pill_x + pill_w > box_x + box_w - 14:
                pill_x = box_x + 24
                pill_y += 22
            canvas.rect(pill_x, pill_y, pill_w, 18, fill=color, opacity=0.16, rx=9)
            canvas.text(pill_x + pill_w / 2, pill_y + 9, label, font_size=12, font_weight=600, anchor="middle")
            pill_x += pill_w + 8


def draw_workflow_phase_timeline(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Workflow Narrative Timeline", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Workflow steps are positioned by their first matched evidence timestamp and extend until the next step begins.",
        cls="caption",
        baseline="hanging",
    )

    segments = context["narrative_steps"]
    start = context["start"]
    end = context["end"]
    bar_x = x + 40
    bar_y = y + 92
    bar_w = width - 80
    bar_h = 36

    canvas.rect(bar_x, bar_y, bar_w, bar_h, fill="#e7e1d5", rx=18)
    for segment in segments:
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        if seg_start is None or seg_end is None or start is None or end is None:
            continue
        px1 = time_to_x(seg_start, start, end, bar_x, bar_w)
        px2 = time_to_x(seg_end, start, end, bar_x, bar_w)
        canvas.rect(px1, bar_y, max(px2 - px1, 2), bar_h, fill=segment["color"], rx=18)
        label = f"{segment['step_id']} {segment['label']}"
        if px2 - px1 > estimate_text_width(label, 12) + 12:
            canvas.text((px1 + px2) / 2, bar_y + bar_h / 2, label, font_size=12, font_weight=700, anchor="middle")

    draw_time_ticks(canvas, bar_x, bar_y + bar_h, bar_w, start=start, end=end)

    legend_top = bar_y + 74
    col_w = (width - 88) / 3
    row_h = 20
    for index, segment in enumerate(segments):
        col = index % 3
        row = index // 3
        lx = x + 28 + col * col_w
        ly = legend_top + row * row_h
        canvas.rect(lx, ly, 12, 12, fill=segment["color"], rx=4)
        canvas.text(lx + 20, ly + 6, trim_text(f"{segment['step_id']} {segment['label']}", max_len=32), cls="small")


def draw_category_density_timeline(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Per-minute Activity Density", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Minute buckets stack reconstructed actions by high-level category, so the color mix shows where the workflow shifts.",
        cls="caption",
        baseline="hanging",
    )

    start = context["start"]
    end = context["end"]
    minute_buckets = context["minute_buckets"]
    chart_x = x + 48
    chart_y = y + 94
    chart_w = width - 96
    chart_h = height - 152
    upper = nice_upper_bound(context["max_minute_activity"])
    category_order = context["activity_categories"] or list(ACTION_CATEGORY_COLORS.keys())

    for step in range(6):
        value = upper * step / 5
        py = chart_y + chart_h - chart_h * step / 5
        canvas.line(chart_x, py, chart_x + chart_w, py, stroke=PALETTE["grid"], stroke_width=1)
        canvas.text(chart_x - 10, py, str(int(value)), cls="axis", anchor="end")

    bucket_indices = sorted(minute_buckets)
    bucket_count = max(bucket_indices[-1] + 1 if bucket_indices else 1, 1)
    bar_gap = 2
    bar_width = max((chart_w - bar_gap * (bucket_count - 1)) / bucket_count, 2)

    for minute_index in range(bucket_count):
        bucket = minute_buckets.get(minute_index, {})
        bx = chart_x + minute_index * (bar_width + bar_gap)
        current_top = chart_y + chart_h
        for category in category_order:
            count = bucket.get(category, 0)
            if count <= 0:
                continue
            block_h = chart_h * count / upper if upper else 0
            by = current_top - block_h
            canvas.rect(
                bx,
                by,
                bar_width,
                max(block_h, 1),
                fill=ACTION_CATEGORY_COLORS.get(category, PALETTE["tool_call"]),
                rx=3,
            )
            current_top = by

    draw_time_ticks(canvas, chart_x, chart_y + chart_h, chart_w, start=start, end=end)

    legend_x = x + 26
    legend_y = y + height - 42
    for index, category in enumerate(category_order):
        lx = legend_x + (index % 3) * 210
        ly = legend_y + (index // 3) * 20
        canvas.rect(lx, ly, 12, 12, fill=ACTION_CATEGORY_COLORS.get(category, PALETTE["tool_call"]), rx=4)
        canvas.text(lx + 20, ly + 6, category, cls="small")


def draw_important_actions_timeline(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    canvas.text(x + 24, y + 26, "Important Actions Called", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        "Detailed raw-action timeline showing anchor commands plus clustered read, discovery, and helper activity on the same axis.",
        cls="caption",
        baseline="hanging",
    )

    start = context["start"]
    end = context["end"]
    events = context["important_actions"]
    layout = context.get("important_action_layout")
    if not layout:
        layout = plan_important_action_layout(events, start=start, end=end, panel_x=x, panel_y=y, panel_width=width)
    lane_x = layout["lane_x"]
    lane_y = layout["lane_y"]
    lane_w = layout["lane_w"]
    lane_h = layout["lane_h"]
    baseline_y = layout["baseline_y"]

    shade_narrative_segments(
        canvas,
        context["narrative_steps"],
        start=start,
        end=end,
        left=lane_x,
        top=lane_y,
        width=lane_w,
        height=lane_h,
        opacity=0.11,
    )
    canvas.line(lane_x, baseline_y, lane_x + lane_w, baseline_y, stroke=PALETTE["muted"], stroke_width=2)

    for event in layout["above"]:
        color = ACTION_CATEGORY_COLORS.get(event["category"], PALETTE["tool_call"])
        lines = event["lines"]
        font_size = event.get("font_size", 12)
        line_height = event.get("line_height", 14)
        box_h = event["box_h"]
        box_x = event["box_x"]
        box_y = event["box_y"]
        canvas.line(event["x"], baseline_y, event["x"], box_y, stroke=color, stroke_width=2)
        canvas.circle(event["x"], baseline_y, 5, fill=color, stroke=PALETTE["bg"], stroke_width=2)
        canvas.rect(box_x, box_y, event["box_width"], box_h, fill="#fbf7ef", stroke=color, stroke_width=1.2, rx=12)
        canvas.text_block(box_x + 12, box_y + 8, lines, font_size=font_size, font_weight=600, line_height=line_height)

    for event in layout["below"]:
        color = ACTION_CATEGORY_COLORS.get(event["category"], PALETTE["tool_call"])
        lines = event["lines"]
        font_size = event.get("font_size", 12)
        line_height = event.get("line_height", 14)
        box_h = event["box_h"]
        box_x = event["box_x"]
        box_y = event["box_y"]
        canvas.line(event["x"], baseline_y, event["x"], box_y, stroke=color, stroke_width=2)
        canvas.circle(event["x"], baseline_y, 5, fill=color, stroke=PALETTE["bg"], stroke_width=2)
        canvas.rect(box_x, box_y, event["box_width"], box_h, fill="#fbf7ef", stroke=color, stroke_width=1.2, rx=12)
        canvas.text_block(box_x + 12, box_y + 8, lines, font_size=font_size, font_weight=600, line_height=line_height)

    draw_time_ticks(canvas, lane_x, lane_y + lane_h + 8, lane_w, start=start, end=end)


def draw_skill_invocation_timeline(canvas: SvgCanvas, x: float, y: float, width: float, height: float, context: dict[str, Any]) -> None:
    canvas.panel(x, y, width, height)
    anchored = len(context["skill_events"])
    omitted = context["skills_without_time"]
    canvas.text(x + 24, y + 26, "Skill Invocation Timeline", cls="section", baseline="hanging")
    canvas.text(
        x + 24,
        y + 54,
        f"Skills are shown when the reconstruction contains a direct temporal anchor from evidence lines or checkpoint timestamps ({anchored} shown, {omitted} without time).",
        cls="caption",
        baseline="hanging",
    )

    start = context["start"]
    end = context["end"]
    events = context["skill_events"]
    lane_x = x + 48
    lane_y = y + 116
    lane_w = width - 96
    lane_h = height - 198
    baseline_y = lane_y + lane_h / 2

    shade_narrative_segments(
        canvas,
        context["narrative_steps"],
        start=start,
        end=end,
        left=lane_x,
        top=lane_y,
        width=lane_w,
        height=lane_h,
        opacity=0.08,
    )
    canvas.line(lane_x, baseline_y, lane_x + lane_w, baseline_y, stroke=PALETTE["muted"], stroke_width=2)

    legend_items = [
        ("Explicitly opened", SKILL_STATUS_COLORS["explicitly_opened"]),
        ("Opened + artifact backed", SKILL_STATUS_COLORS["explicitly_opened_and_artifact_backed"]),
        ("Artifact backed only", SKILL_STATUS_COLORS["artifact_backed_without_explicit_open"]),
    ]
    for index, (label, color) in enumerate(legend_items):
        lx = x + 24 + index * 210
        ly = y + 82
        canvas.rect(lx, ly, 12, 12, fill=color, rx=4)
        canvas.text(lx + 20, ly + 6, label, cls="small")

    above = assign_tracks(events[::2], start=start, end=end, left=lane_x, width=lane_w, max_tracks=5)
    below = assign_tracks(events[1::2], start=start, end=end, left=lane_x, width=lane_w, max_tracks=5)

    for event in above:
        color = SKILL_STATUS_COLORS.get(event["status"], PALETTE["tool_call"])
        box_h = 34
        box_y = baseline_y - 18 - (event["track"] + 1) * 40
        box_x = min(max(event["x"] - event["box_width"] / 2, lane_x), lane_x + lane_w - event["box_width"])
        canvas.line(event["x"], baseline_y, event["x"], box_y + box_h, stroke=color, stroke_width=1.8)
        canvas.circle(event["x"], baseline_y, 4.5, fill=color, stroke=PALETTE["bg"], stroke_width=2)
        canvas.rect(box_x, box_y, event["box_width"], box_h, fill="#fbf7ef", stroke=color, stroke_width=1.2, rx=11)
        canvas.text(box_x + 10, box_y + 17, event["label"], font_size=11, font_weight=700, baseline="middle")

    for event in below:
        color = SKILL_STATUS_COLORS.get(event["status"], PALETTE["tool_call"])
        box_h = 34
        box_y = baseline_y + 12 + event["track"] * 40
        box_x = min(max(event["x"] - event["box_width"] / 2, lane_x), lane_x + lane_w - event["box_width"])
        canvas.line(event["x"], baseline_y, event["x"], box_y, stroke=color, stroke_width=1.8)
        canvas.circle(event["x"], baseline_y, 4.5, fill=color, stroke=PALETTE["bg"], stroke_width=2)
        canvas.rect(box_x, box_y, event["box_width"], box_h, fill="#fbf7ef", stroke=color, stroke_width=1.2, rx=11)
        canvas.text(box_x + 10, box_y + 17, event["label"], font_size=11, font_weight=700, baseline="middle")

    draw_time_ticks(canvas, lane_x, lane_y + lane_h + 8, lane_w, start=start, end=end)


def render_dashboard(context: dict[str, Any], output_path: Path) -> None:
    canvas = SvgCanvas(1600, 1920)
    title = context.get("rollout_file") or "Codex reconstruction"
    canvas.text(60, 52, "Codex Evidence Dashboard", cls="title", baseline="hanging")
    canvas.text_block(
        60,
        94,
        wrap_for_pixels(f"Visualization driven by reconstruct_rollout_evidence output for {title}.", 1480, 16, max_len=220)[:2],
        fill=PALETTE["muted"],
        font_size=16,
        font_weight=500,
        line_height=18,
    )

    metrics = context["metrics"]
    trace_context = context.get("trace_context") or {}
    tool_counts = trace_context.get("tool_counts", {})
    tool_call_total = trace_context.get("tool_call_total")
    total_tokens = trace_context.get("total_tokens")
    wall_clock_seconds = ((trace_context.get("summary") or {}).get("time_span") or {}).get("wall_clock_seconds")
    if wall_clock_seconds is None and trace_context.get("start") and trace_context.get("end"):
        wall_clock_seconds = (trace_context["end"] - trace_context["start"]).total_seconds()
    if tool_call_total is None:
        tool_call_total = sum(tool_counts.values())
    chart_context = {
        "start": trace_context.get("start") or context.get("start"),
        "end": trace_context.get("end") or context.get("end"),
        "phases": trace_context.get("phases", []),
        "token_points": trace_context.get("token_points", []),
        "total_tokens": total_tokens,
    }

    card_y = 130
    card_w = 280
    card_h = 124
    gap = 20
    draw_metric_card(canvas, 60, card_y, card_w, card_h, title="Wall-clock", value=format_duration(wall_clock_seconds) if wall_clock_seconds is not None else "n/a", note="Trace span from first to last recorded event")
    draw_metric_card(canvas, 60 + (card_w + gap), card_y, card_w, card_h, title="Skills", value=str(metrics["skills_total"]), note="Hard-evidenced skill records")
    draw_metric_card(canvas, 60 + 2 * (card_w + gap), card_y, card_w, card_h, title="Workflow steps", value=str(metrics["steps_total"]), note="Grouped flowchart steps")
    draw_metric_card(canvas, 60 + 3 * (card_w + gap), card_y, card_w, card_h, title="Tool calls", value=str(tool_call_total or 0), note="Function and custom tools from the rollout trace")
    draw_metric_card(canvas, 60 + 4 * (card_w + gap), card_y, card_w, card_h, title="Total tokens", value=fmt_tokens(total_tokens), note="Final cumulative token usage")

    bar_data = sorted(tool_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    if not bar_data:
        bar_data = sorted(context["action_kind_counts"].items(), key=lambda item: (-item[1], item[0]))[:5]
    draw_bar_chart(canvas, 60, 286, 700, 340, bar_data)
    draw_line_chart(canvas, 800, 286, 740, 340, chart_context)
    draw_category_panel(canvas, 60, 656, 720, 300, context)
    draw_checkpoint_panel(canvas, 820, 656, 720, 220, context)
    draw_skill_panel(canvas, 60, 986, 1480, 880, context)
    canvas.save(output_path)


def render_timeline(context: dict[str, Any], output_path: Path) -> None:
    action_layout = plan_important_action_layout(
        context["important_actions"],
        start=context["start"],
        end=context["end"],
        panel_x=60,
        panel_y=1530,
        panel_width=1480,
    )
    local_context = dict(context)
    local_context["important_action_layout"] = action_layout
    action_panel_height = max(760, int(math.ceil(action_layout["panel_height"])))
    skill_y = 880
    action_y = skill_y + 620 + 30
    canvas_height = action_y + action_panel_height + 70
    canvas = SvgCanvas(1600, canvas_height)
    canvas.text(60, 52, "Temporal Structure of the Codex Workflow", cls="title", baseline="hanging")
    canvas.text(
        60,
        94,
        "Shared-axis timeline showing workflow phases, per-minute category density, important actions, and skill invocation timing from the reconstruction JSON.",
        cls="subtitle",
        baseline="hanging",
    )

    draw_workflow_phase_timeline(canvas, 60, 130, 1480, 330, local_context)
    draw_category_density_timeline(canvas, 60, 490, 1480, 360, local_context)
    draw_skill_invocation_timeline(canvas, 60, skill_y, 1480, 620, local_context)
    draw_important_actions_timeline(canvas, 60, action_y, 1480, action_panel_height, local_context)
    canvas.save(output_path)


def render_manifest(output_dir: Path, figure_paths: list[Path]) -> None:
    captions = {
        "rollout_codex_dashboard.svg": {
            "title": "Figure 1. Codex Evidence Dashboard",
            "caption": "Dashboard view showing wall-clock duration, tool-call breakdown, token usage over time, action-category counts, checkpoint timing, and cleaned-up skill reconstruction cards.",
        },
        "rollout_codex_workflow.svg": {
            "title": "Figure 2. Codex Workflow Map",
            "caption": "Temporal workflow view showing narrative phases, per-minute action-category density, important action calls, and skill invocation timing on a shared axis.",
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
    parser = argparse.ArgumentParser(
        description="Render Codex SVG visuals from reconstruct_rollout_evidence JSON output."
    )
    parser.add_argument("reconstruction", type=Path, help="Path to reconstruct_rollout_evidence JSON output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/rollout_figures_codex"),
        help="Directory for SVG outputs",
    )
    parser.add_argument("--data-out", type=Path, help="Optional JSON dump of the plotting context")
    args = parser.parse_args()

    context = build_codex_context(args.reconstruction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = args.output_dir / "rollout_codex_dashboard.svg"
    workflow_path = args.output_dir / "rollout_codex_workflow.svg"

    render_dashboard(context, dashboard_path)
    render_timeline(context, workflow_path)
    render_manifest(args.output_dir, [dashboard_path, workflow_path])

    if args.data_out:
        serializable = dict(context)
        serializable["start"] = context["start"].isoformat() if context["start"] else None
        serializable["end"] = context["end"].isoformat() if context["end"] else None
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        args.data_out.write_text(json.dumps(serializable, indent=2) + "\n")


if __name__ == "__main__":
    main()
