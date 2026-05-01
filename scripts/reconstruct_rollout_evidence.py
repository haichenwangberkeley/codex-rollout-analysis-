#!/usr/bin/env python3

import argparse
import ast
import json
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


SKILL_OPEN_RE = re.compile(r"skills/([\w./-]+\.md)")
PATCH_FILE_RE = re.compile(r"^\*\*\* (Add File|Update File|Delete File): (.+)$", re.MULTILINE)
PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)
TITLE_RE = re.compile(r"^# Skill:\s*(.+)$", re.MULTILINE)
LAYER1_RE = re.compile(r"## Layer 1.*?\n(.*?)(?:\n## |\Z)", re.DOTALL)
ARTIFACT_RE = re.compile(r"`([^`]+)`")
CHECKPOINT_STATUS_NAME = "outputs/report/skill_checkpoint_status.json"
FINAL_REVIEW_NAME = "outputs/report/final_report_review.json"

ACTION_CATEGORIES = [
    "Task Intake",
    "Documentation Review",
    "Input Discovery",
    "Spec Alignment",
    "Pipeline Execution",
    "Fit / Significance",
    "Reporting",
    "Governance / Handoff",
    "Verification",
    "Finalization",
]

STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "from",
    "with",
    "by",
    "is",
    "are",
    "be",
    "run",
    "final",
    "report",
    "workflow",
    "skill",
    "skills",
    "core",
    "pipeline",
    "governance",
    "interface",
    "interfaces",
    "meta",
}

COMMENTARY_KEYWORDS = [
    "i noticed",
    "plan for completion",
    "i'm now editing",
    "all-samples run is now in progress",
    "i'm writing a single consolidation script",
    "i'm running the local test suite",
    "completed end-to-end",
]

LOCAL_SKILL_PREFIXES = [
    ("local-core-pipeline-", "core_pipeline"),
    ("local-governance-", "governance"),
    ("local-interfaces-", "interfaces"),
    ("local-meta-", "meta"),
    ("local-analysis-strategy-", "analysis_strategy"),
    ("local-infrastructure-", "infrastructure"),
    ("local-open-data-specific-", "open_data_specific"),
    ("local-physics-facts-", "physics_facts"),
]

DEFAULT_STAGE_BY_SKILL = {
    "bootstrap_repo": "bootstrap_repo",
    "read_summary_and_validate": "summary_validated",
    "agent_pre_flight_fact_check": "preflight_ready",
    "skill_refresh_and_checkpointing": "preflight_ready",
    "final_analysis_report_agent_workflow": "report_complete",
    "final_report_review_and_handoff": "handoff_gate",
    "json_spec_driven_execution": "summary_validation_and_spec_to_runtime_mapping",
    "full_statistics_execution_policy": "pipeline_execution_policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct rollout workflow narrative, skill usage, action flow, and "
            "high-level action categories from rollout JSONL evidence only."
        )
    )
    parser.add_argument("rollout", type=Path, help="Path to rollout JSONL file")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output path prefix. Defaults to <rollout stem>.reconstruction next to the rollout.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []  # type: List[Dict[str, Any]]
    with path.open() as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
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


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        try:
            return datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return None


def isoformat_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone().isoformat().replace("+00:00", "Z")


def trim_text(text: str, max_len: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def extract_message_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts = []  # type: List[str]
    for item in content:
        item_type = item.get("type")
        if item_type in {"output_text", "input_text"}:
            parts.append(item.get("text", ""))
    return "\n".join(part for part in parts if part)


def extract_command_head(command: Optional[str]) -> Optional[str]:
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return None
    head = Path(tokens[0]).name
    if head in {"bash", "sh", "zsh"} and len(tokens) >= 3 and tokens[1] == "-lc":
        return extract_command_head(tokens[2])
    return head


def patch_operations(patch_text: str) -> List[Dict[str, str]]:
    operations = []  # type: List[Dict[str, str]]
    for action, path in PATCH_FILE_RE.findall(patch_text):
        operations.append({"action": action.lower().replace(" ", "_"), "path": path})
    for path in PATCH_MOVE_RE.findall(patch_text):
        operations.append({"action": "move_to", "path": path})
    return operations


def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def extract_literal_after_var(script: str, var_name: str) -> Any:
    marker = f"{var_name} = "
    start = script.find(marker)
    if start == -1:
        return None
    cursor = start + len(marker)
    while cursor < len(script) and script[cursor].isspace():
        cursor += 1
    if cursor >= len(script) or script[cursor] not in "[{":
        return None
    opening = script[cursor]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    string_char = ""
    escape = False
    end = cursor
    while end < len(script):
        char = script[end]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_char:
                in_string = False
        else:
            if char in {"'", '"'}:
                in_string = True
                string_char = char
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    end += 1
                    break
        end += 1
    literal_text = script[cursor:end]
    try:
        return ast.literal_eval(literal_text)
    except Exception:
        return None


def sentence_from_layer1(skill_text: str) -> Optional[str]:
    match = LAYER1_RE.search(skill_text)
    if not match:
        return None
    body = match.group(1)
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("-")
    ]
    if not lines:
        return None
    text = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return trim_text(parts[0], 180) if parts else trim_text(text, 180)


def extract_skill_doc_metadata(skill_output: str) -> Dict[str, Any]:
    title_match = TITLE_RE.search(skill_output)
    title = title_match.group(1).strip() if title_match else None
    layer1 = sentence_from_layer1(skill_output)
    artifacts = sorted(set(ARTIFACT_RE.findall(skill_output)))
    return {
        "title": title,
        "layer1_summary": layer1,
        "artifacts": artifacts,
    }


def pretty_skill_name(skill_name: str) -> str:
    tokens = [token for token in skill_name.replace("-", "_").split("_") if token]
    pretty_tokens = []
    for token in tokens:
        if token in {"mc", "json", "pyhf"}:
            pretty_tokens.append(token.upper())
        elif token == "atlas":
            pretty_tokens.append("ATLAS")
        else:
            pretty_tokens.append(token.capitalize())
    return " ".join(pretty_tokens) if pretty_tokens else skill_name


def canonicalize_skill_path(skill_path: str) -> Dict[str, str]:
    normalized = skill_path.strip()
    path_obj = Path(normalized)
    original_name = path_obj.stem
    if path_obj.name != "SKILL.md":
        return {
            "skill_path": normalized,
            "skill_name": original_name,
            "original_skill_path": normalized,
            "display_name": pretty_skill_name(original_name),
        }

    parent_name = path_obj.parent.name
    for prefix, folder in LOCAL_SKILL_PREFIXES:
        if parent_name.startswith(prefix):
            rest = parent_name[len(prefix) :]
            canonical_name = rest.replace("-", "_")
            canonical_path = f"skills/{folder}/{canonical_name}.md"
            return {
                "skill_path": canonical_path,
                "skill_name": canonical_name,
                "original_skill_path": normalized,
                "display_name": pretty_skill_name(canonical_name),
            }

    canonical_name = parent_name.replace("-", "_")
    return {
        "skill_path": normalized,
        "skill_name": canonical_name,
        "original_skill_path": normalized,
        "display_name": pretty_skill_name(canonical_name),
    }


def classify_exec_command(cmd: str) -> Tuple[Optional[str], Optional[str], str]:
    head = extract_command_head(cmd)
    normalized = cmd.strip()
    if "trigger.md" in normalized:
        return ("read_trigger", "Task Intake", "Read trigger.md")
    if "skills/README.md" in normalized:
        return ("read_skills_index", "Documentation Review", "Read skills/README.md")
    match = SKILL_OPEN_RE.search(normalized)
    if match:
        skill_info = canonicalize_skill_path(f"skills/{match.group(1)}")
        return ("read_skill_doc", "Documentation Review", f"Read skill: {skill_info['display_name']}")
    if re.search(r"analysis\.cli\s+run", normalized):
        return ("run_pipeline", "Pipeline Execution", "Run analysis pipeline")
    if "analysis.stats.roofit_combined" in normalized:
        return ("run_fit", "Fit / Significance", "Run RooFit combined fit/significance")
    if "analysis.report.final_report" in normalized:
        return ("generate_report", "Reporting", "Generate final report")
    if "pytest" in normalized:
        return ("run_tests", "Verification", "Run test suite")
    if "git status" in normalized:
        return ("git_status", "Verification", "Check git status")
    if "required_count" in normalized and "missing_count" in normalized:
        return ("artifact_validation", "Verification", "Validate required artifacts")
    if "skill_refresh_plan" in normalized and "<<'PY'" in normalized:
        return ("governance_consolidation", "Governance / Handoff", "Write governance and checkpoint artifacts")
    if "skill_checkpoint_status.json" in normalized and "handoff_ready" in normalized and "write_text" in normalized:
        return ("checkpoint_repair", "Governance / Handoff", "Repair checkpoint status and review")
    if CHECKPOINT_STATUS_NAME in normalized or FINAL_REVIEW_NAME in normalized:
        return ("inspect_handoff_artifacts", "Governance / Handoff", "Inspect handoff artifacts")
    if "updated skill checkpoint status and synced review/report" in normalized:
        return ("checkpoint_repair", "Governance / Handoff", "Repair checkpoint status")
    if "<<'PY'" in normalized or "python3.11 - <<" in normalized:
        return ("inline_python", "Governance / Handoff", "Run inline Python helper")
    if "input-data" in normalized or re.search(r"\bfind\b.*input-data|\bls\b.*input-data|\brg\b.*input-data", normalized):
        return ("input_discovery", "Input Discovery", "Inspect input data")
    if re.search(r"analysis/[\w.-]+\.json", normalized):
        return ("read_analysis_json", "Documentation Review", "Read analysis JSON")
    if any(token in normalized for token in ["sed -n", "cat ", "tail -n", "head -n"]):
        return ("read_file", "Documentation Review", "Read file")
    if head in {"ls", "find", "rg", "grep"}:
        return ("discover_files", "Input Discovery", "Inspect files")
    return (head, "Documentation Review", head or "exec_command")


def is_significant_commentary(text: str, phase: Optional[str]) -> bool:
    lowered = text.lower()
    if phase == "final_answer":
        return True
    return any(keyword in lowered for keyword in COMMENTARY_KEYWORDS)


def extract_goal_context(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_goal = None
    user_goal_lines = []  # type: List[int]
    trigger_excerpt = None
    trigger_lines = []  # type: List[int]
    assistant_goal = None
    assistant_goal_lines = []  # type: List[int]

    for record in records:
        if record["type"] == "event_msg" and record.get("payload", {}).get("type") == "user_message" and user_goal is None:
            payload = record["payload"]
            user_goal = trim_text(payload.get("message", ""), 260)
            user_goal_lines.append(record["_line_no"])
        if record["type"] == "response_item":
            payload = record["payload"]
            if payload.get("type") == "function_call_output":
                output = payload.get("output", "")
                if "Run the analysis defined in `analysis/" in output and "trigger.md" in output and trigger_excerpt is None:
                    lines = []
                    for line in output.splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("Run the analysis defined in") or stripped.startswith("Use the local skills pack") or stripped.startswith("Treat the analysis JSON") or stripped.startswith("Inputs are in") or stripped.startswith("Produce the standard outputs"):
                            lines.append(stripped)
                    if lines:
                        trigger_excerpt = lines
                        trigger_lines.append(record["_line_no"])
            if payload.get("type") == "message" and payload.get("role") == "assistant" and assistant_goal is None:
                text = extract_message_text(payload)
                if "execute `analysis/Higgs-to-diphoton.json`" in text or "execute the JSON-defined" in text:
                    assistant_goal = trim_text(text, 260)
                    assistant_goal_lines.append(record["_line_no"])

    return {
        "user_goal": user_goal,
        "user_goal_lines": user_goal_lines,
        "trigger_excerpt": trigger_excerpt or [],
        "trigger_excerpt_lines": trigger_lines,
        "assistant_goal": assistant_goal,
        "assistant_goal_lines": assistant_goal_lines,
    }


def collect_call_pairs(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    pairs = {}  # type: Dict[str, Dict[str, Any]]
    for record in records:
        if record["type"] != "response_item":
            continue
        payload = record.get("payload", {})
        ptype = payload.get("type")
        if ptype == "function_call":
            call_id = payload.get("call_id")
            if call_id:
                pairs.setdefault(call_id, {})["call"] = record
        elif ptype in {"function_call_output", "custom_tool_call_output"}:
            call_id = payload.get("call_id")
            if call_id:
                pairs.setdefault(call_id, {})["output"] = record
        elif ptype == "custom_tool_call":
            call_id = payload.get("call_id")
            if call_id:
                pairs.setdefault(call_id, {})["call"] = record
    return pairs


def collect_commentary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    commentary = []  # type: List[Dict[str, Any]]
    for record in records:
        if record["type"] not in {"event_msg", "response_item"}:
            continue
        payload = record.get("payload", {})
        if record["type"] == "event_msg" and payload.get("type") == "agent_message":
            text = payload.get("message", "")
            phase = payload.get("phase")
        elif record["type"] == "response_item" and payload.get("type") == "message":
            text = extract_message_text(payload)
            phase = payload.get("phase")
        else:
            continue
        if not text:
            continue
        commentary.append(
            {
                "line_no": record["_line_no"],
                "timestamp": record.get("timestamp"),
                "phase": phase,
                "text": text,
            }
        )
    return commentary


def collect_skill_reconstruction(
    records: List[Dict[str, Any]],
    call_pairs: Dict[str, Dict[str, Any]],
    commentary: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    explicit_skills = {}  # type: Dict[str, Dict[str, Any]]
    checkpoint_requirements = {}  # type: Dict[str, Any]
    checkpoint_status = {}  # type: Dict[str, Any]
    final_review = {}  # type: Dict[str, Any]
    line_to_commentary = {entry["line_no"]: entry for entry in commentary}

    for call_id, pair in call_pairs.items():
        call_record = pair.get("call")
        if not call_record:
            continue
        payload = call_record["payload"]
        ptype = payload.get("type")

        if ptype == "function_call" and payload.get("name") == "exec_command":
            args = maybe_json_loads(payload.get("arguments", {}))
            if not isinstance(args, dict):
                continue
            cmd = args.get("cmd", "")
            match = SKILL_OPEN_RE.search(cmd)
            if match:
                original_skill_path = f"skills/{match.group(1)}"
                if original_skill_path == "skills/README.md":
                    continue
                skill_identity = canonicalize_skill_path(original_skill_path)
                skill_path = skill_identity["skill_path"]
                skill_name = skill_identity["skill_name"]
                output_text = pair.get("output", {}).get("payload", {}).get("output", "")
                metadata = extract_skill_doc_metadata(output_text)
                explicit_skills.setdefault(
                    skill_path,
                    {
                        "skill_name": skill_name,
                        "skill_path": skill_path,
                        "original_skill_paths": {original_skill_path},
                        "open_line": call_record["_line_no"],
                        "open_timestamp": call_record.get("timestamp"),
                        "metadata": metadata,
                        "evidence_lines": {call_record["_line_no"]},
                        "evidence_type": {"explicit_open"},
                    },
                )
                explicit_skills[skill_path]["original_skill_paths"].add(original_skill_path)
                if pair.get("output"):
                    explicit_skills[skill_path]["evidence_lines"].add(pair["output"]["_line_no"])

            if "checkpoint_requirements" in cmd and "reqs =" in cmd:
                extracted = extract_literal_after_var(cmd, "reqs")
                if isinstance(extracted, dict):
                    checkpoint_requirements = extracted

        if ptype == "function_call_output":
            output_text = payload.get("output", "")
            if CHECKPOINT_STATUS_NAME in output_text:
                parsed = extract_json_blob(output_text)
                if parsed:
                    checkpoint_status = parsed
            elif FINAL_REVIEW_NAME in output_text:
                parsed = extract_json_blob(output_text)
                if parsed:
                    final_review = parsed

    status_by_checkpoint = {
        item.get("checkpoint_id"): item
        for item in checkpoint_status.get("checkpoints", [])
        if isinstance(item, dict)
    }

    skill_records = {}  # type: Dict[str, Dict[str, Any]]

    for skill_path, info in explicit_skills.items():
        skill_records[skill_path] = {
            "skill_name": info["skill_name"],
            "skill_path": skill_path,
            "original_skill_paths": sorted(info.get("original_skill_paths", [])),
            "doc_title": info["metadata"].get("title") or pretty_skill_name(info["skill_name"]),
            "goal": info["metadata"].get("layer1_summary"),
            "purpose": None,
            "trigger": None,
            "trigger_source": None,
            "workflow_stage": DEFAULT_STAGE_BY_SKILL.get(info["skill_name"]),
            "consideration_status": "explicitly_opened",
            "used": True,
            "evidence_lines": set(info["evidence_lines"]),
            "evidence_type": set(info["evidence_type"]),
            "doc_artifacts": info["metadata"].get("artifacts", []),
            "checkpoint_ids": [],
            "checkpoint_timestamps_utc": [],
            "checkpoint_triggers": [],
            "workflow_narrative_connection": None,
        }

    checkpoint_to_skills = {}  # type: Dict[str, List[str]]
    for checkpoint_id, payload in checkpoint_requirements.items():
        if not isinstance(payload, dict):
            continue
        checkpoint_to_skills[checkpoint_id] = list(payload.get("skills", []))
        artifacts = list(payload.get("artifacts", []))
        status_payload = status_by_checkpoint.get(checkpoint_id, {})
        for raw_skill_path in checkpoint_to_skills[checkpoint_id]:
            skill_identity = canonicalize_skill_path(raw_skill_path)
            skill_path = skill_identity["skill_path"]
            skill_name = skill_identity["skill_name"]
            record = skill_records.setdefault(
                skill_path,
                {
                    "skill_name": skill_name,
                    "skill_path": skill_path,
                    "original_skill_paths": [skill_identity["original_skill_path"]],
                    "doc_title": pretty_skill_name(skill_name),
                    "goal": None,
                    "purpose": None,
                    "trigger": None,
                    "trigger_source": None,
                    "workflow_stage": DEFAULT_STAGE_BY_SKILL.get(skill_name),
                    "consideration_status": "artifact_backed_without_explicit_open",
                    "used": True,
                    "evidence_lines": set(),
                    "evidence_type": {"checkpoint_requirement"},
                    "doc_artifacts": [],
                    "checkpoint_ids": [],
                    "checkpoint_timestamps_utc": [],
                    "checkpoint_triggers": [],
                    "workflow_narrative_connection": None,
                },
            )
            record.setdefault("original_skill_paths", [])
            if skill_identity["original_skill_path"] not in record["original_skill_paths"]:
                record["original_skill_paths"].append(skill_identity["original_skill_path"])
            record["checkpoint_ids"].append(checkpoint_id)
            trigger = status_payload.get("trigger")
            if trigger:
                record["checkpoint_triggers"].append(trigger)
            timestamp = status_payload.get("timestamp_utc")
            if timestamp:
                record["checkpoint_timestamps_utc"].append(timestamp)
            if trigger and record["trigger"] is None:
                record["trigger"] = trigger
                record["trigger_source"] = "checkpoint_status"
            if artifacts and record["purpose"] is None:
                artifact_text = ", ".join(artifacts[:4])
                record["purpose"] = f"Support checkpoint `{checkpoint_id}` by producing or validating: {artifact_text}"
            if record["workflow_stage"] is None:
                record["workflow_stage"] = checkpoint_id
            if record["goal"] is None:
                record["goal"] = f"Complete checkpoint `{checkpoint_id}` with its required artifacts"

    plan_commentary_candidates = [
        entry for entry in commentary if "plan for completion" in entry["text"].lower()
    ]
    mismatch_commentary_candidates = [
        entry for entry in commentary if "spec mismatch" in entry["text"].lower()
    ]

    for skill_path, record in skill_records.items():
        skill_name = record["skill_name"]
        checkpoint_ids = record["checkpoint_ids"]

        if checkpoint_ids:
            record["consideration_status"] = (
                "explicitly_opened_and_artifact_backed"
                if "explicit_open" in record["evidence_type"]
                else "artifact_backed_without_explicit_open"
            )
            primary_checkpoint = checkpoint_ids[0]
            if record["workflow_stage"] in {None, primary_checkpoint}:
                record["workflow_stage"] = primary_checkpoint

        if skill_name == "json_spec_driven_execution":
            if mismatch_commentary_candidates:
                entry = mismatch_commentary_candidates[0]
                record["trigger"] = trim_text(entry["text"], 180)
                record["trigger_source"] = "commentary"
                record["evidence_lines"].add(entry["line_no"])
                record["evidence_type"].add("commentary")
            if record["purpose"] is None:
                record["purpose"] = (
                    "Validate the structured analysis JSON, align runtime configuration to it, "
                    "and record mapping/deviation artifacts."
                )
            record["workflow_stage"] = "summary_validation_and_spec_to_runtime_mapping"
            record["workflow_narrative_connection"] = (
                "The rollout explicitly reports a 10-category versus 5-category mismatch, "
                "then patches the runtime YAML files before the main run."
            )
        elif skill_name in {
            "mc_sample_disambiguation_and_nominal_selection",
            "data_mc_discrepancy_sanity_check",
            "skill_refresh_and_checkpointing",
            "final_analysis_report_agent_workflow",
            "extract_new_skill_from_failure",
            "final_report_review_and_handoff",
        }:
            if plan_commentary_candidates:
                entry = plan_commentary_candidates[0]
                text = entry["text"]
                if skill_name == "mc_sample_disambiguation_and_nominal_selection" and "mc_sample_selection" in text:
                    record["trigger"] = trim_text(text, 180)
                    record["trigger_source"] = "plan_commentary"
                    record["evidence_lines"].add(entry["line_no"])
                    record["evidence_type"].add("commentary")
                elif skill_name == "data_mc_discrepancy_sanity_check" and "discrepancy audit" in text:
                    record["trigger"] = trim_text(text, 180)
                    record["trigger_source"] = "plan_commentary"
                    record["evidence_lines"].add(entry["line_no"])
                    record["evidence_type"].add("commentary")
                elif skill_name == "skill_refresh_and_checkpointing" and "skill refresh/checkpoint artifacts" in text:
                    record["trigger"] = trim_text(text, 180)
                    record["trigger_source"] = "plan_commentary"
                    record["evidence_lines"].add(entry["line_no"])
                    record["evidence_type"].add("commentary")
                elif skill_name == "final_analysis_report_agent_workflow" and "Generate publication-style final report" in text:
                    record["trigger"] = trim_text(text, 180)
                    record["trigger_source"] = "plan_commentary"
                    record["evidence_lines"].add(entry["line_no"])
                    record["evidence_type"].add("commentary")
                elif skill_name == "extract_new_skill_from_failure" and "skill extraction summary" in text:
                    record["trigger"] = trim_text(text, 180)
                    record["trigger_source"] = "plan_commentary"
                    record["evidence_lines"].add(entry["line_no"])
                    record["evidence_type"].add("commentary")
                elif skill_name == "final_report_review_and_handoff" and "handoff review" in text:
                    record["trigger"] = trim_text(text, 180)
                    record["trigger_source"] = "plan_commentary"
                    record["evidence_lines"].add(entry["line_no"])
                    record["evidence_type"].add("commentary")

        if record["purpose"] is None and skill_name == "mc_sample_disambiguation_and_nominal_selection":
            record["purpose"] = "Write outputs/report/mc_sample_selection.json and document nominal versus alternative H->gamma gamma sample choices."
        elif record["purpose"] is None and skill_name == "data_mc_discrepancy_sanity_check":
            record["purpose"] = "Write discrepancy audit/check-log artifacts and carry their status into reporting and handoff."
        elif record["purpose"] is None and skill_name == "skill_refresh_and_checkpointing":
            record["purpose"] = "Create the skill-refresh plan, log, checkpoint status, and handoff gate evidence."
        elif record["purpose"] is None and skill_name == "final_analysis_report_agent_workflow":
            record["purpose"] = "Generate the publication-style final report and its required appendices."
        elif record["purpose"] is None and skill_name == "extract_new_skill_from_failure":
            record["purpose"] = "Write the post-run skill-extraction summary even when no new candidate skills are proposed."
        elif record["purpose"] is None and skill_name == "final_report_review_and_handoff":
            record["purpose"] = "Review final artifacts and decide whether the run is handoff-ready."
        elif record["purpose"] is None and skill_name == "read_summary_and_validate":
            record["purpose"] = "Validate the structured analysis summary before execution."
        elif record["purpose"] is None and skill_name == "agent_pre_flight_fact_check":
            record["purpose"] = "Verify preflight assumptions and write the preflight fact-check artifact."
        elif record["purpose"] is None and skill_name == "bootstrap_repo":
            record["purpose"] = "Bootstrap the repository workflow and establish the analysis execution environment."
        elif record["purpose"] is None and skill_name == "full_statistics_execution_policy":
            record["purpose"] = "Enforce the policy that final results must be based on full-statistics execution."
        elif record["purpose"] is None and skill_name == "plotting_and_report":
            record["purpose"] = "Produce the reporting outputs required for the report-complete stage."
        elif record["purpose"] is None and skill_name == "histogramming_and_templates":
            record["purpose"] = "Produce histogram, cutflow, and template-side outputs for the selection stage."
        elif record["purpose"] is None and skill_name == "selection_engine_and_regions":
            record["purpose"] = "Apply selections and region/category partitioning for the analysis workflow."
        elif record["purpose"] is None and skill_name == "profile_likelihood_significance":
            record["purpose"] = "Run the profile-likelihood significance stage and validate its fit outputs."
        elif record["purpose"] is None and skill_name == "workspace_and_fit_pyhf":
            record["purpose"] = "Build workspace and fit artifacts for the statistics stage."

        if record["goal"] is None:
            record["goal"] = record["doc_title"] or pretty_skill_name(skill_name)

        if record["trigger"] is None and record["checkpoint_triggers"]:
            record["trigger"] = record["checkpoint_triggers"][0]
            record["trigger_source"] = "checkpoint_status"

        if record["workflow_narrative_connection"] is None:
            if checkpoint_ids:
                record["workflow_narrative_connection"] = (
                    f"The skill is tied to checkpoint(s) {', '.join(checkpoint_ids)} "
                    "through generated checkpoint requirements and final checkpoint status."
                )
            elif record["consideration_status"].startswith("explicitly_opened"):
                record["workflow_narrative_connection"] = (
                    "The skill was explicitly opened during the rollout and later connected to nearby actions or artifacts."
                )

    normalized_records = []  # type: List[Dict[str, Any]]
    for record in sorted(skill_records.values(), key=lambda item: (item["workflow_stage"] or "", item["skill_path"])):
        normalized_records.append(
            {
                "skill_name": record["skill_name"],
                "skill_path": record["skill_path"],
                "original_skill_paths": record.get("original_skill_paths", []),
                "doc_title": record["doc_title"],
                "goal": record["goal"],
                "trigger": record["trigger"],
                "trigger_source": record["trigger_source"],
                "purpose": record["purpose"],
                "workflow_stage": record["workflow_stage"],
                "consideration_status": record["consideration_status"],
                "used": record["used"],
                "considered_but_not_used": False,
                "checkpoint_ids": record["checkpoint_ids"],
                "checkpoint_timestamps_utc": record["checkpoint_timestamps_utc"],
                "checkpoint_triggers": record["checkpoint_triggers"],
                "evidence_type": sorted(record["evidence_type"]),
                "evidence_lines": sorted(record["evidence_lines"]),
                "workflow_narrative_connection": record["workflow_narrative_connection"],
            }
        )

    aux = {
        "checkpoint_requirements": checkpoint_requirements,
        "checkpoint_status": checkpoint_status,
        "final_review": final_review,
    }
    return normalized_records, aux


def collect_actions(
    records: List[Dict[str, Any]],
    call_pairs: Dict[str, Dict[str, Any]],
    commentary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    actions = []  # type: List[Dict[str, Any]]
    seen_call_ids = set()  # type: Set[str]

    for record in records:
        if record["type"] == "event_msg" and record.get("payload", {}).get("type") == "user_message":
            actions.append(
                {
                    "timestamp": record.get("timestamp"),
                    "line_no": record["_line_no"],
                    "action_kind": "user_request",
                    "category": "Task Intake",
                    "label": "Receive user request",
                    "description": trim_text(record["payload"].get("message", ""), 220),
                    "command": None,
                    "files": [],
                    "evidence_lines": [record["_line_no"]],
                }
            )
            continue

        if record["type"] != "response_item":
            continue
        payload = record.get("payload", {})
        ptype = payload.get("type")

        if ptype == "function_call":
            call_id = payload.get("call_id")
            if not call_id or call_id in seen_call_ids:
                continue
            seen_call_ids.add(call_id)

            if payload.get("name") == "exec_command":
                args = maybe_json_loads(payload.get("arguments", {}))
                if not isinstance(args, dict):
                    continue
                cmd = args.get("cmd", "")
                action_kind, category, label = classify_exec_command(cmd)
                output_record = call_pairs.get(call_id, {}).get("output")
                output_text = ""
                if output_record:
                    output_text = output_record.get("payload", {}).get("output", "")
                files = []  # type: List[str]
                if action_kind == "read_file":
                    files = re.findall(r"([\w./-]+\.(?:md|json|py|yaml))", cmd)
                elif action_kind == "read_skill_doc":
                    match = SKILL_OPEN_RE.search(cmd)
                    if match:
                        files = [canonicalize_skill_path(f"skills/{match.group(1)}")["skill_path"]]
                actions.append(
                    {
                        "timestamp": record.get("timestamp"),
                        "line_no": record["_line_no"],
                        "action_kind": action_kind,
                        "category": category,
                        "label": label,
                        "description": trim_text(output_text or cmd, 220),
                        "command": cmd,
                        "files": files,
                        "evidence_lines": [record["_line_no"]] + ([output_record["_line_no"]] if output_record else []),
                    }
                )

        elif ptype == "custom_tool_call" and payload.get("name") == "apply_patch":
            ops = patch_operations(payload.get("input", ""))
            files = [entry["path"] for entry in ops if entry["action"] != "move_to"]
            actions.append(
                {
                    "timestamp": record.get("timestamp"),
                    "line_no": record["_line_no"],
                    "action_kind": "apply_patch",
                    "category": "Spec Alignment",
                    "label": "Patch runtime configuration",
                    "description": trim_text(", ".join(files), 220),
                    "command": None,
                    "files": files,
                    "evidence_lines": [record["_line_no"]],
                }
            )

    for entry in commentary:
        if not is_significant_commentary(entry["text"], entry["phase"]):
            continue
        lowered = entry["text"].lower()
        if "spec mismatch" in lowered:
            kind = "spec_mismatch"
            category = "Spec Alignment"
            label = "Detect spec mismatch"
        elif "plan for completion" in lowered:
            kind = "plan"
            category = "Pipeline Execution"
            label = "Declare completion plan"
        elif "consolidation script" in lowered:
            kind = "governance_note"
            category = "Governance / Handoff"
            label = "Announce governance artifact write"
        elif "local test suite" in lowered:
            kind = "test_note"
            category = "Verification"
            label = "Announce smoke validation"
        elif entry["phase"] == "final_answer":
            kind = "final_summary"
            category = "Finalization"
            label = "Summarize completed run"
        else:
            kind = "commentary"
            category = "Pipeline Execution"
            label = "Commentary update"
        actions.append(
            {
                "timestamp": entry["timestamp"],
                "line_no": entry["line_no"],
                "action_kind": kind,
                "category": category,
                "label": label,
                "description": trim_text(entry["text"], 240),
                "command": None,
                "files": [],
                "evidence_lines": [entry["line_no"]],
            }
        )

    def sort_key(item: Dict[str, Any]) -> Tuple[Any, int]:
        return (parse_timestamp(item["timestamp"]) or datetime.min, item["line_no"])

    actions.sort(key=sort_key)
    return actions


def build_flow_steps(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    notable_kinds = {
        "user_request",
        "read_trigger",
        "read_skills_index",
        "read_analysis_json",
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
        "final_summary",
    }
    filtered = [action for action in actions if action["action_kind"] in notable_kinds]

    grouped = []  # type: List[List[Dict[str, Any]]]
    for action in filtered:
        if grouped and action["action_kind"] == grouped[-1][-1]["action_kind"] and action["label"] == grouped[-1][-1]["label"]:
            grouped[-1].append(action)
        else:
            grouped.append([action])

    steps = []  # type: List[Dict[str, Any]]
    for index, group in enumerate(grouped, start=1):
        first = group[0]
        last = group[-1]
        files = []  # type: List[str]
        evidence_lines = []  # type: List[int]
        descriptions = []  # type: List[str]
        for item in group:
            files.extend(item["files"])
            evidence_lines.extend(item["evidence_lines"])
            descriptions.append(item["description"])
        files = sorted(set(files))
        evidence_lines = sorted(set(evidence_lines))
        description = trim_text(" | ".join(desc for desc in descriptions if desc), 260)
        label = first["label"]
        if len(group) > 1 and first["action_kind"] == "apply_patch":
            label = "Patch runtime configuration files"
        step = {
            "step_id": f"S{index}",
            "category": first["category"],
            "label": label,
            "description": description,
            "action_kind": first["action_kind"],
            "line_start": first["line_no"],
            "line_end": last["line_no"],
            "evidence_lines": evidence_lines,
            "files": files,
        }
        steps.append(step)
    return steps


def summarize_categories(actions: List[Dict[str, Any]], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions_by_category = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
    steps_by_category = defaultdict(list)  # type: Dict[str, List[str]]
    for action in actions:
        actions_by_category[action["category"]].append(action)
    for step in steps:
        steps_by_category[step["category"]].append(step["step_id"])

    output = []  # type: List[Dict[str, Any]]
    for category in ACTION_CATEGORIES:
        category_actions = actions_by_category.get(category, [])
        if not category_actions:
            continue
        counter = Counter(action["action_kind"] for action in category_actions)
        output.append(
            {
                "category": category,
                "n_actions": len(category_actions),
                "n_steps": len(steps_by_category.get(category, [])),
                "step_ids": steps_by_category.get(category, []),
                "action_kind_counts": dict(counter),
                "example_labels": [action["label"] for action in category_actions[:5]],
            }
        )
    return output


def build_mermaid_flowchart(steps: List[Dict[str, Any]]) -> str:
    lines = ["flowchart TD"]
    for step in steps:
        label = f"{step['step_id']}: {step['label']}\\n[{step['category']}]"
        label = label.replace('"', "'")
        lines.append(f'    {step["step_id"]}["{label}"]')
    for left, right in zip(steps, steps[1:]):
        lines.append(f'    {left["step_id"]} --> {right["step_id"]}')
    return "\n".join(lines)


def build_workflow_narrative(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": step["step_id"],
            "label": step["label"],
            "category": step["category"],
            "description": step["description"],
            "evidence_lines": step["evidence_lines"],
        }
        for step in steps
    ]


def build_machine_readable_output(
    rollout_path: Path,
    goal_context: Dict[str, Any],
    skills: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    aux: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "rollout_file": rollout_path.name,
        "rollout_path": str(rollout_path.resolve()),
        "reconstruction_method": {
            "hard_evidence_only": True,
            "evidence_sources": [
                "user and assistant messages preserved in the rollout",
                "function_call and function_call_output records",
                "custom_tool_call apply_patch records",
                "generated checkpoint requirements embedded in inline Python",
                "rendered checkpoint status and final review artifacts printed in rollout output",
            ],
        },
        "goal_context": goal_context,
        "skill_reconstruction": skills,
        "action_flow": {
            "raw_actions": actions,
            "steps": steps,
            "high_level_categories": categories,
            "workflow_narrative": build_workflow_narrative(steps),
        },
        "governance_artifacts": {
            "checkpoint_status": aux.get("checkpoint_status", {}),
            "final_report_review": aux.get("final_review", {}),
        },
        "negative_findings": {
            "considered_but_not_used_skills": [],
            "note": "This script only marks skills as considered when the rollout contains hard evidence for that claim.",
        },
    }


def evidence_link(rollout_path: Path, line_no: int) -> str:
    return f"[{line_no}]({rollout_path.resolve()}:{line_no})"


def render_markdown(
    rollout_path: Path,
    data: Dict[str, Any],
) -> str:
    lines = []  # type: List[str]
    lines.append(f"# Rollout Evidence Reconstruction: `{rollout_path.stem}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Rollout: [{rollout_path.name}]({rollout_path.resolve()}:1)")
    lines.append("")
    lines.append("## Goal Context")
    lines.append("")
    goal = data["goal_context"]
    if goal.get("user_goal"):
        lines.append(f"- User goal: `{goal['user_goal']}`")
    if goal.get("trigger_excerpt"):
        lines.append("- Trigger requirements:")
        for item in goal["trigger_excerpt"]:
            lines.append(f"  - `{item}`")
    if goal.get("assistant_goal"):
        lines.append(f"- Assistant restatement: `{goal['assistant_goal']}`")
    evidence_lines = sorted(
        set(goal.get("user_goal_lines", []) + goal.get("trigger_excerpt_lines", []) + goal.get("assistant_goal_lines", []))
    )
    if evidence_lines:
        links = ", ".join(evidence_link(rollout_path, line_no) for line_no in evidence_lines)
        lines.append(f"- Goal evidence: {links}")
    lines.append("")
    lines.append("## Action Flowchart")
    lines.append("")
    lines.append("```mermaid")
    lines.append(build_mermaid_flowchart(data["action_flow"]["steps"]))
    lines.append("```")
    lines.append("")
    lines.append("## Workflow Narrative")
    lines.append("")
    for step in data["action_flow"]["workflow_narrative"]:
        links = ", ".join(evidence_link(rollout_path, line_no) for line_no in step["evidence_lines"])
        lines.append(
            f"1. `{step['step_id']}` `{step['label']}` [{step['category']}]: {step['description']} Evidence: {links}"
        )
    lines.append("")
    lines.append("## High-Level Action Categories")
    lines.append("")
    lines.append("| Category | Actions | Steps | Example labels |")
    lines.append("| --- | --- | --- | --- |")
    for entry in data["action_flow"]["high_level_categories"]:
        examples = "; ".join(entry["example_labels"][:3])
        lines.append(
            f"| `{entry['category']}` | {entry['n_actions']} | {entry['n_steps']} | {examples} |"
        )
    lines.append("")
    lines.append("## Skill Reconstruction")
    lines.append("")
    lines.append("| Skill | Goal | Trigger | Purpose | Workflow stage | Status | Evidence |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for skill in data["skill_reconstruction"]:
        evidence = ", ".join(evidence_link(rollout_path, line_no) for line_no in skill["evidence_lines"])
        goal_text = trim_text(skill.get("goal") or "", 120)
        trigger_text = trim_text(skill.get("trigger") or "", 120)
        purpose_text = trim_text(skill.get("purpose") or "", 120)
        stage_text = skill.get("workflow_stage") or ""
        lines.append(
            f"| `{skill['skill_path']}` | {goal_text} | {trigger_text} | {purpose_text} | `{stage_text}` | "
            f"`{skill['consideration_status']}` | {evidence} |"
        )
    checkpoint_status = data.get("governance_artifacts", {}).get("checkpoint_status", {})
    checkpoints = checkpoint_status.get("checkpoints", [])
    if checkpoints:
        lines.append("")
        lines.append("## Checkpoint Timeline")
        lines.append("")
        lines.append("| Checkpoint | Trigger | Timestamp UTC | Status |")
        lines.append("| --- | --- | --- | --- |")
        for item in checkpoints:
            lines.append(
                f"| `{item.get('checkpoint_id', '')}` | `{item.get('trigger', '')}` | `{item.get('timestamp_utc', '')}` | `{item.get('status', '')}` |"
            )
    lines.append("")
    lines.append("## Negative Finding")
    lines.append("")
    lines.append(
        "No skill is emitted as considered-but-unused unless the rollout itself contains hard evidence for that claim."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rollout_path = args.rollout.resolve()
    records = load_jsonl(rollout_path)
    call_pairs = collect_call_pairs(records)
    commentary = collect_commentary(records)
    goal_context = extract_goal_context(records)
    skills, aux = collect_skill_reconstruction(records, call_pairs, commentary)
    actions = collect_actions(records, call_pairs, commentary)
    steps = build_flow_steps(actions)
    categories = summarize_categories(actions, steps)

    data = build_machine_readable_output(
        rollout_path=rollout_path,
        goal_context=goal_context,
        skills=skills,
        actions=actions,
        steps=steps,
        categories=categories,
        aux=aux,
    )

    if args.out_prefix is None:
        out_prefix = rollout_path.with_suffix("")
        out_prefix = out_prefix.parent / f"{out_prefix.name}.reconstruction"
    else:
        out_prefix = args.out_prefix

    json_path = Path(f"{out_prefix}.json")
    md_path = Path(f"{out_prefix}.md")

    json_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    md_path.write_text(render_markdown(rollout_path, data))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
