# Codex Rollout Analysis

Standalone tooling for analyzing Codex session traces stored as `rollout.jsonl`.

The repo contains two dependency-free scripts:

- `scripts/analyze_rollout.py`
  Creates a structured JSON summary and a paper-style Markdown report.
- `scripts/render_rollout_visuals.py`
  Renders SVG figures directly from the trace and writes a figure manifest that the report can embed.

## Design Goals

- No third-party Python dependencies
- Reproducible from a raw `rollout.jsonl`
- Suitable for both quick inspection and paper-style writeups
- Visual outputs remain vector SVG so they embed well in Markdown reports

## Quick Start

From this repo root:

```bash
python3 scripts/render_rollout_visuals.py /path/to/rollout.jsonl \
  --output-dir reports/rollout_figures \
  --data-out reports/rollout_visual_context.json

python3 scripts/analyze_rollout.py /path/to/rollout.jsonl \
  --json-out reports/rollout_examination_summary.json \
  --md-out reports/rollout_examination_report.md \
  --figure-dir reports/rollout_figures
```

## Outputs

- `reports/rollout_examination_summary.json`
- `reports/rollout_examination_report.md`
- `reports/rollout_visual_context.json`
- `reports/rollout_figures/rollout_dashboard.svg`
- `reports/rollout_figures/rollout_timeline.svg`
- `reports/rollout_figures/manifest.json`

## Notes

- The visual renderer uses only the Python standard library and writes plain SVG.
- The report generator can run with or without figures. If `--figure-dir` is omitted, it produces a text-only report.
- The phase timeline in the visual report uses a transparent heuristic:
  audit/repair ends at the last patch event, and the rest of the first turn is treated as the blinded execution phase.
