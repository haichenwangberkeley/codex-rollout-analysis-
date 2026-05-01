# Codex Reconstruction Workflow

Instructions for taking a new Codex rollout `.jsonl` file and producing the reconstruction artifacts plus the dashboard/workflow visualizations.

## What This Produces

Given a new rollout, this workflow creates:

- A machine-readable reconstruction JSON
- A Markdown reconstruction summary
- A Codex dashboard SVG
- A Codex workflow SVG
- A figure manifest JSON

Optional:

- PNG copies of the two SVG figures

## Scripts

The workflow uses these two scripts:

- `scripts/reconstruct_rollout_evidence.py`
- `scripts/render_rollout_visuals_codex.py`

## Run From Repo Root

Example input rollout:

```bash
rollout-2026-05-01T13-14-05-019de52d-51f5-7e10-abaa-27cf600efc0c.jsonl
```

Pick a short output id. The existing convention is to use the middle UUID chunks, for example:

```bash
51f5_7e10
```

Then run:

```bash
mkdir -p reports/<short_id>

python3 scripts/reconstruct_rollout_evidence.py \
  rollout-YYYY-MM-DDT...jsonl \
  --out-prefix reports/<short_id>/rollout_codex_reconstruction

python3 scripts/render_rollout_visuals_codex.py \
  reports/<short_id>/rollout_codex_reconstruction.json \
  --output-dir reports/<short_id>/visuals
```

## Outputs

After the two commands finish, you should have:

- `reports/<short_id>/rollout_codex_reconstruction.json`
- `reports/<short_id>/rollout_codex_reconstruction.md`
- `reports/<short_id>/visuals/rollout_codex_dashboard.svg`
- `reports/<short_id>/visuals/rollout_codex_workflow.svg`
- `reports/<short_id>/visuals/manifest.json`

## Optional PNG Export

If you also want PNG copies of the figures:

```bash
magick reports/<short_id>/visuals/rollout_codex_dashboard.svg \
  reports/<short_id>/visuals/rollout_codex_dashboard.png

magick reports/<short_id>/visuals/rollout_codex_workflow.svg \
  reports/<short_id>/visuals/rollout_codex_workflow.png
```
