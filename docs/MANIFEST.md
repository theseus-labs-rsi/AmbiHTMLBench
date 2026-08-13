# Subjective Arena Manifest v1

The public input is one JSON manifest. Paths are resolved relative to the manifest file.

Required top-level fields:

- `models`: unique model names; an even count is required when the schedule is auto-generated.
- `pages`: final page records.

Optional:

- `reserve_pages`: deterministic fallback pool for preflight failures.
- `judge`: default Judge model metadata.
- `schedule.repeats`: number of complete round-robin blocks.
- `schedule.seed`: fixed schedule seed.
- `pair_sets`: explicit schedule. When omitted, the v1 subsystem generates the balanced schedule.

Each page requires:

- `sample_id`
- `reference_image`
- `candidates`: `{model_name: html_path}` for every model.

For GT axes, either provide `active_axes` directly or provide:

- `gt_json`
- `gt_page_id` (defaults to `sample_id`)

Preflight writes a `resolved_manifest.json`. **Only the resolved manifest should be passed to `arena-run`.**
