# Subjective evidence and versioned results

This directory contains lightweight records of **completed** experiments, not
new measurements. Large judgment JSONL files, human records, model outputs,
screenshots, SQLite checkpoints and their backups are external artifacts.
See [the protocol](../../docs/FROZEN_PROXY.md) and
[provenance](../../docs/SUBJECTIVE_SYNC_PROVENANCE.md).

| Directory | Meaning |
|---|---|
| `original_single/` | Original 10,000-comparison single-vote deployment (Sol BT 154.34) |
| `candidate_refresh/` | Paper's candidate-update version (Sol BT 154.70); separate results and delta |
| `human/` | Aggregate human coverage, agreement and stability evidence, not individual responses |
| `calibration/` | Five-feature OLS, eight-annotation-block validation and recorded uncertainty |
| `committee/` | Historical discrete committee selection, including the pilot-to-holdout reversal |
| `ablation/` | Historical subset/weighting analysis; its estimators and weighting differ from OLS5 |

The copied source evidence preserves historical field names and values. In
human/calibration/ablation records, `paper_id`, `paper` and `leave-one-paper-out`
refer to **annotation blocks**, not source papers or individual pages. Committee
selection fields describe that historical discrete protocol, not the currently
deployed OLS5 proxy. `candidate_refresh/sync_audit.json` records preparation;
its `READY_FOR_QUALITY_GATED_INCREMENTAL_RUN` status is not the final completion
status. The matching completed results contain 50,000 recorded judgments.

Use `dataset_versions.json` to authenticate external input files and the frozen
config before offline replay. It distinguishes the original and candidate-refresh
datasets and records the older subjective GT identity; repository GT v5 is not
a replacement for that snapshot. New candidate HTML and image rendering are
outside this replay release. A hash manifest must itself come from a trusted
repository revision; it is not an independent digital signature.

`release_manifest.json` records the original OLS5 import snapshot and its source
paths/hashes; it is not an inventory of the current tree after code cleanup.
`dataset_versions.json` still identifies the unchanged frozen config and prompt.
`reproduction_check.json` records the offline numeric replay
checks for this migration. Neither authorizes paid calls or changes to the
original results. No full-deployment human Accuracy/Recall/F1 is claimed.
