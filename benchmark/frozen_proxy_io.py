"""Read-only import and validation of the completed subjective experiments.

This module never opens an image, changes a checkpoint, or calls a provider.
Without an artifact manifest, structural validity is not proof of provenance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .frozen_proxy import JUDGES, score_votes

ENCODING = {"A": 1.0, "B": 0.0, "TIE": 0.5}
BASE_AXES = ("GLOBAL_LAYOUT_FIDELITY", "TEXT_VISUAL_PRESENTATION")
ADAPTIVE_AXES = (
    "TABLE_VISUAL_STRUCTURE", "FORMULA_2D_LAYOUT", "ALGORITHM_BLOCK_STRUCTURE",
    "VERTICAL_TEXT_PRESENTATION", "LIST_HIERARCHY_PRESENTATION",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_keys)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON at line {number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            yield row


def verify_artifact(
    path: str | Path, manifest_path: str | Path | None,
    dataset: str | None, role: str = "results",
) -> dict[str, Any]:
    """Compare a supplied artifact with a separately published dataset digest."""
    digest = sha256_file(path)
    status = {"sha256": digest, "bytes": Path(path).stat().st_size,
              "provenance_verified": False}
    if manifest_path is None:
        if dataset is not None:
            raise ValueError("--dataset requires --artifact-manifest")
        return status
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Artifact manifest must contain a datasets object")
    if not dataset or dataset not in manifest.get("datasets", {}):
        raise ValueError("Select a dataset present in the artifact manifest")
    expected = manifest["datasets"][dataset].get(role)
    if not isinstance(expected, dict) or expected.get("sha256") != digest:
        raise ValueError(f"Artifact hash mismatch for {dataset}/{role}")
    if expected.get("bytes", status["bytes"]) != status["bytes"]:
        raise ValueError(f"Artifact size mismatch for {dataset}/{role}")
    return {**status, "provenance_verified": True, "dataset": dataset,
            "manifest_sha256": sha256_file(manifest_path)}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _pair(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"A", "B"}:
        raise ValueError("canonical_pair must contain exactly A and B")
    a, b = _text(value["A"], "model A"), _text(value["B"], "model B")
    if a >= b:
        raise ValueError("Keep the frozen, distinct, lexically ordered canonical model pair")
    return {"A": a, "B": b}


def validate_judgment_row(row: dict[str, Any]) -> tuple[str, str, dict, str]:
    """Verify display/canonical mapping before accepting any overall feature."""
    try:
        job = _text(row["job_id"], "job_id")
        judge = row["judge_id"]
        task, result = row["task"], row["result"]
        if judge not in JUDGES or not isinstance(task, dict) or not isinstance(result, dict):
            raise ValueError("Unknown judge or invalid task/result object")
        _text(task["task_id"], "task_id")
        _text(task["page_id"], "page_id")
        pair = _pair(task["canonical_pair"])
        swapped = task["display_swapped"]
        if type(swapped) is not bool:
            raise ValueError("display_swapped must be a JSON boolean")
        expected_display = {"A": pair["B"], "B": pair["A"]} if swapped else pair
        if "display_pair" in task and task["display_pair"] != expected_display:
            raise ValueError("display_pair disagrees with canonical_pair and direction")
        winner = result["winner_display"]
        if winner not in ENCODING:
            raise ValueError("winner_display must be A/B/TIE")
        canonical = {"A": "B", "B": "A", "TIE": "TIE"}[winner] if swapped else winner
        if result["winner_canonical"] != canonical:
            raise ValueError("winner_canonical does not match the displayed vote")
        if "confidence" in result and (type(result["confidence"]) is not int
                                        or not 1 <= result["confidence"] <= 5):
            raise ValueError("confidence must be an integer in 1..5")
        if "active_axes" in task:
            ids = [a["id"] for a in task["active_axes"]]
            if ids[:2] != list(BASE_AXES) or len(ids) != len(set(ids)):
                raise ValueError("Active axes must start with the two fixed axes, without duplicates")
            if any(axis not in BASE_AXES + ADAPTIVE_AXES for axis in ids):
                raise ValueError("Unknown frozen diagnostic axis")
            displayed, canonical_axes = result["axes_display"], result["axes_canonical"]
            if [a["id"] for a in displayed] != ids or [a["id"] for a in canonical_axes] != ids:
                raise ValueError("Axis membership/order mismatch")
            for axis, normalized in zip(displayed, canonical_axes):
                value = axis["score"]
                if type(value) is not int or value not in range(-2, 3):
                    raise ValueError("Axis scores must be integers in -2..2")
                if type(normalized["score"]) is not int or normalized["score"] != (-value if swapped else value):
                    raise ValueError("Canonical axis sign mismatch")
        return job, judge, task, canonical
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed judgment row: missing or invalid required fields") from exc


def load_single_results(
    path: str | Path, config: dict[str, Any], expected_cases: int | None = 10000,
    strict_schedule: bool = True, artifact_manifest: str | Path | None = None,
    dataset: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance = verify_artifact(path, artifact_manifest, dataset)
    groups: dict[str, dict] = defaultdict(dict)
    metadata: dict[str, dict] = {}
    directions: dict[str, dict] = defaultdict(dict)
    jobs: set[str] = set()
    counts: Counter = Counter()
    for row in read_jsonl(path):
        job, judge, task, canonical = validate_judgment_row(row)
        tid = task["task_id"]
        if type(row.get("repeat_index")) is not int or row["repeat_index"] != 1:
            raise ValueError("Single-vote deployment requires repeat_index=1")
        if job in jobs or judge in groups[tid]:
            raise ValueError("Duplicate job or second judgment for the same case/judge")
        if strict_schedule and task.get("protocol") != "single-vote-v2":
            raise ValueError("Wrong deployment protocol; do not mix calibration or legacy runs")
        if strict_schedule and "active_axes" not in task:
            raise ValueError("Formal recorded judgments must retain their active axes")
        identity = {k: task.get(k) for k in ("page_id", "canonical_pair", "source_kind", "protocol")}
        if tid in metadata and metadata[tid] != identity:
            raise ValueError("Case metadata differs between judges")
        metadata[tid] = identity
        groups[tid][judge] = canonical
        directions[tid][judge] = task["display_swapped"]
        jobs.add(job)
        counts[judge] += 1
    if not groups or (expected_cases is not None and len(groups) != expected_cases):
        raise ValueError(f"Expected {expected_cases or 'nonzero'} complete physical comparisons")
    cases, physical_keys = [], set()
    appearances, pair_counts, sources = Counter(), Counter(), Counter()
    for tid, votes in sorted(groups.items()):
        scored = score_votes(votes, config)  # Missing judges cannot be averaged away.
        meta = metadata[tid]
        pair = meta["canonical_pair"]
        a, b = pair["A"], pair["B"]
        key = (meta["page_id"], a, b)
        if key in physical_keys:
            raise ValueError("Duplicate physical page/model comparison under another task_id")
        physical_keys.add(key)
        appearances.update((a, b))
        pair_counts[a, b] += 1
        sources[meta["source_kind"]] += 1
        cases.append({"task_id": tid, "page_id": meta["page_id"], "canonical_pair": pair,
                      "source_kind": meta["source_kind"], "single_votes": votes,
                      "display_swapped_by_judge": directions[tid],
                      "features": scored["features"], "weighted_terms": scored["contributions"],
                      "raw_prediction": scored["raw"], "p_A": scored["p_A"]})
    if strict_schedule:
        if len(cases) != 10000 or len(jobs) != 50000 or set(counts.values()) != {10000}:
            raise ValueError("Formal deployment must contain 10,000 cases and 50,000 unique votes")
        if len(appearances) != 10 or set(appearances.values()) != {2000}:
            raise ValueError("Formal deployment must have ten models with 2,000 appearances each")
        if len(pair_counts) != 45 or Counter(pair_counts.values()) != Counter({222: 35, 223: 10}):
            raise ValueError("Frozen pair quotas must be 35 x 222 and 10 x 223")
        if sources != Counter({"reuse": 1872, "new": 8128}):
            raise ValueError("Expected 1,872 retained-schedule and 8,128 new comparisons")
    audit = {"status": "VALID", "paper_scale": strict_schedule,
             "physical_comparisons": len(cases), "judgment_rows": len(jobs),
             "judges": dict(counts), "model_appearances": dict(appearances),
             "pair_quota_histogram": dict(Counter(pair_counts.values())),
             "page_count": len({c["page_id"] for c in cases}), "coverage": 1.0,
             "input_artifact": provenance,
             "limitations": ["Structural validation alone does not authenticate candidate images or GT.",
                             "Retained schedule membership is not fresh human validation of changed candidates."]}
    return cases, audit
