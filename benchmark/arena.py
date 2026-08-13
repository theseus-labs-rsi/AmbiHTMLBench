from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Protocol

from .arena_schedule import build_balanced_schedule, choose_reserve_page, validate_schedule
from .rating import compute_batch_elo
from .render import image_size, render_html_jobs
from .subjective import (
    JudgeResult,
    RequestsJudgeClient,
    build_active_axes,
    canonicalize_result,
    estimate_request_bytes,
    load_quality_gt_page,
)
from .utils import logistic_expected


ARENA_METHOD_ID = "subjective_arena_v1"
DEFAULT_SAFE_REQUEST_BYTES = 28_000_000


class JudgeClient(Protocol):
    def judge(
        self,
        reference_image: str | Path,
        candidate_a: str | Path,
        candidate_b: str | Path,
        active_axes: list[dict[str, Any]],
    ) -> JudgeResult: ...


@dataclass(frozen=True)
class ArenaJudgment:
    job_id: str
    pair_set_id: str
    sample_id: str
    canonical_model_a: str
    canonical_model_b: str
    displayed_a: str
    displayed_b: str
    swapped: bool
    winner_displayed: str
    winner_canonical: str
    confidence: int
    reason_displayed: str
    axis_scores_displayed: dict[str, int]
    axis_scores_canonical: dict[str, int]
    active_axes: list[str]
    block_index: int = 0
    round_index: int = 0
    raw_content: str = ""


@dataclass(frozen=True)
class LegacyArenaDecision:
    sample_id: str
    model_a: str
    model_b: str
    winner: str
    confidence: int
    reason: str
    judge_model: str
    round_idx: int
    swapped: bool


# Backward-compatible name used by the original repository.
ArenaDecision = LegacyArenaDecision


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _resolve(base: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def load_records(records_json: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(records_json).read_text(encoding="utf-8-sig"))


def _normalize_page(page: dict[str, Any], base: Path) -> dict[str, Any]:
    p = dict(page)
    p["sample_id"] = str(p["sample_id"])
    ref_key = "reference_image" if "reference_image" in p else "reference"
    p["reference_image"] = str(_resolve(base, p[ref_key]))
    candidates = p.get("candidates", {})
    p["candidates"] = {str(m): str(_resolve(base, html)) for m, html in candidates.items()}

    if "active_axes" not in p:
        if p.get("gt_json") is not None:
            ref_w, ref_h = image_size(p["reference_image"])
            gt_page = load_quality_gt_page(_resolve(base, p["gt_json"]), p.get("gt_page_id", p["sample_id"]))
            p["active_axes"] = build_active_axes(gt_page, ref_w, ref_h)
        else:
            p["active_axes"] = []
    return p


def load_arena_manifest(path: str | Path) -> dict[str, Any]:
    """Load the public RC1 manifest and resolve all filesystem paths."""
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    base = manifest_path.parent

    models = [str(x) for x in raw["models"]]
    if len(set(models)) != len(models):
        raise ValueError("models must be unique")

    pages = [_normalize_page(p, base) for p in raw.get("pages", [])]
    reserve_pages = [_normalize_page(p, base) for p in raw.get("reserve_pages", [])]
    page_map = {p["sample_id"]: p for p in pages}
    reserve_map = {p["sample_id"]: p for p in reserve_pages}
    if len(page_map) != len(pages):
        raise ValueError("page sample_id values must be unique")
    if len(reserve_map) != len(reserve_pages):
        raise ValueError("reserve page sample_id values must be unique")
    overlap = set(page_map) & set(reserve_map)
    if overlap:
        raise ValueError(f"final and reserve page ids must be disjoint: {sorted(overlap)}")

    for page in pages + reserve_pages:
        missing_models = set(models) - set(page["candidates"])
        if missing_models:
            raise ValueError(f"page {page['sample_id']} missing candidates: {sorted(missing_models)}")

    pair_sets = raw.get("pair_sets")
    if pair_sets is None:
        repeats = int(raw.get("schedule", {}).get("repeats", 1))
        seed = int(raw.get("schedule", {}).get("seed", 42))
        pair_sets = build_balanced_schedule(models, [p["sample_id"] for p in pages], repeats=repeats, seed=seed)
    else:
        pair_sets = [dict(x) for x in pair_sets]

    validation = validate_schedule(
        pair_sets,
        models,
        expected_pair_occurrences=raw.get("schedule", {}).get("repeats"),
    )
    if not validation.valid:
        raise ValueError(f"pair schedule is not balanced: {validation}")

    return {
        "version": raw.get("version", "subjective_arena_manifest_v1"),
        "models": models,
        "pages": pages,
        "reserve_pages": reserve_pages,
        "pair_sets": pair_sets,
        "judge": raw.get("judge", {}),
        "schedule": raw.get("schedule", {}),
        "source_manifest": str(manifest_path),
    }


def _page_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(p["sample_id"]): p for p in manifest["pages"]}


def _render_page_candidates(
    page: dict[str, Any],
    models: list[str],
    render_root: Path,
    *,
    report_path: Path,
) -> list[dict[str, Any]]:
    ref_w, ref_h = image_size(page["reference_image"])
    jobs = []
    sid = page["sample_id"]
    for model in models:
        out = render_root / sid / f"{_safe_filename(model)}.png"
        if out.exists():
            continue
        jobs.append({
            "sample_id": sid,
            "model": model,
            "html_path": page["candidates"][model],
            "out_png": str(out),
            "viewport_width": ref_w,
            "viewport_height": ref_h,
            "full_page": True,
        })
    if not jobs:
        return []
    return render_html_jobs(jobs, report_path=report_path, fail_fast=False)


def _evaluate_page_transport(
    page: dict[str, Any],
    page_pair_sets: list[dict[str, Any]],
    render_root: Path,
    *,
    safe_request_bytes: int,
) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    for pair in page_pair_sets:
        a, b = str(pair["model_a"]), str(pair["model_b"])
        ap = render_root / page["sample_id"] / f"{_safe_filename(a)}.png"
        bp = render_root / page["sample_id"] / f"{_safe_filename(b)}.png"
        if not ap.exists() or not bp.exists():
            rows.append({
                "pair_set_id": pair["pair_set_id"],
                "safe": False,
                "reason": "missing_render",
            })
            continue
        estimated = estimate_request_bytes(
            page["reference_image"], ap, bp, page["active_axes"]
        )
        rows.append({
            "pair_set_id": pair["pair_set_id"],
            "estimated_request_bytes": estimated,
            "estimated_request_mib": estimated / (1024 * 1024),
            "safe": estimated <= safe_request_bytes,
            "reason": "ok" if estimated <= safe_request_bytes else "request_too_large",
        })
    return all(r["safe"] for r in rows), rows


def preflight_arena(
    manifest_json: str | Path,
    out_dir: str | Path,
    *,
    safe_request_bytes: int = DEFAULT_SAFE_REQUEST_BYTES,
    auto_replace: bool = True,
) -> dict[str, Any]:
    """Render, validate request size, and resolve deterministic reserve replacements.

    No Judge API calls are made. The resolved manifest is the only manifest that
    should be used for a paid Arena run.
    """
    manifest = load_arena_manifest(manifest_json)
    out = Path(out_dir).resolve()
    render_root = out / "renders"
    report_path = out / "render_report.json"
    render_root.mkdir(parents=True, exist_ok=True)

    models = manifest["models"]
    pages = [dict(x) for x in manifest["pages"]]
    reserves = [dict(x) for x in manifest["reserve_pages"]]
    page_map = {p["sample_id"]: p for p in pages}
    pair_sets = [dict(x) for x in manifest["pair_sets"]]

    used_reserves: set[str] = set()
    replacements: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []

    # Each page participates in a perfect matching, so replacement happens for
    # all pair sets on that page while the model-pair schedule remains unchanged.
    for original_page in pages:
        sid = original_page["sample_id"]
        page_pairs = [x for x in pair_sets if str(x["sample_id"]) == sid]
        render_results = _render_page_candidates(original_page, models, render_root, report_path=report_path)
        render_ok = all(x.get("status") == "ok" for x in render_results) if render_results else all(
            (render_root / sid / f"{_safe_filename(m)}.png").exists() for m in models
        )
        transport_ok, rows = _evaluate_page_transport(
            original_page, page_pairs, render_root, safe_request_bytes=safe_request_bytes
        )
        transport_rows.extend({"sample_id": sid, **r} for r in rows)

        if render_ok and transport_ok:
            continue
        if not auto_replace:
            raise RuntimeError(f"preflight failed for page {sid}; reserve replacement disabled")

        reserve = choose_reserve_page(original_page, reserves, used_ids=used_reserves)
        replacement_ok = False
        while reserve is not None:
            rsid = reserve["sample_id"]
            _render_page_candidates(reserve, models, render_root, report_path=report_path)
            replacement_pairs = [{**x, "sample_id": rsid} for x in page_pairs]
            replacement_ok, replacement_rows = _evaluate_page_transport(
                reserve, replacement_pairs, render_root, safe_request_bytes=safe_request_bytes
            )
            transport_rows.extend({"sample_id": rsid, **r} for r in replacement_rows)
            if replacement_ok:
                used_reserves.add(rsid)
                for x in pair_sets:
                    if str(x["sample_id"]) == sid:
                        x["sample_id"] = rsid
                page_map.pop(sid, None)
                page_map[rsid] = reserve
                replacements.append({
                    "original_sample_id": sid,
                    "replacement_sample_id": rsid,
                    "reason": "render_or_request_preflight_failure",
                    "original_axes": [x["id"] for x in original_page.get("active_axes", [])],
                    "replacement_axes": [x["id"] for x in reserve.get("active_axes", [])],
                })
                break
            used_reserves.add(rsid)
            reserve = choose_reserve_page(original_page, reserves, used_ids=used_reserves)
        if not replacement_ok:
            raise RuntimeError(f"no valid reserve page found for {sid}")

    resolved_pages = list(page_map.values())
    validation = validate_schedule(pair_sets, models)
    if not validation.valid:
        raise RuntimeError(f"resolved schedule invalid: {validation}")

    resolved_pair_sets = []
    for pair in pair_sets:
        page = page_map[str(pair["sample_id"])]
        resolved_pair_sets.append({
            **pair,
            "reference_image": page["reference_image"],
            "active_axes": page["active_axes"],
            "rendered_a": str(render_root / page["sample_id"] / f"{_safe_filename(pair['model_a'])}.png"),
            "rendered_b": str(render_root / page["sample_id"] / f"{_safe_filename(pair['model_b'])}.png"),
        })

    max_request = max((r.get("estimated_request_bytes", 0) for r in transport_rows), default=0)
    resolved = {
        "version": "subjective_arena_resolved_v1",
        "method_id": ARENA_METHOD_ID,
        "models": models,
        "pages": resolved_pages,
        "pair_sets": resolved_pair_sets,
        "judge": manifest["judge"],
        "source_manifest": manifest["source_manifest"],
        "preflight": {
            "status": "PASS",
            "safe_request_bytes": safe_request_bytes,
            "largest_estimated_request_bytes": max_request,
            "largest_estimated_request_mib": max_request / (1024 * 1024),
            "replacements": replacements,
            "schedule_validation": asdict_schedule(validation),
        },
    }
    _atomic_json(out / "resolved_manifest.json", resolved)
    _atomic_json(out / "preflight_summary.json", resolved["preflight"])
    _atomic_json(out / "request_size_report.json", transport_rows)
    return resolved


def asdict_schedule(validation: Any) -> dict[str, Any]:
    return {
        "valid": validation.valid,
        "pages": validation.pages,
        "physical_pair_sets": validation.physical_pair_sets,
        "unique_unordered_pairs": validation.unique_unordered_pairs,
        "pair_occurrence_min": validation.pair_occurrence_min,
        "pair_occurrence_max": validation.pair_occurrence_max,
        "model_comparison_counts": validation.model_comparison_counts,
        "each_model_once_per_page": validation.each_model_once_per_page,
    }


def _load_resolved_manifest(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if raw.get("preflight", {}).get("status") != "PASS":
        raise ValueError("resolved manifest is not preflight PASS")
    return raw


def run_arena_resolved(
    resolved_manifest_json: str | Path,
    decisions_out: str | Path,
    *,
    judge_client: JudgeClient,
    judge_model: str,
    concurrency: int = 4,
) -> list[ArenaJudgment]:
    """Run canonical+swapped judgments with atomic checkpointing."""
    from tqdm import tqdm

    manifest = _load_resolved_manifest(resolved_manifest_json)
    pair_sets = manifest["pair_sets"]
    jobs = [
        {"pair": pair, "swapped": swapped,
         "job_id": f'{pair["pair_set_id"]}::{"swapped" if swapped else "canonical"}'}
        for pair in pair_sets for swapped in (False, True)
    ]

    out = Path(decisions_out)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8-sig"))
        rows = old.get("decisions", [])
        errors = old.get("errors", [])
    done = {x["job_id"] for x in rows}
    lock = threading.Lock()

    def save() -> None:
        _atomic_json(out, {
            "created_or_updated_at": datetime.now(timezone.utc).isoformat(),
            "method_id": ARENA_METHOD_ID,
            "judge_model": judge_model,
            "temperature": 0,
            "expected_jobs": len(jobs),
            "completed_jobs": len(rows),
            "decisions": sorted(rows, key=lambda x: x["job_id"]),
            "errors": errors[-500:],
        })

    def worker(job: dict[str, Any]) -> dict[str, Any]:
        pair = job["pair"]
        swapped = job["swapped"]
        left, right = (
            (pair["model_b"], pair["model_a"]) if swapped
            else (pair["model_a"], pair["model_b"])
        )
        render_dir = Path(pair["rendered_a"]).parent
        left_img = render_dir / f"{_safe_filename(left)}.png"
        right_img = render_dir / f"{_safe_filename(right)}.png"
        result = judge_client.judge(
            pair["reference_image"], left_img, right_img, pair.get("active_axes", [])
        )
        canonical = canonicalize_result(result, swapped)
        return asdict(ArenaJudgment(
            job_id=job["job_id"],
            pair_set_id=pair["pair_set_id"],
            sample_id=str(pair["sample_id"]),
            canonical_model_a=str(pair["model_a"]),
            canonical_model_b=str(pair["model_b"]),
            displayed_a=str(left),
            displayed_b=str(right),
            swapped=swapped,
            winner_displayed=result.winner,
            winner_canonical=canonical.winner,
            confidence=result.confidence,
            reason_displayed=result.reason,
            axis_scores_displayed=result.axis_scores,
            axis_scores_canonical=canonical.axis_scores,
            active_axes=[x["id"] for x in pair.get("active_axes", [])],
            block_index=int(pair.get("block_index", 0)),
            round_index=int(pair.get("round_index", 0)),
            raw_content=result.raw_content,
        ))

    pending = [j for j in jobs if j["job_id"] not in done]
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(worker, job): job for job in pending}
            with tqdm(total=len(pending), desc="subjective-arena") as bar:
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        row = fut.result()
                        with lock:
                            rows.append(row)
                            done.add(row["job_id"])
                            save()
                    except Exception as exc:
                        with lock:
                            errors.append({
                                "time": datetime.now(timezone.utc).isoformat(),
                                "job_id": job["job_id"],
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                            save()
                    finally:
                        bar.update(1)
    save()

    missing = sorted({j["job_id"] for j in jobs} - {r["job_id"] for r in rows})
    if missing:
        raise RuntimeError(f"Arena incomplete: {len(rows)}/{len(jobs)}; pending={missing[:20]}")
    return [ArenaJudgment(**x) for x in sorted(rows, key=lambda x: x["job_id"])]


def run_arena_from_env(
    resolved_manifest_json: str | Path,
    decisions_out: str | Path,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    judge_model: str | None = None,
    concurrency: int = 4,
) -> list[ArenaJudgment]:
    manifest = _load_resolved_manifest(resolved_manifest_json)
    config = manifest.get("judge", {})
    api_base = api_base or os.getenv("VLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = api_key or os.getenv("VLM_API_KEY")
    judge_model = judge_model or os.getenv("VLM_JUDGE_MODEL") or config.get("model", "claude-sonnet-4-6")
    if not api_base:
        raise ValueError("missing API base: pass api_base or set VLM_API_BASE / OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("missing API key: pass api_key or set VLM_API_KEY")
    client = RequestsJudgeClient(api_base, api_key, judge_model)
    return run_arena_resolved(
        resolved_manifest_json,
        decisions_out,
        judge_client=client,
        judge_model=judge_model,
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# Compatibility layer for the original repository API.
# ---------------------------------------------------------------------------

def load_decisions_file(path: str | Path) -> list[LegacyArenaDecision]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [LegacyArenaDecision(**x) for x in raw.get("decisions", [])]


def compute_online_elo_legacy(
    decisions: list[LegacyArenaDecision],
    k_factor: float = 24.0,
    init_rating: float = 1000.0,
) -> dict[str, float]:
    """Deprecated sequential Elo retained only for backward compatibility."""
    ratings: dict[str, float] = {}
    for d in decisions:
        ratings.setdefault(d.model_a, init_rating)
        ratings.setdefault(d.model_b, init_rating)
        ra, rb = ratings[d.model_a], ratings[d.model_b]
        ea = logistic_expected(ra, rb)
        sa = 1.0 if d.winner == "A" else 0.0 if d.winner == "B" else 0.5
        weight = 0.5 + 0.125 * d.confidence
        k = k_factor * weight
        ratings[d.model_a] = ra + k * (sa - ea)
        ratings[d.model_b] = rb + k * ((1.0 - sa) - (1.0 - ea))
    return dict(sorted(ratings.items(), key=lambda kv: kv[1], reverse=True))


def compute_elo(decisions: list[LegacyArenaDecision], k_factor: float = 24.0, init_rating: float = 1000.0) -> dict[str, float]:
    """Deprecated alias for the original online Elo implementation."""
    return compute_online_elo_legacy(decisions, k_factor=k_factor, init_rating=init_rating)


def run_arena(*args, **kwargs):
    raise RuntimeError(
        "The original single-pass run_arena() is deprecated in RC1. "
        "Use preflight_arena() followed by run_arena_resolved(), or the CLI "
        "commands arena-preflight / arena-run."
    )
