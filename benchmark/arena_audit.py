from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import json
import math
import random
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .rating import compute_batch_elo
from .subjective import PairOutcome, collapse_dual_orientation, to_rating_decision


@dataclass(frozen=True)
class ArenaAnalysisSummary:
    raw_judgments: int
    physical_pair_sets: int
    orientation_agreement: float
    orientation_disagreements: int
    reliability_ties: int
    perceptual_ties: int
    collapsed_tie_rate: float
    displayed_a_win_rate: float
    displayed_b_win_rate: float
    raw_tie_rate: float
    mean_confidence: float
    primary_method: str


def _winner_score(winner: str) -> float:
    return 1.0 if winner == "A" else 0.0 if winner == "B" else 0.5


def collapse_decisions(
    decisions: Iterable[dict[str, Any]],
    pair_manifest: dict[str, dict[str, Any]] | None = None,
) -> list[PairOutcome]:
    by = defaultdict(dict)
    for row in decisions:
        by[str(row["pair_set_id"])][bool(row["swapped"])] = row

    out: list[PairOutcome] = []
    for pair_set_id in sorted(by):
        rr = by[pair_set_id]
        if set(rr) != {False, True}:
            raise ValueError(f"pair_set {pair_set_id} is not dual-orientation complete")
        can = rr[False]
        swp = rr[True]
        model_a = str(can.get("canonical_model_a") or can.get("model_a"))
        model_b = str(can.get("canonical_model_b") or can.get("model_b"))
        if pair_manifest and pair_set_id in pair_manifest:
            model_a = str(pair_manifest[pair_set_id]["model_a"])
            model_b = str(pair_manifest[pair_set_id]["model_b"])
        out.append(collapse_dual_orientation(
            pair_set_id=pair_set_id,
            model_a=model_a,
            model_b=model_b,
            canonical_winner=str(can["winner_canonical"]),
            swapped_winner=str(swp["winner_canonical"]),
        ))
    return out


def primary_rating(pair_outcomes: Iterable[PairOutcome]):
    decisions = [to_rating_decision(x) for x in pair_outcomes]
    return compute_batch_elo(decisions, confidence_weighted=False)


def raw_rating(decisions: Iterable[dict[str, Any]], *, confidence_weighted: bool):
    rows = [
        SimpleNamespace(
            model_a=str(x.get("canonical_model_a") or x["model_a"]),
            model_b=str(x.get("canonical_model_b") or x["model_b"]),
            winner=str(x["winner_canonical"]),
            confidence=int(x.get("confidence", 3)),
        )
        for x in decisions
    ]
    return compute_batch_elo(rows, confidence_weighted=confidence_weighted)


def summarize_reliability(decisions: list[dict[str, Any]], outcomes: list[PairOutcome]) -> ArenaAnalysisSummary:
    if not decisions:
        raise ValueError("no decisions")
    disagreements = sum(x.tie_type == "reliability" for x in outcomes)
    perceptual_ties = sum(x.tie_type == "perceptual" for x in outcomes)
    displayed_a = sum(x.get("winner_displayed") == "A" for x in decisions)
    displayed_b = sum(x.get("winner_displayed") == "B" for x in decisions)
    raw_ties = sum(x.get("winner_displayed") == "TIE" for x in decisions)
    mean_conf = statistics.fmean(int(x.get("confidence", 3)) for x in decisions)
    total_pairs = len(outcomes)
    return ArenaAnalysisSummary(
        raw_judgments=len(decisions),
        physical_pair_sets=total_pairs,
        orientation_agreement=(total_pairs - disagreements) / total_pairs,
        orientation_disagreements=disagreements,
        reliability_ties=disagreements,
        perceptual_ties=perceptual_ties,
        collapsed_tie_rate=(disagreements + perceptual_ties) / total_pairs,
        displayed_a_win_rate=displayed_a / len(decisions),
        displayed_b_win_rate=displayed_b / len(decisions),
        raw_tie_rate=raw_ties / len(decisions),
        mean_confidence=mean_conf,
        primary_method="pair_consensus_tie + unweighted batch Bradley-Terry",
    )


def model_wtl(outcomes: Iterable[PairOutcome]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = defaultdict(lambda: {
        "wins": 0, "ties": 0, "losses": 0, "games": 0, "score": 0.0,
        "reliability_ties": 0,
    })
    for x in outcomes:
        a, b = x.model_a, x.model_b
        stats[a]["games"] += 1
        stats[b]["games"] += 1
        if x.tie_type == "reliability":
            stats[a]["reliability_ties"] += 1
            stats[b]["reliability_ties"] += 1
        if x.winner == "A":
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
            stats[a]["score"] += 1.0
        elif x.winner == "B":
            stats[b]["wins"] += 1
            stats[a]["losses"] += 1
            stats[b]["score"] += 1.0
        else:
            stats[a]["ties"] += 1
            stats[b]["ties"] += 1
            stats[a]["score"] += 0.5
            stats[b]["score"] += 0.5
    for row in stats.values():
        row["score_rate"] = float(row["score"]) / int(row["games"])
        row["reliability_tie_rate"] = int(row["reliability_ties"]) / int(row["games"])
    return dict(stats)


def model_pair_stats(outcomes: Iterable[PairOutcome]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[PairOutcome]] = defaultdict(list)
    for x in outcomes:
        grouped[tuple(sorted((x.model_a, x.model_b)))].append(x)
    rows = []
    for (m1, m2), items in sorted(grouped.items()):
        w1 = w2 = ties = rel_ties = 0
        score = 0.0
        for x in items:
            if x.model_a == m1:
                s = _winner_score(x.winner)
            else:
                s = 1.0 - _winner_score(x.winner)
            score += s
            if s == 1.0:
                w1 += 1
            elif s == 0.0:
                w2 += 1
            else:
                ties += 1
            rel_ties += int(x.tie_type == "reliability")
        rows.append({
            "model_1": m1,
            "model_2": m2,
            "comparisons": len(items),
            "model_1_wins": w1,
            "ties": ties,
            "model_2_wins": w2,
            "model_1_score_rate": score / len(items),
            "reliability_ties": rel_ties,
        })
    return rows


def axis_reliability(
    decisions: list[dict[str, Any]],
    pair_manifest: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for r in decisions:
        by[str(r["pair_set_id"])][bool(r["swapped"])] = r
    acc = defaultdict(lambda: {"pair_sets": 0, "agree": 0, "reliability_ties": 0})
    for pair_id, rr in by.items():
        if set(rr) != {False, True}:
            continue
        axes = [x["id"] for x in pair_manifest[pair_id].get("active_axes", [])] or ["NO_ACTIVE_AXIS"]
        agree = rr[False]["winner_canonical"] == rr[True]["winner_canonical"]
        for aid in axes:
            acc[aid]["pair_sets"] += 1
            acc[aid]["agree"] += int(agree)
            acc[aid]["reliability_ties"] += int(not agree)
    return [{
        "axis_id": aid,
        "pair_sets": st["pair_sets"],
        "orientation_agreement": st["agree"] / st["pair_sets"],
        "reliability_tie_rate": st["reliability_ties"] / st["pair_sets"],
    } for aid, st in sorted(acc.items())]


def _quantile(vals: list[float], q: float) -> float:
    xs = sorted(vals)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def bootstrap_complete_blocks(
    outcomes: list[PairOutcome],
    pair_manifest: dict[str, dict[str, Any]],
    *,
    replicates: int = 1000,
    seed: int = 20260812,
) -> list[dict[str, Any]]:
    """Bootstrap complete round-robin blocks so pair balance is retained per draw."""
    by_block: dict[int, list[PairOutcome]] = defaultdict(list)
    for x in outcomes:
        block = int(pair_manifest[x.pair_set_id].get("block_index", 0))
        by_block[block].append(x)
    blocks = sorted(by_block)
    if len(blocks) < 2:
        return []

    models = sorted({m for x in outcomes for m in (x.model_a, x.model_b)})
    rng = random.Random(seed)
    elo_samples = {m: [] for m in models}
    rank_samples = {m: [] for m in models}

    success = 0
    attempts = 0
    max_attempts = max(replicates * 4, replicates + 10)
    while success < replicates and attempts < max_attempts:
        attempts += 1
        sampled: list[PairOutcome] = []
        for _ in blocks:
            sampled.extend(by_block[rng.choice(blocks)])
        try:
            fit = primary_rating(sampled)
        except (ValueError, RuntimeError):
            continue
        ranks = {m: i + 1 for i, m in enumerate(fit.ratings)}
        for m in models:
            elo_samples[m].append(float(fit.ratings[m]))
            rank_samples[m].append(ranks[m])
        success += 1

    if success == 0:
        raise RuntimeError("bootstrap produced zero valid replicates")

    rows = []
    for m in models:
        es, rs = elo_samples[m], rank_samples[m]
        rows.append({
            "model": m,
            "replicates": success,
            "elo_p2_5": _quantile(es, 0.025),
            "elo_median": _quantile(es, 0.5),
            "elo_p97_5": _quantile(es, 0.975),
            "mean_rank": statistics.fmean(rs),
            "rank_median": _quantile(rs, 0.5),
            "top1_probability": sum(x == 1 for x in rs) / success,
            "top3_probability": sum(x <= 3 for x in rs) / success,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_report(
    decisions_json: str | Path,
    resolved_manifest_json: str | Path,
    out_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260812,
) -> dict[str, Any]:
    raw = json.loads(Path(decisions_json).read_text(encoding="utf-8-sig"))
    manifest = json.loads(Path(resolved_manifest_json).read_text(encoding="utf-8-sig"))
    decisions = raw.get("decisions", raw if isinstance(raw, list) else [])
    pair_manifest = {str(x["pair_set_id"]): x for x in manifest["pair_sets"]}

    outcomes = collapse_decisions(decisions, pair_manifest)
    summary = summarize_reliability(decisions, outcomes)
    primary = primary_rating(outcomes)
    raw_u = raw_rating(decisions, confidence_weighted=False)
    raw_w = raw_rating(decisions, confidence_weighted=True)
    wtl = model_wtl(outcomes)
    bootstrap = bootstrap_complete_blocks(
        outcomes, pair_manifest, replicates=bootstrap_replicates, seed=bootstrap_seed
    ) if bootstrap_replicates > 0 else []
    boot = {x["model"]: x for x in bootstrap}

    ranking = []
    primary_ranks = {m: i + 1 for i, m in enumerate(primary.ratings)}
    raw_u_ranks = {m: i + 1 for i, m in enumerate(raw_u.ratings)}
    raw_w_ranks = {m: i + 1 for i, m in enumerate(raw_w.ratings)}
    for model, elo in primary.ratings.items():
        row = {
            "rank": primary_ranks[model],
            "model": model,
            "elo": float(elo),
            **wtl[model],
            "raw_dual_unweighted_rank": raw_u_ranks[model],
            "raw_dual_weighted_rank": raw_w_ranks[model],
        }
        if model in boot:
            row.update({
                "elo_95_low": boot[model]["elo_p2_5"],
                "elo_95_high": boot[model]["elo_p97_5"],
                "top1_probability": boot[model]["top1_probability"],
                "top3_probability": boot[model]["top3_probability"],
            })
        ranking.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "pair_outcomes.csv", [asdict(x) for x in outcomes])
    _write_csv(out / "subjective_ranking.csv", ranking)
    _write_csv(out / "model_pair_stats.csv", model_pair_stats(outcomes))
    _write_csv(out / "axis_reliability.csv", axis_reliability(decisions, pair_manifest))
    _write_csv(out / "bootstrap_model_summary.csv", bootstrap)

    payload = {
        "summary": asdict(summary),
        "primary_rating": primary.__dict__,
        "raw_dual_unweighted_rating": raw_u.__dict__,
        "raw_dual_weighted_rating": raw_w.__dict__,
        "ranking": ranking,
    }
    (out / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Subjective Arena Report",
        "",
        f"- Raw judgments: **{summary.raw_judgments}**",
        f"- Physical pair sets: **{summary.physical_pair_sets}**",
        f"- Orientation agreement: **{summary.orientation_agreement:.3f}**",
        f"- Reliability TIEs: **{summary.reliability_ties}**",
        f"- Perceptual TIEs: **{summary.perceptual_ties}**",
        f"- Displayed-A win rate: **{summary.displayed_a_win_rate:.3f}**",
        f"- Displayed-B win rate: **{summary.displayed_b_win_rate:.3f}**",
        "",
        "## Primary leaderboard",
        "",
        "| Rank | Model | Elo | W-T-L | Score |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['elo']:.2f} | "
            f"{row['wins']}-{row['ties']}-{row['losses']} | {row['score_rate']:.3f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
