from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
from typing import Any

from .render import image_size, render_html_to_png
from .utils import logistic_expected


ARENA_JUDGE_PROMPT = """你是文档还原质量评审。
任务：比较候选A和候选B，判断谁更接近参考原图。
评审维度与权重：
1) 版面结构与几何布局(40%)：区块位置、尺寸比例、对齐、列结构、表格网格。
2) 文本内容正确性(35%)：文字是否正确、缺失/多出、错别字、符号数字。
3) 视觉样式(20%)：字体粗细、字号层级、颜色、边框、留白。
4) 其他关键元素(5%)：图形/公式/项目符号等是否保留。
判决规则：
- 仅比较与参考图的一致性，不比较“好看程度”。
- 若差距很小无法稳定区分，输出 TIE。
- 禁止输出解释性长文。
输出必须是严格JSON，且只输出JSON对象：
{"winner":"A|B|TIE","confidence":1-5,"reason":"<=40字"}
"""

JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class ArenaDecision:
    sample_id: str
    model_a: str
    model_b: str
    winner: str
    confidence: int
    reason: str
    judge_model: str
    round_idx: int
    swapped: bool



def _b64_data_uri(image_path: str | Path) -> str:
    p = Path(image_path)
    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    raw = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{raw}"



def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_OBJ_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))



def call_vlm_judge(
    api_base: str,
    api_key: str,
    judge_model: str,
    reference_image: str | Path,
    cand_a_image: str | Path,
    cand_b_image: str | Path,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    import requests

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": judge_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": ARENA_JUDGE_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "第一张是参考原图，第二张是候选A，第三张是候选B。请给出判决JSON。"},
                    {"type": "image_url", "image_url": {"url": _b64_data_uri(reference_image)}},
                    {"type": "image_url", "image_url": {"url": _b64_data_uri(cand_a_image)}},
                    {"type": "image_url", "image_url": {"url": _b64_data_uri(cand_b_image)}},
                ],
            },
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)

    winner = str(parsed.get("winner", "TIE")).upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    confidence = int(parsed.get("confidence", 3))
    confidence = max(1, min(5, confidence))
    reason = str(parsed.get("reason", ""))[:80]

    return {
        "winner": winner,
        "confidence": confidence,
        "reason": reason,
        "raw": data,
    }



def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)



def load_records(records_json: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(records_json).read_text(encoding="utf-8-sig"))



def render_all_from_records(
    records: list[dict[str, Any]],
    base_dir: str | Path,
    out_dir: str | Path,
    full_page: bool = True,
) -> dict[str, dict[str, str]]:
    from tqdm import tqdm

    base = Path(base_dir)
    out = Path(out_dir)
    rendered: dict[str, dict[str, str]] = {}

    grouped: dict[str, dict[str, str]] = {}
    ref_images: dict[str, str] = {}

    for r in records:
        if r.get("status") != "ok":
            continue
        sample_id = f"{int(r['image_index']):02d}"
        model = str(r["model_safe"])
        html_path = str((base / r["html_file"]).resolve())
        ref_img = str((base / r["image_asset"]).resolve())
        grouped.setdefault(sample_id, {})[model] = html_path
        ref_images[sample_id] = ref_img

    for sample_id, model_map in tqdm(grouped.items(), desc="render-html"):
        rendered[sample_id] = {}
        ref_w, ref_h = image_size(ref_images[sample_id])
        for model, html in model_map.items():
            out_png = out / sample_id / f"{_safe_filename(model)}.png"
            render_html_to_png(
                html_path=html,
                out_png=out_png,
                viewport_width=ref_w,
                viewport_height=ref_h,
                full_page=full_page,
            )
            rendered[sample_id][model] = str(out_png)

    return rendered



def run_arena(
    records_json: str | Path,
    base_dir: str | Path,
    render_out_dir: str | Path,
    decisions_out: str | Path,
    api_base: str,
    api_key: str,
    judge_model: str,
    rounds: int = 1,
    seed: int = 42,
) -> list[ArenaDecision]:
    from tqdm import tqdm

    rng = random.Random(seed)

    records = load_records(records_json)
    rendered = render_all_from_records(records, base_dir=base_dir, out_dir=render_out_dir)

    base = Path(base_dir)
    ref_by_sample: dict[str, str] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        sample_id = f"{int(r['image_index']):02d}"
        ref_by_sample[sample_id] = str((base / r["image_asset"]).resolve())

    decisions: list[ArenaDecision] = []

    for sample_id, model_map in tqdm(rendered.items(), desc="arena"):
        models = sorted(model_map.keys())
        if len(models) < 2:
            continue
        ref_img = ref_by_sample[sample_id]

        for model_a, model_b in combinations(models, 2):
            for round_idx in range(rounds):
                swapped = rng.random() < 0.5
                left_model, right_model = (model_b, model_a) if swapped else (model_a, model_b)
                left_img, right_img = model_map[left_model], model_map[right_model]

                judged = call_vlm_judge(
                    api_base=api_base,
                    api_key=api_key,
                    judge_model=judge_model,
                    reference_image=ref_img,
                    cand_a_image=left_img,
                    cand_b_image=right_img,
                )

                winner = judged["winner"]
                if swapped:
                    if winner == "A":
                        winner = "B"
                    elif winner == "B":
                        winner = "A"

                decisions.append(
                    ArenaDecision(
                        sample_id=sample_id,
                        model_a=model_a,
                        model_b=model_b,
                        winner=winner,
                        confidence=judged["confidence"],
                        reason=judged["reason"],
                        judge_model=judge_model,
                        round_idx=round_idx,
                        swapped=swapped,
                    )
                )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "api_base": api_base,
        "rounds": rounds,
        "decisions": [d.__dict__ for d in decisions],
    }
    out = Path(decisions_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return decisions



def _winner_scores(winner: str) -> tuple[float, float]:
    if winner == "A":
        return 1.0, 0.0
    if winner == "B":
        return 0.0, 1.0
    return 0.5, 0.5


def _confidence_weight(confidence: int) -> float:
    bounded = max(1, min(5, int(confidence)))
    return 0.5 + 0.125 * bounded


@dataclass(frozen=True)
class EloEvidence:
    sample_id: str
    model_a: str
    model_b: str
    score_a: float
    update_weight: float
    round_count: int
    consistency: float


def _decision_to_evidence(d: ArenaDecision) -> EloEvidence:
    score_a, _ = _winner_scores(d.winner)

    if d.model_a <= d.model_b:
        model_a = d.model_a
        model_b = d.model_b
        canonical_score_a = score_a
    else:
        model_a = d.model_b
        model_b = d.model_a
        canonical_score_a = 1.0 - score_a

    return EloEvidence(
        sample_id=d.sample_id,
        model_a=model_a,
        model_b=model_b,
        score_a=canonical_score_a,
        update_weight=_confidence_weight(d.confidence),
        round_count=1,
        consistency=1.0,
    )


def _aggregate_round_evidence(
    decisions: list[ArenaDecision],
    consistency_floor: float,
) -> list[EloEvidence]:
    if not 0.0 <= consistency_floor <= 1.0:
        raise ValueError("consistency_floor must be between 0 and 1")

    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}

    for d in decisions:
        evidence = _decision_to_evidence(d)
        key = (evidence.sample_id, evidence.model_a, evidence.model_b)
        grouped.setdefault(key, []).append(
            (evidence.score_a, evidence.update_weight)
        )

    aggregated: list[EloEvidence] = []

    for (sample_id, model_a, model_b), rounds in grouped.items():
        total_weight = sum(weight for _, weight in rounds)
        score_a = sum(
            score * weight
            for score, weight in rounds
        ) / total_weight

        variance = sum(
            weight * (score - score_a) ** 2
            for score, weight in rounds
        ) / total_weight

        consistency = max(0.0, min(1.0, 1.0 - 4.0 * variance))
        mean_confidence_weight = total_weight / len(rounds)

        round_bonus = (
            1.0
            + 0.5 * (1.0 - 1.0 / len(rounds))
        )

        reliability = (
            consistency_floor
            + (1.0 - consistency_floor) * consistency
        )

        aggregated.append(
            EloEvidence(
                sample_id=sample_id,
                model_a=model_a,
                model_b=model_b,
                score_a=score_a,
                update_weight=(
                    mean_confidence_weight
                    * round_bonus
                    * reliability
                ),
                round_count=len(rounds),
                consistency=consistency,
            )
        )

    return aggregated


def _prepare_elo_evidence(
    decisions: list[ArenaDecision],
    round_aggregation: str,
    consistency_floor: float,
) -> list[EloEvidence]:
    if round_aggregation == "none":
        return [_decision_to_evidence(d) for d in decisions]

    if round_aggregation == "sample_pair":
        return _aggregate_round_evidence(
            decisions=decisions,
            consistency_floor=consistency_floor,
        )

    raise ValueError(
        f"unknown round aggregation: {round_aggregation!r}; "
        "expected 'none' or 'sample_pair'"
    )


@dataclass(frozen=True)
class EloStats:
    elo: float
    elo_std: float
    mean_rank: float
    rank_stability: float
    top_rank_probability: float
    trial_count: int


def _apply_sample_balance(
    evidence: list[EloEvidence],
    sample_balance: str,
) -> list[EloEvidence]:
    if sample_balance == "none":
        return list(evidence)

    if sample_balance != "per_model_exposure":
        raise ValueError(
            f"unknown sample balance: {sample_balance!r}; "
            "expected 'none' or 'per_model_exposure'"
        )

    degree_by_sample: dict[str, dict[str, int]] = {}

    for item in evidence:
        degree = degree_by_sample.setdefault(item.sample_id, {})
        degree[item.model_a] = degree.get(item.model_a, 0) + 1
        degree[item.model_b] = degree.get(item.model_b, 0) + 1

    balanced: list[EloEvidence] = []

    for item in evidence:
        degree = degree_by_sample[item.sample_id]
        degree_a = degree[item.model_a]
        degree_b = degree[item.model_b]
        denominator = degree_a + degree_b

        scale = 1.0 if denominator <= 0 else 2.0 / denominator

        balanced.append(
            EloEvidence(
                sample_id=item.sample_id,
                model_a=item.model_a,
                model_b=item.model_b,
                score_a=item.score_a,
                update_weight=item.update_weight * scale,
                round_count=item.round_count,
                consistency=item.consistency,
            )
        )

    return balanced

def _compute_elo_once(
    evidence: list[EloEvidence],
    k_factor: float,
    init_rating: float,
) -> dict[str, float]:
    ratings: dict[str, float] = {}

    def ensure(name: str) -> None:
        if name not in ratings:
            ratings[name] = init_rating

    for item in evidence:
        ensure(item.model_a)
        ensure(item.model_b)

        ra = ratings[item.model_a]
        rb = ratings[item.model_b]
        ea = logistic_expected(ra, rb)
        eb = 1.0 - ea

        sa = item.score_a
        sb = 1.0 - sa
        k = k_factor * item.update_weight

        ratings[item.model_a] = ra + k * (sa - ea)
        ratings[item.model_b] = rb + k * (sb - eb)

    return ratings


def _evidence_order_key(item: EloEvidence) -> tuple[Any, ...]:
    return (
        item.sample_id,
        item.model_a,
        item.model_b,
        item.score_a,
        item.update_weight,
        item.round_count,
        item.consistency,
    )


def _run_elo_trials(
    evidence: list[EloEvidence],
    k_factor: float,
    init_rating: float,
    mode: str,
    shuffle_repeats: int,
    seed: int,
) -> list[dict[str, float]]:
    if mode == "online":
        return [
            _compute_elo_once(
                evidence=list(evidence),
                k_factor=k_factor,
                init_rating=init_rating,
            )
        ]

    if mode != "shuffle_ensemble":
        raise ValueError(
            f"unknown Elo mode: {mode!r}; "
            "expected 'online' or 'shuffle_ensemble'"
        )

    if shuffle_repeats <= 0:
        raise ValueError("shuffle_repeats must be greater than 0")

    canonical_evidence = sorted(evidence, key=_evidence_order_key)
    rng = random.Random(seed)
    trials: list[dict[str, float]] = []

    for _ in range(shuffle_repeats):
        shuffled = list(canonical_evidence)
        rng.shuffle(shuffled)

        trials.append(
            _compute_elo_once(
                evidence=shuffled,
                k_factor=k_factor,
                init_rating=init_rating,
            )
        )
        trials.append(
            _compute_elo_once(
                evidence=list(reversed(shuffled)),
                k_factor=k_factor,
                init_rating=init_rating,
            )
        )

    return trials


def _mean_trial_ratings(
    trials: list[dict[str, float]],
) -> dict[str, float]:
    if not trials:
        return {}

    rating_sum: dict[str, float] = {}

    for trial in trials:
        for model, rating in trial.items():
            rating_sum[model] = rating_sum.get(model, 0.0) + rating

    trial_count = len(trials)

    return {
        model: total / trial_count
        for model, total in rating_sum.items()
    }


def _rating_ranks(
    ratings: dict[str, float],
    tie_tolerance: float = 1e-12,
) -> dict[str, float]:
    ordered = sorted(
        ratings,
        key=lambda model: (-ratings[model], model),
    )
    ranks: dict[str, float] = {}
    index = 0

    while index < len(ordered):
        end = index + 1

        while (
            end < len(ordered)
            and abs(
                ratings[ordered[end]]
                - ratings[ordered[index]]
            ) <= tie_tolerance
        ):
            end += 1

        first_rank = index + 1
        last_rank = end
        average_rank = (first_rank + last_rank) / 2.0

        for position in range(index, end):
            ranks[ordered[position]] = average_rank

        index = end

    return ranks


def compute_elo(
    decisions: list[ArenaDecision],
    k_factor: float = 24.0,
    init_rating: float = 1000.0,
    mode: str = "shuffle_ensemble",
    shuffle_repeats: int = 64,
    seed: int = 42,
    round_aggregation: str = "sample_pair",
    consistency_floor: float = 0.25,
    sample_balance: str = "per_model_exposure",
) -> dict[str, float]:
    if not decisions:
        return {}

    if k_factor <= 0:
        raise ValueError("k_factor must be greater than 0")

    evidence = _prepare_elo_evidence(
        decisions=list(decisions),
        round_aggregation=round_aggregation,
        consistency_floor=consistency_floor,
    )

    evidence = _apply_sample_balance(
        evidence=evidence,
        sample_balance=sample_balance,
    )

    trials = _run_elo_trials(
        evidence=evidence,
        k_factor=k_factor,
        init_rating=init_rating,
        mode=mode,
        shuffle_repeats=shuffle_repeats,
        seed=seed,
    )
    ratings = _mean_trial_ratings(trials)

    return dict(
        sorted(
            ratings.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )


def compute_elo_stats(
    decisions: list[ArenaDecision],
    k_factor: float = 24.0,
    init_rating: float = 1000.0,
    mode: str = "shuffle_ensemble",
    shuffle_repeats: int = 64,
    seed: int = 42,
    round_aggregation: str = "sample_pair",
    consistency_floor: float = 0.25,
    sample_balance: str = "per_model_exposure",
) -> dict[str, EloStats]:
    if not decisions:
        return {}

    if k_factor <= 0:
        raise ValueError("k_factor must be greater than 0")

    evidence = _prepare_elo_evidence(
        decisions=list(decisions),
        round_aggregation=round_aggregation,
        consistency_floor=consistency_floor,
    )

    evidence = _apply_sample_balance(
        evidence=evidence,
        sample_balance=sample_balance,
    )

    trials = _run_elo_trials(
        evidence=evidence,
        k_factor=k_factor,
        init_rating=init_rating,
        mode=mode,
        shuffle_repeats=shuffle_repeats,
        seed=seed,
    )

    models = sorted(trials[0])
    ratings_by_model: dict[str, list[float]] = {
        model: []
        for model in models
    }
    ranks_by_model: dict[str, list[float]] = {
        model: []
        for model in models
    }
    top_count: dict[str, int] = {
        model: 0
        for model in models
    }

    for trial in trials:
        ranks = _rating_ranks(trial)
        highest_rating = max(trial.values())

        for model in models:
            ratings_by_model[model].append(trial[model])
            ranks_by_model[model].append(ranks[model])

            if abs(trial[model] - highest_rating) <= 1e-12:
                top_count[model] += 1

    stats: dict[str, EloStats] = {}
    trial_count = len(trials)

    for model in models:
        rating_values = ratings_by_model[model]
        rank_values = ranks_by_model[model]

        elo = sum(rating_values) / trial_count
        variance = sum(
            (rating - elo) ** 2
            for rating in rating_values
        ) / trial_count
        elo_std = variance ** 0.5
        mean_rank = sum(rank_values) / trial_count

        rank_counts: dict[float, int] = {}
        for rank in rank_values:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        modal_rank_count = max(rank_counts.values())
        rank_stability = modal_rank_count / trial_count
        top_rank_probability = top_count[model] / trial_count

        stats[model] = EloStats(
            elo=elo,
            elo_std=elo_std,
            mean_rank=mean_rank,
            rank_stability=rank_stability,
            top_rank_probability=top_rank_probability,
            trial_count=trial_count,
        )

    return dict(
        sorted(
            stats.items(),
            key=lambda kv: kv[1].elo,
            reverse=True,
        )
    )


def load_decisions_file(path: str | Path) -> list[ArenaDecision]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    out: list[ArenaDecision] = []
    for d in raw.get("decisions", []):
        out.append(ArenaDecision(**d))
    return out
