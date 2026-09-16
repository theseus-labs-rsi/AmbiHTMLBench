"""Offline implementation of the frozen overall-only five-judge proxy.

This module ports the published experiment's arithmetic, not its API runner.
Each judge supplies one *canonicalized overall* A/B/TIE vote. Axis scores,
confidence, repeat means and weight normalization are deliberately absent.
The no-intercept proxy is not complementary under canonical A/B recoding.

Bradley--Terry uses the original ridge-regularized Newton solver. Bootstrap
samples entire pages with replacement, holding predictions and coefficients
fixed. Its intervals do not include proxy refitting or other systematic error.
Expected cross-runtime comparison tolerances are 1e-12 for proxy predictions
and 1e-8 for BT scores; these are numerical checks, not statistical bounds.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


JUDGES = (
    "anthropic_opus5", "anthropic_sonnet5", "gemini31pro", "openai_sol", "openai_terra",
)
ENCODING = {"A": 1.0, "B": 0.0, "TIE": 0.5}
ELO_SCALE = 400.0 / math.log(10.0)
RIDGE = 1e-3
MAX_ITERATIONS = 80
DEFAULT_SEED = 20260831


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Proxy configuration must be an object")
    coefficients = config.get("coefficients")
    if not isinstance(coefficients, dict) or set(coefficients) != set(JUDGES):
        raise ValueError("Exactly the five frozen judge coefficient keys are required")
    for judge in JUDGES:
        _number(coefficients[judge], f"Coefficient for {judge}")
    if _number(config.get("intercept"), "intercept") != 0.0:
        raise ValueError("The frozen proxy has no intercept")
    if config.get("normalize_weights") is not False:
        raise ValueError("normalize_weights must be false; do not renormalize coefficients")
    encoding = config.get("encoding")
    if not isinstance(encoding, dict) or set(encoding) != set(ENCODING):
        raise ValueError("Encoding must contain exactly A, B and TIE")
    if any(_number(encoding[key], f"Encoding {key}") != value
           for key, value in ENCODING.items()):
        raise ValueError("Frozen encoding is A=1, B=0, TIE=0.5")


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def load_proxy_config(path: str | Path) -> dict:
    """Read and validate a frozen configuration without changing its values.

    Coefficients are read from the versioned file, not fitted or normalized.
    File identity and release hashes belong to the calling provenance layer.
    """
    try:
        with Path(path).open(encoding="utf-8") as handle:
            config = json.load(handle, object_pairs_hook=_unique_json_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read proxy configuration: {path}") from exc
    _validate_config(config)
    return config


def score_votes(votes: dict[str, str], config: dict) -> dict:
    """Score exactly one canonical overall vote per judge, in frozen order."""
    _validate_config(config)
    if not isinstance(votes, dict) or set(votes) != set(JUDGES):
        raise ValueError("Exactly five single canonical overall votes are required")
    if any(not isinstance(value, str) or value not in ENCODING for value in votes.values()):
        raise ValueError("Each overall vote must be A, B or TIE")
    features = {judge: ENCODING[votes[judge]] for judge in JUDGES}
    contributions = {judge: config["coefficients"][judge] * features[judge] for judge in JUDGES}
    raw = sum(contributions.values())
    _number(raw, "Raw proxy prediction")
    return {
        "features": features,
        "contributions": contributions,
        "raw": raw,
        "p_A": min(1.0, max(0.0, raw)),
    }


def _connected(records, models):
    neighbors = {model: set() for model in models}
    for a, b, _p, _weight in records:
        neighbors[a].add(b)
        neighbors[b].add(a)
    reached = {models[0]}
    pending = [models[0]]
    while pending:
        for neighbor in neighbors[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return len(reached) == len(models)


def _solve_linear(matrix, vector):
    # Exact elimination order from scripts/build_dual_leaderboard.py.
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Singular Bradley-Terry information matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(n)]


def _bt_fit(records, models):
    """Historical ridge fit; bootstrap intentionally permits disconnected draws."""
    index = {model: i for i, model in enumerate(models)}
    aggregate = defaultdict(lambda: [0.0, 0.0])
    for model_a, model_b, y, weight in records:
        i, j = index[model_a], index[model_b]
        if i > j:
            i, j, y = j, i, 1.0 - y
        aggregate[(i, j)][0] += weight * y
        aggregate[(i, j)][1] += weight
    rows = [(i, j, wins_i, total) for (i, j), (wins_i, total) in aggregate.items() if total > 0]
    if any(not math.isfinite(wins) or not math.isfinite(total) for _, _, wins, total in rows):
        raise ValueError("Non-finite Bradley-Terry aggregate")
    scores = [0.0] * len(models)
    for _ in range(MAX_ITERATIONS):
        gradient = [-RIDGE * score for score in scores]
        information = [[0.0] * len(models) for _ in models]
        for i in range(len(models)):
            information[i][i] = RIDGE
        for i, j, wins_i, total in rows:
            delta = max(-30.0, min(30.0, scores[i] - scores[j]))
            probability = 1.0 / (1.0 + math.exp(-delta))
            residual = wins_i - total * probability
            curvature = total * probability * (1.0 - probability)
            gradient[i] += residual
            gradient[j] -= residual
            information[i][i] += curvature
            information[j][j] += curvature
            information[i][j] -= curvature
            information[j][i] -= curvature
        step = _solve_linear(information, gradient)
        maximum = max(abs(value) for value in step)
        if maximum > 2.0:
            step = [value * 2.0 / maximum for value in step]
        scores = [score + change for score, change in zip(scores, step)]
        center = sum(scores) / len(scores)
        scores = [score - center for score in scores]
        if max(abs(value) for value in step) < 1e-9:
            break
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Non-finite Bradley-Terry result")
    return {model: score * ELO_SCALE for model, score in zip(models, scores)}


def bt_fit(records: list[tuple[str, str, float, float]], models: list[str]) -> dict:
    """Fit connected soft-preference comparisons on the frozen BT scale.

    Input tuple: (model A, model B, p(A), positive weight). Models must be
    unique; observations may repeat the same pair. No sequential Elo updates.
    """
    if not isinstance(models, (list, tuple)) or len(models) < 2:
        raise ValueError("Bradley-Terry requires at least two models")
    for model in models:
        _name(model, "Model")
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model names")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Bradley-Terry requires nonempty records")
    checked = []
    for record in records:
        if not isinstance(record, (tuple, list)) or len(record) != 4:
            raise ValueError("Each BT record must contain A, B, p(A) and weight")
        a, b, probability, weight = record
        if not isinstance(a, str) or not isinstance(b, str) or a not in models or b not in models or a == b:
            raise ValueError("BT records must name two distinct known models")
        probability = _number(probability, "p(A)")
        weight = _number(weight, "Comparison weight")
        if not 0.0 <= probability <= 1.0 or weight <= 0.0:
            raise ValueError("BT requires p(A) in [0, 1] and weight > 0")
        checked.append((a, b, probability, weight))
    if not _connected(checked, models):
        raise ValueError("Comparison graph is disconnected")
    return _bt_fit(checked, models)


def _ranks(scores):
    ordered = sorted(scores, key=lambda model: (-scores[model], model))
    return {model: rank for rank, model in enumerate(ordered, 1)}


def _percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def build_leaderboard(cases: list[dict], bootstrap: int = 2000, seed: int = DEFAULT_SEED) -> dict:
    """Rank complete scored cases with the original whole-page bootstrap.

    A case requires task_id, page_id, canonical_pair={A, B}, and p_A.
    Sorting task IDs before accumulation is part of historical reproducibility.
    Smaller fixtures are allowed: this function does not assert a 10,000-case
    schedule or infer that the corresponding 50,000 judge jobs were complete.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap <= 0:
        raise ValueError("bootstrap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("Leaderboard requires nonempty cases")
    task_ids = set()
    checked = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        task_id = _name(case.get("task_id"), "task_id")
        page_id = _name(case.get("page_id"), "page_id")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        task_ids.add(task_id)
        pair = case.get("canonical_pair")
        if not isinstance(pair, dict) or set(pair) != {"A", "B"}:
            raise ValueError("canonical_pair must contain exactly A and B")
        a, b = _name(pair["A"], "Model A"), _name(pair["B"], "Model B")
        if a == b:
            raise ValueError("A case must compare two distinct models")
        probability = _number(case.get("p_A"), "p_A")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("p_A must be in [0, 1]")
        checked.append((task_id, page_id, a, b, probability))

    records, blocks = [], defaultdict(list)
    points, appearances = Counter(), Counter()
    for _task_id, page_id, a, b, probability in sorted(checked):
        record = (a, b, probability, 1.0)
        records.append(record)
        blocks[page_id].append(record)
        points[a] += probability
        points[b] += 1 - probability
        appearances.update((a, b))
    models = sorted(appearances)
    scores = bt_fit(records, models)
    rank = _ranks(scores)
    samples = {model: [] for model in models}
    rank_samples = {model: [] for model in models}
    compressed = {}
    for page, rows in blocks.items():
        aggregate = defaultdict(lambda: [0.0, 0.0])
        for a, b, probability, _weight in rows:
            aggregate[a, b][0] += probability
            aggregate[a, b][1] += 1
        compressed[page] = aggregate

    rng = random.Random(seed)
    pages = sorted(blocks)
    disconnected_draws = 0
    for _ in range(bootstrap):
        aggregate = defaultdict(lambda: [0.0, 0.0])
        for page, multiplicity in Counter(rng.choices(pages, k=len(pages))).items():
            for pair, (wins, count) in compressed[page].items():
                aggregate[pair][0] += multiplicity * wins
                aggregate[pair][1] += multiplicity * count
        drawn = [(a, b, wins / count, count) for (a, b), (wins, count) in aggregate.items()]
        disconnected_draws += not _connected(drawn, models)
        # No rejection, retries or reseeding: even disconnected bootstrap draws
        # are fitted with the original ridge, including absent model scores.
        drawn_scores = _bt_fit(drawn, models)
        drawn_ranks = _ranks(drawn_scores)
        for model in models:
            samples[model].append(drawn_scores[model])
            rank_samples[model].append(drawn_ranks[model])

    board = []
    for model in sorted(models, key=lambda model: rank[model]):
        board.append({
            "rank": rank[model], "model": model, "bt_score": scores[model],
            "bt_ci95_low": _percentile(samples[model], .025),
            "bt_ci95_high": _percentile(samples[model], .975),
            "rank_ci95_low": _percentile(rank_samples[model], .025),
            "rank_ci95_high": _percentile(rank_samples[model], .975),
            "preference_point_rate": points[model] / appearances[model],
            "adjudicated_comparisons": appearances[model],
            "scheduled_comparisons": appearances[model],
            "top3_probability": sum(value <= 3 for value in rank_samples[model]) / bootstrap,
        })
    return {
        "leaderboard": board,
        "coverage": 1.0,
        "coverage_scope": "All supplied cases are scored; no claim about an external schedule or human labels.",
        "counts": {"cases": len(checked), "pages": len(pages), "models": len(models)},
        "bootstrap": {
            "n": bootstrap, "seed": seed,
            "successful_draws": bootstrap, "failed_draws": 0,
            "disconnected_draws": disconnected_draws,
            "unit": "page blocks; coefficients/predictions fixed",
            "sampling": "Uniform pages with replacement, drawing the number of distinct input pages",
            "interval": "95% percentile, linear interpolation at (n-1)*q",
            "rank_ties": "Exact equal scores are ordered lexicographically by model name (no average ranks)",
            "disconnected_policy": "Retain all ridge-fitted draws; do not discard or resample",
        },
        "method": {
            "ridge": RIDGE, "max_iterations": MAX_ITERATIONS,
            "newton_step_cap": 2.0, "step_tolerance": 1e-9,
            "logit_clip": [-30.0, 30.0], "score_center": "mean zero",
            "elo_scale": ELO_SCALE,
            "numerical_check_atol": {"proxy": 1e-12, "bt_score": 1e-8},
        },
    }
