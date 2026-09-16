"""Offline reproduction of the original five-feature OLS and eight-block OOF.

This is a diagnostic reproduction, never an update to the deployed coefficients.
Published bootstrap intervals remain in the versioned evidence package.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from .frozen_proxy import JUDGES, bt_fit
from .frozen_proxy_io import ENCODING, _pair, read_jsonl, validate_judgment_row


def _solve(matrix, vector):
    # Same pivoted normal-equation solver as the frozen OLS script; no ridge.
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Singular OLS design; do not silently substitute ridge")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [x / scale for x in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [augmented[i][-1] for i in range(n)]


def fit_ols(ids, features, human):
    if not ids:
        raise ValueError("Empty OLS training set")
    n = len(ids)
    gram = [[sum(features[t][i] * features[t][j] for t in ids) / n for j in range(5)]
            for i in range(5)]
    xy = [sum(features[t][i] * human[t]["human_p_a"] for t in ids) / n for i in range(5)]
    beta = _solve(gram, xy)
    residual = max(abs(sum(gram[i][j] * beta[j] for j in range(5)) - xy[i]) for i in range(5))
    if residual >= 1e-9 or not all(math.isfinite(x) for x in beta):
        raise ValueError("OLS normal-equation residual is too large")
    return beta, residual


def calibrate(results: str | Path, human_targets: str | Path, strict: bool = True):
    human = {}
    for row in read_jsonl(human_targets):
        try:
            tid, block, page = row["task_id"], row["paper_id"], row["page_id"]
            if not all(isinstance(x, str) and x for x in (tid, block, page)) or tid in human:
                raise ValueError("Invalid or duplicate human target identity")
            pair = _pair(row["canonical_pair"])
            outcomes = row["outcomes"]
            votes = [outcomes.get(k, 0) for k in ENCODING]
            if any(type(v) is not int or v < 0 for v in votes):
                raise ValueError("Human vote counts must be nonnegative integers")
            count = sum(votes)
            if (count <= 0 or type(row["valid_preference_votes"]) is not int
                    or count != row["valid_preference_votes"]):
                raise ValueError("Invalid human preference denominator")
            target = (outcomes.get("A", 0) + 0.5 * outcomes.get("TIE", 0)) / count
            if (type(row["human_p_a"]) not in (int, float) or not math.isfinite(row["human_p_a"])
                    or abs(target - row["human_p_a"]) > 1e-12):
                raise ValueError("Human mean disagrees with valid A/TIE/B counts")
            # Do not retain participant identifiers, free text or individual records.
            human[tid] = {"paper_id": block, "page_id": page, "canonical_pair": pair,
                          "human_p_a": target, "valid_preference_votes": count}
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed human target") from exc
    sizes = Counter(t["paper_id"] for t in human.values())
    if not human or len(sizes) < 2:
        raise ValueError("At least two annotation blocks are required")
    if strict and (len(human) != 1872 or len(sizes) != 8 or set(sizes.values()) != {234}):
        raise ValueError("Formal calibration requires eight blocks x 234 = 1,872 cases")
    values, seen_jobs = defaultdict(dict), set()
    for row in read_jsonl(results):
        job, judge, task, winner = validate_judgment_row(row)
        tid, repeat = task["task_id"], row.get("repeat_index")
        if tid not in human or type(repeat) is not int or repeat not in (1, 2):
            raise ValueError("Unknown calibration task or invalid repeat_index")
        if any(task.get(k) != human[tid][k] for k in ("paper_id", "page_id", "canonical_pair")):
            raise ValueError("Calibration judgment and human target identities disagree")
        key = (task["display_swapped"], repeat)
        if job in seen_jobs or key in values[tid, judge]:
            raise ValueError("Duplicate calibration job or judge/direction/repeat")
        seen_jobs.add(job)
        values[tid, judge][key] = ENCODING[winner]
    ids, features = sorted(human), {}
    for tid in ids:
        for judge in JUDGES:
            if set(values[tid, judge]) != {(False, 1), (False, 2), (True, 1), (True, 2)}:
                raise ValueError("Calibration requires four distinct votes per case/judge")
        features[tid] = [sum(values[tid, judge].values()) / 4 for judge in JUDGES]
    folds, predictions = [], {}
    for block in sorted(sizes):
        train = [t for t in ids if human[t]["paper_id"] != block]
        test = [t for t in ids if human[t]["paper_id"] == block]
        beta, residual = fit_ols(train, features, human)
        for tid in test:
            raw = sum(x * b for x, b in zip(features[tid], beta))
            predictions[tid] = (raw, min(1.0, max(0.0, raw)))
        folds.append({"held_out_block": block, "training_n": len(train), "test_n": len(test),
                      "coefficients": dict(zip(JUDGES, beta)), "normal_equation_error": residual})
    full_beta, residual = fit_ols(ids, features, human)
    n = len(ids)
    mse = sum((human[t]["human_p_a"] - predictions[t][1]) ** 2 for t in ids) / n
    mae = sum(abs(human[t]["human_p_a"] - predictions[t][1]) for t in ids) / n
    models = sorted({m for h in human.values() for m in h["canonical_pair"].values()})
    rankings = {}
    for key in ("human", "oof"):
        records = [(human[t]["canonical_pair"]["A"], human[t]["canonical_pair"]["B"],
                    human[t]["human_p_a"] if key == "human" else predictions[t][1], 1.0) for t in ids]
        scores = bt_fit(records, models)
        rankings[key] = {m: rank for rank, m in enumerate(sorted(models, key=lambda m: (-scores[m], m)), 1)}
    squared = sum((rankings["human"][m] - rankings["oof"][m]) ** 2 for m in models)
    spearman = 1 - 6 * squared / (len(models) * (len(models) ** 2 - 1))
    report = {"method": "equal-case OLS, no intercept, no ridge, unnormalized",
              "validation": "leave-one-annotation-block-out (not leave-one-page-out)",
              "physical_comparisons": n, "judgment_rows": len(seen_jobs), "coverage": 1.0,
              "paper_scale": strict, "annotation_blocks": dict(sizes),
              "reproduced_coefficients": dict(zip(JUDGES, full_beta)),
              "normal_equation_error": residual, "fold_coefficients": folds,
              "oof_metrics": {"brier": mse, "rmse": mse ** .5, "mae": mae,
                              "leaderboard_spearman": spearman},
              "limitations": ["Internal validation, not an external human test set.",
                              "Four-vote calibration does not validate single-vote deployment.",
                              "No coefficient or frozen result is overwritten.",
                              "This command does not re-estimate the published bootstrap intervals."]}
    cases = [{"task_id": t, **human[t], "features": dict(zip(JUDGES, features[t])),
              "oof_raw": predictions[t][0], "oof_p_A": predictions[t][1]} for t in ids]
    return report, cases
