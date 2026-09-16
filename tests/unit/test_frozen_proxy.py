"""Small, offline regression tests for the frozen overall-only proxy."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark import frozen_proxy as proxy


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "subjective_ols5_frozen.json"
EXPECTED_COEFFICIENTS = {
    "anthropic_opus5": 0.09045204815695966,
    "anthropic_sonnet5": 0.10401091278902799,
    "gemini31pro": 0.37576678499354577,
    "openai_sol": 0.1374207709022521,
    "openai_terra": 0.160124982073635,
}


@pytest.fixture
def config():
    return proxy.load_proxy_config(CONFIG_PATH)


def case(task_id, page_id, a, b, probability):
    return {
        "task_id": task_id,
        "page_id": page_id,
        "canonical_pair": {"A": a, "B": b},
        "p_A": probability,
    }


@pytest.fixture
def cases():
    # The first page contributes two correlated comparisons; other pages one each.
    return [
        case("t1", "page1", "alpha", "beta", 0.9),
        case("t2", "page1", "alpha", "beta", 0.8),
        case("t3", "page2", "alpha", "gamma", 0.4),
        case("t4", "page3", "beta", "gamma", 0.3),
    ]


def test_checked_in_coefficients_are_exact(config):
    assert tuple(EXPECTED_COEFFICIENTS) == proxy.JUDGES
    assert config["coefficients"] == EXPECTED_COEFFICIENTS
    assert config["intercept"] == 0
    assert config["normalize_weights"] is False
    assert config["encoding"] == {"A": 1.0, "B": 0.0, "TIE": 0.5}


@pytest.mark.parametrize("vote,scale", [("A", 1.0), ("TIE", 0.5), ("B", 0.0)])
def test_score_has_no_intercept_or_weight_normalization(config, vote, scale):
    result = proxy.score_votes(dict.fromkeys(proxy.JUDGES, vote), config)
    expected = sum(EXPECTED_COEFFICIENTS.values()) * scale
    assert result["raw"] == pytest.approx(expected, abs=1e-15)
    assert result["p_A"] == pytest.approx(expected, abs=1e-15)
    assert result["features"] == dict.fromkeys(proxy.JUDGES, scale)
    assert result["contributions"] == pytest.approx(
        {judge: coefficient * scale for judge, coefficient in EXPECTED_COEFFICIENTS.items()}
    )


def test_single_vote_is_not_averaged_and_recoding_is_not_complementary(config):
    votes = dict.fromkeys(proxy.JUDGES, "B")
    votes["gemini31pro"] = "A"
    probability = proxy.score_votes(votes, config)["p_A"]
    assert probability == EXPECTED_COEFFICIENTS["gemini31pro"]
    flipped = {j: "B" if vote == "A" else "A" for j, vote in votes.items()}
    complement = proxy.score_votes(flipped, config)["p_A"]
    assert probability + complement == pytest.approx(sum(EXPECTED_COEFFICIENTS.values()))
    assert probability + complement != pytest.approx(1.0)


def test_clipping_happens_after_linear_sum(config):
    custom = copy.deepcopy(config)
    custom["coefficients"] = dict.fromkeys(proxy.JUDGES, 2.0)
    result = proxy.score_votes(dict.fromkeys(proxy.JUDGES, "A"), custom)
    assert result["raw"] == 10.0
    assert result["p_A"] == 1.0
    custom["coefficients"] = dict.fromkeys(proxy.JUDGES, -2.0)
    result = proxy.score_votes(dict.fromkeys(proxy.JUDGES, "A"), custom)
    assert result["raw"] == -10.0
    assert result["p_A"] == 0.0


@pytest.mark.parametrize("defect", ["missing", "extra", "invalid", "numeric", "repeat"])
def test_rejects_non_single_or_incomplete_votes(config, defect):
    votes = dict.fromkeys(proxy.JUDGES, "A")
    if defect == "missing":
        votes.pop(proxy.JUDGES[0])
    elif defect == "extra":
        votes["unknown_judge"] = "A"
    elif defect == "invalid":
        votes[proxy.JUDGES[0]] = "RELIABILITY_TIE"
    elif defect == "numeric":
        votes[proxy.JUDGES[0]] = 1.0
    else:
        votes[proxy.JUDGES[0]] = ["A", "B"]
    with pytest.raises(ValueError):
        proxy.score_votes(votes, config)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "0.1", None])
def test_rejects_invalid_coefficients_on_load_and_score(config, tmp_path, invalid):
    broken = copy.deepcopy(config)
    broken["coefficients"][proxy.JUDGES[0]] = invalid
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        proxy.load_proxy_config(path)
    with pytest.raises(ValueError):
        proxy.score_votes(dict.fromkeys(proxy.JUDGES, "A"), broken)


@pytest.mark.parametrize("defect", ["missing", "extra", "intercept", "normalize", "encoding"])
def test_rejects_protocol_changing_config(config, tmp_path, defect):
    broken = copy.deepcopy(config)
    if defect == "missing":
        broken["coefficients"].pop(proxy.JUDGES[0])
    elif defect == "extra":
        broken["coefficients"]["unknown_judge"] = 0.0
    elif defect == "intercept":
        broken["intercept"] = 0.1
    elif defect == "normalize":
        broken["normalize_weights"] = True
    else:
        broken["encoding"]["TIE"] = 0.0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        proxy.load_proxy_config(path)


def test_rejects_duplicate_json_configuration_keys(config, tmp_path):
    text = '{"intercept": 1.0, ' + json.dumps(config)[1:]
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        proxy.load_proxy_config(path)


def test_continuous_bt_matches_original_frozen_algorithm():
    # Golden values from scripts/build_dual_leaderboard.py:bt_fit (ridge=1e-3).
    records = [("alpha", "beta", 0.8, 2.0), ("alpha", "gamma", 0.65, 3.0),
               ("beta", "gamma", 0.4, 1.0)]
    models = ["alpha", "beta", "gamma"]
    expected = {"alpha": 112.35686949996412, "beta": -106.60176641851766,
                "gamma": -5.7551030814464585}
    actual = proxy.bt_fit(records, models)
    assert actual == pytest.approx(expected, abs=1e-8)
    assert sum(actual.values()) == pytest.approx(0.0, abs=1e-9)
    assert actual["alpha"] > actual["gamma"] > actual["beta"]
    assert proxy.bt_fit(list(reversed(records)), models) == pytest.approx(actual, abs=1e-9)
    reversed_sides = [(b, a, 1.0 - p, w) for a, b, p, w in records]
    assert proxy.bt_fit(reversed_sides, models) == pytest.approx(actual, abs=1e-9)


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_bt_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        proxy.bt_fit([("a", "b", probability, 1.0)], ["a", "b"])


@pytest.mark.parametrize("weight", [-1.0, 0.0, float("nan"), float("inf"), True])
def test_bt_rejects_invalid_weights(weight):
    with pytest.raises(ValueError):
        proxy.bt_fit([("a", "b", 0.5, weight)], ["a", "b"])


def test_bt_rejects_disconnected_graph_and_self_comparison():
    with pytest.raises(ValueError):
        proxy.bt_fit([("a", "b", 0.5, 1.0), ("c", "d", 0.5, 1.0)], ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        proxy.bt_fit([("a", "a", 0.5, 1.0)], ["a", "b"])


def test_page_bootstrap_is_deterministic_and_reports_coverage(cases):
    first = proxy.build_leaderboard(cases, bootstrap=30, seed=7)
    assert first == proxy.build_leaderboard(cases, bootstrap=30, seed=7)
    assert first["coverage"] == 1.0
    assert first["bootstrap"]["n"] == 30
    assert first["bootstrap"]["seed"] == 7
    rows = {row["model"]: row for row in first["leaderboard"]}
    assert {m: row["adjudicated_comparisons"] for m, row in rows.items()} == {
        "alpha": 3, "beta": 3, "gamma": 2,
    }
    assert all(row["scheduled_comparisons"] == row["adjudicated_comparisons"] for row in rows.values())
    assert rows["alpha"]["preference_point_rate"] == pytest.approx(2.1 / 3)
    for row in rows.values():
        assert row["bt_ci95_low"] <= row["bt_ci95_high"]
        assert all(math.isfinite(row[key]) for key in ("bt_score", "bt_ci95_low", "bt_ci95_high"))


def test_bootstrap_resamples_whole_pages_not_individual_comparisons(cases):
    with patch.object(proxy, "_bt_fit", wraps=proxy._bt_fit) as fitted:
        proxy.build_leaderboard(cases, bootstrap=12, seed=7)
    assert fitted.call_count == 13  # one original estimate, then every bootstrap draw
    for call in fitted.call_args_list[1:]:
        records = call.args[0]
        weights = {(a, b): weight for a, b, _, weight in records}
        assert weights.get(("alpha", "beta"), 0) % 2 == 0
        page_draw_count = weights.get(("alpha", "beta"), 0) / 2
        page_draw_count += weights.get(("alpha", "gamma"), 0) + weights.get(("beta", "gamma"), 0)
        assert page_draw_count == 3
        for a, b, probability, _ in records:
            expected = {("alpha", "beta"): 0.85, ("alpha", "gamma"): 0.4,
                        ("beta", "gamma"): 0.3}[a, b]
            assert probability == pytest.approx(expected)


def test_disconnected_bootstrap_draws_are_kept_not_silently_replaced(cases):
    report = proxy.build_leaderboard(cases, bootstrap=30, seed=7)
    assert report["bootstrap"]["disconnected_draws"] > 0
    assert report["bootstrap"]["successful_draws"] == 30
    assert report["bootstrap"]["failed_draws"] == 0


@pytest.mark.parametrize("defect", ["duplicate", "missing_page", "invalid_probability", "self_pair", "disconnected"])
def test_leaderboard_rejects_invalid_case_sets(cases, defect):
    if defect == "duplicate":
        cases.append(copy.deepcopy(cases[0]))
    elif defect == "missing_page":
        cases[0].pop("page_id")
    elif defect == "invalid_probability":
        cases[0]["p_A"] = float("nan")
    elif defect == "self_pair":
        cases[0]["canonical_pair"]["B"] = "alpha"
    else:
        cases.append(case("t5", "page4", "delta", "epsilon", 0.5))
    with pytest.raises(ValueError):
        proxy.build_leaderboard(cases, bootstrap=5)


@pytest.mark.parametrize("bootstrap", [0, -1, 1.5, True])
def test_leaderboard_rejects_invalid_bootstrap_count(cases, bootstrap):
    with pytest.raises(ValueError):
        proxy.build_leaderboard(cases, bootstrap=bootstrap)
