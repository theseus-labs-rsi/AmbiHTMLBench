from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.rating import compute_batch_elo


HERE = Path(__file__).resolve().parent


def test_final_arena90_golden_pair_outcomes_reproduce_frozen_primary_ranking():
    rows = json.loads((HERE / "final_arena90_pair_outcomes.json").read_text(encoding="utf-8"))
    snapshot = json.loads((HERE / "final_arena90_snapshot.json").read_text(encoding="utf-8"))
    ds = [SimpleNamespace(**r) for r in rows]
    fit = compute_batch_elo(ds, confidence_weighted=False)

    expected_order = [x[0] for x in snapshot["ranking"]]
    assert list(fit.ratings) == expected_order
    expected_ratings = dict(snapshot["ranking"])
    for model, rating in fit.ratings.items():
        assert rating == pytest.approx(expected_ratings[model], abs=0.002)
