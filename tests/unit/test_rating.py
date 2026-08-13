from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from benchmark.rating import compute_batch_elo


def decisions():
    return [
        SimpleNamespace(model_a="A", model_b="B", winner="A", confidence=3),
        SimpleNamespace(model_a="A", model_b="B", winner="A", confidence=4),
        SimpleNamespace(model_a="A", model_b="C", winner="B", confidence=3),
        SimpleNamespace(model_a="A", model_b="C", winner="A", confidence=3),
        SimpleNamespace(model_a="B", model_b="C", winner="B", confidence=3),
        SimpleNamespace(model_a="B", model_b="C", winner="A", confidence=3),
        SimpleNamespace(model_a="A", model_b="B", winner="B", confidence=2),
    ]


def test_batch_bt_is_order_invariant():
    ds = decisions()
    base = compute_batch_elo(ds, confidence_weighted=False).ratings
    reverse = compute_batch_elo(list(reversed(ds)), confidence_weighted=False).ratings
    assert base == pytest.approx(reverse, abs=1e-10)

    rng = random.Random(7)
    for _ in range(20):
        shuffled = ds[:]
        rng.shuffle(shuffled)
        got = compute_batch_elo(shuffled, confidence_weighted=False).ratings
        assert got == pytest.approx(base, abs=1e-10)


def test_batch_bt_mean_is_centered_at_1000():
    fit = compute_batch_elo(decisions(), confidence_weighted=False)
    assert sum(fit.ratings.values()) / len(fit.ratings) == pytest.approx(1000.0, abs=1e-9)


def test_disconnected_graph_rejected():
    ds = [
        SimpleNamespace(model_a="A", model_b="B", winner="A", confidence=3),
        SimpleNamespace(model_a="A", model_b="B", winner="B", confidence=3),
        SimpleNamespace(model_a="C", model_b="D", winner="A", confidence=3),
        SimpleNamespace(model_a="C", model_b="D", winner="B", confidence=3),
    ]
    with pytest.raises(ValueError, match="disconnected"):
        compute_batch_elo(ds, confidence_weighted=False)
