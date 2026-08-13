from __future__ import annotations

from types import SimpleNamespace

from benchmark.arena import LegacyArenaDecision, compute_online_elo_legacy
from benchmark.rating import compute_batch_elo


def _legacy(a, b, winner):
    return LegacyArenaDecision("x", a, b, winner, 3, "", "judge", 0, False)


def test_sequential_elo_order_dependence_is_not_present_in_batch_bt():
    legacy = [
        _legacy("A", "B", "A"),
        _legacy("A", "B", "A"),
        _legacy("B", "C", "A"),
        _legacy("B", "C", "B"),
        _legacy("A", "C", "B"),
        _legacy("A", "C", "A"),
        _legacy("A", "B", "B"),
    ]
    fwd = compute_online_elo_legacy(legacy)
    rev = compute_online_elo_legacy(list(reversed(legacy)))
    assert any(abs(fwd[m] - rev[m]) > 0.1 for m in fwd)

    batch_rows = [SimpleNamespace(model_a=d.model_a, model_b=d.model_b, winner=d.winner, confidence=d.confidence) for d in legacy]
    b1 = compute_batch_elo(batch_rows, confidence_weighted=False).ratings
    b2 = compute_batch_elo(list(reversed(batch_rows)), confidence_weighted=False).ratings
    assert all(abs(b1[m] - b2[m]) < 1e-9 for m in b1)
