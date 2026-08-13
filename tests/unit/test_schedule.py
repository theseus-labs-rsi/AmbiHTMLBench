from __future__ import annotations

from benchmark.arena_schedule import build_balanced_schedule, round_robin_rounds, validate_schedule


MODELS = [f"m{i}" for i in range(10)]


def test_round_robin_10_models_covers_45_pairs_once():
    rounds = round_robin_rounds(MODELS)
    assert len(rounds) == 9
    assert all(len(r) == 5 for r in rounds)
    pairs = [tuple(sorted(x)) for r in rounds for x in r]
    assert len(pairs) == 45
    assert len(set(pairs)) == 45


def test_final_scale_schedule_shape():
    pages = [f"{i:04d}" for i in range(1, 91)]
    schedule = build_balanced_schedule(MODELS, pages, repeats=10, seed=20260812)
    v = validate_schedule(schedule, MODELS, expected_pair_occurrences=10)
    assert v.valid
    assert v.pages == 90
    assert v.physical_pair_sets == 450
    assert v.unique_unordered_pairs == 45
    assert set(v.model_comparison_counts.values()) == {90}
