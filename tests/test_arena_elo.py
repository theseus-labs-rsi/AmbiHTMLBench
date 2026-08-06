from __future__ import annotations

import unittest

from benchmark.arena import ArenaDecision, compute_elo


def make_decision(
    sample_id: str,
    winner: str,
    confidence: int = 4,
    model_a: str = "model_a",
    model_b: str = "model_b",
    round_idx: int = 0,
) -> ArenaDecision:
    return ArenaDecision(
        sample_id=sample_id,
        model_a=model_a,
        model_b=model_b,
        winner=winner,
        confidence=confidence,
        reason="test",
        judge_model="test-judge",
        round_idx=round_idx,
        swapped=False,
    )


class ArenaEloTests(unittest.TestCase):
    def test_empty_decisions_return_empty_ranking(self) -> None:
        self.assertEqual(compute_elo([]), {})

    def test_online_elo_is_order_sensitive(self) -> None:
        decisions = [
            make_decision("01", "A"),
            make_decision("02", "A"),
            make_decision("03", "B"),
            make_decision("04", "B"),
        ]

        forward = compute_elo(decisions, mode="online")
        backward = compute_elo(list(reversed(decisions)), mode="online")

        self.assertNotEqual(forward, backward)

    def test_shuffle_ensemble_ignores_input_order(self) -> None:
        decisions = [
            make_decision("01", "A"),
            make_decision("02", "A"),
            make_decision("03", "B"),
            make_decision("04", "B"),
        ]

        forward = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=42,
        )
        backward = compute_elo(
            list(reversed(decisions)),
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=42,
        )

        self.assertEqual(forward, backward)

    def test_shuffle_ensemble_is_reproducible(self) -> None:
        decisions = [
            make_decision("01", "A"),
            make_decision("02", "B"),
            make_decision("03", "A"),
        ]

        result_1 = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=7,
        )
        result_2 = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=7,
        )

        self.assertEqual(result_1, result_2)

    def test_total_rating_is_preserved(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "B", model_a="a", model_b="c"),
            make_decision("03", "A", model_a="b", model_b="c"),
        ]

        ratings = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=42,
        )

        expected_total = 1000.0 * len(ratings)
        self.assertAlmostEqual(sum(ratings.values()), expected_total, places=9)

    def test_high_confidence_has_larger_effect(self) -> None:
        low = compute_elo(
            [make_decision("01", "A", confidence=1)],
            mode="online",
        )
        high = compute_elo(
            [make_decision("01", "A", confidence=5)],
            mode="online",
        )

        low_change = low["model_a"] - 1000.0
        high_change = high["model_a"] - 1000.0
        self.assertGreater(high_change, low_change)

    def test_equal_rating_tie_keeps_initial_rating(self) -> None:
        ratings = compute_elo(
            [make_decision("01", "TIE")],
            mode="online",
        )

        self.assertAlmostEqual(ratings["model_a"], 1000.0, places=9)
        self.assertAlmostEqual(ratings["model_b"], 1000.0, places=9)

    def test_more_wins_produce_higher_rating(self) -> None:
        decisions = [
            make_decision("01", "A"),
            make_decision("02", "A"),
            make_decision("03", "A"),
            make_decision("04", "A"),
            make_decision("05", "B"),
        ]

        ratings = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=42,
        )

        self.assertGreater(ratings["model_a"], ratings["model_b"])

    def test_v2_single_round_equals_no_aggregation(self) -> None:
        decisions = [
            make_decision("01", "A", confidence=4),
        ]

        none = compute_elo(
            decisions,
            mode="online",
            round_aggregation="none",
        )

        aggregated = compute_elo(
            decisions,
            mode="online",
            round_aggregation="sample_pair",
        )

        self.assertEqual(none, aggregated)

    def test_v2_consistent_rounds_do_not_stack_linearly(self) -> None:
        one = compute_elo(
            [
                make_decision("01", "A", confidence=5),
            ],
            mode="online",
            round_aggregation="none",
        )

        three = compute_elo(
            [
                make_decision("01", "A", confidence=5, round_idx=0),
                make_decision("01", "A", confidence=5, round_idx=1),
                make_decision("01", "A", confidence=5, round_idx=2),
            ],
            mode="online",
            round_aggregation="sample_pair",
        )

        stacked = compute_elo(
            [
                make_decision("01", "A", confidence=5, round_idx=0),
                make_decision("01", "A", confidence=5, round_idx=1),
                make_decision("01", "A", confidence=5, round_idx=2),
            ],
            mode="online",
            round_aggregation="none",
        )

        self.assertLess(
            three["model_a"] - 1000,
            3 * (one["model_a"] - 1000),
        )
        self.assertGreater(
            three["model_a"],
            one["model_a"],
        )

    def test_v2_conflicting_rounds_reduce_update(self) -> None:
        consistent = compute_elo(
            [
                make_decision("01", "A", confidence=5),
                make_decision("01", "A", confidence=5, round_idx=1),
            ],
            mode="online",
            round_aggregation="sample_pair",
        )

        conflict = compute_elo(
            [
                make_decision("01", "A", confidence=5),
                make_decision("01", "B", confidence=5, round_idx=1),
            ],
            mode="online",
            round_aggregation="sample_pair",
        )

        self.assertGreater(
            consistent["model_a"] - 1000,
            conflict["model_a"] - 1000,
        )

    def test_v2_round_aggregation_changes_result(self) -> None:
        decisions = [
            make_decision("01", "A", confidence=5),
            make_decision("01", "B", confidence=5, round_idx=1),
        ]

        none = compute_elo(
            decisions,
            mode="online",
            round_aggregation="none",
        )

        aggregated = compute_elo(
            decisions,
            mode="online",
            round_aggregation="sample_pair",
        )

        self.assertNotEqual(none, aggregated)


if __name__ == "__main__":
    unittest.main()
