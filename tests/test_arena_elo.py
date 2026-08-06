from __future__ import annotations

import unittest

from benchmark.arena import ArenaDecision, compute_elo, compute_elo_stats


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

    def test_v3_stats_mean_matches_compute_elo(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="a", model_b="c"),
            make_decision("03", "A", model_a="b", model_b="c"),
        ]

        ratings = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=11,
        )
        stats = compute_elo_stats(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=11,
        )

        for model, rating in ratings.items():
            self.assertAlmostEqual(
                stats[model].elo,
                rating,
                places=9,
            )

    def test_v3_online_stats_have_zero_std(self) -> None:
        stats = compute_elo_stats(
            [make_decision("01", "A")],
            mode="online",
        )

        self.assertAlmostEqual(stats["model_a"].elo_std, 0.0)
        self.assertAlmostEqual(stats["model_b"].elo_std, 0.0)
        self.assertEqual(stats["model_a"].trial_count, 1)
        self.assertEqual(stats["model_a"].rank_stability, 1.0)

    def test_v3_stats_ranges_are_valid(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="b", model_b="c"),
            make_decision("03", "A", model_a="c", model_b="a"),
        ]

        stats = compute_elo_stats(
            decisions,
            shuffle_repeats=32,
            seed=42,
        )

        for item in stats.values():
            self.assertGreaterEqual(item.elo_std, 0.0)
            self.assertGreaterEqual(item.mean_rank, 1.0)
            self.assertLessEqual(item.mean_rank, len(stats))
            self.assertGreaterEqual(item.rank_stability, 0.0)
            self.assertLessEqual(item.rank_stability, 1.0)
            self.assertGreaterEqual(
                item.top_rank_probability,
                0.0,
            )
            self.assertLessEqual(
                item.top_rank_probability,
                1.0,
            )

    def test_v3_cyclic_results_have_nonzero_uncertainty(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="b", model_b="c"),
            make_decision("03", "A", model_a="c", model_b="a"),
        ]

        stats = compute_elo_stats(
            decisions,
            shuffle_repeats=64,
            seed=42,
        )

        self.assertTrue(
            any(item.elo_std > 0.0 for item in stats.values())
        )

    def test_v3_clear_winner_has_full_top_probability(self) -> None:
        decisions = [
            make_decision("01", "A", confidence=5, model_a="a", model_b="b"),
            make_decision("02", "A", confidence=5, model_a="a", model_b="c"),
            make_decision("03", "A", confidence=5, model_a="b", model_b="c"),
        ]

        stats = compute_elo_stats(
            decisions,
            shuffle_repeats=64,
            seed=42,
        )

        self.assertAlmostEqual(stats["a"].top_rank_probability, 1.0)

    def test_v4_two_model_sample_is_unchanged(self) -> None:
        decisions = [
            make_decision("01", "A", confidence=5),
        ]

        none = compute_elo(
            decisions,
            mode="online",
            sample_balance="none",
        )
        balanced = compute_elo(
            decisions,
            mode="online",
            sample_balance="per_model_exposure",
        )

        self.assertEqual(none, balanced)

    def test_v4_dense_sample_reduces_total_movement(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("01", "A", model_a="a", model_b="c"),
            make_decision("01", "A", model_a="a", model_b="d"),
            make_decision("01", "A", model_a="b", model_b="c"),
            make_decision("01", "A", model_a="b", model_b="d"),
            make_decision("01", "A", model_a="c", model_b="d"),
        ]

        none = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=42,
            sample_balance="none",
        )
        balanced = compute_elo(
            decisions,
            mode="shuffle_ensemble",
            shuffle_repeats=32,
            seed=42,
            sample_balance="per_model_exposure",
        )

        none_movement = sum(
            abs(rating - 1000.0)
            for rating in none.values()
        )
        balanced_movement = sum(
            abs(rating - 1000.0)
            for rating in balanced.values()
        )

        self.assertLess(balanced_movement, none_movement)

    def test_v4_sample_balance_preserves_total_rating(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("01", "B", model_a="a", model_b="c"),
            make_decision("01", "A", model_a="b", model_b="c"),
        ]

        ratings = compute_elo(
            decisions,
            shuffle_repeats=32,
            seed=42,
            sample_balance="per_model_exposure",
        )

        self.assertAlmostEqual(
            sum(ratings.values()),
            1000.0 * len(ratings),
            places=9,
        )

    def test_v4_invalid_sample_balance_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_elo(
                [make_decision("01", "A")],
                sample_balance="invalid",
            )

    def test_v5_batch_is_order_independent(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="a", model_b="c"),
            make_decision("03", "A", model_a="b", model_b="c"),
        ]

        forward = compute_elo(decisions, mode="batch")
        backward = compute_elo(
            list(reversed(decisions)),
            mode="batch",
        )

        for model in forward:
            self.assertAlmostEqual(
                forward[model],
                backward[model],
                places=9,
            )

    def test_v5_batch_clear_winner_ranking(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="a", model_b="c"),
            make_decision("03", "A", model_a="b", model_b="c"),
        ]

        ratings = compute_elo(decisions, mode="batch")

        self.assertGreater(ratings["a"], ratings["b"])
        self.assertGreater(ratings["b"], ratings["c"])

    def test_v5_batch_cycle_remains_balanced(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="b", model_b="c"),
            make_decision("03", "A", model_a="c", model_b="a"),
        ]

        ratings = compute_elo(decisions, mode="batch")

        self.assertAlmostEqual(ratings["a"], 1000.0, places=9)
        self.assertAlmostEqual(ratings["b"], 1000.0, places=9)
        self.assertAlmostEqual(ratings["c"], 1000.0, places=9)

    def test_v5_batch_preserves_mean_rating(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "B", model_a="a", model_b="c"),
            make_decision("03", "A", model_a="b", model_b="c"),
        ]

        ratings = compute_elo(decisions, mode="batch")

        self.assertAlmostEqual(
            sum(ratings.values()),
            1000.0 * len(ratings),
            places=9,
        )

    def test_v5_regularization_reduces_rating_spread(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
            make_decision("02", "A", model_a="a", model_b="c"),
            make_decision("03", "A", model_a="b", model_b="c"),
        ]

        weak = compute_elo(
            decisions,
            mode="batch",
            batch_regularization=0.01,
        )
        strong = compute_elo(
            decisions,
            mode="batch",
            batch_regularization=0.20,
        )

        weak_spread = max(weak.values()) - min(weak.values())
        strong_spread = max(strong.values()) - min(strong.values())

        self.assertLess(strong_spread, weak_spread)

    def test_v5_batch_stats_have_one_trial(self) -> None:
        decisions = [
            make_decision("01", "A", model_a="a", model_b="b"),
        ]

        stats = compute_elo_stats(decisions, mode="batch")

        for item in stats.values():
            self.assertEqual(item.trial_count, 1)
            self.assertAlmostEqual(item.elo_std, 0.0, places=12)

    def test_v5_invalid_batch_parameters_raise(self) -> None:
        decisions = [make_decision("01", "A")]

        with self.assertRaises(ValueError):
            compute_elo(
                decisions,
                mode="batch",
                batch_iterations=0,
            )

        with self.assertRaises(ValueError):
            compute_elo(
                decisions,
                mode="batch",
                batch_tolerance=0.0,
            )

        with self.assertRaises(ValueError):
            compute_elo(
                decisions,
                mode="batch",
                batch_regularization=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
