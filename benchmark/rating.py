from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


METHOD_ID = "batch_elo_bt_mle_v1"


@dataclass(frozen=True)
class BatchEloResult:
    ratings: dict[str, float]
    method_id: str
    confidence_weighted: bool
    init_rating: float
    elo_scale: float
    converged: bool
    iterations: int
    tolerance: float
    max_log_strength_delta: float
    log_likelihood: float
    models: int
    decisions: int


def _confidence_weight(confidence: int) -> float:
    """Preserve the frozen pilot's confidence weighting rule exactly."""
    if confidence not in {1, 2, 3, 4, 5}:
        raise ValueError(f"confidence must be 1..5, got {confidence!r}")
    return 0.5 + 0.125 * confidence


def _score_for_a(winner: str) -> float:
    if winner == "A":
        return 1.0
    if winner == "B":
        return 0.0
    if winner == "TIE":
        return 0.5
    raise ValueError(f"winner must be A/B/TIE, got {winner!r}")


def _validate_connected(models: list[str], pair_totals: dict[tuple[str, str], float]) -> None:
    adj = {m: set() for m in models}
    for (a, b), total in pair_totals.items():
        if total > 0:
            adj[a].add(b)
            adj[b].add(a)

    seen = set()
    stack = [models[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur] - seen)

    if len(seen) != len(models):
        missing = sorted(set(models) - seen)
        raise ValueError(
            "comparison graph is disconnected; a single global rating is not identifiable. "
            f"disconnected models={missing}"
        )


def compute_batch_elo(
    decisions: Iterable[Any],
    *,
    init_rating: float = 1000.0,
    elo_scale: float = 400.0,
    confidence_weighted: bool = True,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> BatchEloResult:
    """Order-invariant offline Elo fit via Bradley-Terry maximum likelihood.

    The probability model is exactly the Elo logistic model::

        P(A beats B) = 1 / (1 + 10 ** ((R_B - R_A) / elo_scale))

    Unlike sequential Elo, all decisions are fitted jointly.  Ties contribute 0.5
    score to each side.  When confidence weighting is enabled, the same frozen
    pilot rule is retained as an observation weight::

        weight = 0.5 + 0.125 * confidence

    The Bradley-Terry strengths are estimated by a deterministic MM algorithm and
    converted to the Elo scale.  Ratings are centered to mean ``init_rating``.
    Therefore the result is independent of decision input order.
    """
    ds = list(decisions)
    if not ds:
        return BatchEloResult(
            ratings={}, method_id=METHOD_ID, confidence_weighted=confidence_weighted,
            init_rating=init_rating, elo_scale=elo_scale, converged=True, iterations=0,
            tolerance=tolerance, max_log_strength_delta=0.0, log_likelihood=0.0,
            models=0, decisions=0,
        )
    if elo_scale <= 0:
        raise ValueError("elo_scale must be > 0")
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    model_set: set[str] = set()
    observations: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for d in ds:
        a = str(d.model_a)
        b = str(d.model_b)
        if not a or not b or a == b:
            raise ValueError(f"invalid comparison pair: {a!r} vs {b!r}")
        model_set.update((a, b))

        conf = int(d.confidence)
        weight = _confidence_weight(conf) if confidence_weighted else 1.0
        score_a = _score_for_a(str(d.winner).upper())

        if a < b:
            key = (a, b)
            score_first = score_a
        else:
            key = (b, a)
            score_first = 1.0 - score_a
        observations.setdefault(key, []).append((weight, weight * score_first))

    models = sorted(model_set)
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    if n < 2:
        raise ValueError("at least two models are required")

    # Aggregate in canonical pair order.  All frozen confidence weights are binary
    # fractions, so these sums are deterministic under input permutation.
    pair_totals: dict[tuple[str, str], float] = {}
    pair_score_first: dict[tuple[str, str], float] = {}
    for key in sorted(observations):
        vals = observations[key]
        pair_totals[key] = math.fsum(v[0] for v in vals)
        pair_score_first[key] = math.fsum(v[1] for v in vals)

    _validate_connected(models, pair_totals)

    weighted_scores = [0.0] * n
    weighted_games = [0.0] * n
    for (a, b) in sorted(pair_totals):
        total = pair_totals[(a, b)]
        first = pair_score_first[(a, b)]
        ia, ib = idx[a], idx[b]
        weighted_scores[ia] += first
        weighted_scores[ib] += total - first
        weighted_games[ia] += total
        weighted_games[ib] += total

    # A finite unregularized BT MLE requires every model to have some score and
    # some non-score in a connected graph.  Fail loudly rather than hiding a
    # separation problem behind arbitrary pseudocounts.
    for i, m in enumerate(models):
        if weighted_scores[i] <= 0.0:
            raise ValueError(f"finite batch Elo MLE unavailable: model {m!r} has zero weighted score")
        if weighted_games[i] - weighted_scores[i] <= 0.0:
            raise ValueError(f"finite batch Elo MLE unavailable: model {m!r} has no weighted losses")

    strengths = [1.0] * n
    converged = False
    max_delta = math.inf
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        denom = [0.0] * n
        for (a, b) in sorted(pair_totals):
            ia, ib = idx[a], idx[b]
            total = pair_totals[(a, b)]
            common = total / (strengths[ia] + strengths[ib])
            denom[ia] += common
            denom[ib] += common

        new_strengths = [weighted_scores[i] / denom[i] for i in range(n)]

        # BT strengths are identifiable only up to a common scale.  Geometric-mean
        # normalization fixes that degree of freedom and makes mean Elo exactly the
        # requested init_rating after conversion.
        log_mean = math.fsum(math.log(x) for x in new_strengths) / n
        scale = math.exp(log_mean)
        new_strengths = [x / scale for x in new_strengths]

        max_delta = max(abs(math.log(new_strengths[i] / strengths[i])) for i in range(n))
        strengths = new_strengths
        if max_delta < tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"batch Elo MLE did not converge after {max_iterations} iterations; "
            f"last max_log_strength_delta={max_delta:.3e}"
        )

    factor = elo_scale / math.log(10.0)
    ratings = {
        m: init_rating + factor * math.log(strengths[idx[m]])
        for m in models
    }
    ratings = dict(sorted(ratings.items(), key=lambda kv: (-kv[1], kv[0])))

    log_likelihood = 0.0
    for (a, b) in sorted(pair_totals):
        ia, ib = idx[a], idx[b]
        total = pair_totals[(a, b)]
        first = pair_score_first[(a, b)]
        sa, sb = strengths[ia], strengths[ib]
        log_den = math.log(sa + sb)
        log_p = math.log(sa) - log_den
        log_q = math.log(sb) - log_den
        log_likelihood += first * log_p + (total - first) * log_q

    return BatchEloResult(
        ratings=ratings,
        method_id=METHOD_ID,
        confidence_weighted=confidence_weighted,
        init_rating=init_rating,
        elo_scale=elo_scale,
        converged=True,
        iterations=iteration,
        tolerance=tolerance,
        max_log_strength_delta=max_delta,
        log_likelihood=log_likelihood,
        models=n,
        decisions=len(ds),
    )
