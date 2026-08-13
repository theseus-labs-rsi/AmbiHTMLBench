from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Any, Iterable


SCHEDULE_METHOD_ID = "balanced_round_robin_v1"


@dataclass(frozen=True)
class ScheduleValidation:
    valid: bool
    pages: int
    physical_pair_sets: int
    unique_unordered_pairs: int
    pair_occurrence_min: int
    pair_occurrence_max: int
    model_comparison_counts: dict[str, int]
    each_model_once_per_page: bool


def round_robin_rounds(models: Iterable[str]) -> list[list[tuple[str, str]]]:
    items = list(models)
    if len(items) < 2 or len(items) % 2:
        raise ValueError("round_robin_rounds requires an even number of >=2 models")
    if len(set(items)) != len(items):
        raise ValueError("model names must be unique")

    arr = items[:]
    n = len(arr)
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(n - 1):
        rounds.append([(arr[i], arr[n - 1 - i]) for i in range(n // 2)])
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]

    unordered = [tuple(sorted(pair)) for rnd in rounds for pair in rnd]
    expected = n * (n - 1) // 2
    if len(unordered) != expected or len(set(unordered)) != expected:
        raise RuntimeError("round-robin construction failed pair coverage validation")
    return rounds


def build_balanced_schedule(
    models: Iterable[str],
    page_ids: Iterable[str],
    *,
    repeats: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build complete round-robin blocks using one perfect matching per page.

    With 10 models, one block is 9 pages and covers all 45 unordered pairs once.
    ``repeats=10`` therefore yields 90 pages and 450 physical comparisons.
    """
    models = list(models)
    pages = [str(x) for x in page_ids]
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    rounds = round_robin_rounds(models)
    pages_per_block = len(rounds)
    needed = repeats * pages_per_block
    if len(pages) < needed:
        raise ValueError(f"need at least {needed} pages for repeats={repeats}, got {len(pages)}")

    rng = random.Random(seed)
    pages = pages[:]
    rng.shuffle(pages)
    pages = pages[:needed]

    pair_sets: list[dict[str, Any]] = []
    for block in range(repeats):
        block_pages = pages[block * pages_per_block:(block + 1) * pages_per_block]
        round_order = list(range(pages_per_block))
        rng.shuffle(round_order)
        for page_position, (sample_id, round_index) in enumerate(zip(block_pages, round_order), 1):
            slot_id = f"B{block + 1:02d}_R{round_index + 1:02d}"
            for pair_position, (model_a, model_b) in enumerate(rounds[round_index], 1):
                pair_sets.append({
                    "pair_set_id": f"{slot_id}_P{pair_position}",
                    "slot_id": slot_id,
                    "block_index": block,
                    "round_index": round_index,
                    "block_page_position": page_position,
                    "sample_id": sample_id,
                    "model_a": model_a,
                    "model_b": model_b,
                })

    validation = validate_schedule(pair_sets, models, expected_pair_occurrences=repeats)
    if not validation.valid:
        raise RuntimeError(f"generated schedule failed validation: {validation}")
    return pair_sets


def validate_schedule(
    pair_sets: Iterable[dict[str, Any]],
    models: Iterable[str],
    *,
    expected_pair_occurrences: int | None = None,
) -> ScheduleValidation:
    pair_sets = list(pair_sets)
    models = list(models)
    pair_counts = Counter(tuple(sorted((str(x["model_a"]), str(x["model_b"])))) for x in pair_sets)
    model_counts = Counter()
    page_model = defaultdict(Counter)

    for x in pair_sets:
        a = str(x["model_a"])
        b = str(x["model_b"])
        sid = str(x["sample_id"])
        model_counts[a] += 1
        model_counts[b] += 1
        page_model[sid][a] += 1
        page_model[sid][b] += 1

    each_once = all(
        set(c) == set(models) and set(c.values()) == {1}
        for c in page_model.values()
    ) if page_model else False

    expected_unique = len(models) * (len(models) - 1) // 2
    pair_min = min(pair_counts.values()) if pair_counts else 0
    pair_max = max(pair_counts.values()) if pair_counts else 0
    occurrence_ok = (
        pair_min == pair_max == expected_pair_occurrences
        if expected_pair_occurrences is not None
        else pair_min == pair_max
    )

    valid = (
        len(pair_counts) == expected_unique
        and occurrence_ok
        and each_once
        and len(set(model_counts.values())) == 1
    )
    return ScheduleValidation(
        valid=valid,
        pages=len(page_model),
        physical_pair_sets=len(pair_sets),
        unique_unordered_pairs=len(pair_counts),
        pair_occurrence_min=pair_min,
        pair_occurrence_max=pair_max,
        model_comparison_counts=dict(model_counts),
        each_model_once_per_page=each_once,
    )


def page_axis_ids(page: dict[str, Any]) -> set[str]:
    return {str(x["id"]) for x in page.get("active_axes", [])}


def choose_reserve_page(
    original_page: dict[str, Any],
    reserve_pages: Iterable[dict[str, Any]],
    *,
    used_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Choose the most axis-similar unused reserve deterministically."""
    used_ids = used_ids or set()
    old_axes = page_axis_ids(original_page)
    rare = {"LIST_HIERARCHY_PRESENTATION", "ALGORITHM_BLOCK_STRUCTURE"}

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for page in reserve_pages:
        sid = str(page["sample_id"])
        if sid in used_ids:
            continue
        new_axes = page_axis_ids(page)
        if old_axes & rare and not (old_axes & rare).issubset(new_axes):
            continue
        shared = len(old_axes & new_axes)
        union = len(old_axes | new_axes)
        jaccard = shared / union if union else 1.0
        no_axis_match = int(not old_axes and not new_axes)
        score = 100.0 * shared + 20.0 * no_axis_match + jaccard
        scored.append((score, sid, page))

    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def lightly_stratified_sample(
    pages: Iterable[dict[str, Any]],
    *,
    total: int,
    axis_floors: dict[str, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Fixed-seed light stratification without turning the benchmark into an axis test.

    Each page is expected to have ``sample_id`` and ``active_axes``. Floors are
    minimum page counts per axis; remaining slots are filled uniformly from the
    unused pool.
    """
    pages = list(pages)
    if total <= 0 or total > len(pages):
        raise ValueError("invalid total sample size")
    rng = random.Random(seed)
    by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pages:
        axes = page_axis_ids(p)
        if not axes:
            by_axis["NO_ACTIVE_AXIS"].append(p)
        for aid in axes:
            by_axis[aid].append(p)
    for arr in by_axis.values():
        rng.shuffle(arr)

    selected: dict[str, dict[str, Any]] = {}
    for axis, floor in axis_floors.items():
        while sum(
            (axis == "NO_ACTIVE_AXIS" and not page_axis_ids(p)) or axis in page_axis_ids(p)
            for p in selected.values()
        ) < floor:
            candidate = next((p for p in by_axis.get(axis, []) if str(p["sample_id"]) not in selected), None)
            if candidate is None:
                raise ValueError(f"cannot satisfy axis floor {axis}={floor}")
            selected[str(candidate["sample_id"])] = candidate

    remaining = [p for p in pages if str(p["sample_id"]) not in selected]
    rng.shuffle(remaining)
    for p in remaining:
        if len(selected) >= total:
            break
        selected[str(p["sample_id"])] = p
    return list(selected.values())
