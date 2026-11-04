"""Merge objective page CSVs using the historical experiment report formula.

Page scores omit the table weight when no GT table exists. Historical merged
leaderboards always divide by 2.3; keep that convention for existing reports.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    inputs: list[Path], out_dir: Path, *, excluded_models: set[str] | None = None,
) -> None:
    """Combine page scores and write page_scores.csv and leaderboard.csv."""
    rows = []
    for path in inputs:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(row for row in csv.DictReader(handle)
                        if row["model"] not in (excluded_models or set()))
    if not rows:
        raise ValueError("No page scores remain after filtering")

    by_model = defaultdict(list)
    for row in rows:
        text = _number(row.get("text_edit_distance"))
        layout = _number(row.get("layout_graph_similarity")) or 0.0
        table = _number(row.get("table_teds"))
        text_quality = max(0.0, min(1.0, 1.0 - (text if text is not None else 1.0)))
        total = text_quality + layout + 0.3 * (table or 0.0)
        row["page_weighted_score"] = _fmt(total / (2.3 if table is not None else 2.0))
        by_model[row["model"]].append(row)

    leaderboard = []
    for model in sorted(by_model, key=str.lower):
        model_rows = by_model[model]
        table = _mean(model_rows, "table_teds")
        texts = [_number(row.get("text_edit_distance")) for row in model_rows]
        text = sum(value if value is not None else 1.0 for value in texts) / len(texts)
        layout = sum(_number(row.get("layout_graph_similarity")) or 0.0
                     for row in model_rows) / len(model_rows)
        weighted = (0.3 * (table or 0.0) + max(0.0, min(1.0, 1.0 - text)) + layout) / 2.3
        leaderboard.append({
            "model": model, "pages": len(model_rows),
            "table_pages": sum(_number(row.get("table_teds")) is not None for row in model_rows),
            "table_teds": _fmt(table), "table_teds_s": _fmt(_mean(model_rows, "table_teds_s")),
            "text_edit_distance": _fmt(text), "layout_graph_similarity": _fmt(layout),
            "weighted_score": _fmt(weighted),
        })
    leaderboard.sort(key=lambda row: float(row["weighted_score"]), reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank

    out_dir.mkdir(parents=True, exist_ok=True)
    page_fields = list(dict.fromkeys(key for row in rows for key in row))
    _write_csv(out_dir / "page_scores.csv", rows, page_fields)
    _write_csv(out_dir / "leaderboard.csv", leaderboard, [
        "rank", "model", "pages", "table_pages", "table_teds", "table_teds_s",
        "text_edit_distance", "layout_graph_similarity", "weighted_score",
    ])
    print(out_dir / "leaderboard.csv")
    for row in leaderboard:
        print(f"{row['rank']:>2}. {row['model']:<28} weighted={float(row['weighted_score']):.6f}")
