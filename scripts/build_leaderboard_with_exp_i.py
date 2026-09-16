from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASE_PAGE_SCORES = Path("outputs/objective_v5_all_models_plus_exp_G/page_scores.csv")
NEW_PAGE_SCORES = Path("outputs/objective_v5_exp_I/page_scores.csv")
OUT_DIR = Path("outputs/objective_v5_all_models_plus_exp_I")


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def page_weighted_score(row: dict[str, Any]) -> float:
    table = parse_float(row.get("table_teds"))
    text = parse_float(row.get("text_edit_distance"))
    layout = parse_float(row.get("layout_graph_similarity"))
    total = max(0.0, min(1.0, 1.0 - (text if text is not None else 1.0)))
    total += layout if layout is not None else 0.0
    denominator = 2.0
    if table is not None:
        total += 0.3 * table
        denominator += 0.3
    return total / denominator


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(BASE_PAGE_SCORES) + read_rows(NEW_PAGE_SCORES)

    for row in rows:
        row["page_weighted_score"] = fmt(page_weighted_score(row))

    page_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in page_fields:
                page_fields.append(key)
    with (OUT_DIR / "page_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=page_fields)
        writer.writeheader()
        writer.writerows(rows)

    leaderboard: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}, key=str.lower):
        model_rows = [row for row in rows if row["model"] == model]
        table_teds = mean([
            value for row in model_rows
            if (value := parse_float(row.get("table_teds"))) is not None
        ])
        table_teds_s = mean([
            value for row in model_rows
            if (value := parse_float(row.get("table_teds_s"))) is not None
        ])
        text_edit = mean([
            value if (value := parse_float(row.get("text_edit_distance"))) is not None else 1.0
            for row in model_rows
        ])
        layout = mean([
            value if (value := parse_float(row.get("layout_graph_similarity"))) is not None else 0.0
            for row in model_rows
        ])
        weighted = (
            0.3 * (table_teds or 0.0)
            + (1.0 - (text_edit if text_edit is not None else 1.0))
            + (layout if layout is not None else 0.0)
        ) / 2.3
        leaderboard.append({
            "model": model,
            "pages": len(model_rows),
            "table_pages": sum(
                1 for row in model_rows
                if parse_float(row.get("table_teds")) is not None
            ),
            "table_teds": fmt(table_teds),
            "table_teds_s": fmt(table_teds_s),
            "text_edit_distance": fmt(text_edit),
            "layout_graph_similarity": fmt(layout),
            "weighted_score": fmt(weighted),
        })

    leaderboard.sort(key=lambda row: float(row["weighted_score"]), reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank

    fields = [
        "rank", "model", "pages", "table_pages", "table_teds", "table_teds_s",
        "text_edit_distance", "layout_graph_similarity", "weighted_score",
    ]
    with (OUT_DIR / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(leaderboard)

    print(OUT_DIR / "leaderboard.csv")
    for row in leaderboard:
        print(
            f"{row['rank']:>2}. {row['model']:<28} "
            f"weighted={float(row['weighted_score']):.6f}"
        )


if __name__ == "__main__":
    main()
