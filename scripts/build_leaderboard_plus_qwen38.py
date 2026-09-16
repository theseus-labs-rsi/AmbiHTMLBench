from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


OLD_PAGE_SCORES = Path("outputs/objective_v5_all_models_maxden_v3/page_scores.csv")
NEW_PAGE_SCORES = [
    Path("outputs/objective_v5_qwen38_base/page_scores.csv"),
    Path("outputs/objective_v5_qwen38_lora/page_scores.csv"),
]
OUT_DIR = Path("outputs/objective_v5_all_models_plus_qwen38")


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def page_weighted_score(row: dict[str, Any]) -> float:
    table = parse_float(row.get("table_teds"))
    text_value = parse_float(row.get("text_edit_distance"))
    layout_value = parse_float(row.get("layout_graph_similarity"))
    text_quality = max(0.0, min(1.0, 1.0 - (text_value if text_value is not None else 1.0)))
    layout = layout_value if layout_value is not None else 0.0
    total = text_quality + layout
    weight = 2.0
    if table is not None:
        total += 0.3 * table
        weight += 0.3
    return total / weight


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in [OLD_PAGE_SCORES, *NEW_PAGE_SCORES]:
        rows.extend(read_rows(path))

    for row in rows:
        row["page_weighted_score"] = fmt(page_weighted_score(row))

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with (OUT_DIR / "page_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    leaderboard: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}, key=str.lower):
        model_rows = [r for r in rows if r["model"] == model]
        table_teds = mean([v for r in model_rows if (v := parse_float(r.get("table_teds"))) is not None])
        table_teds_s = mean([v for r in model_rows if (v := parse_float(r.get("table_teds_s"))) is not None])
        text_edit = mean(
            [
                value if (value := parse_float(r.get("text_edit_distance"))) is not None else 1.0
                for r in model_rows
            ]
        )
        layout = mean(
            [
                value if (value := parse_float(r.get("layout_graph_similarity"))) is not None else 0.0
                for r in model_rows
            ]
        )
        text_quality = max(0.0, min(1.0, 1.0 - (text_edit if text_edit is not None else 1.0)))
        weighted = (0.3 * (table_teds or 0.0) + text_quality + (layout or 0.0)) / 2.3
        leaderboard.append(
            {
                "model": model,
                "pages": len(model_rows),
                "table_pages": sum(1 for r in model_rows if parse_float(r.get("table_teds")) is not None),
                "table_teds": fmt(table_teds),
                "table_teds_s": fmt(table_teds_s),
                "text_edit_distance": fmt(text_edit),
                "layout_graph_similarity": fmt(layout),
                "weighted_score": fmt(weighted),
            }
        )

    leaderboard.sort(key=lambda r: float(r["weighted_score"]), reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank

    fields = [
        "rank",
        "model",
        "pages",
        "table_pages",
        "table_teds",
        "table_teds_s",
        "text_edit_distance",
        "layout_graph_similarity",
        "weighted_score",
    ]
    with (OUT_DIR / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(leaderboard)

    print(OUT_DIR / "leaderboard.csv")
    for row in leaderboard:
        print(
            f"{row['rank']:>2}. {row['model']:<28} "
            f"weighted={float(row['weighted_score']):.6f} "
            f"TEDS={float(row['table_teds']):.4f} "
            f"text_edit={float(row['text_edit_distance']):.4f} "
            f"layout={float(row['layout_graph_similarity']):.4f}"
        )


if __name__ == "__main__":
    main()
