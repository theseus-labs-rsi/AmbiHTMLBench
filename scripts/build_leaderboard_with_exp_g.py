from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASE_PAGE_SCORES = Path("outputs/objective_v5_all_models_plus_swift/page_scores.csv")
NEW_PAGE_SCORES = [Path("outputs/objective_v5_exp_G_clean/page_scores.csv")]
OUT_DIR = Path("outputs/objective_v5_all_models_plus_exp_G")


def pf(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(BASE_PAGE_SCORES)
    for path in NEW_PAGE_SCORES:
        rows.extend(read_rows(path))

    for row in rows:
        table = pf(row.get("table_teds"))
        text = pf(row.get("text_edit_distance"))
        layout = pf(row.get("layout_graph_similarity"))
        text_quality = max(0.0, min(1.0, 1.0 - (text if text is not None else 1.0)))
        total = text_quality + (layout if layout is not None else 0.0)
        denominator = 2.0
        if table is not None:
            total += 0.3 * table
            denominator += 0.3
        row["page_weighted_score"] = fmt(total / denominator)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT_DIR / "page_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    leaderboard: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}, key=str.lower):
        model_rows = [row for row in rows if row["model"] == model]
        teds = mean([v for row in model_rows if (v := pf(row.get("table_teds"))) is not None])
        teds_s = mean([v for row in model_rows if (v := pf(row.get("table_teds_s"))) is not None])
        text = mean([
            v if (v := pf(row.get("text_edit_distance"))) is not None else 1.0
            for row in model_rows
        ])
        layout = mean([
            v if (v := pf(row.get("layout_graph_similarity"))) is not None else 0.0
            for row in model_rows
        ])
        weighted = (0.3 * (teds or 0.0) + (1.0 - (text or 1.0)) + (layout or 0.0)) / 2.3
        leaderboard.append({
            "model": model,
            "pages": len(model_rows),
            "table_pages": sum(1 for row in model_rows if pf(row.get("table_teds")) is not None),
            "table_teds": fmt(teds),
            "table_teds_s": fmt(teds_s),
            "text_edit_distance": fmt(text),
            "layout_graph_similarity": fmt(layout),
            "weighted_score": fmt(weighted),
        })

    leaderboard.sort(key=lambda row: float(row["weighted_score"]), reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank

    lb_fields = [
        "rank", "model", "pages", "table_pages", "table_teds", "table_teds_s",
        "text_edit_distance", "layout_graph_similarity", "weighted_score",
    ]
    with (OUT_DIR / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lb_fields)
        writer.writeheader()
        writer.writerows(leaderboard)

    print(OUT_DIR / "leaderboard.csv")
    for row in leaderboard:
        print(f"{row['rank']:>2}. {row['model']:<28} weighted={float(row['weighted_score']):.6f}")


if __name__ == "__main__":
    main()
