from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("outputs/objective_v5_newmatch_all_models_final_v2")
PAGE_CSV = ROOT / "page_scores.csv"
LEADERBOARD_CSV = ROOT / "leaderboard.csv"
LOW_CSV = ROOT / "low_score_pages_top100.csv"


def value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    return None if raw in {"", "None", "nan"} else float(raw)


def weighted_score(row: dict[str, float | int | str | None]) -> float:
    text_quality = 1.0 - float(row["text_edit_distance"])
    layout_quality = float(row["layout_graph_similarity"])
    total = text_quality + layout_quality
    weights = 2.0
    if row["table_teds"] is not None:
        total += 0.3 * float(row["table_teds"])
        weights += 0.3
    return total / weights


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


with PAGE_CSV.open(encoding="utf-8", newline="") as f:
    page_rows = list(csv.DictReader(f))

by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in page_rows:
    by_model[row["model"]].append(row)

leaderboard: list[dict[str, object]] = []
low_rows: list[dict[str, object]] = []
for model, rows in sorted(by_model.items()):
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {
            "model": model,
            "page_id": row["page_id"],
            "table_teds": value(row, "table_teds"),
            "text_edit_distance": float(row["text_edit_distance"]),
            "layout_graph_similarity": float(row["layout_graph_similarity"]),
        }
        parsed_row["page_quality"] = weighted_score(parsed_row)
        parsed.append(parsed_row)

    table_values = [float(r["table_teds"]) for r in parsed if r["table_teds"] is not None]
    leaderboard.append(
        {
            "model": model,
            "pages": len(rows),
            "table_pages": len(table_values),
            "formula_pages": sum(1 for r in rows if r.get("formula_cdm", "") not in {"", "None"}),
            "table_teds": sum(table_values) / len(table_values) if table_values else None,
            "table_teds_s": (
                sum(float(row["table_teds_s"]) for row in rows if row.get("table_teds_s", "") not in {"", "None"})
                / sum(1 for row in rows if row.get("table_teds_s", "") not in {"", "None"})
                if any(row.get("table_teds_s", "") not in {"", "None"} for row in rows)
                else None
            ),
            "formula_cdm": (
                sum(float(row["formula_cdm"]) for row in rows if row.get("formula_cdm", "") not in {"", "None"})
                / sum(1 for row in rows if row.get("formula_cdm", "") not in {"", "None"})
                if any(row.get("formula_cdm", "") not in {"", "None"} for row in rows)
                else None
            ),
            "text_edit_distance": sum(float(r["text_edit_distance"]) for r in parsed) / len(parsed),
            "layout_graph_similarity": sum(float(r["layout_graph_similarity"]) for r in parsed) / len(parsed),
        }
    )
    for rank, row in enumerate(sorted(parsed, key=lambda r: float(r["page_quality"]))[:100], 1):
        row["low_score_rank"] = rank
        low_rows.append(row)

for row in leaderboard:
    row["weighted_score"] = weighted_score(row)
leaderboard.sort(key=lambda row: float(row["weighted_score"]), reverse=True)
for rank, row in enumerate(leaderboard, 1):
    row["rank"] = rank

write_csv(LEADERBOARD_CSV, leaderboard)
write_csv(LOW_CSV, low_rows)
print(f"leaderboard: {LEADERBOARD_CSV}")
print(f"low pages: {LOW_CSV}")
for row in leaderboard:
    print(row["rank"], row["model"], f"{float(row['weighted_score']):.4f}")
