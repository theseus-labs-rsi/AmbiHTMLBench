from __future__ import annotations

import csv
from pathlib import Path


def pf(v: object) -> float | None:
    if v in (None, ""):
        return None
    return float(v)


def page_weighted_score(row: dict[str, str]) -> float:
    table = pf(row.get("table_teds"))
    text_value = pf(row.get("text_edit_distance"))
    layout_value = pf(row.get("layout_graph_similarity"))
    text = 1.0 if text_value is None else text_value
    layout = 0.0 if layout_value is None else layout_value
    text_quality = max(0.0, min(1.0, 1.0 - text))
    total = text_quality + layout
    weight = 2.0
    if table is not None:
        total += 0.3 * table
        weight += 0.3
    return total / weight


def classify_low_row(row: dict[str, str]) -> str:
    table = pf(row.get("table_teds"))
    text_value = pf(row.get("text_edit_distance"))
    layout_value = pf(row.get("layout_graph_similarity"))
    text = 1.0 if text_value is None else text_value
    layout = 0.0 if layout_value is None else layout_value
    gt_tables = int(row.get("gt_tables") or 0)
    pred_tables = int(row.get("pred_tables") or 0)
    gt_text = int(row.get("gt_text_blocks") or 0)
    pred_text = int(row.get("pred_text_blocks") or 0)

    if gt_tables and pred_tables == 0:
        return "model_missing_table"
    if table is not None and table < 0.15 and pred_tables > 0:
        return "table_output_or_matching_poor"
    if text >= 0.80 and layout < 0.75:
        return "model_output_poor"
    if text >= 0.80 and layout >= 0.85:
        return "text_matching_or_reading_order_suspect"
    if pred_text > max(80, 3 * max(gt_text, 1)):
        return "over_fragmented_or_hallucinated_output"
    if gt_text > max(80, 3 * max(pred_text, 1)):
        return "under_generated_output"
    return "mixed_low_score"


def avg_present(rows: list[dict[str, str]], key: str) -> float | None:
    values = [pf(r.get(key)) for r in rows if pf(r.get(key)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.12g}"


def main() -> None:
    base = Path("outputs/objective_v5_all_models_maxden_v3")
    rerun_path = Path("outputs/objective_v5_rerun_lowpages/page_scores.csv")
    orig_path = base / "page_scores.csv"
    merged_path = base / "page_scores_merged_rerun.csv"
    leaderboard_path = base / "leaderboard_merged_rerun.csv"

    orig_rows = list(csv.DictReader(orig_path.open(encoding="utf-8")))
    rerun_rows = list(csv.DictReader(rerun_path.open(encoding="utf-8")))

    by_key: dict[tuple[str, str], dict[str, str]] = {
        (r["model"], r["page_id"]): dict(r) for r in orig_rows
    }
    for r in rerun_rows:
        key = (r["model"], r["page_id"])
        merged = by_key.get(key, {}).copy()
        merged.update(r)
        by_key[key] = merged

    rows = list(by_key.values())
    for r in rows:
        r["page_weighted_score"] = fmt(page_weighted_score(r))
        r["low_score_reason"] = classify_low_row(r)

    with merged_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    models = sorted({r["model"] for r in rows}, key=str.lower)
    leaderboard: list[dict[str, object]] = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        table_teds = avg_present(model_rows, "table_teds")
        table_teds_s = avg_present(model_rows, "table_teds_s")
        text_edit = avg_present(model_rows, "text_edit_distance")
        layout = avg_present(model_rows, "layout_graph_similarity")
        text_quality = max(0.0, min(1.0, 1.0 - (text_edit or 1.0)))
        weighted = (0.3 * (table_teds or 0.0) + text_quality + (layout or 0.0)) / 2.3
        leaderboard.append(
            {
                "model": model,
                "pages": len(model_rows),
                "table_pages": sum(1 for r in model_rows if pf(r.get("table_teds")) is not None),
                "table_teds": table_teds,
                "table_teds_s": table_teds_s,
                "text_edit_distance": text_edit,
                "layout_graph_similarity": layout,
                "weighted_score": weighted,
            }
        )

    leaderboard.sort(key=lambda r: float(r["weighted_score"]), reverse=True)
    for i, row in enumerate(leaderboard, 1):
        row["rank"] = i

    with leaderboard_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "model",
                "pages",
                "table_pages",
                "table_teds",
                "table_teds_s",
                "text_edit_distance",
                "layout_graph_similarity",
                "weighted_score",
            ],
        )
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({k: fmt(v) if isinstance(v, float) or v is None else v for k, v in row.items()})

    print(f"merged_rows={len(rows)} rerun_rows={len(rerun_rows)}")
    print(merged_path)
    print(leaderboard_path)
    for row in leaderboard:
        print(f"{row['rank']:2d}. {row['model']:<24} weighted={float(row['weighted_score']):.6f}")


if __name__ == "__main__":
    main()
