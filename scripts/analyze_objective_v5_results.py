from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def page_weighted_score(row: dict[str, Any]) -> float:
    table = parse_float(row.get("table_teds"))
    text_value = parse_float(row.get("text_edit_distance"))
    layout_value = parse_float(row.get("layout_graph_similarity"))
    text = 1.0 if text_value is None else text_value
    layout = 0.0 if layout_value is None else layout_value
    text_quality = max(0.0, min(1.0, 1.0 - text))
    total = text_quality + layout
    weight = 2.0
    if table is not None:
        total += 0.3 * table
        weight += 0.3
    return total / weight


def classify_low_row(row: dict[str, Any]) -> str:
    table = parse_float(row.get("table_teds"))
    text_value = parse_float(row.get("text_edit_distance"))
    layout_value = parse_float(row.get("layout_graph_similarity"))
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


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def main() -> None:
    root = Path("outputs/objective_v5_all_models_maxden_v3_by_model")
    out = Path("outputs/objective_v5_all_models_maxden_v3")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob("*/page_scores.csv")):
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["page_weighted_score"] = fmt(page_weighted_score(row))
                row["low_score_reason"] = classify_low_row(row)
                rows.append(row)

    fieldnames = list(rows[0].keys())
    with (out / "page_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    models = sorted({r["model"] for r in rows}, key=str.lower)
    leaderboard: list[dict[str, Any]] = []
    for model in models:
        mr = [r for r in rows if r["model"] == model]
        table_teds = mean([parse_float(r["table_teds"]) for r in mr if parse_float(r["table_teds"]) is not None])
        table_teds_s = mean([parse_float(r["table_teds_s"]) for r in mr if parse_float(r["table_teds_s"]) is not None])
        text_edit = mean([parse_float(r["text_edit_distance"]) if parse_float(r["text_edit_distance"]) is not None else 1.0 for r in mr])
        layout = mean([parse_float(r["layout_graph_similarity"]) if parse_float(r["layout_graph_similarity"]) is not None else 0.0 for r in mr])
        text_quality = max(0.0, min(1.0, 1.0 - (text_edit or 1.0)))
        weighted = (0.3 * (table_teds or 0.0) + text_quality + (layout or 0.0)) / 2.3
        leaderboard.append(
            {
                "model": model,
                "pages": len(mr),
                "table_pages": sum(1 for r in mr if parse_float(r["table_teds"]) is not None),
                "table_teds": fmt(table_teds),
                "table_teds_s": fmt(table_teds_s),
                "text_edit_distance": fmt(text_edit),
                "layout_graph_similarity": fmt(layout),
                "weighted_score": fmt(weighted),
            }
        )
    leaderboard.sort(key=lambda r: float(r["weighted_score"]), reverse=True)
    for i, row in enumerate(leaderboard, 1):
        row["rank"] = i
    lb_fields = ["rank", "model", "pages", "table_pages", "table_teds", "table_teds_s", "text_edit_distance", "layout_graph_similarity", "weighted_score"]
    with (out / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lb_fields)
        writer.writeheader()
        writer.writerows(leaderboard)

    low_rows: list[dict[str, Any]] = []
    for model in models:
        mr = [r for r in rows if r["model"] == model]
        low_rows.extend(sorted(mr, key=lambda r: float(r["page_weighted_score"]))[:100])
    low_fields = ["model", "page_id", "page_weighted_score", "low_score_reason", "table_teds", "table_teds_s", "text_edit_distance", "layout_graph_similarity", "gt_tables", "pred_tables", "gt_text_blocks", "pred_text_blocks", "html"]
    with (out / "lowest100_per_model.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=low_fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in low_fields} for r in low_rows])

    by_page: dict[str, list[dict[str, Any]]] = {}
    for r in low_rows:
        by_page.setdefault(r["page_id"], []).append(r)
    summary: list[dict[str, Any]] = []
    for page_id, pr in by_page.items():
        reasons: dict[str, int] = {}
        for r in pr:
            reasons[r["low_score_reason"]] = reasons.get(r["low_score_reason"], 0) + 1
        dominant_reason = max(reasons.items(), key=lambda x: x[1])[0]
        if len(pr) >= 6 and dominant_reason in {"text_matching_or_reading_order_suspect", "mixed_low_score", "table_output_or_matching_poor"}:
            action = "inspect_gt_or_matching"
        elif len(pr) >= 6:
            action = "hard_page_or_common_infer_failure"
        elif dominant_reason in {"model_missing_table", "model_output_poor", "over_fragmented_or_hallucinated_output", "under_generated_output"}:
            action = "rerun_model_infer"
        else:
            action = "inspect_sample_visualization"
        summary.append(
            {
                "page_id": page_id,
                "low_model_count": len(pr),
                "models": ";".join(sorted(r["model"] for r in pr)),
                "avg_page_weighted_score": fmt(mean([float(r["page_weighted_score"]) for r in pr])),
                "dominant_reason": dominant_reason,
                "reason_counts": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
                "recommended_action": action,
            }
        )
    summary.sort(key=lambda r: (-int(r["low_model_count"]), float(r["avg_page_weighted_score"])))
    with (out / "lowest_pages_cross_model_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["page_id", "low_model_count", "avg_page_weighted_score", "dominant_reason", "reason_counts", "recommended_action", "models"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    for filename, actions in [
        ("rerun_infer_candidates.csv", {"rerun_model_infer", "hard_page_or_common_infer_failure"}),
        ("inspect_gt_or_matching_candidates.csv", {"inspect_gt_or_matching", "inspect_sample_visualization"}),
    ]:
        selected = [r for r in summary if r["recommended_action"] in actions]
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            fields = ["page_id", "low_model_count", "avg_page_weighted_score", "dominant_reason", "reason_counts", "recommended_action", "models"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)

    print(f"rows={len(rows)} models={len(models)}")
    print(out / "leaderboard.csv")
    print(out / "lowest100_per_model.csv")
    print(out / "lowest_pages_cross_model_summary.csv")
    print(out / "rerun_infer_candidates.csv")
    print(out / "inspect_gt_or_matching_candidates.csv")


if __name__ == "__main__":
    main()
