from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .objective_v3 import evaluate_all_models_v3


def _cmd_eval_objective_v3(args: argparse.Namespace) -> None:
    models = {x.strip() for x in args.models.split(",") if x.strip()} if args.models else None
    page_ids = {f"{int(x.strip()):04d}" for x in args.page_ids.split(",") if x.strip()} if args.page_ids else None
    leaderboard, page_rows = evaluate_all_models_v3(
        gt_json=args.gt_json,
        infer_dir=args.infer_dir,
        out_dir=args.out_dir,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        visualizations=not args.no_visualizations,
        max_pages=args.max_pages,
        models=models,
        page_ids=page_ids,
        partial_interval=args.partial_interval,
        dataset_dir=args.dataset_dir,
    )
    print(f"Objective v3 complete: models={len(leaderboard)}, page_rows={len(page_rows)}")
    print(f"page scores: {Path(args.out_dir).resolve() / 'page_scores.csv'}")
    print(f"leaderboard: {Path(args.out_dir).resolve() / 'leaderboard.csv'}")
    if not args.no_visualizations:
        print(f"visualizations: {Path(args.out_dir).resolve() / 'visualizations'}")
    def fmt_metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"
    for row in leaderboard:
        print(
            f"{row['rank']:>2}. {row['model']:<28} "
            f"TEDS={fmt_metric(row['table_teds'])} "
            f"TEDS-S={fmt_metric(row['table_teds_s'])} "
            f"CDM={fmt_metric(row['formula_cdm'])} "
            f"text_edit={row['text_edit_distance']:.4f} "
            f"layout_graph={row['layout_graph_similarity']:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AmbiHTMLBench HTML reconstruction benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Objective benchmark: fixed AmbiHTMLBench_GT_v5_1115.json schema,
    # all_model_infer model folders, and the new page-level metrics only.
    p = sub.add_parser("eval-objective-v3", help="evaluate all all_model_infer outputs with AmbiHTMLBench objective metrics")
    p.add_argument("--gt-json", default="AmbiHTMLBench_GT_v5_1115.json")
    p.add_argument("--dataset-dir", help="downloaded Hugging Face dataset root containing index/pages.jsonl and pages/")
    p.add_argument("--infer-dir", default="all_model_infer")
    p.add_argument("--out-dir", default="outputs/objective_v3")
    p.add_argument("--viewport-width", type=int, default=1280)
    p.add_argument("--viewport-height", type=int, default=1800)
    p.add_argument("--max-pages", type=int, help="debug limit per model")
    p.add_argument("--models", help="comma-separated model folder names to evaluate")
    p.add_argument("--page-ids", help="comma-separated 4-digit page IDs to evaluate")
    p.add_argument("--partial-interval", type=int, default=25, help="write partial page scores every N completed pages")
    p.add_argument("--no-visualizations", action="store_true")
    p.set_defaults(func=_cmd_eval_objective_v3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
