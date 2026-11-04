from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .arena import preflight_arena, run_arena_from_env
from .arena_audit import write_analysis_report
from .objective_v3 import evaluate_all_models_v3


def _cmd_arena_preflight(args: argparse.Namespace) -> None:
    resolved = preflight_arena(
        manifest_json=args.manifest,
        out_dir=args.out_dir,
        safe_request_bytes=args.safe_request_bytes,
        auto_replace=not args.no_auto_replace,
    )
    path = Path(args.out_dir).resolve() / "resolved_manifest.json"
    print("Subjective Arena preflight: PASS")
    print(f"resolved pages={len(resolved['pages'])}")
    print(f"physical pair sets={len(resolved['pair_sets'])}")
    print(f"resolved manifest: {path}")


def _cmd_arena_run(args: argparse.Namespace) -> None:
    api_base = args.api_base or os.getenv("VLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("VLM_API_KEY")
    if not api_base:
        raise ValueError("missing API base. pass --api-base or set VLM_API_BASE / OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("missing api key. pass --api-key or set VLM_API_KEY")
    decisions = run_arena_from_env(
        resolved_manifest_json=args.manifest,
        decisions_out=args.decisions_out,
        api_base=api_base,
        api_key=api_key,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
    )
    print(f"Arena complete: {len(decisions)} raw dual-orientation judgments")
    print(f"saved: {args.decisions_out}")


def _cmd_arena_report(args: argparse.Namespace) -> None:
    payload = write_analysis_report(
        decisions_json=args.decisions_json,
        resolved_manifest_json=args.manifest,
        out_dir=args.out_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print("Primary subjective ranking:")
    for row in payload["ranking"]:
        print(f"{row['rank']:>2}. {row['model']:<28} {row['elo']:.2f}")
    print(f"report: {Path(args.out_dir).resolve() / 'report.md'}")


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

    # Historical Arena v1 commands; frozen OLS5 uses frozen_proxy_cli.py.
    p = sub.add_parser("arena-preflight", help="render and validate a subjective Arena manifest without Judge API calls")
    p.add_argument("--manifest", required=True, help="public subjective arena manifest JSON")
    p.add_argument("--out-dir", default="outputs/arena_preflight")
    p.add_argument("--safe-request-bytes", type=int, default=28_000_000)
    p.add_argument("--no-auto-replace", action="store_true", help="fail instead of using reserve pages")
    p.set_defaults(func=_cmd_arena_preflight)

    p = sub.add_parser("arena-run", help="run frozen human-visual Judge with canonical+swapped orientation")
    p.add_argument("--manifest", required=True, help="preflight resolved_manifest.json")
    p.add_argument("--decisions-out", default="outputs/arena/decisions.json")
    p.add_argument("--api-base", help="OpenAI-compatible API base URL; or set VLM_API_BASE / OPENAI_BASE_URL")
    p.add_argument("--api-key")
    p.add_argument("--judge-model", default=os.getenv("VLM_JUDGE_MODEL", "claude-sonnet-4-6"))
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=_cmd_arena_run)

    p = sub.add_parser("arena-report", help="collapse reliability pairs, fit Batch BT, bootstrap and write report")
    p.add_argument("--manifest", required=True, help="preflight resolved_manifest.json")
    p.add_argument("--decisions-json", required=True)
    p.add_argument("--out-dir", default="outputs/arena/report")
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20260812)
    p.set_defaults(func=_cmd_arena_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
