from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from .arena import compute_bootstrap_elo_stats, compute_elo, compute_elo_stats, load_decisions_file, load_records, run_arena
from .html_extractor import extract_html_as_page
from .render import image_size
from .schema import dump_page_json, parse_page_json
from .scorer import evaluate_pages



def _cmd_extract_html(args: argparse.Namespace) -> None:
    page = extract_html_as_page(
        html_path=args.html,
        viewport_width=args.viewport_width or 1280,
        viewport_height=args.viewport_height or 1800,
    )
    dump_page_json(page, args.out)
    print(f"saved: {args.out}")
    print(f"elements: {len(page.elements)}, page={page.width:.1f}x{page.height:.1f}")


def _infer_page_id_from_html_name(html_path: str | Path) -> str | None:
    parts = Path(html_path).stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return None


def _find_record_for_html(records_json: str | None, html_path: str | Path) -> dict[str, Any] | None:
    if not records_json:
        return None
    html_name = Path(html_path).name
    html_stem = Path(html_path).stem
    for record in load_records(records_json):
        record_html = str(record.get("html_file", ""))
        if not record_html:
            continue
        if Path(record_html).name == html_name or Path(record_html).stem == html_stem:
            return record
    return None


def _resolve_record_asset(records_json: str, folder_name: str, file_name: str) -> Path:
    records_path = Path(records_json).resolve()
    return records_path.parent.parent / folder_name / file_name


def _resolve_gt_context(args: argparse.Namespace) -> tuple[str | None, Path | None]:
    record = _find_record_for_html(getattr(args, "records_json", None), args.html)

    page_id = args.gt_page_id
    if page_id is None and record is not None:
        page_id = str(record.get("id", ""))
    if page_id is None:
        page_id = _infer_page_id_from_html_name(args.html)

    gt_image_path: Path | None = None
    if args.gt_image:
        gt_image_path = Path(args.gt_image).resolve()
    elif record is not None and getattr(args, "records_json", None):
        candidate = _resolve_record_asset(args.records_json, "downloaded_images", str(record.get("image_file", "")))
        if candidate.exists():
            gt_image_path = candidate

    return page_id, gt_image_path


def _resolve_viewport(args: argparse.Namespace, gt_image_path: Path | None) -> tuple[int, int]:
    width = args.viewport_width
    height = args.viewport_height
    if gt_image_path is not None and gt_image_path.exists():
        img_w, img_h = image_size(gt_image_path)
        width = width or img_w
        height = height or img_h
    return int(width or 1280), int(height or 1800)


def _save_rows(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _cmd_eval_json_html(args: argparse.Namespace) -> None:
    gt_page_id, gt_image_path = _resolve_gt_context(args)
    gt_page_size = image_size(gt_image_path) if gt_image_path is not None and gt_image_path.exists() else None
    viewport_width, viewport_height = _resolve_viewport(args, gt_image_path)

    gt = parse_page_json(args.gt_json, page_id=gt_page_id, page_size=gt_page_size)
    pred = extract_html_as_page(
        html_path=args.html,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    result = evaluate_pages(
        gt_page=gt,
        pred_page=pred,
        iou_threshold=args.iou_threshold,
        text_threshold=args.text_threshold,
        weight_content=args.weight_content,
        weight_layout=args.weight_layout,
    )

    payload = {
        "gt_json": str(Path(args.gt_json).resolve()),
        "gt_page_id": gt.meta.get("page_id", gt_page_id),
        "gt_image": str(gt_image_path) if gt_image_path is not None else None,
        "html": str(Path(args.html).resolve()),
        "matched": result.matched,
        "gt_total": result.gt_total,
        "pred_total": result.pred_total,
        "content_score": result.content_score,
        "tag_score": result.tag_score,
        "layout_score": result.layout_score,
        "page_iou": result.page_iou,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "total_score": result.total_score,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {args.out}")



def _collect_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []

    if args.pairs_jsonl:
        for line in Path(args.pairs_jsonl).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(
                {
                    "id": str(obj.get("id") or Path(obj["html"]).stem),
                    "gt": str(obj["gt"]),
                    "html": str(obj["html"]),
                    "gt_page_id": str(obj["gt_page_id"]) if obj.get("gt_page_id") is not None else None,
                    "gt_image": str(obj["gt_image"]) if obj.get("gt_image") is not None else None,
                }
            )
        return pairs

    if args.gt_dir and args.html_dir:
        gt_map = {p.stem: p for p in Path(args.gt_dir).glob("*.json")}
        for h in Path(args.html_dir).glob("*.html"):
            if h.stem in gt_map:
                pairs.append({"id": h.stem, "gt": str(gt_map[h.stem]), "html": str(h), "gt_page_id": None, "gt_image": None})
        return sorted(pairs, key=lambda x: x["id"])

    raise ValueError("must provide --pairs-jsonl or (--gt-dir and --html-dir)")



def _cmd_eval_dataset_json_html(args: argparse.Namespace) -> None:
    pairs = _collect_pairs(args)
    rows: list[dict[str, Any]] = []

    for p in pairs:
        gt_image_path = Path(p["gt_image"]).resolve() if p.get("gt_image") else None
        gt_page_size = image_size(gt_image_path) if gt_image_path is not None and gt_image_path.exists() else None
        viewport_width, viewport_height = _resolve_viewport(args, gt_image_path)
        gt = parse_page_json(p["gt"], page_id=p.get("gt_page_id"), page_size=gt_page_size)
        pred = extract_html_as_page(
            html_path=p["html"],
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        r = evaluate_pages(
            gt_page=gt,
            pred_page=pred,
            iou_threshold=args.iou_threshold,
            text_threshold=args.text_threshold,
            weight_content=args.weight_content,
            weight_layout=args.weight_layout,
        )

        rows.append(
            {
                "id": p["id"],
                "gt": p["gt"],
                "gt_page_id": gt.meta.get("page_id", p.get("gt_page_id")),
                "gt_image": str(gt_image_path) if gt_image_path is not None else None,
                "html": p["html"],
                "matched": r.matched,
                "gt_total": r.gt_total,
                "pred_total": r.pred_total,
                "content_score": r.content_score,
                "tag_score": r.tag_score,
                "layout_score": r.layout_score,
                "page_iou": r.page_iou,
                "f1": r.f1,
                "total_score": r.total_score,
            }
        )

    _save_rows(rows, args.out)

    avg = sum(r["total_score"] for r in rows) / max(len(rows), 1)
    print(f"pairs={len(rows)}, avg_total_score={avg:.4f}, out={args.out}")


def _cmd_eval_quality_inspection(args: argparse.Namespace) -> None:
    records = load_records(args.records_json)
    records_path = Path(args.records_json).resolve()
    base_dir = records_path.parent.parent
    rows: list[dict[str, Any]] = []

    for record in records:
        if record.get("status") != "ok":
            continue

        gt_image_path = base_dir / "downloaded_images" / str(record["image_file"])
        gt_page_size = image_size(gt_image_path) if gt_image_path.exists() else None
        viewport_width, viewport_height = _resolve_viewport(args, gt_image_path if gt_image_path.exists() else None)
        html_path = base_dir / "rendered_html" / str(record["html_file"])

        gt = parse_page_json(args.gt_json, page_id=str(record["id"]), page_size=gt_page_size)
        pred = extract_html_as_page(
            html_path=html_path,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        r = evaluate_pages(
            gt_page=gt,
            pred_page=pred,
            iou_threshold=args.iou_threshold,
            text_threshold=args.text_threshold,
            weight_content=args.weight_content,
            weight_layout=args.weight_layout,
        )

        rows.append(
            {
                "index": record["index"],
                "id": record["id"],
                "image_file": str(gt_image_path),
                "html": str(html_path),
                "matched": r.matched,
                "gt_total": r.gt_total,
                "pred_total": r.pred_total,
                "content_score": r.content_score,
                "tag_score": r.tag_score,
                "layout_score": r.layout_score,
                "page_iou": r.page_iou,
                "f1": r.f1,
                "total_score": r.total_score,
            }
        )

    _save_rows(rows, args.out)
    avg = sum(r["total_score"] for r in rows) / max(len(rows), 1)
    print(f"pages={len(rows)}, avg_total_score={avg:.4f}, out={args.out}")


def _cmd_arena_judge(args: argparse.Namespace) -> None:
    api_base = args.api_base or os.getenv("VLM_API_BASE", "https://api.vectorengine.cn/v1")
    api_key = args.api_key or os.getenv("VLM_API_KEY")
    if not api_key:
        raise ValueError("missing api key. pass --api-key or set VLM_API_KEY")

    decisions = run_arena(
        records_json=args.records_json,
        base_dir=args.base_dir,
        render_out_dir=args.render_out,
        decisions_out=args.decisions_out,
        api_base=api_base,
        api_key=api_key,
        judge_model=args.judge_model,
        rounds=args.rounds,
        seed=args.seed,
    )
    print(f"decisions={len(decisions)}, out={args.decisions_out}")



def _cmd_arena_elo(args: argparse.Namespace) -> None:
    decisions = load_decisions_file(args.decisions_json)

    if args.include_stats and args.include_bootstrap_stats:
        raise ValueError(
            "--include-stats and --include-bootstrap-stats "
            "cannot be used together"
        )

    common_args = {
        "k_factor": args.k_factor,
        "init_rating": args.init_rating,
        "mode": args.elo_mode,
        "shuffle_repeats": args.shuffle_repeats,
        "seed": args.elo_seed,
        "round_aggregation": args.round_aggregation,
        "consistency_floor": args.consistency_floor,
        "sample_balance": args.sample_balance,
        "batch_iterations": args.batch_iterations,
        "batch_tolerance": args.batch_tolerance,
        "batch_regularization": args.batch_regularization,
    }

    if args.include_bootstrap_stats:
        stats = compute_bootstrap_elo_stats(
            decisions,
            bootstrap_repeats=args.bootstrap_repeats,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.bootstrap_confidence,
            **common_args,
        )
        rows = [
            {
                "model": model,
                "elo": round(item.elo, 2),
                "bootstrap_mean": round(item.bootstrap_mean, 4),
                "bootstrap_std": round(item.bootstrap_std, 4),
                "ci_lower": round(item.ci_lower, 4),
                "ci_upper": round(item.ci_upper, 4),
                "mean_rank": round(item.mean_rank, 4),
                "rank_stability": round(item.rank_stability, 4),
                "top_rank_probability": round(
                    item.top_rank_probability,
                    4,
                ),
                "bootstrap_repeats": item.bootstrap_repeats,
                "sample_count": item.sample_count,
            }
            for model, item in stats.items()
        ]
    elif args.include_stats:
        stats = compute_elo_stats(decisions, **common_args)
        rows = [
            {
                "model": model,
                "elo": round(item.elo, 2),
                "elo_std": round(item.elo_std, 4),
                "mean_rank": round(item.mean_rank, 4),
                "rank_stability": round(item.rank_stability, 4),
                "top_rank_probability": round(
                    item.top_rank_probability,
                    4,
                ),
                "trial_count": item.trial_count,
            }
            for model, item in stats.items()
        ]
    else:
        elo = compute_elo(decisions, **common_args)
        rows = [
            {
                "model": model,
                "elo": round(rating, 2),
            }
            for model, rating in elo.items()
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Elo ranking: mode={args.elo_mode}, "
        f"round_aggregation={args.round_aggregation}, "
        f"sample_balance={args.sample_balance}"
    )

    for index, row in enumerate(rows, start=1):
        if args.include_bootstrap_stats:
            print(
                f"{index:>2}. {row['model']:<20} "
                f"{row['elo']:.2f} "
                f"bootstrap_std={row['bootstrap_std']:.4f} "
                f"CI=[{row['ci_lower']:.2f}, "
                f"{row['ci_upper']:.2f}] "
                f"top_prob={row['top_rank_probability']:.4f}"
            )
        elif args.include_stats:
            print(
                f"{index:>2}. {row['model']:<20} "
                f"{row['elo']:.2f} "
                f"std={row['elo_std']:.4f} "
                f"mean_rank={row['mean_rank']:.4f} "
                f"stability={row['rank_stability']:.4f}"
            )
        else:
            print(
                f"{index:>2}. {row['model']:<20} "
                f"{row['elo']:.2f}"
            )

    print(f"saved: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HTML reconstruction benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract-html", help="extract html page to normalized json")
    p.add_argument("--html", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--viewport-width", type=int)
    p.add_argument("--viewport-height", type=int)
    p.set_defaults(func=_cmd_extract_html)

    p = sub.add_parser("eval-json-html", help="evaluate one gt json vs one predicted html")
    p.add_argument("--gt-json", required=True)
    p.add_argument("--html", required=True)
    p.add_argument("--out")
    p.add_argument("--gt-page-id")
    p.add_argument("--gt-image")
    p.add_argument("--records-json")
    p.add_argument("--viewport-width", type=int)
    p.add_argument("--viewport-height", type=int)
    p.add_argument("--iou-threshold", type=float, default=0.1)
    p.add_argument("--text-threshold", type=float, default=0.0)
    p.add_argument("--weight-content", type=float, default=0.5)
    p.add_argument("--weight-layout", type=float, default=0.5)
    p.set_defaults(func=_cmd_eval_json_html)

    p = sub.add_parser("eval-dataset-json-html", help="evaluate a dataset of gt json/html pairs")
    p.add_argument("--pairs-jsonl")
    p.add_argument("--gt-dir")
    p.add_argument("--html-dir")
    p.add_argument("--out", required=True)
    p.add_argument("--viewport-width", type=int)
    p.add_argument("--viewport-height", type=int)
    p.add_argument("--iou-threshold", type=float, default=0.1)
    p.add_argument("--text-threshold", type=float, default=0.0)
    p.add_argument("--weight-content", type=float, default=0.5)
    p.add_argument("--weight-layout", type=float, default=0.5)
    p.set_defaults(func=_cmd_eval_dataset_json_html)

    p = sub.add_parser("eval-quality-inspection", help="evaluate Quality_inspection records against the aggregated gt json")
    p.add_argument("--gt-json", required=True)
    p.add_argument("--records-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--viewport-width", type=int)
    p.add_argument("--viewport-height", type=int)
    p.add_argument("--iou-threshold", type=float, default=0.1)
    p.add_argument("--text-threshold", type=float, default=0.0)
    p.add_argument("--weight-content", type=float, default=0.5)
    p.add_argument("--weight-layout", type=float, default=0.5)
    p.set_defaults(func=_cmd_eval_quality_inspection)

    p = sub.add_parser("arena-judge", help="render htmls and run VLM arena judging")
    p.add_argument("--records-json", default="ocr_html_experiment_four_models/logs/records.json")
    p.add_argument("--base-dir", default="ocr_html_experiment_four_models")
    p.add_argument("--render-out", default="outputs/arena_renders")
    p.add_argument("--decisions-out", default="outputs/arena_decisions.json")
    p.add_argument("--api-base")
    p.add_argument("--api-key")
    p.add_argument("--judge-model", default=os.getenv("VLM_JUDGE_MODEL", "gpt-4o"))
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_arena_judge)

    p = sub.add_parser("arena-elo", help="compute elo from arena decisions")
    p.add_argument("--decisions-json", default="outputs/arena_decisions.json")
    p.add_argument("--out", default="outputs/arena_elo.json")
    p.add_argument("--k-factor", type=float, default=24.0)
    p.add_argument("--init-rating", type=float, default=1000.0)
    p.add_argument(
        "--elo-mode",
        choices=["online", "shuffle_ensemble", "batch"],
        default="shuffle_ensemble",
        help=(
            "online uses the stored evidence order; "
            "shuffle_ensemble averages deterministic shuffled "
            "forward/reverse trials"
        ),
    )
    p.add_argument(
        "--shuffle-repeats",
        type=int,
        default=64,
        help="number of shuffled orders used by shuffle_ensemble",
    )
    p.add_argument(
        "--elo-seed",
        type=int,
        default=42,
        help="random seed used by shuffle_ensemble",
    )
    p.add_argument(
        "--round-aggregation",
        choices=["none", "sample_pair"],
        default="sample_pair",
        help="aggregate repeated rounds for the same sample/model pair",
    )
    p.add_argument(
        "--consistency-floor",
        type=float,
        default=0.25,
        help="minimum reliability retained for conflicting round evidence",
    )
    p.add_argument(
        "--sample-balance",
        choices=["none", "per_model_exposure"],
        default="per_model_exposure",
        help=(
            "normalize each model's total Elo exposure "
            "within a sample"
        ),
    )
    p.add_argument(
        "--batch-iterations",
        type=int,
        default=1500,
        help="maximum synchronous batch Elo iterations",
    )
    p.add_argument(
        "--batch-tolerance",
        type=float,
        default=1e-5,
        help="batch Elo convergence tolerance",
    )
    p.add_argument(
        "--batch-regularization",
        type=float,
        default=0.05,
        help="L2 shrinkage toward the initial Elo rating",
    )
    p.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=500,
        help="number of sample bootstrap replicates",
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="random seed for sample bootstrap",
    )
    p.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="percentile confidence interval level",
    )
    p.add_argument(
        "--include-bootstrap-stats",
        action="store_true",
        help="output sample bootstrap Elo uncertainty",
    )
    p.add_argument(
        "--include-stats",
        action="store_true",
        help="output Elo uncertainty and rank stability statistics",
    )
    p.set_defaults(func=_cmd_arena_elo)


    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
