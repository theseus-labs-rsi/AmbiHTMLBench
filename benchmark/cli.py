from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from .arena import load_records, preflight_arena, run_arena_from_env
from .arena_audit import write_analysis_report
from .html_extractor import extract_html_as_page
from .objective_v3 import evaluate_all_models_v3
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

    # RC1 subjective evaluation system.
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
