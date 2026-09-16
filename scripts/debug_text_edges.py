from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmark.objective_v3 import (
    evaluate_page_v3,
    _overlap_min_area,
    _partial_text_similarity,
    extract_html_with_page,
    load_gt_v3,
)


def short(text: str, n: int = 90) -> str:
    value = " ".join(text.split())
    return value if len(value) <= n else value[: n - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json", default="AmbiHTMLBench_GT_v5_1115.json")
    parser.add_argument("--html", required=True)
    parser.add_argument("--page-id", required=True)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=1800)
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    gt_page = load_gt_v3(args.gt_json)[args.page_id]
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": args.viewport_width, "height": args.viewport_height})
        pred_page = extract_html_with_page(
            page,
            Path(args.html),
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
        )
        browser.close()

    _, debug = evaluate_page_v3(gt_page, pred_page)
    gt_text = debug["gt_text"]
    pred_text = debug["pred_text"]
    print(f"GT text blocks: {len(gt_text)}")
    for i, obj in enumerate(gt_text):
        print(f"G{i:02d} box=({obj.bbox.x:.1f},{obj.bbox.y:.1f},{obj.bbox.w:.1f},{obj.bbox.h:.1f}) {short(obj.text)}")
    print(f"Pred text blocks: {len(pred_text)}")
    for j, obj in enumerate(pred_text):
        print(f"P{j:02d} box=({obj.bbox.x:.1f},{obj.bbox.y:.1f},{obj.bbox.w:.1f},{obj.bbox.h:.1f}) {short(obj.text)}")

    edges: list[tuple[float, float, float, int, int]] = []
    for i, g in enumerate(gt_text):
        for j, pred in enumerate(pred_text):
            ov = _overlap_min_area(g.bbox, pred.bbox)
            ts = _partial_text_similarity(g.text, pred.text) / 100.0
            if ov > 0.9 or ts > 0.66:
                edges.append((max(ov, ts), ov, ts, i, j))
    edges.sort(reverse=True)
    print(f"Candidate edges: {len(edges)}")
    for sim, ov, ts, i, j in edges[: args.top]:
        print(f"G{i:02d}-P{j:02d} sim={sim:.3f} ov={ov:.3f} ts={ts:.3f} | {short(gt_text[i].text, 55)} || {short(pred_text[j].text, 55)}")


if __name__ == "__main__":
    main()
