#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def read_ids(path: str | None) -> list[str]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8-sig")
    vals = re.findall(r"\d{1,4}", text)
    return [f"{int(x):04d}" for x in vals]


def find_reference(reference_dir: Path, sid: str) -> Path:
    matches = [p for p in reference_dir.iterdir() if p.is_file() and p.stem == sid]
    if len(matches) != 1:
        raise ValueError(f"expected one Reference image for {sid}, found {len(matches)}")
    return matches[0]


def find_html(html_dir: Path, sid: str) -> Path:
    matches = [p for p in html_dir.glob(f"{sid}*.html") if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one HTML in {html_dir} for {sid}, found {len(matches)}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description="Build an RC1 subjective Arena manifest from directories")
    ap.add_argument("--models-json", required=True, help='JSON mapping {"model": "html_dir"}')
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--page-ids", required=True, help="text/json/csv-like file containing final page ids")
    ap.add_argument("--reserve-ids")
    ap.add_argument("--gt-json")
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    base = out.parent
    model_dirs_raw = json.loads(Path(args.models_json).read_text(encoding="utf-8-sig"))
    model_dirs = {m: Path(d).resolve() for m, d in model_dirs_raw.items()}
    refs = Path(args.reference_dir).resolve()
    gt = Path(args.gt_json).resolve() if args.gt_json else None

    def build_page(sid: str):
        page = {
            "sample_id": sid,
            "reference_image": str(find_reference(refs, sid)),
            "candidates": {m: str(find_html(d, sid)) for m, d in model_dirs.items()},
        }
        if gt:
            page["gt_json"] = str(gt)
            page["gt_page_id"] = str(int(sid))
        else:
            page["active_axes"] = []
        return page

    final_ids = read_ids(args.page_ids)
    reserve_ids = read_ids(args.reserve_ids)
    payload = {
        "version": "subjective_arena_manifest_v1",
        "models": list(model_dirs),
        "judge": {"model": "claude-sonnet-4-6", "temperature": 0},
        "schedule": {"repeats": args.repeats, "seed": args.seed},
        "pages": [build_page(sid) for sid in final_ids],
        "reserve_pages": [build_page(sid) for sid in reserve_ids],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(f"models={len(model_dirs)}, pages={len(final_ids)}, reserves={len(reserve_ids)}")


if __name__ == "__main__":
    main()
