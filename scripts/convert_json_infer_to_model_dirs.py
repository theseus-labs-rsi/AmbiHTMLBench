from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FENCE_RE = re.compile(r"^\s*```(?:html)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def strip_html_fence(value: str) -> str:
    match = FENCE_RE.match(value)
    return match.group(1).strip() if match else value.strip()


def page_id_from_record(record: dict[str, Any]) -> str:
    if "ID" in record and record["ID"] not in (None, ""):
        return f"{int(record['ID']):04d}"
    image_file = str(record.get("image_file", ""))
    if len(image_file) >= 4 and image_file[:4].isdigit():
        return image_file[:4]
    raise ValueError(f"Cannot infer 4-digit page id from record: {record!r}")


def output_stem(record: dict[str, Any], page_id: str) -> str:
    image_file = str(record.get("image_file") or "").strip()
    if image_file:
        if len(image_file) >= 4 and image_file[:4].isdigit():
            return image_file
        return f"{page_id}__{Path(image_file).name}"
    return page_id


def convert_one(json_path: Path, out_root: Path, model_name: str | None = None) -> tuple[str, int]:
    model = model_name or json_path.stem
    model_dir = out_root / model
    raw_dir = model_dir / "raw"
    html_dir = model_dir / "html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{json_path} must contain a list of records")

    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"{json_path} contains a non-object record")
        page_id = page_id_from_record(record)
        if page_id in seen:
            raise ValueError(f"Duplicate page id {page_id} in {json_path}")
        seen.add(page_id)

        stem = output_stem(record, page_id)
        html = strip_html_fence(str(record.get("html") or ""))
        (html_dir / f"{stem}.html").write_text(html, encoding="utf-8", newline="\n")
        (raw_dir / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
    return model, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert batch JSON infer files into all_model_infer model folders.")
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("all_model_infer"))
    parser.add_argument("--model-names", help="Comma-separated output model folder names, aligned to json_paths.")
    args = parser.parse_args()

    names = args.model_names.split(",") if args.model_names else [None] * len(args.json_paths)
    if len(names) != len(args.json_paths):
        raise ValueError("--model-names count must match json_paths count")

    for json_path, model_name in zip(args.json_paths, names):
        model, count = convert_one(json_path, args.out_root, model_name.strip() if model_name else None)
        print(f"{model}: {count} records")


if __name__ == "__main__":
    main()
