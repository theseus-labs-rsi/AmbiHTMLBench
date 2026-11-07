"""Resolve downloaded reference images by annotation ID, without network access."""
from __future__ import annotations

import json
from pathlib import Path


def load_image_paths(dataset_dir: str | Path) -> dict[str, str]:
    root = Path(dataset_dir).resolve()
    paths: dict[str, str] = {}
    for line in (root / "index" / "pages.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        page_id = f"{int(row['page_id']):04d}"
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Image path escapes dataset directory: {page_id}")
        if page_id in paths:
            raise ValueError(f"Duplicate dataset page ID: {page_id}")
        if not image.is_file():
            raise FileNotFoundError(f"Missing image for page {page_id}: {image}")
        paths[page_id] = str(image)
    return paths
