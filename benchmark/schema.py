from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from .utils import BBox, IGNORE_GT_TAGS, clamp01, clean_text, normalize_semantic_tag


@dataclass
class Element:
    tag: str
    text: str
    bbox: BBox
    source_id: str = ""
    raw_html: str = ""
    latex: str = ""
    visual_bbox: BBox | None = None


@dataclass
class PageData:
    width: float
    height: float
    elements: list[Element]
    meta: dict[str, Any]


def _bbox_from_points(points: list[dict[str, Any]], page_w: float, page_h: float) -> BBox:
    xs = [float(p["x"]) for p in points if "x" in p]
    ys = [float(p["y"]) for p in points if "y" in p]
    if not xs or not ys:
        raise ValueError(f"invalid points: {points}")
    return _bbox_from_dict(
        {
            "x": min(xs),
            "y": min(ys),
            "w": max(xs) - min(xs),
            "h": max(ys) - min(ys),
        },
        page_w=page_w,
        page_h=page_h,
    )


def _bbox_from_dict(raw: dict[str, Any], page_w: float, page_h: float) -> BBox:
    x = raw.get("x", raw.get("left"))
    y = raw.get("y", raw.get("top"))
    w = raw.get("w", raw.get("width"))
    h = raw.get("h", raw.get("height"))

    if x is None and raw.get("x1") is not None:
        x = raw.get("x1")
    if y is None and raw.get("y1") is not None:
        y = raw.get("y1")
    if w is None and raw.get("x2") is not None and x is not None:
        w = float(raw.get("x2")) - float(x)
    if h is None and raw.get("y2") is not None and y is not None:
        h = float(raw.get("y2")) - float(y)
    if (x is None or y is None or w is None or h is None) and isinstance(raw.get("points"), list):
        return _bbox_from_points(raw["points"], page_w=page_w, page_h=page_h)

    if x is None or y is None or w is None or h is None:
        raise ValueError(f"invalid bbox: {raw}")

    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    # Absolute coordinates fallback.
    if x > 1.0 or y > 1.0 or w > 1.0 or h > 1.0:
        if page_w <= 0 or page_h <= 0:
            raise ValueError("absolute bbox detected but page size missing")
        x = x / page_w
        y = y / page_h
        w = w / page_w
        h = h / page_h

    return BBox(
        x=clamp01(x),
        y=clamp01(y),
        w=clamp01(w),
        h=clamp01(h),
    )



def _load_quality_inspection_page(
    raw_pages: list[dict[str, Any]],
    page_id: str | None,
    page_size: tuple[float, float] | None,
) -> PageData:
    selected: dict[str, Any] | None = None
    if page_id is None:
        if len(raw_pages) != 1:
            raise ValueError("quality-inspection json contains multiple pages; provide page_id")
        selected = raw_pages[0]
    else:
        target = str(page_id)
        for page in raw_pages:
            if str(page.get("ID")) == target:
                selected = page
                break
        if selected is None:
            raise ValueError(f"page_id not found in quality-inspection json: {page_id}")

    if page_size is not None:
        width = float(page_size[0])
        height = float(page_size[1])
        page_size_source = "image"
    else:
        max_right = 1.0
        max_bottom = 1.0
        for item in selected.get("contentList", []):
            left = float(item.get("left", 0.0))
            top = float(item.get("top", 0.0))
            item_width = float(item.get("width", 0.0))
            item_height = float(item.get("height", 0.0))
            max_right = max(max_right, left + item_width)
            max_bottom = max(max_bottom, top + item_height)
        width = max_right
        height = max_bottom
        page_size_source = "bbox_extent"

    out: list[Element] = []
    skipped = 0
    for i, item in enumerate(selected.get("contentList", [])):
        if not isinstance(item, dict):
            continue
        if item.get("visible") is False:
            continue

        raw_tag = ""
        tag_list = item.get("markUpEntityTagList", [])
        if isinstance(tag_list, list) and tag_list:
            raw_tag = str(tag_list[0].get("tagName", ""))
        raw_tag = raw_tag or str(item.get("name", ""))
        tag = normalize_semantic_tag(raw_tag or str(item.get("type", "unknown")))
        if tag in IGNORE_GT_TAGS:
            skipped += 1
            continue

        text = clean_text(str(item.get("remark", "")))
        if not text:
            full_name = clean_text(str(item.get("fullName", "")))
            text = clean_text(full_name.split(":", 1)[1] if ":" in full_name else full_name)

        bbox = _bbox_from_dict(item, page_w=width, page_h=height)
        out.append(
            Element(
                tag=tag,
                text=text,
                bbox=bbox,
                source_id=str(item.get("id", i)),
            )
        )

    return PageData(
        width=width,
        height=height,
        elements=out,
        meta={
            "format": "quality_inspection_v1",
            "page_id": str(selected.get("ID", "")),
            "page_size_source": page_size_source,
            "skipped_ignored_tags": skipped,
        },
    )


def parse_page_json(
    path: str | Path,
    page_id: str | None = None,
    page_size: tuple[float, float] | None = None,
) -> PageData:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8-sig"))

    # Supported formats:
    # 1) {"width": ..., "height": ..., "elements": [...]}
    # 2) {"page": {"width":...,"height":...}, "items": [...]}
    # 3) [{"ID": ..., "contentList": [...]}, ...] (Quality Inspection GT)
    # 4) [...] (elements only, assumed relative coords)
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and "contentList" in raw[0] and "ID" in raw[0]:
            return _load_quality_inspection_page(raw, page_id=page_id, page_size=page_size)
        elements_raw = raw
        width = 1.0 if page_size is None else float(page_size[0])
        height = 1.0 if page_size is None else float(page_size[1])
        meta = {}
    elif isinstance(raw, dict):
        page = raw.get("page", {}) if isinstance(raw.get("page"), dict) else {}
        fallback_width = 1.0 if page_size is None else float(page_size[0])
        fallback_height = 1.0 if page_size is None else float(page_size[1])
        width = float(raw.get("width", page.get("width", fallback_width)) or fallback_width)
        height = float(raw.get("height", page.get("height", fallback_height)) or fallback_height)
        elements_raw = raw.get("elements", raw.get("items", raw.get("nodes", [])))
        if not isinstance(elements_raw, list):
            raise ValueError(f"elements/items/nodes must be list: {p}")
        meta = {k: v for k, v in raw.items() if k not in {"width", "height", "page", "elements", "items", "nodes"}}
    else:
        raise ValueError(f"unsupported json format: {p}")

    out: list[Element] = []
    for i, e in enumerate(elements_raw):
        if not isinstance(e, dict):
            continue
        bbox_raw = e.get("bbox", e.get("box", e))
        try:
            bbox = _bbox_from_dict(bbox_raw, page_w=width, page_h=height)
        except Exception:
            continue
        tag = normalize_semantic_tag(str(e.get("tag", e.get("name", e.get("type", "unknown")))))
        text = clean_text(str(e.get("text", e.get("remark", e.get("content", "")))))
        source_id = str(e.get("id", i))
        out.append(Element(tag=tag, text=text, bbox=bbox, source_id=source_id))

    return PageData(width=width, height=height, elements=out, meta=meta)



def dump_page_json(page: PageData, path: str | Path) -> None:
    p = Path(path)
    payload = {
        "width": page.width,
        "height": page.height,
        "elements": [
            {
                "id": e.source_id,
                "tag": e.tag,
                "text": e.text,
                "bbox": {
                    "x": e.bbox.x,
                    "y": e.bbox.y,
                    "w": e.bbox.w,
                    "h": e.bbox.h,
                },
            }
            for e in page.elements
        ],
        "meta": page.meta,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
