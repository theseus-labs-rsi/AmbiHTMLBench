from __future__ import annotations

from dataclasses import dataclass
import csv
import html
from itertools import combinations, permutations
import os
from html.parser import HTMLParser
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .html_extractor import extract_html_with_page
from .schema import Element, PageData
from .utils import BBox, clean_text, iou, safe_div


TEXT_WILDCARD = "\u03be"
TEXT_DP_SPARSE_PRODUCT = 300
TEXT_DP_LARGE_PRODUCT = 2_000
TEXT_DP_LONG_CHAR_TOTAL = 1_500
TEXT_DP_HUGE_PRODUCT = 10_000
TEXT_DP_HUGE_BLOCKS = 500
TEXT_DP_HUGE_CHAR_TOTAL = 50_000
TABLE_MATCH_MIN_SCORE = 0.40
TEXT_LAYOUT_MAX_DISTANCE_WITH_ANCHORS = 0.75
TEXT_GLOBAL_MAX_PRED_BLOCKS = 22
TEXT_GLOBAL_MAX_TOTAL_BLOCKS = 34
TEXT_EXACT_MAX_PRED_BLOCKS = 8
TEXT_EXACT_MAX_TOTAL_BLOCKS = 14
TEXT_GLOBAL_MAX_SEG_PRED = 3
TEXT_GLOBAL_LONG_SEG_PRED = 6
TEXT_GLOBAL_CANDIDATES_PER_GT = 8
TEXT_GLOBAL_BEAM = 1200
TEXT_FALLBACK_MAX_GT_SEGMENT = 3
TEXT_FALLBACK_MAX_PRED_SEGMENT = 8
TEXT_FALLBACK_MAX_PRODUCT = 1600
TEXT_PAGE_ORDER_EXACT_ROWS = 5
TEXT_PAGE_ORDER_BEAM_ROWS = 16
TEXT_PAGE_ORDER_BEAM_SIZE = 32
TEXT_EDGE_THRESHOLD = 0.66
TEXT_REORDER_MAX_BLOCKS = 36
TEXT_REORDER_ROUNDS = 1
TEXT_REORDER_WINDOW = 3
TEXT_MAX_SEGMENT = 3


TABLE_NAMES = {"table", "表格table"}
FORMULA_NAMES = {"display_formula", "inline_formula", "equation_isolated", "equation_merged"}


@dataclass(frozen=True)
class Obj:
    id: str
    kind: str
    text: str
    bbox: BBox
    raw_html: str = ""
    latex: str = ""
    visual_bbox: BBox | None = None


@dataclass(frozen=True)
class Match:
    gt_index: int
    pred_index: int
    score: float


@dataclass(frozen=True)
class PageObjectiveScores:
    page_id: str
    table_teds: float | None
    table_teds_s: float | None
    formula_cdm: float | None
    text_edit_distance: float
    layout_graph_similarity: float
    gt_tables: int
    pred_tables: int
    gt_formulas: int
    pred_formulas: int
    gt_text_blocks: int
    pred_text_blocks: int


def load_gt_v3(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("AmbiHTMLBench GT must be a list of page objects")
    pages: dict[str, dict[str, Any]] = {}
    for page in raw:
        if not isinstance(page, dict) or set(page.keys()) != {"ID", "contentList", "imageUrl"}:
            raise ValueError("AmbiHTMLBench GT page must contain exactly ID/contentList/imageUrl")
        if not isinstance(page["contentList"], list):
            raise ValueError(f"page {page.get('ID')} contentList must be a list")
        pid = f"{int(page['ID']):04d}"
        pages[pid] = page
    return pages


def _bbox_from_item(item: dict[str, Any], page_w: float, page_h: float) -> BBox:
    return BBox(
        x=max(0.0, min(1.0, float(item.get("left", 0.0)) / page_w)),
        y=max(0.0, min(1.0, float(item.get("top", 0.0)) / page_h)),
        w=max(0.0, min(1.0, float(item.get("width", 0.0)) / page_w)),
        h=max(0.0, min(1.0, float(item.get("height", 0.0)) / page_h)),
    )


def _raw_bbox_from_item(item: dict[str, Any]) -> BBox:
    return BBox(
        x=float(item.get("left", 0.0)),
        y=float(item.get("top", 0.0)),
        w=float(item.get("width", 0.0)),
        h=float(item.get("height", 0.0)),
    )


def _strip_formula_delimiters(text: str) -> str:
    s = clean_text(text)
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return s


def _table_text_rows_from_source(source: str) -> list[list[str]]:
    text = source or ""
    if "<table" not in text.lower():
        value = clean_text(text)
        return [[value]] if value else []

    class _CellParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.rows: list[list[str]] = []
            self.current_row: list[str] = []
            self.buffer: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            tag = tag.lower()
            if tag in {"tr", "td", "th"}:
                self.stack.append(tag)
                if tag == "tr":
                    self.current_row = []
                elif tag in {"td", "th"}:
                    self.buffer = []

        def handle_data(self, data: str) -> None:
            if self.stack and self.stack[-1] in {"td", "th"}:
                self.buffer.append(data)

        def handle_endtag(self, tag: str) -> None:
            tag = tag.lower()
            while self.stack:
                top = self.stack.pop()
                if top != tag:
                    continue
                if tag in {"td", "th"}:
                    cell = clean_text("".join(self.buffer))
                    if cell:
                        self.current_row.append(cell)
                    self.buffer = []
                elif tag == "tr":
                    if self.current_row:
                        self.rows.append(self.current_row)
                    self.current_row = []
                break

    parser = _CellParser()
    parser.feed(text)
    rows = [row for row in parser.rows if any(cell for cell in row)]
    if rows:
        return rows

    tr_rows: list[list[str]] = []
    for tr_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.I | re.S):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr_html, flags=re.I | re.S)
        if not cells:
            plain = clean_text(re.sub(r"<[^>]+>", " ", tr_html))
            if plain:
                tr_rows.append([plain])
            continue
        row: list[str] = []
        for cell_html in cells:
            cell = clean_text(re.sub(r"<[^>]+>", " ", cell_html))
            if cell:
                row.append(cell)
        if row:
            tr_rows.append(row)
    return tr_rows


def _table_text_from_source(source: str) -> str:
    rows = _table_text_rows_from_source(source)
    cells = [cell for row in rows for cell in row if cell]
    return clean_text(" ".join(cells)) if cells else clean_text(source or "")


def _sequence_similarity(tokens_a: list[str], tokens_b: list[str], sub_cost_fn) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    m, n = len(tokens_a), len(tokens_b)
    dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = float(i)
    for j in range(1, n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = float(sub_cost_fn(tokens_a[i - 1], tokens_b[j - 1]))
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + sub,
            )
    return max(0.0, 1.0 - dp[m][n] / max(m, n, 1))


def _gt_page_size(page: dict[str, Any]) -> tuple[float, float]:
    max_right = 1.0
    max_bottom = 1.0
    for item in page["contentList"]:
        if not isinstance(item, dict) or item.get("visible") is False:
            continue
        max_right = max(max_right, float(item.get("left", 0.0)) + float(item.get("width", 0.0)))
        max_bottom = max(max_bottom, float(item.get("top", 0.0)) + float(item.get("height", 0.0)))
    return max_right, max_bottom


def gt_objects(page: dict[str, Any]) -> list[Obj]:
    page_w, page_h = _gt_page_size(page)
    out: list[Obj] = []
    for i, item in enumerate(page["contentList"]):
        if not isinstance(item, dict) or item.get("visible") is False:
            continue
        name = str(item.get("name", ""))
        text = clean_text(str(item.get("remark", "")))
        kind = "other"
        if name in TABLE_NAMES:
            kind = "table"
        elif name in FORMULA_NAMES:
            kind = "formula"
        elif text:
            kind = "text"
        out.append(
            Obj(
                id=str(item.get("id", f"gt_{i}")),
                kind=kind,
                text=text,
                bbox=_bbox_from_item(item, page_w, page_h),
                raw_html=text if kind == "table" else "",
                latex=text if kind == "formula" else "",
                visual_bbox=_raw_bbox_from_item(item),
            )
        )
    return out


def pred_objects(page: PageData) -> list[Obj]:
    out: list[Obj] = []
    for e in page.elements:
        tag = e.tag
        kind = "other"
        if tag == "table":
            kind = "table"
        elif tag in {"equation_isolated", "equation_merged"} or e.latex:
            kind = "formula"
        elif clean_text(e.text):
            kind = "text"
        out.append(
            Obj(
                e.source_id,
                kind,
                clean_text(e.text),
                e.bbox,
                raw_html=e.raw_html,
                latex=e.latex or e.text,
                visual_bbox=e.visual_bbox,
            )
        )
    return out


def _formula_as_text_obj(obj: Obj, suffix: str = "formula_text") -> Obj:
    text = _strip_formula_delimiters(obj.latex or obj.text)
    return Obj(
        id=f"{obj.id}:{suffix}",
        kind="text",
        text=text,
        bbox=obj.bbox,
        raw_html="",
        latex="",
        visual_bbox=obj.visual_bbox,
    )


def _table_as_text_objs(obj: Obj, suffix: str = "unmatched_table_text") -> list[Obj]:
    rows = _table_text_rows_from_source(obj.raw_html or obj.text)
    if not rows:
        text = _table_text_from_source(obj.raw_html or obj.text)
        if not text:
            return []
        return [
            Obj(
                id=f"{obj.id}:{suffix}",
                kind="text",
                text=text,
                bbox=obj.bbox,
                raw_html="",
                latex="",
                visual_bbox=obj.visual_bbox,
            )
        ]

    row_count = len(rows)
    row_heights = [max(1, len(row)) for row in rows]
    max_cols = max(row_heights)
    step_y = obj.bbox.h / max(row_count, 1)
    step_x = obj.bbox.w / max(max_cols, 1)
    out: list[Obj] = []
    for r_idx, row in enumerate(rows):
        y = obj.bbox.y + step_y * r_idx
        h = obj.bbox.h - step_y * r_idx if r_idx == row_count - 1 else step_y
        vb_row = None
        if obj.visual_bbox is not None:
            vb = obj.visual_bbox
            vstep_y = vb.h / max(row_count, 1)
            vy = vb.y + vstep_y * r_idx
            vh = vb.h - vstep_y * r_idx if r_idx == row_count - 1 else vstep_y
            vb_row = BBox(vb.x, vy, vb.w, vh)
        for c_idx, cell in enumerate(row):
            text = clean_text(cell)
            if not text:
                continue
            x = obj.bbox.x + step_x * c_idx
            w = obj.bbox.w - step_x * c_idx if c_idx == len(row) - 1 else step_x
            visual_bbox = None
            if vb_row is not None:
                vstep_x = vb_row.w / max(len(row), 1)
                vx = vb_row.x + vstep_x * c_idx
                vw = vb_row.w - vstep_x * c_idx if c_idx == len(row) - 1 else vstep_x
                visual_bbox = BBox(vx, vb_row.y, vw, vb_row.h)
            out.append(
                Obj(
                    id=f"{obj.id}:{suffix}:{r_idx}:{c_idx}",
                    kind="text",
                    text=text,
                    bbox=BBox(x, y, w, h),
                    raw_html="",
                    latex="",
                    visual_bbox=visual_bbox,
                )
            )
    return out


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[list[Any]] = [["root", ""]]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}:
            self.stack.append([tag.lower(), ""])

    def handle_data(self, data: str) -> None:
        if self.stack:
            self.stack[-1][1] += data

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag not in {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i][0] == tag:
                node = self.stack.pop(i)
                self.stack[-1].append(node)
                break

    @property
    def tree(self) -> list[Any]:
        return self.stack[0]


def _markdown_table_to_html(text: str) -> str:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return f"<table><tr><td>{html.escape(clean_text(text))}</td></tr></table>"
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
    return "<table>" + "".join(body) + "</table>"


def _table_tree(source: str, *, structure_only: bool) -> list[Any]:
    text = source or ""
    parser = _TableParser()
    parser.feed(text if "<table" in text.lower() else _markdown_table_to_html(text))

    def normalize(node: list[Any]) -> tuple[Any, ...]:
        tag = node[0]
        content = "" if structure_only else clean_text(node[1])
        children = [normalize(x) for x in node[2:] if isinstance(x, list)]
        return (tag, content, tuple(children))

    return list(normalize(parser.tree))


def _tree_size(node: Any) -> int:
    return 1 + sum(_tree_size(c) for c in node[2])


def _tree_distance(a: Any, b: Any) -> int:
    root_cost = 0 if a[:2] == b[:2] else 1
    ca = list(a[2])
    cb = list(b[2])
    m, n = len(ca), len(cb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + _tree_size(ca[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + _tree_size(cb[j - 1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + _tree_size(ca[i - 1]),
                dp[i][j - 1] + _tree_size(cb[j - 1]),
                dp[i - 1][j - 1] + _tree_distance(ca[i - 1], cb[j - 1]),
            )
    return root_cost + dp[m][n]


def table_teds(a: str, b: str, *, structure_only: bool = False) -> float:
    rows_a = _table_text_rows_from_source(a)
    rows_b = _table_text_rows_from_source(b)
    if not rows_a and not rows_b:
        return 1.0
    if not rows_a or not rows_b:
        return 0.0

    if structure_only:
        tokens_a = [str(len(row)) for row in rows_a]
        tokens_b = [str(len(row)) for row in rows_b]
        return _sequence_similarity(tokens_a, tokens_b, lambda x, y: abs(int(x) - int(y)) / max(int(x), int(y), 1))

    structure_sim = _sequence_similarity(
        [str(len(row)) for row in rows_a],
        [str(len(row)) for row in rows_b],
        lambda x, y: abs(int(x) - int(y)) / max(int(x), int(y), 1),
    )
    content_sim = _sequence_similarity(
        ["|".join(row) for row in rows_a],
        ["|".join(row) for row in rows_b],
        lambda x, y: _text_distance_with_wildcards(_text_match_key(x), _text_match_key(y)),
    )
    return 0.4 * structure_sim + 0.6 * content_sim


def _canonical_latex(text: str) -> str:
    s = clean_text(text)
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def cdm_score(a: str, b: str) -> float:
    aa = _canonical_latex(a)
    bb = _canonical_latex(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return max(0.0, 1.0 - Levenshtein.distance(aa, bb) / max(len(aa), len(bb), 1))


@lru_cache(maxsize=32768)
def _text_match_key(text: str) -> str:
    return re.sub(r"\s+", "", clean_text(text)).replace("$", "")


def _has_text_wildcard(text: str) -> bool:
    return TEXT_WILDCARD in text


def _is_text_wildcard(ch: str) -> bool:
    return ch == TEXT_WILDCARD


@lru_cache(maxsize=8192)
def _text_distance_with_wildcards(gt_norm: str, pred_norm: str) -> float:
    if not gt_norm and not pred_norm:
        return 0.0
    if not gt_norm or not pred_norm:
        return 1.0
    if not _has_text_wildcard(gt_norm) and not _has_text_wildcard(pred_norm):
        return Levenshtein.distance(gt_norm, pred_norm) / max(len(gt_norm), len(pred_norm), 1)

    product = len(gt_norm) * len(pred_norm)
    if product > 250_000:
        gt_wildcards = gt_norm.count(TEXT_WILDCARD)
        pred_wildcards = pred_norm.count(TEXT_WILDCARD)
        gt_plain = gt_norm.replace(TEXT_WILDCARD, "")
        pred_plain = pred_norm.replace(TEXT_WILDCARD, "")
        base = Levenshtein.distance(gt_plain, pred_plain)
        length_floor = max(
            0,
            len(gt_norm) - len(pred_norm) - pred_wildcards,
            len(pred_norm) - len(gt_norm) - gt_wildcards,
        )
        adjusted = max(length_floor, base - gt_wildcards - pred_wildcards)
        return adjusted / max(len(gt_norm), len(pred_norm), 1)

    m = len(pred_norm)
    prev = list(range(m + 1))
    for i, gc in enumerate(gt_norm, 1):
        cur = [i] + [0] * m
        for j, pc in enumerate(pred_norm, 1):
            sub_cost = 0 if gc == pc or _is_text_wildcard(gc) or _is_text_wildcard(pc) else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + sub_cost,
            )
        prev = cur
    return prev[m] / max(len(gt_norm), len(pred_norm), 1)


@lru_cache(maxsize=32768)
def _partial_text_similarity(a: str, b: str) -> float:
    aa = _text_match_key(a)
    bb = _text_match_key(b)
    if not aa and not bb:
        return 100.0
    if not aa or not bb:
        return 0.0
    return float(fuzz.partial_ratio(aa, bb))


def _overlap_min_area(a: BBox, b: BBox) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h
    iw = max(0.0, min(ax2, bx2) - max(a.x, b.x))
    ih = max(0.0, min(ay2, by2) - max(a.y, b.y))
    inter = iw * ih
    denom = min(a.area, b.area)
    if inter <= 0.0 or denom <= 0.0:
        return 0.0
    return inter / denom


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h
    iw = max(0.0, min(ax2, bx2) - max(a.x, b.x))
    ih = max(0.0, min(ay2, by2) - max(a.y, b.y))
    inter = iw * ih
    union = a.area + b.area - inter
    return safe_div(inter, union)


def _center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.x + a.w / 2.0, a.y + a.h / 2.0
    bx, by = b.x + b.w / 2.0, b.y + b.h / 2.0
    return math.hypot(ax - bx, ay - by)


def _center_similarity(a: BBox, b: BBox) -> float:
    return max(0.0, 1.0 - _center_distance(a, b) / math.sqrt(2.0))


def _area_ratio(a: BBox, b: BBox) -> float:
    if a.area <= 0.0 or b.area <= 0.0:
        return math.inf
    r = a.area / b.area
    return max(r, 1.0 / r)


def _text_len(o: Obj) -> int:
    return len(_text_match_key(o.text))


def _is_tiny_text(o: Obj) -> bool:
    return _text_len(o) < 6


def _is_large_text(o: Obj, median_area: float) -> bool:
    return _text_len(o) >= 20 or o.bbox.area >= 0.01 or (median_area > 0 and o.bbox.area >= median_area * 2.0)


@lru_cache(maxsize=131072)
def _text_pair_features(g: Obj, p: Obj) -> tuple[float, float, float, float, float]:
    ts = _partial_text_similarity(g.text, p.text) / 100.0
    ov = _overlap_min_area(g.bbox, p.bbox)
    box_iou = _bbox_iou(g.bbox, p.bbox)
    center = _center_similarity(g.bbox, p.bbox)
    size = _size_sim(g.bbox, p.bbox)
    return ts, ov, box_iou, center, size


def _text_pair_supported(g: Obj, p: Obj) -> bool:
    ts, ov, box_iou, center, _ = _text_pair_features(g, p)
    if _is_tiny_text(g) or _is_tiny_text(p):
        return ts >= 0.95 and (center >= 0.55 or box_iou >= 0.05 or ov >= 0.85)
    if ts >= 0.78:
        return True
    if ts >= 0.62 and (center >= 0.65 or box_iou >= 0.06 or ov >= 0.85):
        return True
    if ov >= 0.88 and ts >= 0.35 and _area_ratio(g.bbox, p.bbox) <= 4.0:
        return True
    return False


def _text_pair_affinity(g: Obj, p: Obj) -> float:
    ts, ov, box_iou, center, size = _text_pair_features(g, p)
    if not _text_pair_supported(g, p):
        return 0.0
    return 0.55 * ts + 0.18 * center + 0.12 * box_iou + 0.10 * size + 0.05 * ov


def _greedy_one_to_one(gt: list[Obj], pred: list[Obj], score_fn) -> list[Match]:
    candidates: list[Match] = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            candidates.append(Match(i, j, float(score_fn(g, p))))
    candidates.sort(key=lambda x: x.score, reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    out: list[Match] = []
    for m in candidates:
        if m.gt_index in used_g or m.pred_index in used_p:
            continue
        used_g.add(m.gt_index)
        used_p.add(m.pred_index)
        out.append(m)
    return out


def _hungarian_one_to_one(gt: list[Obj], pred: list[Obj], score_fn, *, min_score: float = 0.0) -> list[Match]:
    if not gt and not pred:
        return []
    if not gt or not pred:
        return []
    m, n = len(gt), len(pred)
    size = m + n
    dummy_cost = 1.0 - min_score
    cost = [[dummy_cost for _ in range(size)] for _ in range(size)]
    for i in range(m):
        for j in range(n):
            score = float(score_fn(gt[i], pred[j]))
            if score >= min_score:
                cost[i][j] = 1.0 - score
            else:
                cost[i][j] = 1.0
    row_ind, col_ind = linear_sum_assignment(cost)
    out: list[Match] = []
    for r, c in zip(row_ind, col_ind):
        if r < m and c < n:
            score = 1.0 - float(cost[r][c])
            if score >= min_score:
                out.append(Match(r, c, score))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _typed_score(gt: list[Obj], pred: list[Obj], value_fn, match_fn, *, min_score: float = 0.0) -> tuple[float, list[Match]]:
    if not gt and not pred:
        return 1.0, []
    if not gt or not pred:
        return 0.0, []
    matches = _hungarian_one_to_one(gt, pred, match_fn, min_score=min_score)
    values = [float(value_fn(gt[m.gt_index], pred[m.pred_index])) for m in matches[: len(gt)]]
    values.extend([0.0] * max(0, len(gt) - len(values)))
    if len(pred) > len(gt):
        values.extend([0.0] * (len(pred) - len(gt)))
    return safe_div(sum(values), len(values)), matches


def _table_match_score(g: Obj, p: Obj) -> float:
    gt_html = g.raw_html or g.text
    pred_html = p.raw_html or p.text
    content = table_teds(gt_html, pred_html, structure_only=False)
    structure = table_teds(gt_html, pred_html, structure_only=True)
    text_sim = _partial_text_similarity(_table_text_from_source(gt_html), _table_text_from_source(pred_html)) / 100.0
    pos = _overlap_min_area(g.bbox, p.bbox)
    return 0.45 * content + 0.25 * structure + 0.20 * text_sim + 0.10 * pos


def _text_distance_from_norm(gt_norm: str, pred_norm: str) -> float:
    return _text_distance_with_wildcards(gt_norm, pred_norm)


def _reading_order(ids: list[int], objs: list[Obj]) -> list[int]:
    return sorted(ids, key=lambda k: (objs[k].bbox.y, objs[k].bbox.x))


def _x_order(ids: list[int], objs: list[Obj]) -> list[int]:
    return sorted(ids, key=lambda k: (objs[k].bbox.x, objs[k].bbox.y))


def _concat_norm(ids: list[int], objs: list[Obj]) -> str:
    return "".join(_text_match_key(objs[i].text) for i in ids)


def _unique_orders(candidates: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for c in candidates:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _segment_has_candidate(gt_seg: list[int], pred_seg: list[int], scores: list[list[float]]) -> bool:
    for i in gt_seg:
        for j in pred_seg:
            if scores[i][j] > 0.0:
                return True
    return False


def _component_text_dp(
    gt_ids: list[int],
    pred_ids: list[int],
    gt: list[Obj],
    pred: list[Obj],
    *,
    allow_sparse_split: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    gt_order = _reading_order(gt_ids, gt)
    pred_order = _reading_order(pred_ids, pred)
    m, n = len(gt_order), len(pred_order)
    total_char_len = sum(len(_text_match_key(gt[i].text)) for i in gt_order) + sum(len(_text_match_key(pred[j].text)) for j in pred_order)
    if m == 0 and n == 0:
        return 0.0, []
    if m == 0:
        rows = [{"gt": [], "pred": [j], "distance": 1.0, "dropped_gt": [], "dropped_pred": []} for j in pred_order]
        return sum(len(_concat_norm([j], pred)) for j in pred_order), rows
    if n == 0:
        rows = [{"gt": [i], "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []} for i in gt_order]
        return sum(len(_concat_norm([i], gt)) for i in gt_order), rows

    product = m * n
    if (
        product > TEXT_DP_HUGE_PRODUCT
        or max(m, n) > TEXT_DP_HUGE_BLOCKS
        or total_char_len > TEXT_DP_HUGE_CHAR_TOTAL
        or (total_char_len > TEXT_DP_LONG_CHAR_TOTAL and product > TEXT_DP_SPARSE_PRODUCT)
    ):
        dist, go, po = _best_text_alignment(gt_order, pred_order, gt, pred)
        gt_norm = _concat_norm(go, gt)
        pred_norm = _concat_norm(po, pred)
        return dist * max(len(gt_norm), len(pred_norm), 1), [
            {"gt": go, "pred": po, "distance": dist, "dropped_gt": [], "dropped_pred": []}
        ]

    # Pairwise candidate similarities, used both for component pruning and DP transitions.
    pair_sim = [[0.0 for _ in range(n)] for _ in range(m)]
    pair_ok = [[False for _ in range(n)] for _ in range(m)]
    top_k = 3
    gt_rank: list[list[int]] = [[] for _ in range(m)]
    pred_rank: list[list[int]] = [[] for _ in range(n)]
    for i, gi in enumerate(gt_order):
        for j, pj in enumerate(pred_order):
            ts, ov, box_iou, center, size = _text_pair_features(gt[gi], pred[pj])
            sim = _text_pair_affinity(gt[gi], pred[pj])
            pair_sim[i][j] = sim
            pair_ok[i][j] = sim > 0.0
        gt_rank[i] = sorted(range(n), key=lambda j: pair_sim[i][j], reverse=True)[:top_k]
    for j in range(n):
        pred_rank[j] = sorted(range(m), key=lambda i: pair_sim[i][j], reverse=True)[:top_k]

    long_text_dp = total_char_len > TEXT_DP_LONG_CHAR_TOTAL
    sparse_dp = product > TEXT_DP_SPARSE_PRODUCT or long_text_dp
    large_dp = product > TEXT_DP_LARGE_PRODUCT
    if allow_sparse_split and sparse_dp:
        parent = list(range(m + n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        edge_count = 0
        for i in range(m):
            for j in range(n):
                if pair_ok[i][j]:
                    union(i, m + j)
                    edge_count += 1

        if edge_count:
            comp_gt: dict[int, list[int]] = {}
            comp_pred: dict[int, list[int]] = {}
            for i in range(m):
                comp_gt.setdefault(find(i), []).append(gt_order[i])
            for j in range(n):
                comp_pred.setdefault(find(m + j), []).append(pred_order[j])

            all_roots = set(comp_gt) | set(comp_pred)
            splittable = len(all_roots) > 1
            if splittable:
                total_cost = 0.0
                rows: list[dict[str, Any]] = []
                for root in sorted(
                    all_roots,
                    key=lambda r: (
                        min([gt[i].bbox.y for i in comp_gt.get(r, [])] + [pred[j].bbox.y for j in comp_pred.get(r, [])]),
                        min([gt[i].bbox.x for i in comp_gt.get(r, [])] + [pred[j].bbox.x for j in comp_pred.get(r, [])]),
                    ),
                ):
                    cg = comp_gt.get(root, [])
                    cp = comp_pred.get(root, [])
                    if cg and cp:
                        cost, comp_rows = _component_text_dp(cg, cp, gt, pred, allow_sparse_split=False)
                        total_cost += cost
                        rows.extend(comp_rows)
                    elif cg:
                        for gi in _reading_order(cg, gt):
                            total_cost += len(_concat_norm([gi], gt))
                            rows.append({"gt": [gi], "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []})
                    else:
                        for pj in _reading_order(cp, pred):
                            total_cost += len(_concat_norm([pj], pred))
                            rows.append({"gt": [], "pred": [pj], "distance": 1.0, "dropped_gt": [], "dropped_pred": []})
                return total_cost, rows

        if large_dp:
            # If the candidate graph is still one giant component, avoid an
            # exponential-looking segment search. Treat it as one page-level
            # alignment and choose the best cheap ordering available.
            dist, go, po = _best_text_alignment(gt_order, pred_order, gt, pred)
            gt_norm = _concat_norm(go, gt)
            pred_norm = _concat_norm(po, pred)
            return dist * max(len(gt_norm), len(pred_norm), 1), [
                {"gt": go, "pred": po, "distance": dist, "dropped_gt": [], "dropped_pred": []}
            ]

    if large_dp and not allow_sparse_split:
        dist, go, po = _best_text_alignment(gt_order, pred_order, gt, pred)
        gt_norm = _concat_norm(go, gt)
        pred_norm = _concat_norm(po, pred)
        return dist * max(len(gt_norm), len(pred_norm), 1), [
            {"gt": go, "pred": po, "distance": dist, "dropped_gt": [], "dropped_pred": []}
        ]

    seg_cache: dict[tuple[int, int, int, int], tuple[float, float, list[int], list[int]]] = {}

    def seg_score(i0: int, i1: int, j0: int, j1: int) -> tuple[float, float, list[int], list[int]]:
        key = (i0, i1, j0, j1)
        cached = seg_cache.get(key)
        if cached is not None:
            return cached
        gt_seg = gt_order[i0:i1]
        pred_seg = pred_order[j0:j1]
        if not gt_seg and not pred_seg:
            result = (0.0, 1.0, [], [])
            seg_cache[key] = result
            return result
        if gt_seg and pred_seg:
            if not _segment_has_candidate(list(range(i0, i1)), list(range(j0, j1)), pair_sim):
                result = (math.inf, 0.0, gt_seg, pred_seg)
                seg_cache[key] = result
                return result
            # Allow only reasonably local, mutually-supported segment alignments.
            supported = False
            for ii, gi in enumerate(gt_seg):
                for jj, pj in enumerate(pred_seg):
                    if pair_ok[i0 + ii][j0 + jj] and (j0 + jj in gt_rank[i0 + ii]) and (i0 + ii in pred_rank[j0 + jj]):
                        supported = True
                        break
                if supported:
                    break
            if not supported:
                result = (math.inf, 0.0, gt_seg, pred_seg)
                seg_cache[key] = result
                return result
            # Block grouping is only used to produce interpretable matches
            # and layout nodes.  The final text metric is computed later on
            # the whole page with the exact wildcard-aware edit distance.
            # Using that quadratic distance for every candidate segment makes
            # long-text pages disproportionately expensive, so use the cheap
            # ratio proxy to select a grouping here.
            go = _reading_order(gt_seg, gt)
            po = _reading_order(pred_seg, pred)
            gt_norm = _concat_norm(go, gt)
            pred_norm = _concat_norm(po, pred)
            dist = safe_div(
                _fast_text_cost(gt_norm, pred_norm),
                max(len(gt_norm), len(pred_norm), 1),
            )
            cost = dist * max(len(gt_norm), len(pred_norm), 1)
            result = (cost, dist, go, po)
            seg_cache[key] = result
            return result
        if gt_seg:
            gt_norm = _concat_norm(gt_seg, gt)
            result = (len(gt_norm), 1.0, gt_seg, [])
            seg_cache[key] = result
            return result
        pred_norm = _concat_norm(pred_seg, pred)
        result = (len(pred_norm), 1.0, [], pred_seg)
        seg_cache[key] = result
        return result

    band = max(12 if sparse_dp else 24, int(math.sqrt(max(m, n))) + (6 if sparse_dp else 12))

    def in_band(i: int, j: int) -> bool:
        if not large_dp:
            return True
        if (i == 0 and j == 0) or (i == m and j == n):
            return True
        center = round((i / max(m, 1)) * n)
        return abs(j - center) <= band

    max_seg = min(1 if long_text_dp else 2 if large_dp else 3 if sparse_dp else 8, max(m, n))
    dp = [[math.inf for _ in range(n + 1)] for _ in range(m + 1)]
    back: list[list[tuple[int, int, list[int], list[int], float] | None]] = [[None for _ in range(n + 1)] for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            base = dp[i][j]
            if not math.isfinite(base):
                continue
            if not in_band(i, j):
                continue
            for a in range(1, min(max_seg, m - i) + 1):
                if not in_band(i + a, j):
                    continue
                cost, dist, go, po = seg_score(i, i + a, j, j)
                if not math.isfinite(cost):
                    continue
                cand = base + cost
                if cand < dp[i + a][j]:
                    dp[i + a][j] = cand
                    back[i + a][j] = (i, j, go, po, dist)
            for b in range(1, min(max_seg, n - j) + 1):
                if not in_band(i, j + b):
                    continue
                cost, dist, go, po = seg_score(i, i, j, j + b)
                if not math.isfinite(cost):
                    continue
                cand = base + cost
                if cand < dp[i][j + b]:
                    dp[i][j + b] = cand
                    back[i][j + b] = (i, j, go, po, dist)
            for a in range(1, min(max_seg, m - i) + 1):
                for b in range(1, min(max_seg, n - j) + 1):
                    if not in_band(i + a, j + b):
                        continue
                    cost, dist, go, po = seg_score(i, i + a, j, j + b)
                    if not math.isfinite(cost):
                        continue
                    cand = base + cost
                    if cand < dp[i + a][j + b]:
                        dp[i + a][j + b] = cand
                        back[i + a][j + b] = (i, j, go, po, dist)

    if not math.isfinite(dp[m][n]):
        # Fallback to reading-order unmatched rows if the component could not be aligned.
        rows = [{"gt": [i], "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []} for i in gt_order]
        rows.extend({"gt": [], "pred": [j], "distance": 1.0, "dropped_gt": [], "dropped_pred": []} for j in pred_order)
        total_len = sum(len(_concat_norm([i], gt)) for i in gt_order) + sum(len(_concat_norm([j], pred)) for j in pred_order)
        return float(total_len), rows

    rows: list[dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj, go, po, dist = prev
        rows.append({"gt": go, "pred": po, "distance": dist, "dropped_gt": [], "dropped_pred": []})
        i, j = pi, pj
    rows.reverse()
    return dp[m][n], rows


def _best_text_alignment(gt_ids: list[int], pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> tuple[float, list[int], list[int]]:
    gt_read = _reading_order(gt_ids, gt)
    pred_read = _reading_order(pred_ids, pred)
    gt_orders = _unique_orders([gt_read, _x_order(gt_ids, gt)])
    pred_orders = _unique_orders([pred_read, _x_order(pred_ids, pred)])

    if len(gt_ids) <= 6:
        by_pred_similarity = sorted(
            gt_ids,
            key=lambda i: max((_partial_text_similarity(gt[i].text, pred[j].text) for j in pred_ids), default=0.0),
            reverse=True,
        )
        gt_orders = _unique_orders(gt_orders + [by_pred_similarity])
    if len(pred_ids) <= 6:
        by_gt_similarity = sorted(
            pred_ids,
            key=lambda j: max((_partial_text_similarity(gt[i].text, pred[j].text) for i in gt_ids), default=0.0),
            reverse=True,
        )
        pred_orders = _unique_orders(pred_orders + [by_gt_similarity])

    if 1 < len(gt_ids) <= 5 and math.factorial(len(gt_ids)) * max(1, len(pred_orders)) <= 5000:
        gt_orders = _unique_orders(gt_orders + [list(x) for x in permutations(gt_ids)])
    if 1 < len(pred_ids) <= 5 and math.factorial(len(pred_ids)) * max(1, len(gt_orders)) <= 5000:
        pred_orders = _unique_orders(pred_orders + [list(x) for x in permutations(pred_ids)])

    best: tuple[float, list[int], list[int]] | None = None
    for go in gt_orders:
        gt_norm = _concat_norm(go, gt)
        for po in pred_orders:
            dist = _text_distance_from_norm(gt_norm, _concat_norm(po, pred))
            if best is None or dist < best[0]:
                best = (dist, go, po)
    return best if best is not None else (1.0, gt_read, pred_read)


def _match_row_cost(gt_ids: list[int], pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> tuple[float, dict[str, Any]]:
    if gt_ids and pred_ids:
        dist, go, po = _best_text_alignment(gt_ids, pred_ids, gt, pred)
        gt_norm = _concat_norm(go, gt)
        pred_norm = _concat_norm(po, pred)
        return dist * max(len(gt_norm), len(pred_norm), 1), {
            "gt": go,
            "pred": po,
            "distance": dist,
            "dropped_gt": [],
            "dropped_pred": [],
        }
    if gt_ids:
        go = _reading_order(gt_ids, gt)
        return len(_concat_norm(go, gt)), {"gt": go, "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
    po = _reading_order(pred_ids, pred)
    return len(_concat_norm(po, pred)), {"gt": [], "pred": po, "distance": 1.0, "dropped_gt": [], "dropped_pred": []}


def _candidate_preds_for_gt(gi: int, pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> list[int]:
    scored: list[tuple[float, int]] = []
    for pj in pred_ids:
        affinity = _text_pair_affinity(gt[gi], pred[pj])
        if affinity > 0.0:
            scored.append((affinity, pj))
    scored.sort(reverse=True)
    return [pj for _, pj in scored[:TEXT_GLOBAL_CANDIDATES_PER_GT]]


def _pred_subset_options(candidates: list[int], pred_pos: dict[int, int], used_mask: int, max_pred_seg: int) -> list[tuple[int, list[int]]]:
    available = [p for p in candidates if not (used_mask & (1 << pred_pos[p]))]
    options: list[tuple[int, list[int]]] = [(0, [])]
    max_size = min(max_pred_seg, len(available))
    seen = {0}

    def add(pred_seg: list[int]) -> None:
        mask = 0
        for p in pred_seg:
            mask |= 1 << pred_pos[p]
        if mask not in seen:
            seen.add(mask)
            options.append((mask, pred_seg))

    for p in available:
        add([p])
    if max_pred_seg <= TEXT_GLOBAL_MAX_SEG_PRED and len(available) <= 7:
        for size in range(2, max_size + 1):
            for combo in combinations(available, size):
                add(list(combo))
    else:
        for size in range(2, max_size + 1):
            for start in range(0, len(available) - size + 1):
                add(available[start : start + size])
    return options


def _global_text_subset_search(gt_ids: list[int], pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    gt_order = _reading_order(gt_ids, gt)
    pred_order = _reading_order(pred_ids, pred)
    m = len(gt_order)
    n = len(pred_order)
    if n > TEXT_GLOBAL_MAX_PRED_BLOCKS or m + n > TEXT_GLOBAL_MAX_TOTAL_BLOCKS:
        return _component_text_dp(gt_ids, pred_ids, gt, pred)

    pred_pos = {pid: i for i, pid in enumerate(pred_order)}
    gt_candidates = {gi: _candidate_preds_for_gt(gi, pred_order, gt, pred) for gi in gt_order}
    dp: list[dict[int, tuple[float, list[dict[str, Any]]]]] = [dict() for _ in range(m + 1)]
    dp[0][0] = (0.0, [])
    max_gt_seg = min(3, max(1, m))
    row_cost_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, dict[str, Any]]] = {}

    def cached_row_cost(gt_seg: list[int], pred_seg: list[int]) -> tuple[float, dict[str, Any]]:
        key = (tuple(gt_seg), tuple(sorted(pred_seg)))
        cached = row_cost_cache.get(key)
        if cached is not None:
            cost, row = cached
            return cost, dict(row)
        cost, row = _match_row_cost(gt_seg, pred_seg, gt, pred)
        row_cost_cache[key] = (cost, dict(row))
        return cost, row

    for i in range(m + 1):
        if len(dp[i]) > TEXT_GLOBAL_BEAM:
            keep = sorted(dp[i].items(), key=lambda kv: kv[1][0])[:TEXT_GLOBAL_BEAM]
            dp[i] = dict(keep)
        for mask, (base_cost, base_rows) in list(dp[i].items()):
            if i >= m:
                continue
            for a in range(1, min(max_gt_seg, m - i) + 1):
                gt_seg = gt_order[i : i + a]
                cand_set: set[int] = set()
                for gi in gt_seg:
                    cand_set.update(gt_candidates.get(gi, []))
                gt_chars = len(_concat_norm(gt_seg, gt))
                max_pred_seg = TEXT_GLOBAL_LONG_SEG_PRED if gt_chars >= 35 or len(gt_seg) > 1 else TEXT_GLOBAL_MAX_SEG_PRED
                for add_mask, pred_seg in _pred_subset_options(_reading_order(list(cand_set), pred), pred_pos, mask, max_pred_seg):
                    cost, row = cached_row_cost(gt_seg, pred_seg)
                    new_mask = mask | add_mask
                    new_cost = base_cost + cost
                    old = dp[i + a].get(new_mask)
                    if old is None or new_cost < old[0]:
                        dp[i + a][new_mask] = (new_cost, base_rows + [row])

    if not dp[m]:
        return _component_text_dp(gt_ids, pred_ids, gt, pred)

    best_mask, (best_cost, rows) = min(dp[m].items(), key=lambda kv: kv[1][0])
    leftover = [pid for pid in pred_order if not (best_mask & (1 << pred_pos[pid]))]
    for pj in leftover:
        cost, row = cached_row_cost([], [pj])
        best_cost += cost
        rows.append(row)
    return best_cost, rows


def _greedy_text_set_cover(gt_ids: list[int], pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    gt_order = _reading_order(gt_ids, gt)
    pred_order = _reading_order(pred_ids, pred)
    if not gt_order and not pred_order:
        return 0.0, []
    if not gt_order:
        return _component_text_dp([], pred_order, gt, pred)
    if not pred_order:
        return _component_text_dp(gt_order, [], gt, pred)

    pred_pos = {pid: i for i, pid in enumerate(pred_order)}
    gt_candidates = {gi: _candidate_preds_for_gt(gi, pred_order, gt, pred) for gi in gt_order}
    row_cost_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, dict[str, Any]]] = {}

    def cached_row_cost(gt_seg: list[int], pred_seg: list[int]) -> tuple[float, dict[str, Any]]:
        key = (tuple(gt_seg), tuple(sorted(pred_seg)))
        cached = row_cost_cache.get(key)
        if cached is not None:
            cost, row = cached
            return cost, dict(row)
        cost, row = _match_row_cost(gt_seg, pred_seg, gt, pred)
        row_cost_cache[key] = (cost, dict(row))
        return cost, row

    options: list[tuple[float, float, float, tuple[int, ...], tuple[int, ...], dict[str, Any]]] = []
    max_gt_seg = 1
    for i in range(len(gt_order)):
        for a in range(1, min(max_gt_seg, len(gt_order) - i) + 1):
            gt_seg = gt_order[i : i + a]
            cand_set: set[int] = set()
            for gi in gt_seg:
                cand_set.update(gt_candidates.get(gi, []))
            if not cand_set:
                continue
            gt_chars = len(_concat_norm(gt_seg, gt))
            max_pred_seg = TEXT_GLOBAL_LONG_SEG_PRED if gt_chars >= 20 or len(gt_seg) > 1 else TEXT_GLOBAL_MAX_SEG_PRED
            for _, pred_seg in _pred_subset_options(_reading_order(list(cand_set), pred), pred_pos, 0, max_pred_seg):
                if not pred_seg:
                    continue
                cost, row = cached_row_cost(gt_seg, pred_seg)
                if not math.isfinite(cost):
                    continue
                base_cost = gt_chars + sum(len(_concat_norm([pj], pred)) for pj in pred_seg)
                gain = base_cost - cost
                if gain <= 0.0:
                    continue
                coverage = gt_chars + sum(len(_concat_norm([pj], pred)) for pj in pred_seg)
                options.append((gain, safe_div(gain, coverage), cost, tuple(gt_seg), tuple(pred_seg), row))

    options.sort(key=lambda x: (x[1], x[0], -len(x[3]), -len(x[4])), reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for gain, _, cost, gt_seg_t, pred_seg_t, row in options:
        if any(gi in used_gt for gi in gt_seg_t) or any(pj in used_pred for pj in pred_seg_t):
            continue
        used_gt.update(gt_seg_t)
        used_pred.update(pred_seg_t)
        total_cost += cost
        rows.append(row)

    for gi in gt_order:
        if gi in used_gt:
            continue
        cost, row = cached_row_cost([gi], [])
        total_cost += cost
        rows.append(row)
    for pj in pred_order:
        if pj in used_pred:
            continue
        cost, row = cached_row_cost([], [pj])
        total_cost += cost
        rows.append(row)

    def row_key(row: dict[str, Any]) -> tuple[float, float]:
        ids = list(row.get("gt", [])) + list(row.get("pred", []))
        boxes = [gt[i].bbox for i in row.get("gt", [])] + [pred[j].bbox for j in row.get("pred", [])]
        if not boxes:
            return (1e9, 1e9)
        return (min(b.y for b in boxes), min(b.x for b in boxes))

    rows.sort(key=row_key)
    return total_cost, rows


def _median_area(objs: list[Obj]) -> float:
    areas = sorted(o.bbox.area for o in objs if o.bbox.area > 0)
    if not areas:
        return 0.0
    return areas[len(areas) // 2]


def _anchor_text_score(g: Obj, p: Obj, gt_median_area: float, pred_median_area: float) -> float:
    if not (_is_large_text(g, gt_median_area) and _is_large_text(p, pred_median_area)):
        return 0.0
    ts, ov, box_iou, center, size = _text_pair_features(g, p)
    if ts < 0.70:
        return 0.0
    if not (center >= 0.62 or box_iou >= 0.08 or (ov >= 0.85 and _area_ratio(g.bbox, p.bbox) <= 4.0)):
        return 0.0
    return 0.55 * ts + 0.20 * center + 0.10 * box_iou + 0.10 * size + 0.05 * ov


def _nearest_anchor(obj: Obj, anchors: list[tuple[int, int]], side: str, gt: list[Obj], pred: list[Obj]) -> int | None:
    best: tuple[float, int] | None = None
    for k, (gi, pj) in enumerate(anchors):
        anchor_obj = gt[gi] if side == "gt" else pred[pj]
        dist = _center_distance(obj.bbox, anchor_obj.bbox)
        if best is None or dist < best[0]:
            best = (dist, k)
    if best is None or best[0] > 0.28:
        return None
    return best[1]


def _monotone_anchors(anchors: list[tuple[int, int]], gt_ids: list[int], pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> list[tuple[int, int]]:
    if len(anchors) <= 1:
        return anchors
    gt_rank = {idx: pos for pos, idx in enumerate(_reading_order(gt_ids, gt))}
    pred_rank = {idx: pos for pos, idx in enumerate(_reading_order(pred_ids, pred))}
    ordered = sorted(anchors, key=lambda pair: gt_rank.get(pair[0], 0))
    n = len(ordered)
    dp = [1] * n
    prev = [-1] * n
    for i in range(n):
        pi = pred_rank.get(ordered[i][1], 0)
        for j in range(i):
            pj = pred_rank.get(ordered[j][1], 0)
            if pj < pi and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: dp[i])
    keep: list[tuple[int, int]] = []
    while end >= 0:
        keep.append(ordered[end])
        end = prev[end]
    keep.reverse()
    return keep


def _solve_text_bucket(gt_bucket: list[int], pred_bucket: list[int], gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    if not gt_bucket and not pred_bucket:
        return 0.0, []
    if len(pred_bucket) <= TEXT_EXACT_MAX_PRED_BLOCKS and len(gt_bucket) + len(pred_bucket) <= TEXT_EXACT_MAX_TOTAL_BLOCKS:
        return _global_text_subset_search(gt_bucket, pred_bucket, gt, pred)
    return _greedy_text_set_cover(gt_bucket, pred_bucket, gt, pred)


def _anchored_text_divide(gt_ids: list[int], pred_ids: list[int], gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    gt_median = _median_area([gt[i] for i in gt_ids])
    pred_median = _median_area([pred[j] for j in pred_ids])
    large_gt = [i for i in gt_ids if _is_large_text(gt[i], gt_median)]
    large_pred = [j for j in pred_ids if _is_large_text(pred[j], pred_median)]
    if not large_gt or not large_pred:
        return _component_text_dp(gt_ids, pred_ids, gt, pred)

    large_matches = _hungarian_one_to_one(
        [gt[i] for i in large_gt],
        [pred[j] for j in large_pred],
        lambda g, p: _anchor_text_score(g, p, gt_median, pred_median),
        min_score=0.65,
    )
    anchors = [(large_gt[m.gt_index], large_pred[m.pred_index]) for m in large_matches]
    if not anchors:
        return _greedy_text_set_cover(gt_ids, pred_ids, gt, pred)
    anchors = _monotone_anchors(anchors, gt_ids, pred_ids, gt, pred)
    if not anchors:
        return _greedy_text_set_cover(gt_ids, pred_ids, gt, pred)

    gt_order = _reading_order(gt_ids, gt)
    pred_order = _reading_order(pred_ids, pred)
    gt_rank = {idx: pos for pos, idx in enumerate(gt_order)}
    pred_rank = {idx: pos for pos, idx in enumerate(pred_order)}
    anchors = sorted(anchors, key=lambda pair: gt_rank[pair[0]])
    used_gt = {gi for gi, _ in anchors}
    used_pred = {pj for _, pj in anchors}
    total_cost = 0.0
    rows: list[dict[str, Any]] = []

    prev_gt_pos = -1
    prev_pred_pos = -1
    for gi, pj in anchors:
        gt_pos = gt_rank[gi]
        pred_pos = pred_rank[pj]
        gt_bucket = [idx for idx in gt_order if idx not in used_gt and prev_gt_pos < gt_rank[idx] < gt_pos]
        pred_bucket = [idx for idx in pred_order if idx not in used_pred and prev_pred_pos < pred_rank[idx] < pred_pos]
        cost, sub_rows = _solve_text_bucket(gt_bucket, pred_bucket, gt, pred)
        total_cost += cost
        rows.extend(sub_rows)
        cost, row = _match_row_cost([gi], [pj], gt, pred)
        total_cost += cost
        rows.append(row)
        prev_gt_pos = gt_pos
        prev_pred_pos = pred_pos

    gt_bucket = [idx for idx in gt_order if idx not in used_gt and gt_rank[idx] > prev_gt_pos]
    pred_bucket = [idx for idx in pred_order if idx not in used_pred and pred_rank[idx] > prev_pred_pos]
    cost, sub_rows = _solve_text_bucket(gt_bucket, pred_bucket, gt, pred)
    total_cost += cost
    rows.extend(sub_rows)
    return total_cost, rows


def _fallback_segment_allowed(
    gt_seg: list[int],
    pred_seg: list[int],
    gt: list[Obj],
    pred: list[Obj],
) -> bool:
    """Keep the exhaustive fallback from aligning unrelated page regions.

    The fallback deliberately does not require an edge for every pair: this is
    what lets it recover one-to-many and many-to-one fragments.  It still
    requires either a plausible local pair or a plausible similarity between
    the concatenated segments.
    """
    if not gt_seg or not pred_seg:
        return True

    best_pair = 0.0
    best_local = 0.0
    for gi in gt_seg:
        for pj in pred_seg:
            ts, ov, box_iou, center, _ = _text_pair_features(gt[gi], pred[pj])
            best_pair = max(best_pair, ts)
            best_local = max(best_local, 0.55 * ts + 0.25 * ov + 0.20 * center)

    gt_norm = _concat_norm(gt_seg, gt)
    pred_norm = _concat_norm(pred_seg, pred)
    # This is only a segment-admission proxy. Pairwise text affinity above
    # remains partial_ratio-based; using partial_ratio again on every long
    # candidate segment makes the fallback quadratic search unnecessarily
    # expensive.
    aggregate_ts = fuzz.ratio(gt_norm, pred_norm) / 100.0 if gt_norm and pred_norm else 0.0
    gt_box = _union_bbox([gt[i] for i in gt_seg])
    pred_box = _union_bbox([pred[j] for j in pred_seg])
    spatial = max(
        _overlap_min_area(gt_box, pred_box),
        _bbox_iou(gt_box, pred_box),
        _center_similarity(gt_box, pred_box),
    )
    if len(gt_seg) == 1 and len(pred_seg) > 1:
        # A one-to-many split must be supported by every fragment.  Checking
        # only the best pair lets an unrelated block be inserted into an
        # otherwise valid group and was the cause of the 0006 date mismatch.
        for pj in pred_seg:
            ts, ov, box_iou, center, _ = _text_pair_features(gt[gt_seg[0]], pred[pj])
            if ts < 0.66 and ov < 0.90:
                return False
        return aggregate_ts >= 0.66 or spatial >= 0.85
    if len(gt_seg) > 1 and len(pred_seg) == 1:
        for gi in gt_seg:
            ts, ov, box_iou, center, _ = _text_pair_features(gt[gi], pred[pred_seg[0]])
            if ts < 0.66 and ov < 0.90:
                return False
        return aggregate_ts >= 0.66 or spatial >= 0.85
    return (
        best_pair >= 0.66
        or best_local >= 0.50
        or (aggregate_ts >= 0.55 and spatial >= 0.28)
        or (len(gt_seg) == 1 and len(pred_seg) == 1 and spatial >= 0.85)
    )


def _unconstrained_order_fallback(
    gt_ids: list[int],
    pred_ids: list[int],
    gt: list[Obj],
    pred: list[Obj],
    *,
    gt_order: list[int] | None = None,
    pred_order: list[int] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Align all remaining text in reading order with bounded segment DP.

    Unlike the regular candidate graph, this routine can cross a missing
    edge and therefore recover fragments that were split by the HTML
    extractor.  It optimizes the same character-level cost as the page
    metric.  Single-sided transitions model deletions/insertions, while
    non-empty segment transitions model all four cardinalities.
    """
    gt_order = list(gt_order) if gt_order is not None else _reading_order(gt_ids, gt)
    pred_order = list(pred_order) if pred_order is not None else _reading_order(pred_ids, pred)
    m, n = len(gt_order), len(pred_order)
    if not gt_order and not pred_order:
        return 0.0, []
    if not gt_order:
        return sum(len(_concat_norm([j], pred)) for j in pred_order), [
            {"gt": [], "pred": [j], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
            for j in pred_order
        ]
    if not pred_order:
        return sum(len(_concat_norm([i], gt)) for i in gt_order), [
            {"gt": [i], "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
            for i in gt_order
        ]
    if m * n > TEXT_FALLBACK_MAX_PRODUCT:
        return math.inf, []

    if m + n <= 24 and m * n <= 400:
        max_gt_seg = min(4, m)
        max_pred_seg = min(8, n)
    else:
        max_gt_seg = min(2, m)
        max_pred_seg = min(TEXT_FALLBACK_MAX_PRED_SEGMENT, n)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple[int, int, dict[str, Any]] | None]] = [
        [None] * (n + 1) for _ in range(m + 1)
    ]
    dp[0][0] = 0.0
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, dict[str, Any]]] = {}

    def row_cost(gt_seg: list[int], pred_seg: list[int]) -> tuple[float, dict[str, Any]]:
        key = (tuple(gt_seg), tuple(pred_seg))
        if key not in cache:
            # For one-to-many and many-to-one fragments, the spatial reading
            # order is the useful order and avoids repeatedly exploring
            # factorial permutations.  Only small genuine many-to-many
            # groups need the more expensive order search.
            if gt_seg and pred_seg and (len(gt_seg) == 1 or len(pred_seg) == 1):
                go = _reading_order(gt_seg, gt)
                po = _reading_order(pred_seg, pred)
                gt_norm = _concat_norm(go, gt)
                pred_norm = _concat_norm(po, pred)
                distance = safe_div(
                    _fast_text_cost(gt_norm, pred_norm),
                    max(len(gt_norm), len(pred_norm), 1),
                )
                cache[key] = (
                    distance * max(len(gt_norm), len(pred_norm), 1),
                    {
                        "gt": go,
                        "pred": po,
                        "distance": distance,
                        "dropped_gt": [],
                        "dropped_pred": [],
                    },
                )
            else:
                go = _reading_order(gt_seg, gt)
                po = _reading_order(pred_seg, pred)
                gt_norm = _concat_norm(go, gt)
                pred_norm = _concat_norm(po, pred)
                distance = safe_div(
                    _fast_text_cost(gt_norm, pred_norm),
                    max(len(gt_norm), len(pred_norm), 1),
                )
                cache[key] = (
                    distance * max(len(gt_norm), len(pred_norm), 1),
                    {
                        "gt": go,
                        "pred": po,
                        "distance": distance,
                        "dropped_gt": [],
                        "dropped_pred": [],
                    },
                )
        cost, row = cache[key]
        return cost, dict(row)

    for i in range(m + 1):
        for j in range(n + 1):
            if not math.isfinite(dp[i][j]):
                continue
            if i < m:
                cost, row = row_cost([gt_order[i]], [])
                if dp[i][j] + cost < dp[i + 1][j]:
                    dp[i + 1][j] = dp[i][j] + cost
                    back[i + 1][j] = (i, j, row)
            if j < n:
                cost, row = row_cost([], [pred_order[j]])
                if dp[i][j] + cost < dp[i][j + 1]:
                    dp[i][j + 1] = dp[i][j] + cost
                    back[i][j + 1] = (i, j, row)

            for a in range(1, min(max_gt_seg, m - i) + 1):
                for b in range(1, min(max_pred_seg, n - j) + 1):
                    gseg = gt_order[i : i + a]
                    pseg = pred_order[j : j + b]
                    if not _fallback_segment_allowed(gseg, pseg, gt, pred):
                        continue
                    cost, row = row_cost(gseg, pseg)
                    if dp[i][j] + cost < dp[i + a][j + b]:
                        dp[i + a][j + b] = dp[i][j] + cost
                        back[i + a][j + b] = (i, j, row)

    if not math.isfinite(dp[m][n]):
        return math.inf, []
    rows: list[dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        previous = back[i][j]
        if previous is None:
            return math.inf, []
        pi, pj, row = previous
        rows.append(row)
        i, j = pi, pj
    rows.reverse()
    return dp[m][n], rows


def _strong_fragment_recovery(
    gt_ids: list[int],
    pred_ids: list[int],
    gt: list[Obj],
    pred: list[Obj],
) -> tuple[float, list[dict[str, Any]], set[int], set[int]]:
    """Recover high-confidence one-to-many groups before order alignment.

    A fragmented infer result can be impossible to recover from a monotone
    page alignment when blocks from another column interleave by y-coordinate.
    Lock only groups with near-exact fragment text and high aggregate coverage;
    this prevents a short common token from stealing a larger, correct group.
    """
    candidates: list[tuple[float, int, list[int], dict[str, Any]]] = []
    for gi in gt_ids:
        gt_norm = _concat_norm([gi], gt)
        if len(gt_norm) < 8:
            continue
        fragment_ids = [
            pj
            for pj in pred_ids
            if len(_concat_norm([pj], pred)) <= max(2, int(len(gt_norm) * 0.65))
            and _partial_text_similarity(gt[gi].text, pred[pj].text) >= 90.0
        ]
        if len(fragment_ids) < 2:
            continue
        po = sorted(fragment_ids, key=lambda pj: (pred[pj].bbox.y, pred[pj].bbox.x))
        pred_norm = _concat_norm(po, pred)
        distance = _text_distance_from_norm(gt_norm, pred_norm)
        coverage = safe_div(len(pred_norm), len(gt_norm))
        if distance > 0.25 or coverage < 0.70:
            continue
        row = {
            "gt": [gi],
            "pred": po,
            "distance": distance,
            "dropped_gt": [],
            "dropped_pred": [],
        }
        # Prefer near-exact, high-coverage groups and then larger groups.
        candidates.append((-coverage + distance, gi, po, row))

    candidates.sort(key=lambda x: (x[0], x[1]))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for _, gi, po, row in candidates:
        if gi in used_gt or any(pj in used_pred for pj in po):
            continue
        gt_norm = _concat_norm([gi], gt)
        pred_norm = _concat_norm(po, pred)
        used_gt.add(gi)
        used_pred.update(po)
        total_cost += _text_distance_from_norm(gt_norm, pred_norm) * max(len(gt_norm), len(pred_norm), 1)
        rows.append(row)
    return total_cost, rows, used_gt, used_pred


def _whole_page_text_cost_from_orders(
    gt_order: list[int],
    pred_order: list[int],
    gt: list[Obj],
    pred: list[Obj],
) -> float:
    gt_norm = _concat_norm(gt_order, gt)
    pred_norm = _concat_norm(pred_order, pred)
    return _text_distance_from_norm(gt_norm, pred_norm) * max(len(gt_norm), len(pred_norm), 1)


def _whole_page_text_cost_from_rows(
    rows: list[dict[str, Any]],
    gt: list[Obj],
    pred: list[Obj],
) -> float:
    gt_norm = "".join(_concat_norm(row.get("gt", []), gt) for row in rows)
    pred_norm = "".join(_concat_norm(row.get("pred", []), pred) for row in rows)
    return _text_distance_from_norm(gt_norm, pred_norm) * max(len(gt_norm), len(pred_norm), 1)


def _fast_text_cost(gt_norm: str, pred_norm: str) -> float:
    """Cheap proxy used while searching block groupings.

    The final page score still uses `_text_distance_from_norm`, including the
    special `xi` wildcard semantics.  Recomputing that quadratic DP for every
    candidate segment makes dataset-scale evaluation impractical.
    """
    if not gt_norm and not pred_norm:
        return 0.0
    if not gt_norm or not pred_norm:
        return float(max(len(gt_norm), len(pred_norm)))
    return (1.0 - fuzz.ratio(gt_norm, pred_norm) / 100.0) * max(
        len(gt_norm), len(pred_norm), 1
    )


def _best_whole_page_text_cost(
    gt_ids: list[int],
    pred_ids: list[int],
    gt: list[Obj],
    pred: list[Obj],
    rows: list[dict[str, Any]],
) -> float:
    # The metric no longer sums per-group edit costs.  It evaluates the page
    # text once after merging all groups, then picks the best deterministic
    # whole-page ordering hypothesis.  For a fixed order, the optimal merge
    # strategy is the single merged group because group boundaries can only add
    # edit-script constraints.
    candidates = [
        _whole_page_text_cost_from_orders(_reading_order(gt_ids, gt), _reading_order(pred_ids, pred), gt, pred),
        _whole_page_text_cost_from_orders(_x_order(gt_ids, gt), _x_order(pred_ids, pred), gt, pred),
    ]
    if rows:
        candidates.append(_whole_page_text_cost_from_rows(rows, gt, pred))
    return min(x for x in candidates if math.isfinite(x))


def _row_order_text_cost(
    rows: list[dict[str, Any]],
    order: tuple[int, ...],
    gt: list[Obj],
    pred: list[Obj],
) -> float:
    ordered = [rows[i] for i in order]
    return _whole_page_text_cost_from_rows(ordered, gt, pred)


def _optimize_match_row_order(
    rows: list[dict[str, Any]],
    gt: list[Obj],
    pred: list[Obj],
) -> tuple[float, list[dict[str, Any]]]:
    """Search the order of existing groups against the page-level ED.

    Group boundaries do not affect the final concatenated strings; their
    order does.  Exact permutation search is used for small pages.  For
    larger pages a beam search keeps the best partial orders according to a
    prefix edit cost plus the unavoidable remaining-length penalty.
    """
    if not rows:
        return math.inf, rows
    if len(rows) <= 1:
        return _whole_page_text_cost_from_rows(rows, gt, pred), rows
    if len(rows) > TEXT_PAGE_ORDER_BEAM_ROWS:
        return _whole_page_text_cost_from_rows(rows, gt, pred), rows

    def row_gt(i: int) -> str:
        return _concat_norm(rows[i].get("gt", []), gt)

    def row_pred(i: int) -> str:
        return _concat_norm(rows[i].get("pred", []), pred)

    gt_parts = [row_gt(i) for i in range(len(rows))]
    pred_parts = [row_pred(i) for i in range(len(rows))]
    baseline_cost = _whole_page_text_cost_from_rows(rows, gt, pred)
    if len(rows) <= TEXT_PAGE_ORDER_EXACT_ROWS:
        best: tuple[float, tuple[int, ...]] | None = None
        for order in permutations(range(len(rows))):
            cost = _row_order_text_cost(rows, order, gt, pred)
            if best is None or cost < best[0]:
                best = (cost, order)
        assert best is not None
        return best[0], [rows[i] for i in best[1]]

    # State: (heuristic, exact_prefix_cost, order, remaining_mask, gt_text, pred_text)
    # Prefix strings are retained to avoid reconstructing them repeatedly.
    states: list[tuple[float, float, tuple[int, ...], int, str, str]] = [
        (0.0, 0.0, (), (1 << len(rows)) - 1, "", "")
    ]
    for _ in range(len(rows)):
        expanded: list[tuple[float, float, tuple[int, ...], int, str, str]] = []
        for _, _, order, remaining, gt_text, pred_text in states:
            candidates = [i for i in range(len(rows)) if remaining & (1 << i)]
            for i in candidates:
                next_gt = gt_text + gt_parts[i]
                next_pred = pred_text + pred_parts[i]
                prefix_cost = _fast_text_cost(next_gt, next_pred)
                rem = remaining & ~(1 << i)
                rem_gt_len = sum(len(gt_parts[j]) for j in range(len(rows)) if rem & (1 << j))
                rem_pred_len = sum(len(pred_parts[j]) for j in range(len(rows)) if rem & (1 << j))
                heuristic = prefix_cost + abs(rem_gt_len - rem_pred_len)
                expanded.append(
                    (
                        heuristic,
                        prefix_cost,
                        order + (i,),
                        rem,
                        next_gt,
                        next_pred,
                    )
                )
        expanded.sort(key=lambda x: (x[0], x[1], x[2]))
        states = expanded[:TEXT_PAGE_ORDER_BEAM_SIZE]
        if not states:
            break

    best_final: tuple[float, tuple[int, ...]] | None = None
    for _, _, order, _, gt_text, pred_text in states:
        cost = _text_distance_from_norm(gt_text, pred_text) * max(len(gt_text), len(pred_text), 1)
        if best_final is None or cost < best_final[0]:
            best_final = (cost, order)
    if best_final is None or baseline_cost <= best_final[0]:
        return baseline_cost, rows
    return best_final[0], [rows[i] for i in best_final[1]]


def text_group_matches(gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    if not gt and not pred:
        return 0.0, []

    gt_ids = list(range(len(gt)))
    pred_ids = list(range(len(pred)))
    # Restore the previous connected-component behavior on small/medium
    # pages.  On larger pages use the bounded whole-pool fallback directly;
    # running several exponential-style matchers in sequence is both slow and
    # can make a fragmented page lose valid many-to-one alignments.
    product = len(gt_ids) * len(pred_ids)
    fragment_cost, fragment_rows, used_fragment_gt, used_fragment_pred = _strong_fragment_recovery(
        gt_ids, pred_ids, gt, pred
    )
    if used_fragment_gt:
        remaining_gt = [i for i in gt_ids if i not in used_fragment_gt]
        remaining_pred = [j for j in pred_ids if j not in used_fragment_pred]
        remainder_cost, remainder_rows = _unconstrained_order_fallback(
            remaining_gt, remaining_pred, gt, pred
        )
        if math.isfinite(remainder_cost):
            total_cost, rows = fragment_cost + remainder_cost, fragment_rows + remainder_rows
        else:
            total_cost, rows = math.inf, []
    elif product <= 120 and len(gt_ids) + len(pred_ids) <= 30:
        total_cost, rows = _component_text_dp(gt_ids, pred_ids, gt, pred, allow_sparse_split=True)
    else:
        total_cost, rows = _unconstrained_order_fallback(gt_ids, pred_ids, gt, pred)
        # A page can contain columns or overlapping rows.  In that case a
        # strict y-then-x order can interleave fragments from two columns and
        # make a valid one-to-many match impossible.  Evaluate a horizontal
        # order as a second global hypothesis; this is still polynomial and
        # avoids factorial permutation search.
        x_cost, x_rows = _unconstrained_order_fallback(
            gt_ids,
            pred_ids,
            gt,
            pred,
            gt_order=_x_order(gt_ids, gt),
            pred_order=_x_order(pred_ids, pred),
        )
        if math.isfinite(x_cost) and x_cost < total_cost:
            total_cost, rows = x_cost, x_rows

    # If strong fragments were recovered, the remainder DP already provides
    # the relevant fallback. Running another full-pool DP here duplicates a
    # quadratic search and is especially expensive on fragmented pages.
    if not used_fragment_gt and product <= TEXT_FALLBACK_MAX_PRODUCT:
        fallback_cost, fallback_rows = _unconstrained_order_fallback(gt_ids, pred_ids, gt, pred)
        if math.isfinite(fallback_cost) and fallback_cost < total_cost:
            total_cost, rows = fallback_cost, fallback_rows
    total_gt_len = sum(len(_concat_norm([i], gt)) for i in gt_ids)
    total_pred_len = sum(len(_concat_norm([j], pred)) for j in pred_ids)
    for comp in rows:
        gt_norm = _concat_norm(comp["gt"], gt)
        pred_norm = _concat_norm(comp["pred"], pred)
        comp["distance"] = _text_distance_from_norm(gt_norm, pred_norm) if gt_norm or pred_norm else 0.0
    optimized_cost, optimized_rows = _optimize_match_row_order(rows, gt, pred)
    whole_page_cost = _best_whole_page_text_cost(gt_ids, pred_ids, gt, pred, rows)
    if optimized_cost <= whole_page_cost:
        total_cost = optimized_cost
        rows = optimized_rows
    else:
        total_cost = whole_page_cost
    return safe_div(total_cost, max(total_gt_len, total_pred_len, 1)), rows


def _text_candidate_graph(
    gt: list[Obj], pred: list[Obj]
) -> tuple[dict[tuple[int, int], float], list[tuple[list[int], list[int]]], set[int], set[int]]:
    """Build the text-only bipartite graph and return its connected components."""
    edges: dict[tuple[int, int], float] = {}
    parent = list(range(len(gt) + len(pred)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for gi, g in enumerate(gt):
        for pj, p in enumerate(pred):
            score = _partial_text_similarity(g.text, p.text) / 100.0
            if score >= TEXT_EDGE_THRESHOLD:
                edges[(gi, pj)] = score
                union(gi, len(gt) + pj)

    groups: dict[int, tuple[list[int], list[int]]] = {}
    for gi in range(len(gt)):
        root = find(gi)
        groups.setdefault(root, ([], []))[0].append(gi)
    for pj in range(len(pred)):
        root = find(len(gt) + pj)
        groups.setdefault(root, ([], []))[1].append(pj)

    components = [
        (sorted(g), sorted(p))
        for g, p in groups.values()
        if g and p
    ]
    connected_gt = {gi for g, _ in components for gi in g}
    connected_pred = {pj for _, p in components for pj in p}
    return edges, components, connected_gt, connected_pred


def _text_component_orders(
    gt_ids: list[int],
    pred_ids: list[int],
    gt: list[Obj],
    pred: list[Obj],
    edges: dict[tuple[int, int], float],
) -> tuple[list[list[int]], list[list[int]]]:
    """Create a few deterministic orders, including graph-guided orders."""
    gt_read = _reading_order(gt_ids, gt)
    pred_read = _reading_order(pred_ids, pred)
    gt_x = _x_order(gt_ids, gt)
    pred_x = _x_order(pred_ids, pred)
    gt_rank = {i: k for k, i in enumerate(gt_read)}
    pred_rank = {j: k for k, j in enumerate(pred_read)}

    def graph_order(
        ids: list[int],
        side: str,
        fallback_rank: dict[int, int],
    ) -> list[int]:
        keys = []
        for idx in ids:
            if side == "gt":
                partners = [(pred_rank[j], -score) for (i, j), score in edges.items() if i == idx]
            else:
                partners = [(gt_rank[i], -score) for (i, j), score in edges.items() if j == idx]
            keys.append((min(partners) if partners else (10**9, 0.0), fallback_rank.get(idx, idx), idx))
        return [idx for _, _, idx in sorted(keys)]

    def unique(orders: list[list[int]]) -> list[list[int]]:
        out: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for order in orders:
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                out.append(order)
        return out

    gt_orders = unique([
        gt_read,
        gt_x,
        list(reversed(gt_read)),
        graph_order(gt_ids, "gt", gt_rank),
    ])
    pred_orders = unique([
        pred_read,
        pred_x,
        list(reversed(pred_read)),
        graph_order(pred_ids, "pred", pred_rank),
    ])
    return gt_orders, pred_orders


def _local_order_moves(order: list[int], max_moves: int = 40) -> list[list[int]]:
    """Generate small swaps, insertions, and local reversals."""
    moves: list[list[int]] = []
    n = len(order)
    for i in range(n):
        for j in range(max(0, i - TEXT_REORDER_WINDOW), min(n, i + TEXT_REORDER_WINDOW + 1)):
            if i == j:
                continue
            candidate = order[:]
            value = candidate.pop(i)
            candidate.insert(j, value)
            moves.append(candidate)
        for width in (2, 3, 4):
            if i + width <= n:
                candidate = order[:]
                candidate[i : i + width] = reversed(candidate[i : i + width])
                moves.append(candidate)
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for move in moves:
        key = tuple(move)
        if key not in seen:
            seen.add(key)
            unique.append(move)
        if len(unique) >= max_moves:
            return unique
    return unique


def _ordered_text(ids: list[int], objs: list[Obj]) -> str:
    return _concat_norm(ids, objs)


def _order_pair_cost(
    gt_order: list[int],
    pred_order: list[int],
    gt: list[Obj],
    pred: list[Obj],
) -> float:
    return _text_distance_from_norm(
        _ordered_text(gt_order, gt),
        _ordered_text(pred_order, pred),
    )


def _local_reorder_pair(
    gt_orders: list[list[int]],
    pred_orders: list[list[int]],
    gt: list[Obj],
    pred: list[Obj],
) -> tuple[list[int], list[int]]:
    """Optimize two block orders with bounded coordinate-descent search."""
    best: tuple[float, list[int], list[int]] | None = None
    starts = [(g, p) for g in gt_orders for p in pred_orders]
    for start_gt, start_pred in starts[:8]:
        current_gt, current_pred = start_gt[:], start_pred[:]
        current_cost = _order_pair_cost(current_gt, current_pred, gt, pred)
        if best is None or current_cost < best[0]:
            best = (current_cost, current_gt[:], current_pred[:])
        if len(current_gt) + len(current_pred) > TEXT_REORDER_MAX_BLOCKS:
            continue
        for _ in range(TEXT_REORDER_ROUNDS):
            candidate: tuple[float, list[int], list[int]] | None = None
            for order in _local_order_moves(current_gt):
                cost = _order_pair_cost(order, current_pred, gt, pred)
                if candidate is None or cost < candidate[0]:
                    candidate = (cost, order, current_pred[:])
            for order in _local_order_moves(current_pred):
                cost = _order_pair_cost(current_gt, order, gt, pred)
                if candidate is None or cost < candidate[0]:
                    candidate = (cost, current_gt[:], order)
            if candidate is None or candidate[0] >= current_cost:
                break
            current_cost, current_gt, current_pred = candidate
            if best is None or current_cost < best[0]:
                best = (current_cost, current_gt[:], current_pred[:])
    assert best is not None
    return best[1], best[2]


def _segment_supported(
    gt_seg: list[int],
    pred_seg: list[int],
    edges: dict[tuple[int, int], float],
) -> bool:
    if not gt_seg or not pred_seg:
        return False
    gt_degree = {gi: False for gi in gt_seg}
    pred_degree = {pj: False for pj in pred_seg}
    for gi in gt_seg:
        for pj in pred_seg:
            if (gi, pj) in edges:
                gt_degree[gi] = True
                pred_degree[pj] = True
    return all(gt_degree.values()) and all(pred_degree.values())


def _segment_rows(
    gt_order: list[int],
    pred_order: list[int],
    gt: list[Obj],
    pred: list[Obj],
    edges: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    """Cut two fixed orders into non-empty or single-sided groups."""
    m, n = len(gt_order), len(pred_order)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple[int, int, dict[str, Any]] | None]] = [
        [None] * (n + 1) for _ in range(m + 1)
    ]
    dp[0][0] = 0.0

    def row_cost(gs: list[int], ps: list[int]) -> tuple[float, dict[str, Any]]:
        gt_text = _ordered_text(gs, gt)
        pred_text = _ordered_text(ps, pred)
        if gs and ps:
            distance = _text_distance_from_norm(gt_text, pred_text)
            support = _segment_supported(gs, ps, edges)
            if not support:
                return inf, {}
            edge_score = sum(edges.get((gi, pj), 0.0) for gi in gs for pj in ps)
            edge_count = max(1, len(gs) * len(ps))
            bonus = 0.02 * min(len(gt_text), len(pred_text)) * edge_score / edge_count
            cost = distance * max(len(gt_text), len(pred_text), 1) - bonus
        else:
            distance = 1.0
            cost = len(gt_text) + len(pred_text)
        return cost, {
            "gt": gs[:],
            "pred": ps[:],
            "distance": distance,
            "dropped_gt": [],
            "dropped_pred": [],
        }

    for i in range(m + 1):
        for j in range(n + 1):
            if not math.isfinite(dp[i][j]):
                continue
            if i < m:
                cost, row = row_cost(gt_order[i : i + 1], [])
                if dp[i][j] + cost < dp[i + 1][j]:
                    dp[i + 1][j] = dp[i][j] + cost
                    back[i + 1][j] = (i, j, row)
            if j < n:
                cost, row = row_cost([], pred_order[j : j + 1])
                if dp[i][j] + cost < dp[i][j + 1]:
                    dp[i][j + 1] = dp[i][j] + cost
                    back[i][j + 1] = (i, j, row)
            for a in range(1, min(TEXT_MAX_SEGMENT, m - i) + 1):
                for b in range(1, min(TEXT_MAX_SEGMENT, n - j) + 1):
                    cost, row = row_cost(gt_order[i : i + a], pred_order[j : j + b])
                    if not math.isfinite(cost):
                        continue
                    if dp[i][j] + cost < dp[i + a][j + b]:
                        dp[i + a][j + b] = dp[i][j] + cost
                        back[i + a][j + b] = (i, j, row)

    if not math.isfinite(dp[m][n]):
        rows = [
            {"gt": [gi], "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
            for gi in gt_order
        ]
        rows.extend(
            {"gt": [], "pred": [pj], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
            for pj in pred_order
        )
        return rows

    rows: list[dict[str, Any]] = []
    i, j = m, n
    while i or j:
        previous = back[i][j]
        if previous is None:
            break
        pi, pj, row = previous
        rows.append(row)
        i, j = pi, pj
    rows.reverse()
    return rows


def _reorder_rows(rows: list[dict[str, Any]], gt: list[Obj], pred: list[Obj]) -> list[dict[str, Any]]:
    if len(rows) <= 1:
        return rows

    def row_box(row: dict[str, Any], side: str) -> BBox:
        objs = [gt[i] for i in row["gt"]] if side == "gt" else [pred[j] for j in row["pred"]]
        return _union_bbox(objs) if objs else BBox(0.0, 0.0, 0.0, 0.0)

    reading = sorted(range(len(rows)), key=lambda k: (
        min(row_box(rows[k], "gt").y, row_box(rows[k], "pred").y),
        min(row_box(rows[k], "gt").x, row_box(rows[k], "pred").x),
    ))
    x_order = sorted(range(len(rows)), key=lambda k: (
        min(row_box(rows[k], "gt").x, row_box(rows[k], "pred").x),
        min(row_box(rows[k], "gt").y, row_box(rows[k], "pred").y),
    ))

    def cost(order: list[int]) -> float:
        gs = [i for k in order for i in rows[k]["gt"]]
        ps = [j for k in order for j in rows[k]["pred"]]
        return _order_pair_cost(gs, ps, gt, pred)

    starts = [reading, x_order, list(reversed(reading))]
    best = min(starts, key=cost)
    if len(rows) <= TEXT_REORDER_MAX_BLOCKS:
        for _ in range(TEXT_REORDER_ROUNDS):
            candidate = min(
                _local_order_moves(best, max_moves=60),
                key=cost,
                default=best,
            )
            if cost(candidate) >= cost(best):
                break
            best = candidate
    return [rows[k] for k in best]


def _graph_text_matches(gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    if not gt and not pred:
        return 0.0, []

    edges, components, connected_gt, connected_pred = _text_candidate_graph(gt, pred)
    rows: list[dict[str, Any]] = []
    components.sort(key=lambda pair: min(
        [gt[i].bbox.y for i in pair[0]] + [pred[j].bbox.y for j in pair[1]]
    ))

    for gt_ids, pred_ids in components:
        gt_orders, pred_orders = _text_component_orders(gt_ids, pred_ids, gt, pred, edges)
        gt_order, pred_order = _local_reorder_pair(gt_orders, pred_orders, gt, pred)
        rows.extend(_segment_rows(gt_order, pred_order, gt, pred, edges))

    residual_gt = [i for i in range(len(gt)) if i not in connected_gt]
    residual_pred = [j for j in range(len(pred)) if j not in connected_pred]
    rows.extend(
        {"gt": [i], "pred": [], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
        for i in _reading_order(residual_gt, gt)
    )
    rows.extend(
        {"gt": [], "pred": [j], "distance": 1.0, "dropped_gt": [], "dropped_pred": []}
        for j in _reading_order(residual_pred, pred)
    )

    rows = _reorder_rows(rows, gt, pred)
    gt_text = "".join(_ordered_text(row["gt"], gt) for row in rows)
    pred_text = "".join(_ordered_text(row["pred"], pred) for row in rows)
    distance = _text_distance_from_norm(gt_text, pred_text)
    for row in rows:
        gs, ps = row["gt"], row["pred"]
        if gs and ps:
            row["distance"] = _text_distance_from_norm(
                _ordered_text(gs, gt), _ordered_text(ps, pred)
            )
    return distance, rows


# This is the public implementation used by evaluate_page_v3.  The older
# experimental matchers above are kept only for historical reproducibility.
def text_group_matches(gt: list[Obj], pred: list[Obj]) -> tuple[float, list[dict[str, Any]]]:
    return _graph_text_matches(gt, pred)


def _union_bboxes(boxes: list[BBox]) -> BBox:
    if not boxes:
        return BBox(0.0, 0.0, 0.0, 0.0)
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x + b.w for b in boxes)
    y2 = max(b.y + b.h for b in boxes)
    return BBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def _union_bbox(objs: list[Obj]) -> BBox:
    return _union_bboxes([o.bbox for o in objs])


def _union_visual_bbox(objs: list[Obj]) -> BBox:
    return _union_bboxes([o.visual_bbox for o in objs if o.visual_bbox is not None])


def _center(b: BBox) -> tuple[float, float]:
    return b.x + b.w / 2.0, b.y + b.h / 2.0


def _size_sim(gt_box: BBox, pred_box: BBox) -> float:
    eps = 1e-6
    w = abs(math.log((pred_box.w + eps) / (gt_box.w + eps)))
    h = abs(math.log((pred_box.h + eps) / (gt_box.h + eps)))
    return math.exp(-(w + h) / 2.0)


def _build_layout_nodes(debug: dict[str, Any]) -> list[dict[str, Any]]:
    gt_all: list[Obj] = debug["gt"]
    pred_all: list[Obj] = debug["pred"]
    gt_index = {id(o): i for i, o in enumerate(gt_all)}
    pred_index = {id(o): i for i, o in enumerate(pred_all)}
    nodes: list[dict[str, Any]] = []

    for group_name, gt_name, pred_name, prefix in [
        ("tables", "gt_tables", "pred_tables", "T"),
        ("formulas", "gt_formulas", "pred_formulas", "F"),
    ]:
        typed_gt: list[Obj] = debug[gt_name]
        typed_pred: list[Obj] = debug[pred_name]
        for k, m in enumerate(debug[group_name]):
            gt_ids = [gt_index[id(typed_gt[m.gt_index])]]
            pred_ids = [pred_index[id(typed_pred[m.pred_index])]]
            nodes.append(
                {
                    "id": f"{prefix}{k}",
                    "kind": group_name[:-1],
                    "gt": gt_ids,
                    "pred": pred_ids,
                    "gt_bbox": _union_bbox([gt_all[i] for i in gt_ids]),
                    "pred_bbox": _union_bbox([pred_all[i] for i in pred_ids]),
                    "gt_visual_bbox": _union_visual_bbox([gt_all[i] for i in gt_ids]),
                    "pred_visual_bbox": _union_visual_bbox([pred_all[i] for i in pred_ids]),
                }
            )

    typed_gt_text: list[Obj] = debug["gt_text"]
    typed_pred_text: list[Obj] = debug["pred_text"]
    has_non_text_anchor = bool(nodes)
    for k, comp in enumerate(debug["texts"]):
        if not comp["gt"] or not comp["pred"]:
            continue
        if has_non_text_anchor and float(comp.get("distance", 1.0)) > TEXT_LAYOUT_MAX_DISTANCE_WITH_ANCHORS:
            continue
        gt_ids = [gt_index[id(typed_gt_text[i])] for i in comp["gt"]]
        pred_ids = [pred_index[id(typed_pred_text[i])] for i in comp["pred"]]
        nodes.append(
            {
                "id": f"X{k}",
                "kind": "text",
                "gt": gt_ids,
                "pred": pred_ids,
                "gt_bbox": _union_bbox([gt_all[i] for i in gt_ids]),
                "pred_bbox": _union_bbox([pred_all[i] for i in pred_ids]),
                "gt_visual_bbox": _union_visual_bbox([gt_all[i] for i in gt_ids]),
                "pred_visual_bbox": _union_visual_bbox([pred_all[i] for i in pred_ids]),
            }
        )
    return nodes


def _knn_edges(nodes: list[dict[str, Any]], k: int = 3) -> list[tuple[int, int]]:
    if len(nodes) <= 1:
        return []
    edges: set[tuple[int, int]] = set()
    centers = [_center(n["gt_bbox"]) for n in nodes]
    for i, ci in enumerate(centers):
        neighbors = []
        for j, cj in enumerate(centers):
            if i == j:
                continue
            neighbors.append((math.hypot(ci[0] - cj[0], ci[1] - cj[1]), j))
        for _, j in sorted(neighbors)[:k]:
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def _edge_similarity(gt_a: BBox, gt_b: BBox, pred_a: BBox, pred_b: BBox) -> float:
    eps = 1e-6
    ga = _center(gt_a)
    gb = _center(gt_b)
    pa = _center(pred_a)
    pb = _center(pred_b)
    vg = (gb[0] - ga[0], gb[1] - ga[1])
    vp = (pb[0] - pa[0], pb[1] - pa[1])
    lg = math.hypot(vg[0], vg[1])
    lp = math.hypot(vp[0], vp[1])
    if lg <= eps and lp <= eps:
        return 1.0
    if lg <= eps or lp <= eps:
        return 0.0
    dir_sim = max(0.0, (vg[0] * vp[0] + vg[1] * vp[1]) / (lg * lp))
    len_sim = math.exp(-abs(math.log((lp + eps) / (lg + eps))))
    vec_err = math.hypot(vp[0] - vg[0], vp[1] - vg[1]) / math.sqrt(2.0)
    vec_sim = max(0.0, 1.0 - vec_err)
    return 0.4 * dir_sim + 0.3 * len_sim + 0.3 * vec_sim


def layout_graph_similarity(
    gt_all: list[Obj],
    pred_all: list[Obj],
    debug: dict[str, Any],
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    if not gt_all and not pred_all:
        return 1.0, [], []
    nodes = _build_layout_nodes(debug)
    matched_gt = sum(len(n["gt"]) for n in nodes)
    matched_pred = sum(len(n["pred"]) for n in nodes)
    matched_text_gt = {
        idx
        for n in nodes
        if n["kind"] == "text"
        for idx in n["gt"]
    }
    matched_text_pred = {
        idx
        for n in nodes
        if n["kind"] == "text"
        for idx in n["pred"]
    }
    gt_denominator = sum(1 for i, o in enumerate(gt_all) if o.kind != "text" or i in matched_text_gt)
    pred_denominator = sum(1 for i, o in enumerate(pred_all) if o.kind != "text" or i in matched_text_pred)
    recall = matched_gt / max(gt_denominator, 1)
    precision = matched_pred / max(pred_denominator, 1)
    coverage_f1 = safe_div(2 * precision * recall, precision + recall)
    if not nodes:
        return 0.15 * coverage_f1, [], []

    node_scores = []
    for n in nodes:
        cg = _center(n["gt_bbox"])
        cp = _center(n["pred_bbox"])
        center_sim = max(0.0, 1.0 - math.hypot(cp[0] - cg[0], cp[1] - cg[1]) / math.sqrt(2.0))
        node_scores.append(0.5 * center_sim + 0.5 * _size_sim(n["gt_bbox"], n["pred_bbox"]))
    node_score = safe_div(sum(node_scores), len(node_scores))

    raw_edges = _knn_edges(nodes)
    edges: list[dict[str, Any]] = []
    for i, j in raw_edges:
        sim = _edge_similarity(nodes[i]["gt_bbox"], nodes[j]["gt_bbox"], nodes[i]["pred_bbox"], nodes[j]["pred_bbox"])
        edges.append({"i": i, "j": j, "similarity": sim})
    edge_score = safe_div(sum(e["similarity"] for e in edges), len(edges)) if edges else node_score
    score = 0.55 * edge_score + 0.30 * node_score + 0.15 * coverage_f1
    return score, nodes, edges


def evaluate_page_v3(gt_page: dict[str, Any], pred_page: PageData) -> tuple[PageObjectiveScores, dict[str, Any]]:
    raw_gt_all = gt_objects(gt_page)
    raw_pred_all = pred_objects(pred_page)
    gt_tables = [x for x in raw_gt_all if x.kind == "table"]
    pred_tables = [x for x in raw_pred_all if x.kind == "table"]
    gt_formulas = [x for x in raw_gt_all if x.kind == "formula"]
    pred_formulas = [x for x in raw_pred_all if x.kind == "formula"]
    gt_text = [x for x in raw_gt_all if x.kind == "text"]
    pred_text = [x for x in raw_pred_all if x.kind == "text"]

    if gt_tables:
        teds, table_matches = _typed_score(
            gt_tables,
            pred_tables,
            lambda g, p: table_teds(g.raw_html or g.text, p.raw_html or p.text, structure_only=False),
            _table_match_score,
            min_score=TABLE_MATCH_MIN_SCORE,
        )
        teds_s, _ = _typed_score(
            gt_tables,
            pred_tables,
            lambda g, p: table_teds(g.raw_html or g.text, p.raw_html or p.text, structure_only=True),
            _table_match_score,
            min_score=TABLE_MATCH_MIN_SCORE,
        )
    else:
        teds = None
        teds_s = None
        table_matches = []

    matched_gt_tables = {m.gt_index for m in table_matches}
    matched_pred_tables = {m.pred_index for m in table_matches}
    gt_formula_text = [_formula_as_text_obj(x) for x in gt_formulas]
    pred_formula_text = [_formula_as_text_obj(x) for x in pred_formulas]
    gt_unmatched_table_text: list[Obj] = []
    pred_unmatched_table_text: list[Obj] = []
    for i, x in enumerate(gt_tables):
        if i not in matched_gt_tables:
            gt_unmatched_table_text.extend(_table_as_text_objs(x))
    for i, x in enumerate(pred_tables):
        if i not in matched_pred_tables:
            pred_unmatched_table_text.extend(_table_as_text_objs(x))
    gt_text.extend(gt_formula_text)
    pred_text.extend(pred_formula_text)
    gt_text.extend(gt_unmatched_table_text)
    pred_text.extend(pred_unmatched_table_text)
    cdm = None
    formula_matches: list[Match] = []
    text_dist, text_matches = text_group_matches(gt_text, pred_text)
    gt_matched_tables = [x for i, x in enumerate(gt_tables) if i in matched_gt_tables]
    pred_matched_tables = [x for i, x in enumerate(pred_tables) if i in matched_pred_tables]
    gt_all = gt_matched_tables + [x for x in raw_gt_all if x.kind not in {"table", "formula"}] + gt_formula_text + gt_unmatched_table_text
    pred_all = pred_matched_tables + [x for x in raw_pred_all if x.kind not in {"table", "formula"}] + pred_formula_text + pred_unmatched_table_text
    debug = {
        "gt": gt_all,
        "pred": pred_all,
        "gt_tables": gt_tables,
        "pred_tables": pred_tables,
        "gt_formulas": gt_formulas,
        "pred_formulas": pred_formulas,
        "gt_text": gt_text,
        "pred_text": pred_text,
        "tables": table_matches,
        "formulas": formula_matches,
        "texts": text_matches,
    }
    layout, layout_nodes, layout_edges = layout_graph_similarity(gt_all, pred_all, debug)
    debug["layout_nodes"] = layout_nodes
    debug["layout_edges"] = layout_edges

    scores = PageObjectiveScores(
        page_id=f"{int(gt_page['ID']):04d}",
        table_teds=teds,
        table_teds_s=teds_s,
        formula_cdm=cdm,
        text_edit_distance=text_dist,
        layout_graph_similarity=layout,
        gt_tables=len(gt_tables),
        pred_tables=len(pred_tables),
        gt_formulas=len(gt_formulas),
        pred_formulas=len(pred_formulas),
        gt_text_blocks=len(gt_text),
        pred_text_blocks=len(pred_text),
    )
    return scores, debug


def _visual_box(o: Obj) -> BBox:
    return o.visual_bbox if o.visual_bbox is not None else o.bbox


def _rect(o: Obj, color: str, label: str) -> str:
    b = _visual_box(o)
    tip = html.escape(f"{label}: {o.text or o.latex or o.raw_html}"[:500])
    return (
        f'<div class="box pixel-box" title="{tip}" data-x="{b.x:.6f}" data-y="{b.y:.6f}" data-w="{b.w:.6f}" data-h="{b.h:.6f}" style="border-color:{color}">'
        f'<span style="background:{color}">{html.escape(label)}</span></div>'
    )


def _line(a: BBox, b: BBox, color: str, title: str) -> str:
    ax, ay = _center(a)
    bx, by = _center(b)
    return (
        f'<line class="pixel-line" data-x1="{ax:.6f}" data-y1="{ay:.6f}" data-x2="{bx:.6f}" data-y2="{by:.6f}" '
        f'stroke="{color}" stroke-width="1.4" opacity="0.55"><title>{html.escape(title)}</title></line>'
    )


def _href_for_target(base: Path, target: str | Path) -> str:
    value = str(target)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return value
    target_path = Path(value)
    if not target_path.is_absolute():
        target_path = target_path.resolve()
    return os.path.relpath(target_path, start=base.parent).replace(os.sep, "/")


def write_match_visualization(
    path: str | Path,
    model: str,
    scores: PageObjectiveScores,
    debug: dict[str, Any],
    *,
    gt_image: str | None = None,
    rendered_html: str | None = None,
    html_href: str | None = None,
) -> None:
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b", "#e377c2"]
    gt: list[Obj] = debug["gt"]
    pred: list[Obj] = debug["pred"]
    labels_gt: dict[tuple[str, int], tuple[str, str]] = {}
    labels_pred: dict[tuple[str, int], tuple[str, str]] = {}
    gt_index = {id(o): i for i, o in enumerate(gt)}
    pred_index = {id(o): i for i, o in enumerate(pred)}
    for group_name, gt_name, pred_name in [
        ("tables", "gt_tables", "pred_tables"),
        ("formulas", "gt_formulas", "pred_formulas"),
    ]:
        typed_gt: list[Obj] = debug[gt_name]
        typed_pred: list[Obj] = debug[pred_name]
        for k, m in enumerate(debug[group_name]):
            color = colors[k % len(colors)]
            labels_gt[(group_name, gt_index[id(typed_gt[m.gt_index])])] = (color, f"{group_name[0].upper()}{k}")
            labels_pred[(group_name, pred_index[id(typed_pred[m.pred_index])])] = (color, f"{group_name[0].upper()}{k}")
    text_offset = len(labels_gt)
    typed_gt_text: list[Obj] = debug["gt_text"]
    typed_pred_text: list[Obj] = debug["pred_text"]
    for k, comp in enumerate(debug["texts"]):
        color = colors[(k + text_offset) % len(colors)]
        for gi in comp["gt"]:
            labels_gt[("texts", gt_index[id(typed_gt_text[gi])])] = (color, f"X{k}")
        for pj in comp["pred"]:
            labels_pred[("texts", pred_index[id(typed_pred_text[pj])])] = (color, f"X{k}")

    gt_boxes = []
    for i, o in enumerate(gt):
        label = labels_gt.get((o.kind + "s", i))
        if label:
            gt_boxes.append(_rect(o, label[0], label[1]))
    pred_boxes = []
    for i, o in enumerate(pred):
        label = labels_pred.get((o.kind + "s", i))
        if label:
            pred_boxes.append(_rect(o, label[0], label[1]))

    layout_nodes: list[dict[str, Any]] = debug.get("layout_nodes", [])
    layout_edges: list[dict[str, Any]] = debug.get("layout_edges", [])
    gt_lines = []
    pred_lines = []
    edge_rows = []
    for e in layout_edges:
        i = int(e["i"])
        j = int(e["j"])
        sim = float(e["similarity"])
        color = "#333333" if sim >= 0.75 else "#cc7a00" if sim >= 0.45 else "#b00020"
        ni = layout_nodes[i]
        nj = layout_nodes[j]
        label = f"{ni['id']} - {nj['id']}"
        gt_lines.append(_line(ni["gt_visual_bbox"], nj["gt_visual_bbox"], color, f"{label} GT"))
        pred_lines.append(_line(ni["pred_visual_bbox"], nj["pred_visual_bbox"], color, f"{label} infer sim={sim:.4f}"))
        edge_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{sim:.4f}</td>"
            f"<td>{html.escape(ni['kind'])}</td>"
            f"<td>{html.escape(nj['kind'])}</td>"
            "</tr>"
        )

    def snippet(text: str, limit: int = 500) -> str:
        value = clean_text(text)
        if len(value) > limit:
            value = value[:limit] + "..."
        return html.escape(value)

    out = Path(path)
    gt_href = html.escape(_href_for_target(out, gt_image) if gt_image else "", quote=True)
    rendered_href = html.escape(_href_for_target(out, rendered_html) if rendered_html else "", quote=True)
    html_link = html.escape(_href_for_target(out, html_href) if html_href else "", quote=True)

    match_rows = []
    for group_name, gt_name, pred_name, title in [
        ("tables", "gt_tables", "pred_tables", "Table"),
        ("formulas", "gt_formulas", "pred_formulas", "Formula"),
    ]:
        typed_gt: list[Obj] = debug[gt_name]
        typed_pred: list[Obj] = debug[pred_name]
        for k, m in enumerate(debug[group_name]):
            color = colors[k % len(colors)]
            g = typed_gt[m.gt_index]
            p = typed_pred[m.pred_index]
            g_text = g.raw_html or g.latex or g.text if title == "Table" else g.latex or g.raw_html or g.text
            p_text = p.raw_html or p.latex or p.text if title == "Table" else p.latex or p.raw_html or p.text
            match_rows.append(
                "<tr>"
                f'<td><span class="chip" style="background:{color}">{html.escape(title[0] + str(k))}</span></td>'
                f"<td>{html.escape(title)}</td>"
                f"<td>{m.score:.4f}</td>"
                f"<td>{snippet(g_text)}</td>"
                f"<td>{snippet(p_text)}</td>"
                "</tr>"
            )
    for k, comp in enumerate(debug["texts"]):
        color = colors[(k + text_offset) % len(colors)]
        gt_text = " ".join(typed_gt_text[i].text for i in comp["gt"])
        pred_text = " ".join(typed_pred_text[i].text for i in comp["pred"])
        match_rows.append(
            "<tr>"
            f'<td><span class="chip" style="background:{color}">X{k}</span></td>'
            "<td>Text</td>"
            f"<td>edit_dist={comp['distance']:.4f}</td>"
            f"<td>{snippet(gt_text)}</td>"
            f"<td>{snippet(pred_text)}</td>"
            "</tr>"
        )

    def metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    content = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(model)} {scores.page_id} objective matches</title>
<style>
body{{font-family:Arial,sans-serif;margin:20px;color:#222}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.panel{{position:relative;border:1px solid #bbb;background:#fafafa;overflow:hidden}}
.scene{{position:relative;width:100%;display:block}}
.scene img{{display:block;width:100%;height:auto}}
.box{{position:absolute;border:2px solid;box-sizing:border-box;background:rgba(255,255,255,.08);z-index:2}}
.box span{{color:white;font-size:11px;padding:1px 3px}}
.edges{{position:absolute;inset:0;pointer-events:none;z-index:3}}
table{{border-collapse:collapse;margin:12px 0;width:100%;table-layout:fixed}} td,th{{border:1px solid #ccc;padding:5px 7px;vertical-align:top;word-break:break-word}} code{{white-space:pre-wrap}}
.chip{{display:inline-block;color:white;font-weight:700;font-size:12px;min-width:28px;text-align:center;padding:2px 5px;border-radius:3px}}
.meta{{margin:8px 0 14px;color:#555;font-size:13px}}
</style>
<script>
function scaleCoord(value, base) {{
  return Math.abs(value) <= 1 ? value * 100 : value / Math.max(base, 1) * 100;
}}
function alignOverlay(scene) {{
  const img = scene.querySelector('img');
  const iw = img.naturalWidth || img.clientWidth || 1;
  const ih = img.naturalHeight || img.clientHeight || 1;
  for (const box of scene.querySelectorAll('.pixel-box')) {{
    const x = Number(box.dataset.x || 0), y = Number(box.dataset.y || 0);
    const w = Number(box.dataset.w || 0), h = Number(box.dataset.h || 0);
    box.style.left = scaleCoord(x, iw).toFixed(4) + '%';
    box.style.top = scaleCoord(y, ih).toFixed(4) + '%';
    box.style.width = scaleCoord(w, iw).toFixed(4) + '%';
    box.style.height = scaleCoord(h, ih).toFixed(4) + '%';
  }}
  for (const line of scene.querySelectorAll('.pixel-line')) {{
    const x1 = Number(line.dataset.x1 || 0), y1 = Number(line.dataset.y1 || 0);
    const x2 = Number(line.dataset.x2 || 0), y2 = Number(line.dataset.y2 || 0);
    line.setAttribute('x1', scaleCoord(x1, iw).toFixed(4) + '%');
    line.setAttribute('y1', scaleCoord(y1, ih).toFixed(4) + '%');
    line.setAttribute('x2', scaleCoord(x2, iw).toFixed(4) + '%');
    line.setAttribute('y2', scaleCoord(y2, ih).toFixed(4) + '%');
  }}
}}
window.addEventListener('load', () => {{
  for (const scene of document.querySelectorAll('.scene')) alignOverlay(scene);
}});
</script>
<h1>{html.escape(model)} / page {scores.page_id}</h1>
<p>table_teds={metric(scores.table_teds)}, table_teds_s={metric(scores.table_teds_s)}, formula_cdm={metric(scores.formula_cdm)}, text_edit_distance={scores.text_edit_distance:.4f}, layout_graph_similarity={scores.layout_graph_similarity:.4f}</p>
<p class="meta">GT image: {('<a href="' + gt_href + '" target="_blank">open</a>') if gt_href else 'n/a'} | rendered HTML: {('<a href="' + rendered_href + '" target="_blank">open screenshot</a>') if rendered_href else 'n/a'} | source HTML: {('<a href="' + html_link + '" target="_blank">open file</a>') if html_link else 'n/a'}</p>
<div class="grid"><section><h2>GT image + layout graph</h2><div class="panel"><div class="scene"><img src="{gt_href}" alt="gt image">{''.join(gt_boxes)}<svg class="edges" width="100%" height="100%">{''.join(gt_lines)}</svg></div></div></section><section><h2>Rendered HTML + layout graph</h2><div class="panel"><div class="scene"><img src="{rendered_href}" alt="rendered html screenshot">{''.join(pred_boxes)}<svg class="edges" width="100%" height="100%">{''.join(pred_lines)}</svg></div></div></section></div>
<h2>Matched Content</h2>
<table><thead><tr><th style="width:64px">ID</th><th style="width:90px">Type</th><th style="width:110px">Score</th><th>GT text / code</th><th>Infer text / code</th></tr></thead><tbody>{''.join(match_rows)}</tbody></table>
<h2>Layout Graph Edges</h2>
<table><thead><tr><th>Edge</th><th>Similarity</th><th>Node A</th><th>Node B</th></tr></thead><tbody>{''.join(edge_rows)}</tbody></table>
</html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["model"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) not in {None, ""}]
    if not values:
        return None
    return sum(values) / len(values)


def _weighted_quality_score(row: dict[str, Any]) -> float:
    table = row.get("table_teds")
    text_quality = 1.0 - float(row["text_edit_distance"])
    layout_quality = float(row["layout_graph_similarity"])
    total = text_quality + layout_quality
    weights = 2.0
    if table not in (None, ""):
        total += 0.3 * float(table)
        weights += 0.3
    return safe_div(total, weights)


def evaluate_all_models_v3(
    gt_json: str | Path = "AmbiHTMLBench_GT_v5_1115.json",
    infer_dir: str | Path = "all_model_infer",
    out_dir: str | Path = "outputs/objective_v3",
    *,
    viewport_width: int = 1280,
    viewport_height: int = 1800,
    visualizations: bool = True,
    max_pages: int | None = None,
    models: set[str] | None = None,
    page_ids: set[str] | None = None,
    partial_interval: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_pages = load_gt_v3(gt_json)
    root = Path(infer_dir)
    out = Path(out_dir)
    page_rows: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    partial_path = out / "page_scores.partial.csv"
    if partial_path.exists():
        with partial_path.open(encoding="utf-8") as f:
            page_rows = list(csv.DictReader(f))
        completed = {(str(r.get("model", "")), str(r.get("page_id", ""))) for r in page_rows}
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        browser_page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
        try:
            for model_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
                if models is not None and model_dir.name not in models:
                    continue
                html_dir = model_dir / "html"
                if not html_dir.exists():
                    continue
                model_rows: list[dict[str, Any]] = [r for r in page_rows if r.get("model") == model_dir.name]
                files = sorted(html_dir.glob("*.html"), key=lambda p: p.name)
                if page_ids is not None:
                    files = [p for p in files if p.name[:4] in page_ids]
                if max_pages is not None:
                    files = files[:max_pages]
                for html_path in files:
                    page_id = html_path.name[:4]
                    if (model_dir.name, page_id) in completed:
                        continue
                    gt_page = gt_pages.get(page_id)
                    if gt_page is None:
                        continue
                    pred = extract_html_with_page(browser_page, html_path, viewport_width=viewport_width, viewport_height=viewport_height)
                    scores, debug = evaluate_page_v3(gt_page, pred)
                    rendered_path = out / "visualizations" / model_dir.name / "_assets" / f"{page_id}.png"
                    rendered_image: str | None = str(rendered_path)
                    if visualizations:
                        rendered_path.parent.mkdir(parents=True, exist_ok=True)
                        if not rendered_path.exists():
                            try:
                                browser_page.screenshot(path=str(rendered_path), full_page=True)
                            except Exception:
                                rendered_image = None
                    row = {
                        "model": model_dir.name,
                        "page_id": page_id,
                        "html": str(html_path),
                        **scores.__dict__,
                    }
                    page_rows.append(row)
                    model_rows.append(row)
                    completed.add((model_dir.name, page_id))
                    if visualizations:
                        write_match_visualization(
                            out / "visualizations" / model_dir.name / f"{page_id}.html",
                            model_dir.name,
                            scores,
                            debug,
                            gt_image=str(gt_page.get("imageUrl", "")) or None,
                            rendered_html=rendered_image,
                            html_href=str(html_path.resolve()),
                        )
                    if partial_interval <= 1 or len(page_rows) % partial_interval == 0:
                        write_csv(out / "page_scores.partial.csv", page_rows)
                if model_rows:
                    table_teds_avg = _mean_present(model_rows, "table_teds")
                    table_teds_s_avg = _mean_present(model_rows, "table_teds_s")
                    formula_cdm_avg = _mean_present(model_rows, "formula_cdm")
                    leaderboard.append(
                        {
                            "model": model_dir.name,
                            "pages": len(model_rows),
                            "table_pages": sum(1 for r in model_rows if r.get("table_teds") is not None),
                            "formula_pages": sum(1 for r in model_rows if r.get("formula_cdm") not in (None, "")),
                            "table_teds": table_teds_avg,
                            "table_teds_s": table_teds_s_avg,
                            "formula_cdm": formula_cdm_avg,
                            "text_edit_distance": _mean_present(model_rows, "text_edit_distance"),
                            "layout_graph_similarity": _mean_present(model_rows, "layout_graph_similarity"),
                        }
                    )
                    leaderboard[-1]["weighted_score"] = _weighted_quality_score(leaderboard[-1])
                    write_csv(out / "page_scores.partial.csv", page_rows)
                    write_csv(out / "leaderboard.partial.csv", leaderboard)
        finally:
            browser.close()
    leaderboard.sort(
        key=lambda r: (
            -r["weighted_score"],
            r["text_edit_distance"],
            -r["layout_graph_similarity"],
        )
    )
    for i, row in enumerate(leaderboard, 1):
        row["rank"] = i
    write_csv(out / "page_scores.csv", page_rows)
    write_csv(out / "leaderboard.csv", leaderboard)
    return leaderboard, page_rows
