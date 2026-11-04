"""Objective scoring: table assignment, text graph alignment, layout, and reports.

Only the active matching implementation lives here; earlier experiments remain
available in Git history.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import html
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
from .schema import PageData
from .utils import BBox, clean_text, safe_div


TEXT_WILDCARD = "\u03be"
TABLE_MATCH_MIN_SCORE = 0.40
TEXT_LAYOUT_MAX_DISTANCE_WITH_ANCHORS = 0.75
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
                            "table_pages": sum(1 for r in model_rows if r.get("table_teds") not in (None, "")),
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
