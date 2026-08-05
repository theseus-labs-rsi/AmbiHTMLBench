from __future__ import annotations

from dataclasses import dataclass

from .schema import Element, PageData
from .utils import iou, norm_edit_similarity, safe_div, tag_similarity


@dataclass
class Match:
    gt_idx: int
    pred_idx: int
    iou: float
    text_sim: float
    tag_sim: float
    score: float


@dataclass
class EvalResult:
    matched: int
    gt_total: int
    pred_total: int
    content_score: float
    tag_score: float
    layout_score: float
    page_iou: float
    precision: float
    recall: float
    f1: float
    total_score: float
    matches: list[Match]



def _pair_score(
    gt: Element,
    pred: Element,
    w_iou: float,
    w_text: float,
    w_tag: float,
) -> tuple[float, float, float, float]:
    ov = iou(gt.bbox, pred.bbox)
    ts = norm_edit_similarity(gt.text, pred.text)
    gs = tag_similarity(gt.tag, pred.tag)
    return ov, ts, gs, w_iou * ov + w_text * ts + w_tag * gs



def greedy_match(
    gt: list[Element],
    pred: list[Element],
    iou_threshold: float = 0.1,
    text_threshold: float = 0.0,
    w_iou: float = 0.6,
    w_text: float = 0.25,
    w_tag: float = 0.15,
) -> list[Match]:
    candidates: list[Match] = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            ov, ts, gs, s = _pair_score(g, p, w_iou=w_iou, w_text=w_text, w_tag=w_tag)
            if ov < iou_threshold and ts < text_threshold:
                continue
            candidates.append(Match(gt_idx=i, pred_idx=j, iou=ov, text_sim=ts, tag_sim=gs, score=s))

    candidates.sort(key=lambda x: x.score, reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    picked: list[Match] = []

    for c in candidates:
        if c.gt_idx in used_gt or c.pred_idx in used_pred:
            continue
        used_gt.add(c.gt_idx)
        used_pred.add(c.pred_idx)
        picked.append(c)

    return picked



def evaluate_pages(
    gt_page: PageData,
    pred_page: PageData,
    iou_threshold: float = 0.1,
    text_threshold: float = 0.0,
    weight_content: float = 0.5,
    weight_layout: float = 0.5,
) -> EvalResult:
    gt = gt_page.elements
    pred = pred_page.elements

    matches = greedy_match(
        gt,
        pred,
        iou_threshold=iou_threshold,
        text_threshold=text_threshold,
    )

    matched = len(matches)
    gt_total = len(gt)
    pred_total = len(pred)

    content_score = safe_div(sum(m.text_sim for m in matches), max(gt_total, 1))
    tag_score = safe_div(sum(m.tag_sim for m in matches), max(gt_total, 1))

    total_inter = 0.0
    for m in matches:
        g = gt[m.gt_idx].bbox
        p = pred[m.pred_idx].bbox
        ov = m.iou
        inter = ov * (g.area + p.area) / (1.0 + ov) if ov > 0 else 0.0
        total_inter += inter

    sum_gt = sum(e.bbox.area for e in gt)
    sum_pred = sum(e.bbox.area for e in pred)
    union = sum_gt + sum_pred - total_inter
    page_iou = safe_div(total_inter, union)

    precision = safe_div(matched, pred_total)
    recall = safe_div(matched, gt_total)
    f1 = safe_div(2 * precision * recall, precision + recall)
    layout_score = 0.5 * page_iou + 0.5 * f1

    total_score = weight_content * content_score + weight_layout * layout_score

    return EvalResult(
        matched=matched,
        gt_total=gt_total,
        pred_total=pred_total,
        content_score=content_score,
        tag_score=tag_score,
        layout_score=layout_score,
        page_iou=page_iou,
        precision=precision,
        recall=recall,
        f1=f1,
        total_score=total_score,
        matches=matches,
    )
