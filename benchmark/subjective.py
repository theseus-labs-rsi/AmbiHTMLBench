from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import mimetypes
from pathlib import Path
import re
from typing import Any, Iterable


SUBJECTIVE_METHOD_ID = "human_visual_fidelity_v1"
PROTOCOL_ID = "AmbiHTMLBench-subjective-v1"
PROMPT_VERSION = "AmbiHTMLBench-pilot-elo-v2-human-visual"
PROMPT_SHA256 = "786c9de357c08373de9158145f639b666650f3bcfe81a1f8c59c600ccee7d3bb"

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "subjective_v1.md"
SUBJECTIVE_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

# The prompt used in the validated benchmark is an immutable protocol artifact.
# Fail loudly if a repository edit changes it without an explicit protocol/version bump.
import hashlib as _hashlib
if _hashlib.sha256(SUBJECTIVE_PROMPT.encode("utf-8")).hexdigest() != PROMPT_SHA256:
    raise RuntimeError(
        "subjective_v1.md hash does not match the validated Prompt SHA256; "
        "bump the protocol/version and revalidate before changing the prompt"
    )


AXIS_DEFINITIONS: dict[str, dict[str, Any]] = {
    "IMAGE_VISUAL_FIDELITY": {
        "description": "图片、插画、Logo 或其他独立图像的主体形态、内部组成、裁切构图、关键图形信息与视觉完整性。",
        "names": {"image", "图片figure", "header_image", "footer_image"},
        "role": "image",
    },
    "TABLE_VISUAL_STRUCTURE": {
        "description": "表格内部行列组织、合并与分组、多级表头、网格层级、视觉密度与结构完整性。",
        "names": {"table", "表格table"},
        "role": "table",
    },
    "CHART_VISUAL_ENCODING": {
        "description": "图表内部的 series、mark、axis、legend、主要数据形态及其视觉对应关系。",
        "names": {"chart"},
        "role": "chart",
    },
    "FORMULA_2D_LAYOUT": {
        "description": "分式、上下标、根式、矩阵、对齐、公式编号及公式组的二维数学结构。",
        "names": {"display_formula", "inline_formula"},
        "role": "formula",
    },
    "ALGORITHM_BLOCK_STRUCTURE": {
        "description": "代码或伪代码块内部的缩进、换行、行包装、步骤层级、等宽排版与连续结构。",
        "names": {"algorithm"},
        "role": "algorithm",
    },
    "VERTICAL_TEXT_PRESENTATION": {
        "description": "竖排文本方向、字列排列、列间关系与阅读流。",
        "names": {"vertical_text"},
        "role": "vertical",
    },
    "LIST_HIERARCHY_PRESENTATION": {
        "description": "列表或目录的缩进层级、编号/项目符号、嵌套、分组与连续组织。",
        "names": {"列表类（目录）list", "合并列表类list_merge"},
        "role": "list",
    },
}


@dataclass(frozen=True)
class JudgeResult:
    winner: str
    confidence: int
    reason: str
    axis_scores: dict[str, int]
    raw_content: str = ""


@dataclass(frozen=True)
class PairOutcome:
    pair_set_id: str
    model_a: str
    model_b: str
    winner: str
    tie_type: str | None
    canonical_winner: str
    swapped_winner: str


@dataclass(frozen=True)
class PairRatingDecision:
    model_a: str
    model_b: str
    winner: str
    confidence: int = 3


def build_active_axes(gt_page: dict[str, Any], page_width: int, page_height: int) -> list[dict[str, Any]]:
    """Build GT-driven visual focus axes from a Quality Inspection page record."""
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page dimensions must be positive")

    content = gt_page.get("contentList", [])
    out: list[dict[str, Any]] = []
    for axis_id, spec in AXIS_DEFINITIONS.items():
        objects = [x for x in content if isinstance(x, dict) and x.get("name") in spec["names"]]
        if not objects:
            continue
        targets = []
        for i, obj in enumerate(objects, 1):
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            width = float(obj.get("width", 0.0))
            height = float(obj.get("height", 0.0))
            bbox = [
                round(1000 * left / page_width),
                round(1000 * top / page_height),
                round(1000 * (left + width) / page_width),
                round(1000 * (top + height) / page_height),
            ]
            bbox = [int(max(0, min(1000, v))) for v in bbox]
            targets.append({
                "target": f"T{i}",
                "source_id": str(obj.get("id", "")),
                "role": spec["role"],
                "bbox_1000": bbox,
            })
        out.append({"id": axis_id, "description": spec["description"], "targets": targets})
    return out


def load_quality_gt_page(gt_json: str | Path, page_id: str | int) -> dict[str, Any]:
    raw = json.loads(Path(gt_json).read_text(encoding="utf-8-sig"))
    pages = raw if isinstance(raw, list) else [raw]
    target = str(page_id).strip()
    target_num = int(target) if target.isdigit() else None
    for page in pages:
        if not isinstance(page, dict):
            continue
        value = str(page.get("ID", "")).strip()
        if value == target:
            return page
        if target_num is not None and value.isdigit() and int(value) == target_num:
            return page
    raise ValueError(f"GT page_id not found: {page_id}")


def build_task_text(active_axes: list[dict[str, Any]]) -> str:
    return (
        "系统根据 Reference GT 得到以下 active_axes / targets：\n"
        + json.dumps(active_axes, ensure_ascii=False, indent=2)
        + "\n\n请比较 Reference、Candidate A、Candidate B，并严格按系统要求输出 JSON。"
    )


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.I | re.S)
    return (m.group(1) if m else s).strip()


def _validate_result(obj: dict[str, Any], active_axes: list[dict[str, Any]]) -> JudgeResult:
    winner = str(obj.get("winner", "")).upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"invalid winner={winner!r}")
    confidence = int(obj.get("confidence"))
    if confidence not in range(1, 6):
        raise ValueError(f"invalid confidence={confidence}")

    expected = [str(x["id"]) for x in active_axes]
    axes = obj.get("axes", [])
    if not isinstance(axes, list):
        raise ValueError("axes must be list")
    got = [str(x.get("id", "")) for x in axes]
    if got != expected:
        raise ValueError(f"axis ids/order mismatch expected={expected}, got={got}")

    scores: dict[str, int] = {}
    for item in axes:
        score = int(item["score"])
        if score not in {-2, -1, 0, 1, 2}:
            raise ValueError(f"invalid axis score={score}")
        scores[str(item["id"])] = score

    return JudgeResult(
        winner=winner,
        confidence=confidence,
        reason=str(obj.get("reason", "")).strip()[:420],
        axis_scores=scores,
    )


def _tolerant_parse(text: str, active_axes: list[dict[str, Any]]) -> JudgeResult:
    """Schema-specific recovery for malformed-but-unambiguous Judge JSON.

    It never invents a winner or an axis score. If required fields cannot be
    extracted exactly, parsing fails loudly.
    """
    s = _strip_json_fence(text)
    wm = re.search(r'"winner"\s*:\s*"?(A|B|TIE)"?', s, re.I)
    cm = re.search(r'"confidence"\s*:\s*([1-5])', s, re.I)
    if not wm or not cm:
        raise ValueError("could not recover winner/confidence")

    reason = ""
    rm = re.search(r'"reason"\s*:\s*"', s, re.I)
    if rm:
        rest = s[rm.end():]
        am = re.search(r'"\s*,\s*"axes"\s*:', rest, re.I | re.S)
        reason = (rest[:am.start()] if am else rest[:320]).replace('\\"', '"').strip()

    axes = []
    for axis in active_axes:
        aid = str(axis["id"])
        patterns = [
            r'\{\s*"id"\s*:\s*"' + re.escape(aid) + r'"\s*,\s*"score"\s*:\s*(-?[0-2])\s*\}',
            r'\{\s*"score"\s*:\s*(-?[0-2])\s*,\s*"id"\s*:\s*"' + re.escape(aid) + r'"\s*\}',
        ]
        match = next((re.search(p, s, re.I) for p in patterns if re.search(p, s, re.I)), None)
        if not match:
            raise ValueError(f"could not recover axis score for {aid}")
        axes.append({"id": aid, "score": int(match.group(1))})

    return _validate_result({
        "winner": wm.group(1).upper(),
        "confidence": int(cm.group(1)),
        "reason": reason,
        "axes": axes,
    }, active_axes)


def parse_judge_response(text: str, active_axes: list[dict[str, Any]]) -> JudgeResult:
    s = _strip_json_fence(text)
    try:
        result = _validate_result(json.loads(s), active_axes)
    except Exception:
        result = _tolerant_parse(text, active_axes)
    return JudgeResult(**{**result.__dict__, "raw_content": text})


def canonicalize_result(result: JudgeResult, swapped: bool) -> JudgeResult:
    if not swapped:
        return result
    winner = "B" if result.winner == "A" else "A" if result.winner == "B" else "TIE"
    return JudgeResult(
        winner=winner,
        confidence=result.confidence,
        reason=result.reason,
        axis_scores={k: -int(v) for k, v in result.axis_scores.items()},
        raw_content=result.raw_content,
    )


def collapse_dual_orientation(
    pair_set_id: str,
    model_a: str,
    model_b: str,
    canonical_winner: str,
    swapped_winner: str,
) -> PairOutcome:
    """Collapse two canonicalized orientation judgments into one physical outcome."""
    for winner in (canonical_winner, swapped_winner):
        if winner not in {"A", "B", "TIE"}:
            raise ValueError(f"invalid winner={winner}")

    if canonical_winner == swapped_winner:
        winner = canonical_winner
        tie_type = "perceptual" if winner == "TIE" else None
    else:
        winner = "TIE"
        tie_type = "reliability"
    return PairOutcome(
        pair_set_id=pair_set_id,
        model_a=model_a,
        model_b=model_b,
        winner=winner,
        tie_type=tie_type,
        canonical_winner=canonical_winner,
        swapped_winner=swapped_winner,
    )


def to_rating_decision(outcome: PairOutcome) -> PairRatingDecision:
    return PairRatingDecision(outcome.model_a, outcome.model_b, outcome.winner, confidence=3)


def data_uri(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _base64_len(n: int) -> int:
    return 4 * math.ceil(n / 3)


def estimate_request_bytes(
    reference_image: str | Path,
    candidate_a: str | Path,
    candidate_b: str | Path,
    active_axes: list[dict[str, Any]],
    *,
    prompt: str = SUBJECTIVE_PROMPT,
    json_overhead_bytes: int = 96_000,
) -> int:
    paths = [Path(reference_image), Path(candidate_a), Path(candidate_b)]
    image_bytes = sum(_base64_len(p.stat().st_size) for p in paths)
    task = build_task_text(active_axes)
    return image_bytes + len(prompt.encode("utf-8")) + len(task.encode("utf-8")) + json_overhead_bytes


class RequestsJudgeClient:
    """OpenAI-compatible VLM Judge client used by the production Arena runner."""

    def __init__(self, api_base: str, api_key: str, model: str, *, timeout_sec: int = 180, retries: int = 3):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.retries = retries

    def judge(
        self,
        reference_image: str | Path,
        candidate_a: str | Path,
        candidate_b: str | Path,
        active_axes: list[dict[str, Any]],
    ) -> JudgeResult:
        import requests
        import time

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SUBJECTIVE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_task_text(active_axes)},
                        {"type": "text", "text": "第一张图片：Reference"},
                        {"type": "image_url", "image_url": {"url": data_uri(reference_image)}},
                        {"type": "text", "text": "第二张图片：Candidate A"},
                        {"type": "image_url", "image_url": {"url": data_uri(candidate_a)}},
                        {"type": "text", "text": "第三张图片：Candidate B"},
                        {"type": "image_url", "image_url": {"url": data_uri(candidate_b)}},
                    ],
                },
            ],
        }
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.api_base + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return parse_judge_response(content, active_axes)
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 6))
        assert last is not None
        raise last
