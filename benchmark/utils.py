from __future__ import annotations

from dataclasses import dataclass
import re

from rapidfuzz.distance import Levenshtein


WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

HTML_TAG_TO_SEMANTIC = {
    "h1": "title",
    "h2": "title",
    "h3": "title",
    "h4": "title",
    "h5": "title",
    "h6": "title",
    "p": "text_block",
    "div": "text_block",
    "span": "text_block",
    "section": "text_block",
    "article": "text_block",
    "aside": "text_block",
    "blockquote": "text_block",
    "pre": "text_block",
    "code": "text_block",
    "ul": "list",
    "ol": "list",
    "li": "list",
    "table": "table",
    "caption": "table_caption",
    "figure": "figure",
    "figcaption": "figure_caption",
    "img": "figure",
    "svg": "figure",
    "canvas": "figure",
    "video": "figure",
    "header": "header",
    "footer": "footer",
    "math": "equation_isolated",
}

SEMANTIC_KEYWORDS = [
    ("equation_isolated", "equation_isolated"),
    ("equation_merged", "equation_merged"),
    ("equation_caption", "equation_caption"),
    ("page_number", "page_number"),
    ("page_footnote", "page_footnote"),
    ("figure_caption", "figure_caption"),
    ("figure_footnote", "figure_footnote"),
    ("table_caption", "table_caption"),
    ("table_footnote", "table_footnote"),
    ("text_ignore", "text_ignore"),
    ("text_block", "text_block"),
    ("list_merge", "list"),
    ("reference", "reference"),
    ("vertical", "vertical"),
    ("rotate", "rotate"),
    ("header", "header"),
    ("footer", "footer"),
    ("figure", "figure"),
    ("table", "table"),
    ("title", "title"),
    ("vague", "vague"),
    ("list", "list"),
]

TEXTUAL_TAGS = {
    "title",
    "text_block",
    "list",
    "reference",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "figure_caption",
    "figure_footnote",
    "table_caption",
    "table_footnote",
    "equation_caption",
    "vertical",
    "rotate",
}

NON_TEXTUAL_TAGS = {
    "figure",
    "table",
    "equation_isolated",
    "equation_merged",
}

IGNORE_GT_TAGS = {
    "text_ignore",
}


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)



def clean_text(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())



def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize_semantic_tag(raw: str) -> str:
    text = clean_text(raw).lower()
    if not text:
        return "unknown"

    if text in HTML_TAG_TO_SEMANTIC:
        return HTML_TAG_TO_SEMANTIC[text]

    squashed = NON_ALNUM_RE.sub("_", text).strip("_")
    if squashed in HTML_TAG_TO_SEMANTIC:
        return HTML_TAG_TO_SEMANTIC[squashed]

    for keyword, canonical in SEMANTIC_KEYWORDS:
        if keyword in squashed or keyword in text:
            return canonical

    return squashed or "unknown"


def tag_similarity(a: str, b: str) -> float:
    aa = normalize_semantic_tag(a)
    bb = normalize_semantic_tag(b)

    if aa == bb:
        if aa == "unknown":
            return 0.0
        return 1.0
    if aa in IGNORE_GT_TAGS or bb in IGNORE_GT_TAGS:
        return 0.0
    if aa in NON_TEXTUAL_TAGS or bb in NON_TEXTUAL_TAGS:
        families = {
            "figure": {"figure", "figure_caption", "figure_footnote"},
            "table": {"table", "table_caption", "table_footnote"},
            "equation": {"equation_isolated", "equation_merged", "equation_caption"},
        }
        for family in families.values():
            if aa in family and bb in family:
                return 0.7
        return 0.0
    if aa in TEXTUAL_TAGS and bb in TEXTUAL_TAGS:
        return 0.6
    return 0.0



def iou(a: BBox, b: BBox) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union



def norm_edit_similarity(a: str, b: str) -> float:
    aa = clean_text(a)
    bb = clean_text(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    dist = Levenshtein.distance(aa, bb)
    denom = max(len(aa), len(bb), 1)
    return 1.0 - (dist / denom)



def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b
