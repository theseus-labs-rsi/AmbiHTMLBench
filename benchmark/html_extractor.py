from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import Element, PageData
from .utils import BBox, clamp01, clean_text, normalize_semantic_tag


EXTRACT_JS = r"""
() => {
  const docEl = document.documentElement;
  const body = document.body || docEl;
  const pageWidth = Math.max(docEl.scrollWidth, body.scrollWidth, docEl.clientWidth, body.clientWidth);
  const pageHeight = Math.max(docEl.scrollHeight, body.scrollHeight, docEl.clientHeight, body.clientHeight);
  const blockishDisplays = new Set(['block', 'list-item', 'table', 'table-caption', 'flex', 'grid']);
  const preferredTags = new Set([
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'li', 'header', 'footer', 'blockquote', 'pre', 'code',
    'div', 'section', 'article', 'aside', 'figcaption', 'caption',
    'figure', 'table', 'img', 'svg', 'canvas', 'video', 'math'
  ]);
  const blocks = new Map();

  const isVisibleRect = (el, rect) => {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    if (parseFloat(cs.opacity || '1') <= 0.001) return false;
    if (rect.width < 2 || rect.height < 2) return false;
    return true;
  };

  const clean = (text) => (text || '').replace(/\s+/g, ' ').trim();

  const semanticRole = (el) => {
    if (!el || !el.tagName) return 'unknown';
    const tag = el.tagName.toLowerCase();
    if (/^h[1-6]$/.test(tag)) return 'title';
    if (tag === 'header') return 'header';
    if (tag === 'footer') return 'footer';
    if (tag === 'figcaption') return 'figure_caption';
    if (tag === 'caption') return 'table_caption';
    if (tag === 'figure' || tag === 'img' || tag === 'svg' || tag === 'canvas' || tag === 'video') return 'figure';
    if (tag === 'table') return 'table';
    if (tag === 'ul' || tag === 'ol' || tag === 'li') return 'list';
    if (tag === 'math' || el.closest('.MathJax, .katex')) return 'equation_isolated';
    return 'text_block';
  };

  const semanticRoot = (node) => {
    let el = node.parentElement;
    if (!el) return null;
    const figcaption = el.closest('figcaption');
    if (figcaption) return figcaption;
    const caption = el.closest('caption');
    if (caption) return caption;
    const mathRoot = el.closest('math, .MathJax, .katex');
    if (mathRoot) return mathRoot;
    const table = el.closest('table');
    if (table) return table;
    const figure = el.closest('figure');
    if (figure) return figure;

    let cur = el;
    while (cur && cur !== body) {
      const rect = cur.getBoundingClientRect();
      if (!isVisibleRect(cur, rect)) {
        cur = cur.parentElement;
        continue;
      }
      const tag = cur.tagName.toLowerCase();
      const display = getComputedStyle(cur).display;
      if (preferredTags.has(tag) || blockishDisplays.has(display)) {
        return cur;
      }
      cur = cur.parentElement;
    }
    return el;
  };

  const upsert = (el, textPart = '') => {
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (!isVisibleRect(el, rect)) return;

    let entry = blocks.get(el);
    if (!entry) {
      entry = {
        tag: semanticRole(el),
        rawTag: el.tagName.toLowerCase(),
        texts: [],
        x: rect.left + window.scrollX,
        y: rect.top + window.scrollY,
        w: rect.width,
        h: rect.height,
      };
      blocks.set(el, entry);
    }

    const text = clean(textPart);
    if (text) entry.texts.push(text);
  };

  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const txt = clean(node.textContent || '');
    if (!txt) continue;
    upsert(semanticRoot(node), txt);
  }

  const structuralNodes = document.querySelectorAll('img,svg,canvas,video,figure,figcaption,table,caption,header,footer,math,.MathJax,.katex,ul,ol');
  for (const el of structuralNodes) {
    upsert(el, el.innerText || el.textContent || '');
  }

  const elements = Array.from(blocks.values())
    .map((entry) => ({
      tag: entry.tag,
      rawTag: entry.rawTag,
      text: clean(entry.texts.join(' ')),
      x: entry.x,
      y: entry.y,
      w: entry.w,
      h: entry.h,
    }))
    .filter((entry) => entry.text || entry.tag === 'figure' || entry.tag === 'table' || entry.tag === 'equation_isolated');

  return { pageWidth, pageHeight, elements };
}
"""


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        key = (
            round(float(it["x"]), 1).__int__(),
            round(float(it["y"]), 1).__int__(),
            round(float(it["w"]), 1).__int__(),
            round(float(it["h"]), 1).__int__(),
            clean_text(str(it.get("text", "")))[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out



def extract_html_as_page(html_path: str | Path, viewport_width: int = 1280, viewport_height: int = 1800) -> PageData:
    from playwright.sync_api import sync_playwright

    html = Path(html_path).resolve()
    if not html.exists():
        raise FileNotFoundError(html)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
        page.goto(html.as_uri(), wait_until="load")
        page.wait_for_timeout(250)

        payload = page.evaluate(EXTRACT_JS)
        browser.close()

    page_w = float(payload.get("pageWidth", viewport_width) or viewport_width)
    page_h = float(payload.get("pageHeight", viewport_height) or viewport_height)
    raw_elements = _dedupe(payload.get("elements", []))

    elements: list[Element] = []
    for i, e in enumerate(raw_elements):
        x = float(e.get("x", 0.0))
        y = float(e.get("y", 0.0))
        w = float(e.get("w", 0.0))
        h = float(e.get("h", 0.0))
        if w <= 0 or h <= 0:
            continue

        bbox = BBox(
            x=clamp01(x / page_w),
            y=clamp01(y / page_h),
            w=clamp01(w / page_w),
            h=clamp01(h / page_h),
        )

        tag = normalize_semantic_tag(str(e.get("tag", e.get("rawTag", "unknown"))))
        text = clean_text(str(e.get("text", "")))
        source_id = f"pred_{i}"
        elements.append(Element(tag=tag, text=text, bbox=bbox, source_id=source_id))

    return PageData(
        width=page_w,
        height=page_h,
        elements=elements,
        meta={
            "html_path": str(html),
            "extractor": "playwright_block_semantic_v2",
        },
    )
