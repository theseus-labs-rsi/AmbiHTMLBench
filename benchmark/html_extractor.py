from __future__ import annotations

from pathlib import Path
import re
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

  const isPlaceholder = (el) => {
    if (!el || !el.closest) return false;
    return Boolean(el.closest('.placeholder'));
  };

  const colorAlpha = (color) => {
    if (!color || color === 'transparent') return 0;
    const match = color.match(/rgba?\(([^)]+)\)/i);
    if (!match) return 1;
    const parts = match[1].split(',').map((x) => x.trim());
    if (parts.length < 4) return 1;
    const alpha = parseFloat(parts[3]);
    return Number.isFinite(alpha) ? alpha : 1;
  };

  const isRenderedTextNode = (node) => {
    const el = node && node.parentElement;
    if (!el) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    if (parseFloat(cs.opacity || '1') <= 0.001) return false;
    if (parseFloat(cs.fontSize || '0') <= 0.5) return false;
    if (colorAlpha(cs.color) <= 0.001) return false;
    return true;
  };

  const extractLatex = (el) => {
    if (!el) return '';
    const attrs = ['data-latex', 'data-tex', 'alttext', 'aria-label'];
    for (const name of attrs) {
      const value = el.getAttribute && el.getAttribute(name);
      if (clean(value)) return clean(value);
    }
    const annotation = el.querySelector && el.querySelector('annotation[encoding*="TeX"], annotation[encoding*="tex"]');
    if (annotation && clean(annotation.textContent || '')) return clean(annotation.textContent || '');
    const script = el.querySelector && el.querySelector('script[type*="math/tex"]');
    if (script && clean(script.textContent || '')) return clean(script.textContent || '');
    return clean(el.textContent || '');
  };

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
    if (isPlaceholder(el)) return null;
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
    if (isPlaceholder(el)) return;
    const rect = el.getBoundingClientRect();
    if (!isVisibleRect(el, rect)) return;

    let entry = blocks.get(el);
    if (!entry) {
      entry = {
        tag: semanticRole(el),
        rawTag: el.tagName.toLowerCase(),
        rawHtml: el.outerHTML || '',
        latex: semanticRole(el) === 'equation_isolated' ? extractLatex(el) : '',
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
    if (node.parentElement && isPlaceholder(node.parentElement)) continue;
    if (!isRenderedTextNode(node)) continue;
    if (node.parentElement && node.parentElement.closest('table')) continue;
    upsert(semanticRoot(node), txt);
  }

  const structuralNodes = document.querySelectorAll('img,svg,canvas,video,figure,figcaption,table,caption,header,footer,math,.MathJax,.katex,ul,ol');
  for (const el of structuralNodes) {
    if (isPlaceholder(el)) continue;
    upsert(el, el.innerText || el.textContent || '');
  }

  const elements = Array.from(blocks.values())
    .map((entry) => ({
      tag: entry.tag,
      rawTag: entry.rawTag,
      rawHtml: entry.rawHtml,
      latex: entry.latex,
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


_REMOTE_SCRIPT_RE = re.compile(
    r"<script\b(?=[^>]*\bsrc\s*=\s*['\"]https?://)[^>]*>.*?</script\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _set_static_html_content(browser_page: Any, html: Path, *, timeout_ms: int = 10_000) -> None:
    raw = html.read_text(encoding="utf-8", errors="ignore")
    raw = _REMOTE_SCRIPT_RE.sub("", raw)
    base_tag = f'<base href="{html.parent.as_uri().rstrip("/")}/">'
    if re.search(r"<head\b[^>]*>", raw, flags=re.IGNORECASE):
        raw = re.sub(r"(<head\b[^>]*>)", r"\1" + base_tag, raw, count=1, flags=re.IGNORECASE)
    else:
        raw = base_tag + raw
    browser_page.set_content(raw, wait_until="domcontentloaded", timeout=timeout_ms)


def _page_data_from_payload(payload: dict[str, Any], html_path: str | Path, viewport_width: int, viewport_height: int) -> PageData:
    html = Path(html_path).resolve()
    raw_elements = _dedupe(payload.get("elements", []))
    page_w = max(
        1.0,
        max(
            (
                float(e.get("x", 0.0)) + float(e.get("w", 0.0))
                for e in raw_elements
                if float(e.get("w", 0.0)) > 0 and float(e.get("h", 0.0)) > 0
            ),
            default=float(payload.get("pageWidth", viewport_width) or viewport_width),
        ),
    )
    page_h = max(
        1.0,
        max(
            (
                float(e.get("y", 0.0)) + float(e.get("h", 0.0))
                for e in raw_elements
                if float(e.get("w", 0.0)) > 0 and float(e.get("h", 0.0)) > 0
            ),
            default=float(payload.get("pageHeight", viewport_height) or viewport_height),
        ),
    )

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
        visual_bbox = BBox(x=x, y=y, w=w, h=h)

        tag = normalize_semantic_tag(str(e.get("tag", e.get("rawTag", "unknown"))))
        text = clean_text(str(e.get("text", "")))
        raw_html = str(e.get("rawHtml", ""))
        latex = clean_text(str(e.get("latex", "")))
        source_id = f"pred_{i}"
        elements.append(
            Element(
                tag=tag,
                text=text,
                bbox=bbox,
                source_id=source_id,
                raw_html=raw_html,
                latex=latex,
                visual_bbox=visual_bbox,
            )
        )

    return PageData(
        width=page_w,
        height=page_h,
        elements=elements,
        meta={
            "html_path": str(html),
            "extractor": "playwright_block_semantic_v2",
        },
    )


def extract_html_with_page(
    browser_page: Any,
    html_path: str | Path,
    viewport_width: int = 1280,
    viewport_height: int = 1800,
) -> PageData:
    html = Path(html_path).resolve()
    if not html.exists():
        raise FileNotFoundError(html)
    browser_page.set_viewport_size({"width": viewport_width, "height": viewport_height})
    if not getattr(browser_page, "_ambi_block_external_requests", False):
        try:
            browser_page.route("http://**/*", lambda route: route.abort())
            browser_page.route("https://**/*", lambda route: route.abort())
            setattr(browser_page, "_ambi_block_external_requests", True)
        except Exception:
            pass
    try:
        browser_page.goto(html.as_uri(), wait_until="domcontentloaded", timeout=5_000)
    except Exception:
        _set_static_html_content(browser_page, html)
    browser_page.wait_for_timeout(50)
    payload = browser_page.evaluate(EXTRACT_JS)
    return _page_data_from_payload(payload, html, viewport_width, viewport_height)



def extract_html_as_page(html_path: str | Path, viewport_width: int = 1280, viewport_height: int = 1800) -> PageData:
    from playwright.sync_api import sync_playwright

    html = Path(html_path).resolve()
    if not html.exists():
        raise FileNotFoundError(html)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
        try:
            page.route("http://**/*", lambda route: route.abort())
            page.route("https://**/*", lambda route: route.abort())
        except Exception:
            pass
        try:
            page.goto(html.as_uri(), wait_until="domcontentloaded", timeout=5_000)
        except Exception:
            _set_static_html_content(page, html)
        page.wait_for_timeout(50)

        payload = page.evaluate(EXTRACT_JS)
        browser.close()

    return _page_data_from_payload(payload, html, viewport_width, viewport_height)
