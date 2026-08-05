from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image



def image_size(path: str | Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size



def render_html_to_png(
    html_path: str | Path,
    out_png: str | Path,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
    full_page: bool = True,
) -> Path:
    from playwright.sync_api import sync_playwright

    html = Path(html_path).resolve()
    out = Path(out_png).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    vw = viewport_width or 1280
    vh = viewport_height or 1800

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": int(vw), "height": int(vh)})
        page.goto(html.as_uri(), wait_until="load")
        page.wait_for_timeout(250)
        page.screenshot(path=str(out), full_page=full_page)
        browser.close()

    return out
