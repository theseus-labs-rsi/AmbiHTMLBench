from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Tuple

from PIL import Image


def image_size(path: str | Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _browser_executable() -> str | None:
    """Prefer an explicit/system Chromium when Playwright's managed browser is absent."""
    env = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if env and Path(env).exists():
        return env

    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found

    for candidate in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def _launch_browser(playwright):
    executable = _browser_executable()
    launch_kwargs: dict[str, Any] = {
        "headless": True,
        "args": [
            "--allow-file-access-from-files",
            "--disable-background-networking",
            "--disable-default-apps",
        ],
    }
    if executable:
        launch_kwargs["executable_path"] = executable
    return playwright.chromium.launch(**launch_kwargs)


def _install_page_routes(page, *, block_mathjax: bool = False) -> None:
    """Neutralize known blocking third-party scripts without blocking page assets.

    polyfill.io is unnecessary in modern Chromium and is known to hang/fail in
    some networks. MathJax is normally allowed; it is blocked only on fallback
    after a navigation timeout so that the rest of the static page can render.
    """

    def handler(route):
        url = route.request.url
        lower = url.lower()
        if "polyfill.io/" in lower:
            route.fulfill(status=200, content_type="application/javascript", body="/* polyfill bypassed by Arena renderer */")
            return
        if block_mathjax and ("cdn.jsdelivr.net/npm/mathjax" in lower or "cdnjs.cloudflare.com" in lower and "mathjax" in lower):
            route.fulfill(status=200, content_type="application/javascript", body="/* MathJax network fallback */")
            return
        route.continue_()

    page.route("**/*", handler)


def _best_effort_settle(page, *, settle_ms: int = 900) -> None:
    """Give local images/fonts/MathJax a bounded chance to settle."""
    try:
        page.wait_for_function(
            "Array.from(document.images || []).every(img => img.complete)",
            timeout=3000,
        )
    except Exception:
        pass
    try:
        page.evaluate(
            """async () => {
                if (document.fonts && document.fonts.ready) {
                    await Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 2000))]);
                }
            }"""
        )
    except Exception:
        pass
    page.wait_for_timeout(max(0, int(settle_ms)))


_EXTERNAL_SCRIPT_RE = re.compile(
    r"<script\b(?=[^>]*\bsrc\s*=\s*['\"]https?://)[^>]*>.*?</script\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _static_html_fallback(page, html: Path, *, timeout_ms: int) -> None:
    """Last-resort static render: strip only remote script tags, keep HTML/CSS/assets."""
    raw = html.read_text(encoding="utf-8", errors="ignore")
    raw = _EXTERNAL_SCRIPT_RE.sub("", raw)
    base_tag = f'<base href="{html.parent.as_uri().rstrip("/")}/">'
    if re.search(r"<head\b[^>]*>", raw, flags=re.IGNORECASE):
        raw = re.sub(r"(<head\b[^>]*>)", r"\1" + base_tag, raw, count=1, flags=re.IGNORECASE)
    else:
        raw = base_tag + raw
    page.set_content(raw, wait_until="domcontentloaded", timeout=timeout_ms)


def _render_one_with_context(
    context,
    *,
    html_path: str | Path,
    out_png: str | Path,
    viewport_width: int,
    viewport_height: int,
    full_page: bool,
    navigation_timeout_ms: int = 15000,
) -> dict[str, Any]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    html = Path(html_path).resolve()
    out = Path(out_png).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    viewport = {"width": int(viewport_width), "height": int(viewport_height)}

    # Attempt 1: normal page, but neutralize polyfill.io. This preserves MathJax
    # and all ordinary remote/local assets when the network is healthy.
    page = context.new_page()
    page.set_viewport_size(viewport)
    _install_page_routes(page, block_mathjax=False)
    try:
        page.goto(html.as_uri(), wait_until="domcontentloaded", timeout=navigation_timeout_ms)
        _best_effort_settle(page)
        page.screenshot(path=str(out), full_page=full_page)
        return {"status": "ok", "fallback": "none", "message": ""}
    except PlaywrightTimeoutError as exc:
        first_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        first_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            page.close()
        except Exception:
            pass

    # Attempt 2: if a synchronous remote MathJax script is the blocker, fulfill
    # it with an empty response. The rest of the page should then parse normally.
    page = context.new_page()
    page.set_viewport_size(viewport)
    _install_page_routes(page, block_mathjax=True)
    try:
        page.goto(html.as_uri(), wait_until="domcontentloaded", timeout=10000)
        _best_effort_settle(page)
        page.screenshot(path=str(out), full_page=full_page)
        return {"status": "ok", "fallback": "block_remote_mathjax", "message": first_error}
    except Exception as exc:
        second_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            page.close()
        except Exception:
            pass

    # Attempt 3: static HTML fallback. Remove only remote script tags and keep
    # markup, inline CSS and relative image assets. This is preferable to aborting
    # the entire 2250-job experiment because one candidate depends on a bad CDN.
    page = context.new_page()
    page.set_viewport_size(viewport)
    try:
        _static_html_fallback(page, html, timeout_ms=10000)
        _best_effort_settle(page)
        page.screenshot(path=str(out), full_page=full_page)
        return {
            "status": "ok",
            "fallback": "strip_remote_scripts",
            "message": first_error + " | retry: " + second_error,
        }
    except Exception as exc:
        third_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"render failed after 3 attempts for {html}: "
            f"primary=[{first_error}] retry=[{second_error}] static=[{third_error}]"
        ) from exc
    finally:
        try:
            page.close()
        except Exception:
            pass


def render_html_jobs(
    jobs: list[dict[str, Any]],
    *,
    report_path: str | Path | None = None,
    fail_fast: bool = True,
) -> list[dict[str, Any]]:
    """Render many HTML files using one browser process.

    Each job requires html_path, out_png, viewport_width, viewport_height and may
    include arbitrary metadata. Browser reuse makes the 500-page render much
    faster and the three-stage fallback prevents a single CDN timeout from
    killing the full Elo run.
    """
    from playwright.sync_api import sync_playwright
    from tqdm import tqdm

    results: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = _launch_browser(p)
        context = browser.new_context()
        try:
            for job in tqdm(jobs, desc="render-candidates"):
                meta = {k: v for k, v in job.items() if k not in {"html_path", "out_png", "viewport_width", "viewport_height", "full_page"}}
                try:
                    outcome = _render_one_with_context(
                        context,
                        html_path=job["html_path"],
                        out_png=job["out_png"],
                        viewport_width=int(job["viewport_width"]),
                        viewport_height=int(job["viewport_height"]),
                        full_page=bool(job.get("full_page", True)),
                    )
                    row = {**meta, "html_path": str(job["html_path"]), "out_png": str(job["out_png"]), **outcome}
                except Exception as exc:
                    row = {**meta, "html_path": str(job["html_path"]), "out_png": str(job["out_png"]), "status": "error", "fallback": "failed", "message": f"{type(exc).__name__}: {exc}"}
                    results.append(row)
                    if report_path is not None:
                        rp = Path(report_path)
                        rp.parent.mkdir(parents=True, exist_ok=True)
                        rp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    if fail_fast:
                        raise
                    continue
                results.append(row)
                if report_path is not None:
                    rp = Path(report_path)
                    rp.parent.mkdir(parents=True, exist_ok=True)
                    rp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            context.close()
            browser.close()
    return results


def render_html_to_png(
    html_path: str | Path,
    out_png: str | Path,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
    full_page: bool = True,
) -> Path:
    """Single-file compatibility wrapper using the same robust renderer."""
    from playwright.sync_api import sync_playwright

    html = Path(html_path).resolve()
    out = Path(out_png).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = int(viewport_width or 1280)
    vh = int(viewport_height or 1800)

    with sync_playwright() as p:
        browser = _launch_browser(p)
        context = browser.new_context()
        try:
            _render_one_with_context(
                context,
                html_path=html,
                out_png=out,
                viewport_width=vw,
                viewport_height=vh,
                full_page=full_page,
            )
        finally:
            context.close()
            browser.close()
    return out
