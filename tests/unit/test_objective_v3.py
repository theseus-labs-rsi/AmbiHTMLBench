from __future__ import annotations

import pytest

from benchmark.objective_v3 import (
    cdm_score,
    evaluate_page_v3,
    load_gt_v3,
    table_teds,
    _text_match_key,
    text_group_matches,
)
from benchmark.html_extractor import extract_html_with_page
from benchmark.schema import Element, PageData
from benchmark.utils import BBox


def test_gt_loader_rejects_non_list_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"elements":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="GT"):
        load_gt_v3(p)


def test_table_and_formula_scores_are_exact_for_identical_content():
    table = "<table><tr><td>A</td><td>B</td></tr></table>"
    assert table_teds(table, table) == 1.0
    assert table_teds(table, table, structure_only=True) == 1.0
    assert cdm_score(r"$$ x^2 + y^2 $$", r"x^2+y^2") == 1.0


def test_text_match_key_preserves_punctuation_symbols_and_cjk():
    assert _text_match_key("a-b, c.") == "a-b,c."
    assert _text_match_key(r"$x^2+y^2$") == r"x^2+y^2"
    assert _text_match_key("こんにちは,안녕!") == "こんにちは,안녕!"


def test_text_group_matching_supports_many_to_one():
    gt_page = {
        "ID": 1,
        "imageUrl": "",
        "contentList": [
            {"id": "a", "left": 0, "top": 0, "width": 100, "height": 20, "visible": True, "name": "text", "remark": "hello"},
            {"id": "b", "left": 0, "top": 20, "width": 100, "height": 20, "visible": True, "name": "text", "remark": "world"},
        ],
    }
    pred = PageData(
        width=100,
        height=100,
        elements=[
            Element("text_block", "hello world", BBox(0, 0, 1, 0.4), source_id="p0"),
        ],
        meta={},
    )
    scores, debug = evaluate_page_v3(gt_page, pred)
    assert scores.text_edit_distance < 0.1
    assert scores.layout_graph_similarity > 0
    assert len(debug["texts"][0]["gt"]) == 2
    assert debug["layout_nodes"]


def test_text_group_matching_treats_xi_as_wildcard():
    gt_page = {
        "ID": 1,
        "imageUrl": "",
        "contentList": [
            {"id": "a", "left": 0, "top": 0, "width": 100, "height": 20, "visible": True, "name": "text", "remark": "a\u03bec"},
        ],
    }
    pred = PageData(
        width=100,
        height=100,
        elements=[
            Element("text_block", "abc", BBox(0, 0, 1, 0.2), source_id="p0"),
        ],
        meta={},
    )
    scores, _ = evaluate_page_v3(gt_page, pred)
    assert scores.text_edit_distance == 0.0


def test_text_group_distance_uses_max_length_normalization():
    gt = [Element("text_block", "abc", BBox(0, 0, 1, 0.2), source_id="g0")]
    pred = PageData(
        width=100,
        height=100,
        elements=[Element("text_block", "xyz", BBox(0, 0, 1, 0.2), source_id="p0")],
        meta={},
    )
    gt_page = {
        "ID": 1,
        "imageUrl": "",
        "contentList": [
            {"id": "g0", "left": 0, "top": 0, "width": 100, "height": 20, "visible": True, "name": "text", "remark": gt[0].text},
        ],
    }
    scores, _ = evaluate_page_v3(gt_page, pred)
    assert scores.text_edit_distance == 1.0


def test_placeholder_class_text_is_not_extracted(tmp_path):
    html = tmp_path / "placeholder.html"
    html.write_text(
        """<!DOCTYPE html><html><body>
        <div class="placeholder scene-background">[Game scene: full-frame background]</div>
        <p>Visible text</p>
        </body></html>""",
        encoding="utf-8",
    )
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 800, "height": 600})
        data = extract_html_with_page(page, html, viewport_width=800, viewport_height=600)
        browser.close()
    assert all("Game scene" not in e.text for e in data.elements)
    assert any(e.text == "Visible text" for e in data.elements)


def test_formula_is_scored_as_text_not_cdm():
    gt_page = {
        "ID": 1,
        "imageUrl": "",
        "contentList": [
            {"id": "f", "left": 0, "top": 0, "width": 100, "height": 20, "visible": True, "name": "inline_formula", "remark": "$x+y$"},
        ],
    }
    pred = PageData(
        width=100,
        height=100,
        elements=[Element("equation_isolated", "$x+y$", BBox(0, 0, 1, 0.2), source_id="p0", latex="$x+y$")],
        meta={},
    )
    scores, debug = evaluate_page_v3(gt_page, pred)
    assert scores.formula_cdm is None
    assert scores.text_edit_distance == 0.0
    assert not debug["formulas"]


def test_unmatched_table_cells_fall_back_to_text_matching():
    gt_page = {
        "ID": 1,
        "imageUrl": "",
        "contentList": [
            {
                "id": "t",
                "left": 0,
                "top": 0,
                "width": 100,
                "height": 30,
                "visible": True,
                "name": "table",
                "remark": "<table><tr><td>Alpha</td><td>Beta</td></tr></table>",
            },
        ],
    }
    pred = PageData(
        width=100,
        height=100,
        elements=[Element("text_block", "Alpha Beta", BBox(0, 0, 1, 0.3), source_id="p0")],
        meta={},
    )
    scores, debug = evaluate_page_v3(gt_page, pred)
    assert scores.table_teds == 0.0
    assert scores.text_edit_distance == 0.0
    gt_texts = [obj.text for obj in debug["gt_text"]]
    assert "Alpha" in gt_texts
    assert "Beta" in gt_texts
    assert len(gt_texts) >= 2
