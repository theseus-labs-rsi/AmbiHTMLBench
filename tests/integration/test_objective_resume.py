import json

from benchmark.objective_v3 import evaluate_all_models_v3


def test_resumed_csv_preserves_table_page_count_and_scores(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([
        {"ID": 1, "imageUrl": "", "contentList": [
            {"id": "a", "name": "text", "remark": "Hello", "left": 0,
             "top": 0, "width": 100, "height": 30},
        ]},
        {"ID": 2, "imageUrl": "", "contentList": [
            {"id": "b", "name": "table", "left": 0, "top": 0,
             "width": 100, "height": 30,
             "remark": "<table><tr><td>42</td></tr></table>"},
        ]},
    ]), encoding="utf-8")
    infer = tmp_path / "infer"
    html = infer / "model" / "html"
    html.mkdir(parents=True)
    (html / "0001_text.html").write_text("<p>Hello</p>", encoding="utf-8")
    (html / "0002_table.html").write_text(
        "<table><tr><td>42</td></tr></table>", encoding="utf-8")
    out = tmp_path / "scores"
    first, _ = evaluate_all_models_v3(gt, infer, out, visualizations=False)
    resumed, rows = evaluate_all_models_v3(gt, infer, out, visualizations=False)
    assert first[0]["table_pages"] == 1
    assert resumed == first
    assert len(rows) == 2
