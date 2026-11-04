import csv

import pytest

from benchmark.objective_reports import build_report


def test_perfect_text_is_not_treated_as_missing_and_empty_tables_are_excluded(tmp_path):
    source = tmp_path / "pages.csv"
    source.write_text(
        "model,page_id,table_teds,table_teds_s,text_edit_distance,layout_graph_similarity\n"
        "perfect,0001,,,0,1\n"
        "mixed,0001,0,0,1,0\n"
        "mixed,0002,,,0,1\n"
        "excluded,0001,1,1,0,1\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"
    build_report([source], out, excluded_models={"excluded"})
    with (out / "leaderboard.csv").open(encoding="utf-8", newline="") as handle:
        perfect, mixed = list(csv.DictReader(handle))
    assert perfect["model"] == "perfect"
    assert float(perfect["weighted_score"]) == pytest.approx(2 / 2.3)
    assert perfect["table_pages"] == "0"
    assert perfect["table_teds"] == ""
    assert mixed["table_pages"] == "1"
    assert float(mixed["table_teds"]) == 0
    with (out / "page_scores.csv").open(encoding="utf-8", newline="") as handle:
        pages = list(csv.DictReader(handle))
    assert [float(row["page_weighted_score"]) for row in pages] == [1, 0, 1]
