from __future__ import annotations

from pathlib import Path

from benchmark.subjective import (
    AXIS_DEFINITIONS,
    JudgeResult,
    build_active_axes,
    canonicalize_result,
    collapse_dual_orientation,
    estimate_request_bytes,
    parse_judge_response,
)


def test_all_seven_axes_are_frozen():
    assert set(AXIS_DEFINITIONS) == {
        "IMAGE_VISUAL_FIDELITY",
        "TABLE_VISUAL_STRUCTURE",
        "CHART_VISUAL_ENCODING",
        "FORMULA_2D_LAYOUT",
        "ALGORITHM_BLOCK_STRUCTURE",
        "VERTICAL_TEXT_PRESENTATION",
        "LIST_HIERARCHY_PRESENTATION",
    }


def test_build_active_axes_bbox_1000():
    gt = {
        "contentList": [
            {"id": "img1", "name": "image", "left": 10, "top": 20, "width": 30, "height": 40},
            {"id": "t1", "name": "table", "left": 50, "top": 100, "width": 100, "height": 200},
        ]
    }
    axes = build_active_axes(gt, 200, 400)
    assert [x["id"] for x in axes] == ["IMAGE_VISUAL_FIDELITY", "TABLE_VISUAL_STRUCTURE"]
    assert axes[0]["targets"][0]["bbox_1000"] == [50, 50, 200, 150]


def test_parse_strict_and_tolerant_json():
    axes = [{"id": "IMAGE_VISUAL_FIDELITY", "description": "", "targets": []}]
    strict = '{"winner":"A","confidence":4,"reason":"ok","axes":[{"id":"IMAGE_VISUAL_FIDELITY","score":2}]}'
    r = parse_judge_response(strict, axes)
    assert r.winner == "A" and r.axis_scores["IMAGE_VISUAL_FIDELITY"] == 2

    malformed = '{"winner":"B","confidence":3,"reason":"含有"引号"但可恢复","axes":[{"id":"IMAGE_VISUAL_FIDELITY","score":-1}]}'
    r2 = parse_judge_response(malformed, axes)
    assert r2.winner == "B" and r2.axis_scores["IMAGE_VISUAL_FIDELITY"] == -1


def test_canonicalize_swapped_reverses_winner_and_axis_sign():
    r = JudgeResult("A", 5, "", {"TABLE_VISUAL_STRUCTURE": 2})
    out = canonicalize_result(r, True)
    assert out.winner == "B"
    assert out.axis_scores == {"TABLE_VISUAL_STRUCTURE": -2}


def test_pair_collapse_distinguishes_reliability_and_perceptual_ties():
    same = collapse_dual_orientation("p1", "m1", "m2", "A", "A")
    assert same.winner == "A" and same.tie_type is None

    reliability = collapse_dual_orientation("p2", "m1", "m2", "A", "B")
    assert reliability.winner == "TIE" and reliability.tie_type == "reliability"

    perceptual = collapse_dual_orientation("p3", "m1", "m2", "TIE", "TIE")
    assert perceptual.winner == "TIE" and perceptual.tie_type == "perceptual"


def test_request_size_estimator_detects_large_payload(tmp_path: Path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        p.write_bytes(b"x" * 1_000_000)
        paths.append(p)
    estimated = estimate_request_bytes(paths[0], paths[1], paths[2], [])
    assert estimated > 4_000_000
