from __future__ import annotations

import pytest

from benchmark.schema import Element, PageData
from benchmark.scorer import evaluate_pages
from benchmark.utils import BBox


def test_existing_objective_scorer_unchanged_smoke():
    page = PageData(
        width=100,
        height=100,
        elements=[Element(tag="text_block", text="hello", bbox=BBox(0.1, 0.1, 0.5, 0.2))],
        meta={},
    )
    result = evaluate_pages(page, page)
    assert result.content_score == 1.0
    assert result.f1 == 1.0
    assert result.total_score == pytest.approx(1.0, abs=1e-12)
