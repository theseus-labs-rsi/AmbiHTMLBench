from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from benchmark.arena import run_arena_resolved
from benchmark.arena_audit import collapse_decisions
from benchmark.subjective import JudgeResult


class IdentityAwareMockJudge:
    def judge(self, reference_image, candidate_a, candidate_b, active_axes):
        # model_good should win regardless of whether it is displayed as A or B.
        if "model_good" in Path(candidate_a).name:
            winner = "A"
        else:
            winner = "B"
        return JudgeResult(winner, 5, "mock", {})


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


def test_dual_orientation_checkpoint_and_pair_collapse(tmp_path: Path):
    ref = tmp_path / "ref.png"
    good = tmp_path / "renders" / "p1" / "model_good.png"
    bad = tmp_path / "renders" / "p1" / "model_bad.png"
    for p in (ref, good, bad):
        _png(p)

    manifest = {
        "version": "subjective_arena_resolved_v1",
        "preflight": {"status": "PASS"},
        "models": ["model_good", "model_bad"],
        "pages": [],
        "pair_sets": [{
            "pair_set_id": "P1",
            "slot_id": "S1",
            "block_index": 0,
            "round_index": 0,
            "sample_id": "p1",
            "model_a": "model_good",
            "model_b": "model_bad",
            "reference_image": str(ref),
            "active_axes": [],
            "rendered_a": str(good),
            "rendered_b": str(bad),
        }],
    }
    mp = tmp_path / "resolved.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "decisions.json"

    rows = run_arena_resolved(mp, out, judge_client=IdentityAwareMockJudge(), judge_model="mock", concurrency=2)
    assert len(rows) == 2
    raw = json.loads(out.read_text())
    outcomes = collapse_decisions(raw["decisions"])
    assert len(outcomes) == 1
    assert outcomes[0].winner == "A"
    assert outcomes[0].tie_type is None

    # Rerun must reuse checkpoint rather than append duplicates.
    rows2 = run_arena_resolved(mp, out, judge_client=IdentityAwareMockJudge(), judge_model="mock", concurrency=2)
    assert len(rows2) == 2
