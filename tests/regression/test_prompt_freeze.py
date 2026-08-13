from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark.subjective import PROMPT_SHA256, SUBJECTIVE_PROMPT


def test_frozen_prompt_hash_matches_final_experiment():
    assert hashlib.sha256(SUBJECTIVE_PROMPT.encode("utf-8")).hexdigest() == PROMPT_SHA256
    prompt_file = Path(__file__).resolve().parents[2] / "benchmark" / "prompts" / "subjective_v1.md"
    assert prompt_file.read_text(encoding="utf-8") == SUBJECTIVE_PROMPT
