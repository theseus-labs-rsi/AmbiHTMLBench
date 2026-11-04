"""Keep the paper's deployed configuration and prompt tied to their digests."""
import json
from pathlib import Path

import pytest

from benchmark.frozen_proxy_io import sha256_file


@pytest.mark.parametrize("artifact", ["configuration", "prompt"])
def test_frozen_artifacts_match_published_dataset_identity(artifact):
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "evidence/subjective/dataset_versions.json").read_text(encoding="utf-8"))
    expected = manifest[artifact]
    path = root / expected["path"]
    assert sha256_file(path) == expected["sha256"]
    assert path.stat().st_size == expected["bytes"]
