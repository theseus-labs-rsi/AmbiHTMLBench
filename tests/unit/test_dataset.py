import json

import pytest

from benchmark.dataset import load_image_paths
from benchmark.objective_v3 import load_gt_v3


def test_gt_without_image_url_loads_by_padded_id(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"ID": 1, "contentList": []}]), encoding="utf-8")
    assert load_gt_v3(gt) == {"0001": {"ID": 1, "contentList": []}}


def test_local_images_resolve_by_id_and_original_extension(tmp_path):
    (tmp_path / "index").mkdir()
    image = tmp_path / "pages" / "0001" / "image.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    (tmp_path / "index" / "pages.jsonl").write_text(
        json.dumps({"page_id": "0001", "image_path": "pages/0001/image.webp"}) + "\n",
        encoding="utf-8",
    )
    assert load_image_paths(tmp_path) == {"0001": str(image.resolve())}
    image.unlink()
    with pytest.raises(FileNotFoundError, match="0001"):
        load_image_paths(tmp_path)
