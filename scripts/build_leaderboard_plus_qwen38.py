"""Historical experiment recipe; use build_leaderboard.py for new inputs."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.objective_reports import build_report


OLD_PAGE_SCORES = Path("outputs/objective_v5_all_models_maxden_v3/page_scores.csv")
NEW_PAGE_SCORES = [
    Path("outputs/objective_v5_qwen38_base/page_scores.csv"),
    Path("outputs/objective_v5_qwen38_lora/page_scores.csv"),
]
OUT_DIR = Path("outputs/objective_v5_all_models_plus_qwen38")


def main() -> None:
    build_report([OLD_PAGE_SCORES, *NEW_PAGE_SCORES], OUT_DIR)


if __name__ == "__main__":
    main()
