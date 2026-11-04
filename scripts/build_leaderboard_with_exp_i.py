"""Historical experiment recipe; use build_leaderboard.py for new inputs."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.objective_reports import build_report


BASE_PAGE_SCORES = Path("outputs/objective_v5_all_models_plus_exp_G/page_scores.csv")
NEW_PAGE_SCORES = Path("outputs/objective_v5_exp_I/page_scores.csv")
OUT_DIR = Path("outputs/objective_v5_all_models_plus_exp_I")


def main() -> None:
    build_report([BASE_PAGE_SCORES, NEW_PAGE_SCORES], OUT_DIR)


if __name__ == "__main__":
    main()
