"""Historical experiment recipe; use build_leaderboard.py for new inputs."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.objective_reports import build_report


BASE_PAGE_SCORES = Path("outputs/objective_v5_all_models_plus_qwen38/page_scores.csv")
NEW_PAGE_SCORES = [
    Path("outputs/objective_v5_swift_e1/page_scores.csv"),
    Path("outputs/objective_v5_swift_e2/page_scores.csv"),
    Path("outputs/objective_v5_swift_e3/page_scores.csv"),
]
REMOVED_MODELS = {"infer_lora_qwen38_27b"}
OUT_DIR = Path("outputs/objective_v5_all_models_plus_swift")


def main() -> None:
    build_report([BASE_PAGE_SCORES, *NEW_PAGE_SCORES], OUT_DIR, excluded_models=REMOVED_MODELS)


if __name__ == "__main__":
    main()
