"""Merge objective page-score CSVs into a historical experiment leaderboard."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.objective_reports import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Input page_scores.csv files")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-model", action="append", default=[])
    args = parser.parse_args()
    build_report(args.inputs, args.out_dir, excluded_models=set(args.exclude_model))


if __name__ == "__main__":
    main()
