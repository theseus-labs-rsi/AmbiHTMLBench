"""Explicitly offline CLI for frozen subjective results; no paid-run command."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .frozen_calibration import calibrate
from .frozen_proxy import build_leaderboard, load_proxy_config
from .frozen_proxy_io import _unique_keys, load_single_results, sha256_file, verify_artifact


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")


def _new_output(path):
    # Never overwrite a frozen or previously generated run, even if it is empty.
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _config_identity(config_path, manifest_path):
    digest = sha256_file(config_path)
    verified = False
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        if (not isinstance(manifest, dict) or not isinstance(manifest.get("configuration"), dict)
                or manifest["configuration"].get("sha256") != digest):
            raise ValueError("Frozen configuration hash mismatch")
        verified = True
    return {"sha256": digest, "provenance_verified": verified}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "replay", "calibrate"):
        command = sub.add_parser(name)
        command.add_argument("--results", required=True, help="External completed JSONL; read-only")
        command.add_argument("--artifact-manifest", help="Published dataset_versions.json")
        command.add_argument("--dataset", help="Dataset identifier in the artifact manifest")
        command.add_argument("--fixture", action="store_true", help="Allow small test fixtures; not a paper run")
        if name != "calibrate":
            command.add_argument("--config", required=True, help="Frozen five-judge JSON configuration")
        else:
            command.add_argument("--human-targets", required=True)
        if name != "validate":
            command.add_argument("--out-dir", required=True, help="A NEW directory; existing paths are refused")
        if name == "replay":
            command.add_argument("--bootstrap", type=int, default=2000)
            command.add_argument("--seed", type=int, default=20260831)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            if args.dataset and args.dataset != "calibration":
                raise ValueError("calibrate requires the calibration dataset, not deployment results")
            identity = verify_artifact(args.results, args.artifact_manifest, args.dataset)
            human_identity = verify_artifact(args.human_targets, args.artifact_manifest,
                                             args.dataset, "human_targets")
            if Path(args.out_dir).exists():
                raise ValueError("Output directory already exists; frozen outputs are never overwritten")
            report, cases = calibrate(args.results, args.human_targets, strict=not args.fixture)
            report["input_artifacts"] = {"results": identity, "human_targets": human_identity}
            out = _new_output(args.out_dir)
            _json(out / "calibration.json", report)
            _jsonl(out / "calibration_cases.jsonl", cases)
            print(json.dumps({"status": "COMPLETE", "output": str(out),
                              "paper_scale": not args.fixture, "api_calls": 0}))
            return 0
        config = load_proxy_config(args.config)
        config_identity = _config_identity(args.config, args.artifact_manifest)
        cases, audit = load_single_results(
            args.results, config, expected_cases=None if args.fixture else 10000,
            strict_schedule=not args.fixture, artifact_manifest=args.artifact_manifest,
            dataset=args.dataset,
        )
        audit["configuration"] = config_identity
        audit["api_calls"] = 0
        if args.command == "validate":
            print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if Path(args.out_dir).exists():
            raise ValueError("Output directory already exists; frozen outputs are never overwritten")
        report = build_leaderboard(cases, bootstrap=args.bootstrap, seed=args.seed)
        report.update({"protocol": "SINGLE direction, SINGLE vote per judge/case",
                       "coefficients": config["coefficients"], "input_validation": audit,
                       "paper_replay": bool(not args.fixture and config_identity["provenance_verified"]
                                            and audit["input_artifact"]["provenance_verified"]
                                            and args.bootstrap == 2000 and args.seed == 20260831),
                       "limitations": [
                           "No new human accuracy/recall/F1 is measured.",
                           "Training used four-vote means; deployment uses one vote.",
                           "Completion-based historical direction selection may reflect latency, not winners.",
                           "No-intercept unnormalized OLS is not complementary under canonical A/B recoding.",
                           "Bootstrap fixes predictions and coefficients; it does not cover refitting or all system errors.",
                           "Retained human-study task IDs do not validate subsequently modified candidates.",
                       ]})
        if audit["input_artifact"].get("dataset") == "candidate_refresh":
            report["limitations"].append(
                "Candidate refresh changed 21 retained human-study cases without new human labels.")
        out = _new_output(args.out_dir)
        _json(out / "validation.json", audit)
        _json(out / "leaderboard.json", report)
        _jsonl(out / "composite_cases.jsonl", cases)
        with (out / "leaderboard.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["leaderboard"][0]))
            writer.writeheader()
            writer.writerows(report["leaderboard"])
        print(json.dumps({"status": "COMPLETE", "output": str(out),
                          "paper_replay": report["paper_replay"], "api_calls": 0}))
        return 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
