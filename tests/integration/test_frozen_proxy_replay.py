"""Replay synthetic saved votes only: sockets and HTTP are forbidden."""
from __future__ import annotations

import copy
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.frozen_proxy import JUDGES, build_leaderboard, load_proxy_config
from benchmark.frozen_proxy_io import BASE_AXES, load_single_results, sha256_file


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "subjective_ols5_frozen.json"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NETWORK FORBIDDEN in frozen-proxy replay tests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def rows():
    result = []
    pairs = [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")]
    for number, (a, b) in enumerate(pairs):
        for judge_index, judge in enumerate(JUDGES):
            swapped = judge_index % 2 == 1
            canonical = ("A", "TIE", "B")[(judge_index + number) % 3]
            display = {"A": "B", "B": "A", "TIE": "TIE"}[canonical] if swapped else canonical
            result.append({
                "job_id": f"task{number}::{judge}",
                "judge_id": judge,
                "repeat_index": 1,
                "task": {
                    "task_id": f"task{number}",
                    "page_id": f"page{number}",
                    "canonical_pair": {"A": a, "B": b},
                    "display_swapped": swapped,
                    "active_axes": [{"id": axis} for axis in BASE_AXES],
                    "source_kind": "synthetic_fixture",
                },
                "result": {
                    "winner_display": display,
                    "winner_canonical": canonical,
                    "axes_display": [{"id": axis, "score": 1} for axis in BASE_AXES],
                    "axes_canonical": [{"id": axis, "score": -1 if swapped else 1} for axis in BASE_AXES],
                },
            })
    return result


def write_rows(tmp_path, rows):
    path = tmp_path / "single_results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def load_fixture(path):
    return load_single_results(
        path, load_proxy_config(CONFIG_PATH), expected_cases=None, strict_schedule=False,
    )


def test_saved_votes_replay_to_continuous_leaderboard_without_network(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    source_bytes = path.read_bytes()
    cases, audit = load_fixture(path)
    assert len(cases) == 3
    assert isinstance(audit, dict)
    assert {case["task_id"] for case in cases} == {"task0", "task1", "task2"}
    assert all(set(case["single_votes"]) == set(JUDGES) for case in cases)
    coefficients = load_proxy_config(CONFIG_PATH)["coefficients"]
    for case in cases:
        expected = sum(coefficients[j] * {"A": 1.0, "TIE": 0.5, "B": 0.0}[vote]
                       for j, vote in case["single_votes"].items())
        assert case["p_A"] == pytest.approx(expected)
    board = build_leaderboard(cases, bootstrap=15, seed=42)
    assert len(board["leaderboard"]) == 3
    assert all(row["adjudicated_comparisons"] == 2 for row in board["leaderboard"])
    assert path.read_bytes() == source_bytes


def test_replay_is_independent_of_input_row_order(tmp_path, rows):
    first, _ = load_fixture(write_rows(tmp_path, rows))
    second, _ = load_fixture(write_rows(tmp_path, list(reversed(rows))))
    assert first == second


def test_duplicate_json_fields_are_rejected_instead_of_last_value_winning(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    text = path.read_text(encoding="utf-8")
    text = text.replace('"repeat_index": 1', '"repeat_index": 2, "repeat_index": 1', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture(path)


def test_fixture_cannot_pass_formal_schedule_validation(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    with pytest.raises(ValueError):
        load_single_results(path, load_proxy_config(CONFIG_PATH))


@pytest.mark.parametrize("defect", [
    "missing_vote", "duplicate_row", "duplicate_judge", "unknown_judge", "repeat",
    "wrong_canonical", "pair_mismatch", "page_mismatch", "non_boolean_orientation",
    "axis_sign", "axis_membership", "axis_score", "display_pair",
])
def test_invalid_saved_results_are_rejected(tmp_path, rows, defect):
    if defect == "missing_vote":
        rows.pop()
    elif defect == "duplicate_row":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "duplicate_judge":
        duplicate = copy.deepcopy(rows[0])
        duplicate["job_id"] = "second-job-same-judge-and-task"
        rows.append(duplicate)
    elif defect == "unknown_judge":
        rows[0]["judge_id"] = "unknown_judge"
    elif defect == "repeat":
        rows[0]["repeat_index"] = 2
    elif defect == "wrong_canonical":
        rows[0]["result"]["winner_canonical"] = "B"
    elif defect == "pair_mismatch":
        rows[0]["task"]["canonical_pair"]["A"] = "different_model"
    elif defect == "page_mismatch":
        rows[0]["task"]["page_id"] = "different_page"
    elif defect == "non_boolean_orientation":
        rows[0]["task"]["display_swapped"] = "false"
    elif defect == "axis_sign":
        rows[1]["result"]["axes_canonical"][0]["score"] = 1
    elif defect == "axis_membership":
        rows[0]["result"]["axes_display"].pop()
    elif defect == "axis_score":
        rows[0]["result"]["axes_display"][0]["score"] = 3
    else:
        rows[0]["task"]["display_pair"] = {"A": "beta", "B": "alpha"}
    with pytest.raises(ValueError):
        load_fixture(write_rows(tmp_path, rows))


def test_duplicate_physical_comparison_with_different_task_id_is_rejected(tmp_path, rows):
    duplicate = copy.deepcopy(rows[:len(JUDGES)])
    for row in duplicate:
        row["job_id"] += "-duplicate"
        row["task"]["task_id"] = "another-id-for-same-page-and-pair"
    with pytest.raises(ValueError):
        load_fixture(write_rows(tmp_path, rows + duplicate))


def test_diagnostic_axes_do_not_change_proxy_predictions(tmp_path, rows):
    before, _ = load_fixture(write_rows(tmp_path, rows))
    for row in rows:
        for key in ("axes_display", "axes_canonical"):
            for axis in row["result"][key]:
                axis["score"] *= -1
    after, _ = load_fixture(write_rows(tmp_path, rows))
    assert before == after


def test_artifact_digest_authenticates_input_and_rejects_modified_file(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(json.dumps({"datasets": {"fixture": {"results": {
        "sha256": sha256_file(path), "bytes": path.stat().st_size,
    }}}}), encoding="utf-8")
    _, audit = load_single_results(
        path, load_proxy_config(CONFIG_PATH), expected_cases=None, strict_schedule=False,
        artifact_manifest=manifest, dataset="fixture",
    )
    assert audit["input_artifact"]["provenance_verified"] is True
    # Even a semantically irrelevant newline invalidates byte-level provenance.
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_single_results(
            path, load_proxy_config(CONFIG_PATH), expected_cases=None, strict_schedule=False,
            artifact_manifest=manifest, dataset="fixture",
        )



def calibration_fixture(tmp_path):
    """Two full-rank blocks; four-vote means are 0.5, not last-vote labels."""
    judgments, targets = [], []
    for block in ("block1", "block2"):
        for feature_index in range(len(JUDGES)):
            tid = f"{block}-task{feature_index}"
            target = {
                "task_id": tid, "paper_id": block, "page_id": f"page{feature_index}",
                "canonical_pair": {"A": "alpha", "B": "beta"},
                "outcomes": {"A": feature_index + 1, "B": 9 - feature_index, "TIE": 0},
                "valid_preference_votes": 10, "human_p_a": (feature_index + 1) / 10,
            }
            targets.append(target)
            for judge_index, judge in enumerate(JUDGES):
                for swapped in (False, True):
                    for repeat in (1, 2):
                        canonical = "A" if judge_index == feature_index and not swapped else "B"
                        display = {"A": "B", "B": "A"}[canonical] if swapped else canonical
                        judgments.append({
                            "job_id": f"{tid}::{judge}::{swapped}::{repeat}",
                            "judge_id": judge, "repeat_index": repeat,
                            "task": {
                                "task_id": tid, "paper_id": block, "page_id": target["page_id"],
                                "canonical_pair": target["canonical_pair"], "display_swapped": swapped,
                            },
                            "result": {"winner_display": display, "winner_canonical": canonical},
                        })
    result_path, target_path = tmp_path / "calibration.jsonl", tmp_path / "human.jsonl"
    result_path.write_text("".join(json.dumps(row) + "\n" for row in judgments), encoding="utf-8")
    target_path.write_text("".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8")
    return result_path, target_path, judgments, targets


def test_calibration_replays_four_votes_and_holds_out_annotation_blocks(tmp_path):
    from benchmark.frozen_calibration import calibrate

    results, targets, _, _ = calibration_fixture(tmp_path)
    originals = (results.read_bytes(), targets.read_bytes(), CONFIG_PATH.read_bytes())
    report, cases = calibrate(results, targets, strict=False)
    assert report["reproduced_coefficients"] == pytest.approx(dict(zip(JUDGES, (0.2, 0.4, 0.6, 0.8, 1.0))))
    assert report["physical_comparisons"] == 10
    assert report["judgment_rows"] == 200
    assert report["paper_scale"] is False
    assert report["oof_metrics"]["brier"] == pytest.approx(0.0, abs=1e-15)
    assert {fold["held_out_block"] for fold in report["fold_coefficients"]} == {"block1", "block2"}
    assert all(fold["training_n"] == fold["test_n"] == 5 for fold in report["fold_coefficients"])
    for row in cases:
        assert sorted(row["features"].values()) == [0.0, 0.0, 0.0, 0.0, 0.5]
        assert row["oof_p_A"] == pytest.approx(row["human_p_a"])
    assert originals == (results.read_bytes(), targets.read_bytes(), CONFIG_PATH.read_bytes())


@pytest.mark.parametrize("defect", [
    "missing_vote", "duplicate_vote", "human_mean", "metadata", "boolean_target", "boolean_denominator",
])
def test_calibration_rejects_incomplete_or_mismatched_inputs(tmp_path, defect):
    from benchmark.frozen_calibration import calibrate

    results, targets, judgments, human = calibration_fixture(tmp_path)
    if defect == "missing_vote":
        judgments.pop()
    elif defect == "duplicate_vote":
        judgments.append(copy.deepcopy(judgments[0]))
    elif defect == "human_mean":
        human[0]["human_p_a"] = 0.99
    elif defect == "boolean_target":
        human[0]["human_p_a"] = True
    elif defect == "boolean_denominator":
        human[0]["valid_preference_votes"] = True
    else:
        judgments[0]["task"]["paper_id"] = "wrong-block"
    results.write_text("".join(json.dumps(row) + "\n" for row in judgments), encoding="utf-8")
    targets.write_text("".join(json.dumps(row) + "\n" for row in human), encoding="utf-8")
    with pytest.raises(ValueError):
        calibrate(results, targets, strict=False)


def test_calibration_rejects_singular_design_instead_of_adding_ridge():
    from benchmark.frozen_calibration import fit_ols

    with pytest.raises(ValueError, match="Singular"):
        fit_ols(["t"], {"t": [0.5] * 5}, {"t": {"human_p_a": 0.5}})


def run_offline_cli(*arguments):
    # The guard runs inside the child interpreter, not only in pytest's process.
    bootstrap = """
import socket
import runpy
def forbidden(*args, **kwargs):
    raise AssertionError('NETWORK FORBIDDEN in CLI subprocess')
socket.create_connection = forbidden
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
runpy.run_module('benchmark.frozen_proxy_cli', run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-c", bootstrap, *map(str, arguments)],
        cwd=CONFIG_PATH.parents[1], text=True, capture_output=True, timeout=30,
    )


def test_cli_replay_is_offline_and_refuses_existing_output_directory(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    out = tmp_path / "replayed"
    originals = (path.read_bytes(), CONFIG_PATH.read_bytes())
    arguments = ["replay", "--results", path, "--config", CONFIG_PATH,
                 "--out-dir", out, "--fixture", "--bootstrap", "12"]
    completed = run_offline_cli(*arguments)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["api_calls"] == 0
    report = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert report["paper_replay"] is False
    assert report["input_validation"]["paper_scale"] is False
    assert report["bootstrap"]["n"] == 12
    assert not any("21 retained" in text for text in report["limitations"])
    saved = {p.name: p.read_bytes() for p in out.iterdir()}
    second = run_offline_cli(*arguments)
    assert second.returncode == 2
    assert "already exists" in second.stderr
    assert saved == {p.name: p.read_bytes() for p in out.iterdir()}
    assert originals == (path.read_bytes(), CONFIG_PATH.read_bytes())


def test_cli_verifies_configuration_digest_not_only_result_digest(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    config_copy = tmp_path / "config.json"
    config_copy.write_bytes(CONFIG_PATH.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "configuration": {"sha256": sha256_file(config_copy)},
        "datasets": {"fixture": {"results": {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        }}},
    }), encoding="utf-8")
    arguments = ["validate", "--results", path, "--config", config_copy,
                 "--artifact-manifest", manifest, "--dataset", "fixture", "--fixture"]
    valid = run_offline_cli(*arguments)
    assert valid.returncode == 0, valid.stderr
    audit = json.loads(valid.stdout)
    assert audit["configuration"]["provenance_verified"] is True
    assert audit["input_artifact"]["provenance_verified"] is True
    config_copy.write_bytes(config_copy.read_bytes() + b"\n")
    rejected = run_offline_cli(*arguments)
    assert rejected.returncode == 2
    assert "configuration hash mismatch" in rejected.stderr


def test_candidate_refresh_caveat_is_tied_to_authenticated_dataset(tmp_path, rows):
    path = write_rows(tmp_path, rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "configuration": {"sha256": sha256_file(CONFIG_PATH)},
        "datasets": {"candidate_refresh": {"results": {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        }}},
    }), encoding="utf-8")
    out = tmp_path / "refreshed"
    completed = run_offline_cli(
        "replay", "--results", path, "--config", CONFIG_PATH, "--out-dir", out,
        "--artifact-manifest", manifest, "--dataset", "candidate_refresh", "--fixture", "--bootstrap", "5",
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert any("21 retained" in text for text in report["limitations"])
    assert report["paper_replay"] is False


def test_cli_calibration_is_offline_and_cannot_overwrite_results(tmp_path):
    results, targets, _, _ = calibration_fixture(tmp_path)
    out = tmp_path / "calibrated"
    arguments = ["calibrate", "--results", results, "--human-targets", targets,
                 "--out-dir", out, "--fixture"]
    completed = run_offline_cli(*arguments)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["api_calls"] == 0
    report_path = out / "calibration.json"
    original = report_path.read_bytes()
    assert json.loads(original)["paper_scale"] is False
    repeated = run_offline_cli(*arguments)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert report_path.read_bytes() == original


def test_offline_cli_exposes_no_paid_run_command():
    help_result = run_offline_cli("--help")
    assert help_result.returncode == 0
    assert "{validate,replay,calibrate}" in help_result.stdout
    assert "--api-key" not in help_result.stdout
    assert run_offline_cli("run").returncode == 2
