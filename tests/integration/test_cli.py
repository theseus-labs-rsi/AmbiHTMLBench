import pytest

from benchmark.cli import build_parser


def test_objective_cli_defaults_remain_available():
    args = build_parser().parse_args(["eval-objective-v3"])
    assert args.gt_json == "AmbiHTMLBench_GT_v5_1115.json"
    assert args.infer_dir == "all_model_infer"


@pytest.mark.parametrize("command", ["arena-preflight", "arena-run", "arena-report"])
def test_removed_arena_commands_are_rejected(command):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command])
    assert exc.value.code == 2
