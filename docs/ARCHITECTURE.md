# Architecture

The repository has two current evaluation paths and a historical Arena protocol.
They use separate entry points because their inputs and scoring rules differ.

## Objective evaluation

`python main.py eval-objective-v3` runs the current objective evaluator:

```text
cli.py
  -> html_extractor.py: Chromium rendering and visible DOM blocks
  -> objective_v3.py: GT conversion, table assignment, text graph alignment,
                     layout similarity, CSV output and visualizations
```

`schema.py` defines rendered pages and elements; `utils.py` contains shared text
and geometry helpers. `objective_v3.py` contains only the active matcher. Earlier
experiments remain in Git history. The legacy `scorer.py` Python API is retained.

`objective_reports.py` merges existing page-score CSVs through
`scripts/build_leaderboard.py`. Four experiment scripts are short recipes using
the same implementation. Historical merged reports always divide model scores
by 2.3. The main evaluator omits the table weight when a model has no GT tables.

## Frozen subjective proxy

`python -m benchmark.frozen_proxy_cli` provides offline `validate`, `replay` and
`calibrate` commands:

- `frozen_proxy_io.py`: record validation, canonical orientation, completeness
  and artifact hashes.
- `frozen_proxy.py`: frozen coefficients, soft Bradley–Terry fitting and page
  bootstrap.
- `frozen_calibration.py`: offline calibration and held-out evaluation.

Frozen configurations, prompts and provenance are in `configs/`,
`benchmark/prompts/` and `evidence/subjective/`. External raw experiment artifacts
are supplied separately. See [FROZEN_PROXY.md](FROZEN_PROXY.md).

## Historical Arena v1

`arena-preflight`, `arena-run` and `arena-report` support historical single-judge
experiments:

- `arena.py`: manifest loading, rendering preflight, API orchestration and resume.
- `arena_schedule.py`: balanced round-robin schedules and reserve selection.
- `subjective.py`: diagnostic axes, frozen prompt, judge client and parsing.
- `arena_audit.py`: orientation collapse, reliability, bootstrap and reports.
- `rating.py`: discrete Bradley–Terry fitting.
- `render.py`: browser reuse and bounded rendering fallbacks.

`arena-run` calls the configured paid API. Its rules differ from the current
frozen OLS5 protocol; see [METHOD.md](METHOD.md).
