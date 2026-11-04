# Architecture

The repository has two evaluation paths: objective scoring and frozen OLS5.
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

The historical Arena v1 implementation, commands, renderer, schedule builder,
configuration and dedicated tests have been removed. OLS5 has no dependency on
them. `original_single` and `candidate_refresh` remain separate input/result
versions of the same frozen OLS5 method; the paper uses `candidate_refresh`.

Runtime dependencies are limited to Playwright for objective rendering,
RapidFuzz for text comparison and SciPy for table assignment. The offline OLS5
modules use Python's standard library. Historical code is available in Git
history.
