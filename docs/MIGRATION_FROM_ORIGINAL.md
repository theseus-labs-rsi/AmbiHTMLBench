# Migration from the supplied original repository

This release is intentionally a drop-in evolution of the original work directory.

## Existing objective files preserved byte-for-byte

- `benchmark/schema.py`
- `benchmark/scorer.py`
- `benchmark/html_extractor.py`
- `benchmark/utils.py`

## Existing files upgraded

- `benchmark/arena.py` - becomes the subjective Arena orchestration/compatibility layer.
- `benchmark/render.py` - replaces per-page fragile rendering with browser reuse and fallbacks.
- `benchmark/cli.py` - keeps objective commands and adds `arena-preflight`, `arena-run`, `arena-report`.
- `benchmark/__init__.py` - exports the new modules/version.

## New production modules

- `benchmark/subjective.py`
- `benchmark/rating.py`
- `benchmark/arena_schedule.py`
- `benchmark/arena_audit.py`
- `benchmark/prompts/subjective_v1.md`

## New repository support files

- `configs/subjective_v1.json`
- `examples/subjective_manifest.example.json`
- `scripts/build_manifest_from_dirs.py`
- `tests/`
- `docs/`
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`

The root entry style remains unchanged.

## Recommended merge procedure

1. Create a branch from the current AmbiHTMLBench repository.
2. Replace the four upgraded files and add the production/support files listed above.
3. Reuse the parent repository license and project metadata as appropriate.
4. Install dependencies and Chromium.
5. Run `python -m pytest`.
6. Run `python main.py arena-preflight --help`, `arena-run --help`, and `arena-report --help`.
7. Keep `benchmark/prompts/subjective_v1.md` unchanged unless a new protocol is intentionally created and revalidated.
