# Subjective Evaluation Architecture v1

`benchmark.arena` is the orchestration layer. Algorithms that require independent testing are separated into explicit modules.

```text
benchmark.cli
    |
    +-- arena-preflight
    |      +-- arena.load_arena_manifest
    |      +-- arena_schedule: balanced schedule / reserve selection
    |      +-- render: robust Chromium rendering
    |      +-- subjective: GT axes / request-size estimation
    |
    +-- arena-run
    |      +-- subjective.RequestsJudgeClient
    |      +-- canonical + swapped observation
    |      +-- atomic checkpoint / resume
    |
    +-- arena-report
           +-- arena_audit: pair collapse / TIE semantics
           +-- rating: Batch Bradley-Terry
           +-- sensitivity fits
           +-- block bootstrap / diagnostics
```

## Modules

- `arena.py`: manifest normalization, preflight, paid Judge orchestration, checkpointing, legacy compatibility.
- `subjective.py`: prompt/specification, seven axes, GT targets, Judge parser, orientation canonicalization, request-size estimation.
- `rating.py`: deterministic order-invariant Batch Bradley-Terry MLE.
- `arena_schedule.py`: round-robin construction, schedule validation, light stratification and reserve matching.
- `arena_audit.py`: physical-pair collapse, reliability statistics, ranking sensitivities, bootstrap and report writer.
- `render.py`: browser reuse and bounded fallback rendering.
- `cli.py`: repository-facing commands.

The existing objective modules (`schema.py`, `scorer.py`, `html_extractor.py`, `utils.py`) are kept unchanged from the supplied original repository.
