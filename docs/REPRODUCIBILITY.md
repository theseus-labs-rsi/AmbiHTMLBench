# Reproducibility and validation artifacts

The repository includes the frozen Final Arena90 replay inputs and expected snapshot under `tests/golden/`, together with regression, integration and unit tests under `tests/`. The complete validation conclusions and claim boundary are recorded in `docs/FINAL_SYSTEM_VALIDATION_REPORT.md` and `docs/VALIDATION_CLAIMS.md`.

The full validation sequence contained:

1. **Golden replay / regression** - RC1 reproduced the frozen Final Arena90 ranking and exposed legacy sequential-Elo order dependence.
2. **Fresh45 end-to-end holdout** - 45 fresh pages, 225 physical pairs, 450 dual-orientation Judge observations, full production CLI path.
3. **Stress-panel blind audit** - used for failure-mode inspection only, not a global accuracy estimate.
4. **Unbiased random human-validity control** - 30 random Fresh45 physical pairs, 60 original-prompt control observations, blind human labels fixed before unblinding.

Important final numbers:

- Final Arena90 orientation agreement: 80.67%.
- Fresh45 orientation agreement: 84.89%.
- Fresh45 vs Final Arena90 rank Spearman: 0.915.
- Unbiased panel orientation agreement: RC1 93.3%, original prompt 83.3%.
- Unbiased panel collapsed human exact agreement: RC1 66.7%, original prompt 70.0%.

The last comparison does not establish prompt-level human-accuracy superiority. See `docs/FINAL_SYSTEM_VALIDATION_REPORT.md`.
