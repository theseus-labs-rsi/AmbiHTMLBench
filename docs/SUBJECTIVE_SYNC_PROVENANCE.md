# Subjective evaluation: versions and synchronization scope

The repository retains the paper's frozen OLS5 scoring logic, offline validation/replay/calibration interfaces, tests and lightweight evidence. It does not migrate the entire experiment workstation, run new paid judgments, rewrite historical freezes or bundle candidate data.

Before OLS5 was synchronized, `paper_summary.md` already reported the candidate-refresh leaderboard while the recommended commands implemented historical Arena v1. The offline path closed that gap. Arena v1 code, commands, configuration and dedicated tests have since been removed; objective evaluation is retained separately.

## Three data versions

| Version | Scope and identity | Human-label boundary |
|---|---|---|
| Human calibration | 1,872 comparisons; five judges × two orders × two repeats = 37,440 judgments; eight annotation blocks | Same-comparison human means train and internally validate the proxy. Not an independent deployment test. |
| `original_single` | Frozen 10,000-comparison schedule; five single votes per comparison = 50,000; GPT-5.6-Sol BT 154.34 [140.25, 168.75] | Retains the 1,872 calibration comparisons; 8,128 additional comparisons have no new human labels. All 10,000 are proxy-scored. |
| `candidate_refresh` | Same schedule, coefficients and single-vote protocol; 54 accepted model-page updates affect 98 comparisons; GPT-5.6-Sol BT 154.70 [140.59, 168.82] | 21 affected comparisons belong to the old human subset; old human labels cannot validate their changed candidates. |

Both deployment releases have complete 50,000/50,000 recorded judgments, 10,000 physical comparisons and 2,000/2,000 appearances per generation model. Both report 2,000 whole-page bootstrap replicates with fixed predictions and coefficients. The two Sol scores identify distinct candidate versions; their difference is not an independently validated performance improvement.

The earlier dual-direction deployment is retired. Historical helper code or source hashes referring to it document ancestry, not a recommendation to resume it.

## Candidate-refresh accounting

The rerun archive contains 152 model-page HTML candidates. Of these, 58 enter the frozen 10,000-comparison schedule: 54 passed the existing visible-text and render-geometry screens and four were rejected. The other 94 were outside this update's schedule. Accepted replacements affect 98 comparisons: 77 from the original additional set and 21 from the old human subset.

The refreshed result set contains 490 new logical judge tasks and 49,510 reused results, totaling 50,000. “Logical tasks” is not a claim about the number of HTTP attempts or billable retries. Model appearances, pair quotas, displayed-order selection and repeat counts remain unchanged. Original and refreshed results must be kept as separate artifacts, with a replacement/reuse ledger; never overwrite the original result and call it the same release.

The candidate-update filter is part of the recorded experiment. Bootstrap intervals do not include uncertainty from that filter. This code synchronization does not rerun the filter or authorize further candidate replacement.

## Repository HTML and GT are not the frozen subjective input bundle

The historical audit examined 11,150 HTML files: ten models × 1,115 pages. All matched the original inference archive by SHA-256. The 54 accepted refreshed HTML files were **not** synchronized into that tree. `all_model_infer/` has since been removed from the current main branch and is ignored by Git. The candidate-refresh leaderboard cannot be assigned to locally supplied HTML merely because model names and page IDs match.

Keep these subjective inputs in a versioned external artifact with per-file hashes and an explicit replacement ledger. The offline scorer does not replace candidates.

The repository's current objective GT is v5; the frozen subjective pipeline used an older GT snapshot. Both contain the same 1,115 page IDs and reference image URLs, but 649 `contentList` records changed. An audit with the frozen axis-mapping rules found four changed special-axis sets: pages 0065, 0690, 1006 and 1047. Thus matching IDs or image URLs is insufficient evidence of protocol identity. Do not silently rebuild frozen active axes from GT v5 or reselect comparisons using the current objective inputs.

## Lightweight evidence included here

| Repository artifact | Purpose |
|---|---|
| [Frozen config](../configs/subjective_ols5_frozen.json) and [prompt](../benchmark/prompts/subjective_ols5_frozen.txt) | Exact score definition and versioned judgment instructions |
| [Human summary](../evidence/subjective/human/evidence.json) | Aggregate human-study evidence, without participant-level records |
| [Calibration evidence](../evidence/subjective/calibration/evidence.json) | Recorded coefficients, internal validation and uncertainty |
| [Committee evidence](../evidence/subjective/committee/evidence.json) | Pilot/holdout selection, coverage and limitations |
| [Ablation evidence](../evidence/subjective/ablation/evidence.json) | Recorded judge-subset comparisons, not new API experiments |
| [Original leaderboard](../evidence/subjective/original_single/leaderboard.json) | Original single-deployment results; CSV alongside |
| [Candidate-refresh leaderboard](../evidence/subjective/candidate_refresh/leaderboard.json) | Current paper results; CSV alongside |
| [Leaderboard delta](../evidence/subjective/candidate_refresh/leaderboard_delta.json) | Recorded comparison of the two candidate versions |
| [Dataset versions](../evidence/subjective/dataset_versions.json) | External data identities, source/result/GT hashes and version metadata |
| [Release manifest](../evidence/subjective/release_manifest.json) | Hashes of the lightweight release files |

The release manifest records the original OLS5 import snapshot, including documentation that has since changed; it is not an inventory of the current code tree. The dataset manifest identifies external experiment artifacts and the unchanged frozen configuration and prompt. Neither means that full artifacts have been published or that their privacy and redistribution rights have been cleared.

Not bundled: large judgment JSONL files, SQLite checkpoints, raw participant records, candidate/reference image collections, rerun archives, rendering caches, local credentials or proprietary system fonts. Existing absolute-path freeze files remain historical evidence outside this release; they are not rewritten to appear portable. External datasets need an approved storage location, a privacy/license review and portable manifests before distribution.

## Committee-selection evidence

The old three-judge committee (Sonnet + Gemini + Terra) ranked first on the 320-comparison pilot, but fourth among four eligible cross-family committees on the independent 1,552-comparison holdout. On holdout its abstention-aware Brier was 0.1293 with 78.99% resolved coverage. Opus + Gemini + Sol achieved 0.1240 with 80.35% coverage.

The paired difference, old minus alternative, was +0.0053 with 95% CI [-0.0001, +0.0106]. The alternative ranked first in 76.81% of 10,000 page-block bootstrap samples. These numbers do not establish statistically significant superiority over every competitor. The alternative was chosen after inspecting holdout, so that holdout is selection evidence, not fresh independent confirmation of the chosen alternative.

Rare special axes were concentrated in the pilot; the holdout did not cover formula, algorithm, vertical-text or list axes. For the three full-data special-axis slices with at least 30 cases, the old committee had lower descriptive macro-Brier (0.1268 versus 0.1324). The overall and special-axis findings must be reported together.

This study uses discrete committee aggregation and human-vote weighting. Its Brier values are not directly interchangeable with the equal-comparison OLS5 validation results. The final deployed paper proxy uses all five recorded judges and fixed regression coefficients, not the old three-judge vote. Retaining the committee result is necessary provenance, not a claim that it validates the final proxy externally.

## Remaining limitations and reproducibility boundary

1. Calibration uses four-vote means; deployment uses one vote. No new experiment closes that feature-distribution gap.
2. The no-intercept, unnormalized proxy need not be complementary under canonical A/B re-encoding. Replay retains the published convention.
3. Historical reuse chose directions from completion availability, never by inspecting winners; it may still reflect response latency.
4. No new human labels cover the 8,128 additional comparisons. Calibration overlap and the 21 refreshed old-human cases prevent claims of independent full-deployment human accuracy.
5. Deployment bootstrap fixes proxy coefficients and predictions. It measures page-sampling variation, not retraining, candidate cleaning/selection or all systematic uncertainty.
6. In the original single deployment, 18 repair jobs used high-quality compressed reference-image copies; the last job additionally reduced the long edge from 6,320 to 6,000 pixels (about 5.1%). Originals and comparison identities were preserved. Their repair audit belongs with the external input artifact, not an undocumented preprocessing default.
7. This release supports offline reconstruction from recorded judgments. New image rendering, cleaning, provider requests, resumable paid execution and full input-asset provenance verification are not a newly migrated end-to-end service.

The offline layer does not need a new browser or model call. The existing full-repository tests are a separate environment boundary: the objective module imports SciPy, which was not declared in the supplied dependency files. SciPy was supplied only in the temporary verification environment; this subjective synchronization does not change the owner's objective code or dependency declarations.

See [FROZEN_PROXY.md](FROZEN_PROXY.md) for commands and scoring semantics. Historical Arena v1 code and documentation remain available in Git history. The committee and calibration evidence above retain the studies' recorded findings and boundaries.
