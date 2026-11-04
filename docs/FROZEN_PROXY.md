# Frozen OLS5 subjective evaluation

This is the subjective evaluation implementation for the paper's human-calibrated preference proxy. It reads recorded judgments; it does not render new candidates, contact model providers, or launch paid experiments. The historical Arena v1 implementation and its interfaces have been removed.

## Protocol and score

Each comparison has a fixed canonical A/B identity. A displayed-order judgment is mapped back to that identity before scoring. The five overall labels are encoded as `A = 1`, `TIE = 0.5`, `B = 0` and combined as

```text
p_A = clip(sum(beta[j] * x[j] for j in the five judges), 0, 1)
```

The coefficients in [subjective_ols5_frozen.json](../configs/subjective_ols5_frozen.json) are:

| Judge ID | Coefficient |
|---|---:|
| anthropic_opus5 | 0.09045204815695966 |
| anthropic_sonnet5 | 0.10401091278902799 |
| gemini31pro | 0.37576678499354577 |
| openai_sol | 0.1374207709022521 |
| openai_terra | 0.160124982073635 |

There is no intercept and no weight normalization. The ten-model representation has five additional zero entries; these are not five additional observed judges. Axis scores and confidence do not enter the proxy. The [frozen prompt](../benchmark/prompts/subjective_ols5_frozen.txt) defines the recorded OLS5 judgments.

In particular, unanimous A does not force `p_A = 1`, and five explicit TIE labels do not force `p_A = 0.5`. Their outputs are approximately 0.8677755 and 0.4338877. Re-encoding canonical A/B and recomputing the regression is not guaranteed to yield complementary predictions. The implementation preserves the frozen convention rather than silently normalizing or symmetrizing it. Within a scored comparison, B receives the complementary soft score `1 - p_A` for ranking.

## Calibration and deployment are different observation protocols

| Stage | Comparisons | Observations per judge/comparison | Total machine judgments |
|---|---:|---|---:|
| Human calibration | 1,872 | 2 displayed orders × 2 repeats | 37,440 |
| Original single deployment | 10,000 | 1 selected order × 1 vote | 50,000 |
| Candidate-refresh deployment | 10,000 | 1 selected order × 1 vote | 50,000 |

For calibration, a human target is the mean of valid A/TIE/B labels for the same comparison within its annotation block. `CANNOT_JUDGE` is not a preference label. The eight blocks contain 234 different comparisons each: their per-comparison means are concatenated, not averaged across unrelated questions. OLS gives each comparison equal weight, so blocks have equal total weight despite different annotator counts.

Each regression feature is the mean of a judge's four canonicalized overall votes. Internal validation leaves out one of the eight **annotation blocks**, trains on 1,638 comparisons and predicts the other 234. Historical source fields named `paper_id` and labels such as “leave-one-paper-out” refer to these annotation blocks, not source papers or individual rendered pages.

The recorded out-of-fold results cover all 1,872 comparisons: Brier 0.07783 [0.07216, 0.08361], MAE 0.21151 [0.20307, 0.21999], and ten-model rank Spearman 0.91515 [0.86667, 0.95152]. These are internal, four-vote-feature results, not independently measured single-vote deployment accuracy. Their recorded intervals use 5,000 joint page resamples stratified by annotation block, with stored out-of-fold predictions fixed. See [calibration evidence](../evidence/subjective/calibration/evidence.json).

## Validate and replay recorded deployment results

The offline modules use the Python standard library and do not need a browser installation, browser session or model API credentials. Supply an authorized external artifact containing the original recorded `single_results.jsonl`; this file is not bundled with the lightweight release. Paths below are examples relative to the repository, not downloadable files.

```bash
python -m benchmark.frozen_proxy_cli validate \
  --results artifacts/candidate_refresh/single_results.jsonl \
  --config configs/subjective_ols5_frozen.json \
  --artifact-manifest evidence/subjective/dataset_versions.json \
  --dataset candidate_refresh

python -m benchmark.frozen_proxy_cli replay \
  --results artifacts/candidate_refresh/single_results.jsonl \
  --config configs/subjective_ols5_frozen.json \
  --artifact-manifest evidence/subjective/dataset_versions.json \
  --dataset candidate_refresh \
  --out-dir outputs/candidate_refresh_replay \
  --bootstrap 2000
```

To reproduce the original single deployment instead, use its own results artifact and `--dataset original_single`. Do not relabel an original artifact as the candidate-refresh version. The dataset manifest binds the recorded-results and frozen-config hashes to a release identity. Without it, a structurally valid input is only a validated input, not an authenticated paper release; omit both `--artifact-manifest` and `--dataset` in that mode. `--fixture` relaxes release-size requirements for synthetic tests; fixture results must not be described as paper reproduction.

The recorded input has one JSON object per judgment, including `job_id`, `judge_id`, `repeat_index`, `task` and `result`. Task metadata identifies the physical comparison, page, canonical pair and displayed order; result metadata retains the displayed and canonical overall labels. Deployment requires exactly one result for each of the five judges per comparison, with no duplicate or missing entries. All five records for a comparison must identify the same page and canonical pair. Different selected display orders across judges, where recorded by the historical reuse ledger, do not authorize extra votes.

The full deployment contains 10,000 unique physical comparisons and ten generation models with 2,000 appearances each. Of the 45 unordered model pairs, 35 occur 222 times and ten occur 223 times. This is not the older complete-round-robin-block schedule. Validation must not replace frozen comparisons to satisfy the v1 scheduler.

Replay writes to a **new output directory**. It must not overwrite the source judgments, frozen coefficients or released reports. The output includes the per-comparison composite and leaderboard; compare it only with the matching version's [original-single results](../evidence/subjective/original_single/leaderboard.json) or [candidate-refresh results](../evidence/subjective/candidate_refresh/leaderboard.json).

## Ranking and uncertainty

The soft Bradley–Terry fit consumes `p_A` directly, with unit weight per physical comparison; it does not threshold predictions into wins/ties/losses. The frozen solver uses ridge `1e-3` on latent strengths, centers them at zero, and converts them using `400 / log(10)`. This ranking regularization is distinct from calibration OLS, which has no ridge penalty. BT values are relative ability scores, not accuracy percentages.

The soft preference rate is a model's accumulated continuous score divided by its scheduled appearances. It depends on its scheduled opponents and is not a measured human win rate.

Deployment uncertainty uses 2,000 whole-page bootstrap replicates, seed `20260831`. All comparisons from a sampled page move together. Each draw refits BT, but does **not** refit OLS or regenerate model judgments. The resulting 95% score and rank intervals describe page-sampling variation conditional on the frozen proxy and candidates. They exclude calibration uncertainty, candidate-selection uncertainty and other systematic errors. Changing the replicate count is useful for tests, but does not reproduce the released intervals.

Replay reports successful, failed and disconnected-draw counts. A disconnected bootstrap draw is retained and fitted with the original ridge, including absent models; it is not discarded or replaced by a new draw. Percentile intervals use linear interpolation at `(n - 1) * q`. Exact score ties receive deterministic lexicographic ranks rather than average ranks. These details preserve the frozen sampling sequence and ranking convention.

## Offline calibration reproduction

With separately authorized calibration judgments and human targets:

```bash
python -m benchmark.frozen_proxy_cli calibrate \
  --results artifacts/calibration/full_results.jsonl \
  --human-targets artifacts/calibration/human_consensus.jsonl \
  --artifact-manifest evidence/subjective/dataset_versions.json \
  --dataset calibration \
  --out-dir outputs/calibration_replay
```

This authenticates both the recorded judgments and human targets, then reproduces no-intercept OLS and leave-one-annotation-block-out predictions from existing observations. It is not a new human study or a new API experiment. `calibration.json` and `calibration_cases.jsonl` go to a new directory; the command does not update the frozen deployment config or re-estimate the published calibration bootstrap intervals. A discrepancy is a reason to check the input version and implementation, not to replace published coefficients. Human-target artifacts need a separate privacy review before distribution; this code release includes aggregate evidence, not participant-level records.

## Interpretation boundaries

- Four-vote means trained the proxy; deployment uses one vote. Internal validation metrics cannot be transferred unchanged to the deployed protocol.
- All 10,000 deployment comparisons use proxy scores. The original 8,128 additional comparisons have no new human labels, and the original 1,872 overlap model development. No full-deployment human Accuracy/Recall/F1 was measured.
- The candidate-refresh release changes 21 comparisons from the old human subset. Their old labels do not validate the new candidates.
- Historical direction reuse was selected from completion metadata, not winners. It can still reflect response latency; no winner-based selection should be introduced during replay.
- Diagnostic axes explain local errors but do not enter the overall formula. Their sample counts and rare-axis coverage must accompany axis-specific claims.
- The committee holdout study is evidence about a different, discrete committee protocol, not independent validation of OLS5. Its negative result is retained in the [provenance notes](SUBJECTIVE_SYNC_PROVENANCE.md#committee-selection-evidence).

The full input/version chain, candidate-refresh boundary and omitted artifacts are documented in [SUBJECTIVE_SYNC_PROVENANCE.md](SUBJECTIVE_SYNC_PROVENANCE.md).
