# AmbiHTMLBench Subjective RC1 — Final System Validation Report

## 1. Final decision

**RC1 release gate: PASS WITH CLAIM LIMITATION.**

The integrated RC1 functionality has passed engineering, regression, algorithmic, end-to-end Fresh45, ranking-stability, and reliability validation. The final unbiased random human-validity panel closes the last evidence gap and shows a nuanced result:

- RC1 is **more orientation-stable** on the unbiased panel: **93.3% vs 83.3%**.
- RC1 is **not demonstrated to be more human-aligned** than the original prompt on this 30-pair single-reviewer audit.
- Human-agreement point estimates are slightly higher for the original prompt, but the paired differences are small and statistically inconclusive at this sample size.
- Therefore the principal validated improvement is the **benchmark system and reliability methodology**, not a proven prompt-accuracy gain.

No further API experiment is required to release RC1, provided the public report uses the supported claims below.

## 2. What RC1 has validated

RC1 integrates and validates the following production mechanisms:

- corrected subjective benchmark definition: human-like visual reconstruction fidelity;
- frozen Reference-anchored Prompt with GT-driven active axes and targets;
- robust Reference / Candidate A / Candidate B rendering;
- canonical + swapped dual-orientation judging;
- canonicalization and explicit `reliability TIE` / `perceptual TIE` semantics;
- balanced round-robin Arena scheduling;
- deterministic request-size preflight and checkpoint/resume;
- tolerant Judge JSON parsing;
- order-invariant Batch Bradley–Terry Elo;
- bootstrap uncertainty;
- machine diagnostics and audit/report generation.

## 3. Regression and algorithm evidence

From the RC1 System Validation R1 replay:

- SUT identity / pytest / compileall / CLI smoke: **PASS**
- Final Arena90 replay ranking: **exact**
- Final Arena90 max Elo replay difference: **0.000415**
- Legacy sequential Elo forward/reverse maximum rank movement: **2 positions**
- Legacy sequential Elo forward/reverse maximum Elo movement: **56.88**

This is direct evidence that the legacy online Elo was inappropriate for the offline benchmark and that Batch Bradley–Terry fixes a real order-dependence failure.

## 4. Fresh45 end-to-end system experiment

- Fresh pages: **45**
- Physical pairs: **225**
- RC1 Judge calls: **450**
- Preflight: **PASS**
- Replacements: **0**
- Largest request: **4.32 MiB**
- Orientation agreement: **84.9%**
- Reliability TIEs: **34**
- Perceptual TIEs: **3**
- Displayed A / B rates: **47.1% / 50.2%**
- Fresh45 vs Final Arena90 rank Spearman: **0.915**
- Primary vs raw-dual unweighted max rank delta: **0**
- Primary vs raw-dual confidence-weighted max rank delta: **0**

### Fresh45 ranking

| Rank | Model | Elo | 95% bootstrap interval | W-T-L |
|---:|---|---:|---|---:|
| 1 | gpt-5.6-terra | 1246.49 | [1152.74, 1389.76] | 32-7-6 |
| 2 | gpt-5.6-sol | 1186.58 | [1099.48, 1312.07] | 31-3-11 |
| 3 | qwen3.7-plus | 1080.50 | [1037.75, 1134.57] | 22-9-14 |
| 4 | Kimi-K3 | 1072.04 | [989.14, 1164.16] | 22-8-15 |
| 5 | claude-opus-5 | 1063.60 | [942.42, 1173.12] | 21-9-15 |
| 6 | claude-sonnet-5 | 1063.60 | [1038.71, 1099.85] | 20-11-14 |
| 7 | GLM-5V-Turbo | 996.07 | [907.00, 1050.32] | 16-11-18 |
| 8 | gemini-3.1-pro-preview | 943.92 | [898.57, 976.50] | 15-7-23 |
| 9 | ERNIE-5.0 | 772.14 | [723.27, 826.51] | 6-8-31 |
| 10 | grok-4.5 | 575.06 | [252.05, 707.41] | 3-1-41 |

The middle ranks move across page samples, while the macro ranking remains strongly correlated. Bootstrap intervals should accompany leaderboard claims.

## 5. Final unbiased human-validity panel

### Design

The final panel is a **simple random sample without replacement of 30 physical pairs from all 225 Fresh45 pair sets**.

- seed: `2026081203`
- no filtering by RC1 winner;
- no filtering by orientation agreement;
- no filtering by confidence;
- no filtering by active axis;
- no filtering by model identity or page type;
- RC1 observations reused from Fresh45;
- original prompt newly evaluated on canonical + swapped: **60 calls**;
- blind human labels fixed before system outputs were unblinded;
- reviewer: **one blind visual reviewer**;
- human labels: **17 A / 12 B / 1 TIE**.

### Machine reliability

| Metric | RC1 | Original prompt |
|---|---:|---:|
| Orientation agreement | **93.3%** | 83.3% |
| Orientation disagreement pairs | 2 / 30 | 5 / 30 |
| Displayed-A win rate | 45.0% | 41.7% |
| Displayed-B win rate | 51.7% | 58.3% |
| Mean confidence | 3.53 | 4.03 |

RC1's orientation-agreement point advantage is **+10.0 percentage points**. The paired bootstrap 95% interval is **[-3.3, 23.3] pp**; exact McNemar p = **0.375**. This is encouraging but not statistically decisive with n=30.

### Blind human alignment

| Metric | RC1 | Original prompt |
|---|---:|---:|
| Raw orientation-level exact agreement | 70.0% | **73.3%** |
| Collapsed exact agreement | 66.7% | **70.0%** |
| Collapsed soft alignment (TIE=0.5) | 73.3% | **76.7%** |
| Directional coverage | **90.0%** | 83.3% |
| Directional precision on human-directional cases | 76.9% | **80.0%** |
| Harmful opposite-direction outcome / all pairs | 20.0% | **16.7%** |

The point estimates do **not** show prompt-level human-alignment superiority for RC1.

The collapsed exact-agreement difference (RC1 minus original) is **-3.3 pp**, with paired bootstrap 95% interval **[-16.7, 10.0] pp** and exact McNemar p = **1.000**.

The raw orientation-level agreement difference is also **-3.3 pp**, with paired physical-pair bootstrap 95% interval **[-15.0, 8.3] pp**.

These results are compatible with no meaningful human-agreement difference between the prompts in this sample.

## 6. How the unbiased result changes the earlier stress-panel interpretation

The earlier 24-case panel deliberately oversampled RC1 disagreement / confidence strata. It suggested that RC1's reliability layer was more conservative and reduced harmful stable wrong-direction decisions on those hard cases.

The new unbiased panel shows that this finding should **not** be generalized into a universal prompt-accuracy claim:

- RC1 is substantially more stable to A/B orientation in this random sample.
- The original prompt has slightly higher single-reviewer visual agreement point estimates.
- Only **6 / 30** collapsed pair outcomes differ between the two systems, and the advantages are mixed.
- Thus the validated improvement is best described as **evaluation specification + reliability control + ranking/statistical engineering**, not “the new prompt wins on human accuracy”.

## 7. Prompt decision

The RC1 prompt should still be retained in the release because it fixes specification-level problems in the legacy design:

- it removes the arbitrary global `40% layout / 35% text / 20% style / 5% other` weighting;
- it no longer frames subjective evaluation as something that must be artificially decoupled from objective metrics;
- it explicitly defines human-visible fidelity to Reference;
- it supports GT-driven active axes and targets without turning them into a hard scoring formula;
- it distinguishes visible text/layout failures from OCR-level character checking.

However, the release documentation should describe this as a **benchmark-definition redesign**, not as an empirically proven human-accuracy improvement.

## 8. Final claim boundary

### Supported

1. RC1 is a mature integrated subjective-evaluation subsystem.
2. Batch Bradley–Terry fixes demonstrated sequential-Elo order dependence.
3. Dual orientation supplies an explicit reproducibility/reliability signal.
4. RC1 achieved higher orientation agreement than the original prompt on the unbiased 30-pair sample.
5. Fresh45 and Final Arena90 rankings are strongly consistent at the macro level.
6. Preflight, balanced scheduling, checkpointing, parsing, bootstrap and reporting operate end to end.

### Not supported

1. “RC1 Prompt is more human-aligned than the original Prompt.”
2. “RC1 Prompt has higher human accuracy.”
3. “All reliability TIEs mean humans perceive the candidates as equally good.”
4. “The VLM Judge is equivalent to human evaluation.”

## 9. Remaining limitation

The final human comparison is based on **one blind reviewer and 30 random physical pairs**. A multi-annotator human study would be required for a publication-level estimate of human inter-rater agreement and Judge-human agreement. That is a future validation extension, not a blocker for the RC1 software release.

## 10. Final release gate

**PASS WITH CLAIM LIMITATION.**

No further algorithm/prompt/view-layout experiment is justified by the current evidence. The experimental line can now be closed.

The next stage is engineering release work:

1. freeze the tested RC1 code and method IDs;
2. remove validation-only packaging and historical experiment scaffolding;
3. convert RC1 into a clean repository-native open-source implementation;
4. publish the final System Validation Report alongside the code;
5. preserve the raw validation manifests/results as reproducibility artifacts.
