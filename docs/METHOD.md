# Subjective Evaluation Method v1

## 1. Evaluation objective

The subjective benchmark asks a VLM Judge to approximate a normal human visual reconstruction judgment: **which candidate looks more like the Reference and more completely reconstructs the page?**

This definition deliberately rejects the earlier idea that subjective evaluation must be artificially decoupled from objective metrics. If an error is visually obvious - for example a missing table, missing image, severe page-height distortion, broken text block, or lost chart - it can legitimately affect both objective and subjective evaluation from different perspectives.

The Judge is also instructed not to behave like OCR. Tiny character differences that a normal viewer would not notice should not dominate.

## 2. Prompt

The authoritative prompt is stored at:

`benchmark/prompts/subjective_v1.md`

Prompt version: `AmbiHTMLBench-pilot-elo-v2-human-visual`
Prompt SHA256: `786c9de357c08373de9158145f639b666650f3bcfe81a1f8c59c600ccee7d3bb`

`benchmark.subjective` loads this file directly and checks the hash at import time. A prompt edit therefore requires an explicit protocol/version change and revalidation.

## 3. GT-driven visual focus axes

Seven axes are supported:

1. `IMAGE_VISUAL_FIDELITY`
2. `TABLE_VISUAL_STRUCTURE`
3. `CHART_VISUAL_ENCODING`
4. `FORMULA_2D_LAYOUT`
5. `ALGORITHM_BLOCK_STRUCTURE`
6. `VERTICAL_TEXT_PRESENTATION`
7. `LIST_HIERARCHY_PRESENTATION`

Each active axis is derived from Reference GT annotations. Targets are localized with `bbox_1000` coordinates in Reference space. The boxes help the Judge locate the Reference object; the candidate is not required to reproduce the same absolute coordinates.

Axes are **focus and diagnostic channels**. They are not an additive formula and do not have global hard-coded weights. Overall winner remains a holistic Reference-fidelity judgment.

## 4. Dual-orientation reliability protocol

Each physical model pair is judged twice:

```text
canonical: Reference | model A | model B
swapped:   Reference | model B | model A
```

The swapped result is canonicalized back to the same physical model order.

Pair collapse:

```text
A / A       -> A
B / B       -> B
TIE / TIE   -> TIE, tie_type=perceptual
any mismatch -> TIE, tie_type=reliability
```

A reliability TIE does **not** mean the candidates are visually indistinguishable. It means the Judge did not reproduce the same directional preference after the displayed order was exchanged, so the pair contributes neutral rating evidence.

## 5. Primary rating algorithm

The production ranking uses one collapsed result per physical pair and fits all comparisons jointly with a Bradley-Terry maximum-likelihood model on the standard Elo logistic scale:

`P(A > B) = 1 / (1 + 10^((R_B - R_A)/400))`

Implementation properties:

- offline joint fit, not sequential updates;
- deterministic MM solver;
- no K-factor;
- no input-order dependence;
- ties contribute 0.5 / 0.5;
- ratings centered at 1000;
- Elo scale 400;
- comparison graph connectivity is validated.

The primary fit is **unweighted**. Judge confidence is retained in raw data and used only for sensitivity analysis.

## 6. Balanced Arena design

For an even number of models, `arena_schedule.py` constructs complete round-robin perfect-match rounds. With 10 models:

- 9 rounds;
- 5 physical pairs per page;
- 9 pages cover all 45 unordered model pairs once.

Repeating the 9-page block produces balanced repeated measurements across different pages without evaluating all 45 pairs on every page.

## 7. Engineering safeguards

The production path also includes:

- single-browser reuse for candidate rendering;
- polyfill and remote-MathJax fallbacks;
- static HTML fallback for blocked remote scripts;
- request-size estimation before paid Judge calls;
- deterministic reserve-page replacement;
- atomic decision checkpointing and resume;
- strict JSON parsing with schema-specific tolerant recovery;
- bootstrap intervals over complete round-robin blocks;
- model-pair and axis reliability reports.

## 8. Validated claim boundary

The final experiments validate the system-level redesign and reliability/rating improvements. The final unbiased blind panel did **not** prove that the new prompt has higher human-agreement accuracy than the original prompt. Therefore the release should describe the prompt as a corrected benchmark specification, not a proven accuracy improvement.
