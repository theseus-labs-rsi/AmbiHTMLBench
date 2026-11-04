# AmbiHTMLBench

AmbiHTMLBench evaluates how well models reconstruct page images as executable HTML. It provides two complementary evaluation paths:

- **Objective evaluation:** render predicted HTML in Chromium, extract visible blocks, align them with annotated ground truth (GT), and measure text, table, and layout fidelity.
- **Subjective evaluation:** replay the paper's frozen, human-calibrated five-judge OLS preference proxy, then fit a soft Bradley?Terry ranking with page-level bootstrap intervals.

The paper uses the **`candidate_refresh` release of the frozen OLS5 protocol**. Historical Arena v1 code and commands have been removed. The subjective tools in this repository operate on recorded judgments; they do not collect new judgments through model APIs.

## Installation

Python 3.10 or later is required.

```bash
python -m pip install -e .
python -m playwright install chromium
```

Chromium is required for objective HTML evaluation. The offline subjective modules use only the Python standard library and do not require a browser or API key.

To install development dependencies and run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Inspect the available commands:

```bash
python main.py --help
python -m benchmark.frozen_proxy_cli --help
```

Installed entry points are `ambihtmlbench` (alias `ambibench`) for objective evaluation and `ambihtmlbench-proxy` for the frozen subjective proxy.

## Objective evaluation

### Inputs

The default ground-truth file is `AmbiHTMLBench_GT_v5_1115.json`. It contains a list of pages with exactly three top-level fields: `ID`, `imageUrl`, and `contentList`. Each block in `contentList` describes its type in `name`, its text, table HTML, or formula content in `remark`, and its bounding box in `left`, `top`, `width`, and `height`.

Prepare model outputs locally in this layout:

```text
all_model_infer/
  model_a/
    html/
      0001_example.html
      0002_example.html
  model_b/
    html/
      0001_example.html
```

Directory names become model names in the leaderboard. The first four characters of an HTML filename identify its page: `0006_example.html` corresponds to GT page `ID=6`.

`all_model_infer/` is excluded from the current main branch and ignored by Git. Supply the inference files separately. Generated results under `outputs/` are also ignored.

### Run the evaluator

```bash
python main.py eval-objective-v3 --gt-json AmbiHTMLBench_GT_v5_1115.json --infer-dir all_model_infer --out-dir outputs/objective_v3
```

Evaluate selected models and pages:

```bash
python main.py eval-objective-v3 --models gpt-5.6-sol,gemini-3.1-pro-preview --page-ids 0006,0359 --out-dir outputs/objective_debug
```

For a small run without visualization output:

```bash
python main.py eval-objective-v3 --max-pages 20 --no-visualizations --out-dir outputs/objective_smoke
```

`--max-pages` limits the number of pages per model. Partial CSV files support resuming interrupted runs; use a separate output directory for a different evaluation scope.

### Outputs

| Output | Contents |
|---|---|
| `page_scores.csv` | Metrics for each model and page |
| `leaderboard.csv` | Aggregated model scores and ranks |
| `page_scores.partial.csv`, `leaderboard.partial.csv` | Intermediate results |
| `visualizations/<model>/<page_id>.html` | Reference and prediction boxes, text/table matches, and layout edges |
| `visualizations/<model>/_assets/<page_id>.png` | Rendered prediction screenshots |

Visualizations are enabled by default. Use them to inspect alignment quality, or pass `--no-visualizations` to reduce runtime and output size.

### Extraction and alignment

The evaluator renders HTML in headless Chromium and extracts visible DOM blocks. Text nodes are grouped into semantic blocks. Tables remain whole objects during extraction, while images, SVG, canvas, video, figures, and mathematical structures are extracted as structural blocks. Elements with `placeholder` in their class names, hidden or transparent text, and text below the visibility thresholds are excluded.

**Tables.** Candidate table matches combine content similarity, structural similarity, text agreement, and geometry. Hungarian assignment produces a global one-to-one matching. Matched tables receive TEDS and TEDS-S scores. Cells from unmatched tables enter the text pool.

**Formulas.** Formula content enters the text pool after dollar delimiters are removed. The current evaluator does not compute a separate formula CDM score; the compatibility field `formula_cdm` is normally empty.

**Text.** A bipartite graph links GT and predicted text using `partial_ratio` similarity. Connected components support one-to-one, one-to-many, many-to-one, and many-to-many alignment. The matcher considers reading, horizontal, reversed, and graph-guided orders, with bounded local reordering. Segmentation dynamic programming forms aligned groups. Page text distance uses character-level Levenshtein distance with denominator `max(len(gt), len(pred), 1)`. The GT character `?` acts as a wildcard for one character.

### Metrics and aggregation

| Metric | Meaning | Better |
|---|---|---|
| `table_teds` | Approximate table content and structure similarity using normalized row/cell sequence edits | Higher |
| `table_teds_s` | Approximate table structure similarity, excluding cell text | Higher |
| `text_edit_distance` | Normalized page text edit distance | Lower |
| `layout_graph_similarity` | Similarity of aligned block geometry and spatial relationships | Higher |

Table metrics are averaged over pages with GT tables. Text and layout metrics are averaged over all evaluated pages. The model score is:

```text
text_quality = 1 - text_edit_distance
weighted_score = (text_quality + layout_graph_similarity + 0.3 * table_teds) / 2.3
```

When none of a model's evaluated pages has GT tables, the evaluator omits the table term and uses denominator `2.0`. TEDS-S remains a diagnostic metric and does not contribute an additional weight.

Layout nodes represent matched tables and text groups. Groups containing multiple blocks use an enclosing box on each side; groups with an empty side are excluded. The scorer compares node centers and sizes, constructs a `k=3` nearest-neighbor graph on GT centers, and compares corresponding predicted edges by direction, length, and displacement:

```text
layout_graph_similarity = 0.55 * edge_score + 0.30 * node_score + 0.15 * coverage_f1
```

### Merge existing reports

Combine page-score CSVs without rendering HTML again:

```bash
python -m scripts.build_leaderboard outputs/run_a/page_scores.csv outputs/run_b/page_scores.csv --out-dir outputs/merged
```

Repeat `--exclude-model MODEL_NAME` to exclude models. The four experiment-specific `build_leaderboard_*` scripts retain their default paths and share the same report implementation.

Historical merged reports always use denominator `2.3` for model scores, including models without GT tables. The main evaluator uses `2.0` in that case. Use a consistent scoring convention when comparing reports.

## Frozen OLS5 subjective evaluation

### Paper protocol and result versions

The human calibration set contains 1,872 physical comparisons. Five judges each evaluated both display orders twice, producing 37,440 judgments. The mean of each judge's four canonical overall votes formed one OLS feature.

Deployment uses one display order and one vote per judge for each comparison: 10,000 comparisons and 50,000 judgments. Overall labels are encoded as `A=1`, `TIE=0.5`, and `B=0`, then combined using the frozen coefficients:

```text
p_A = clip(sum(beta[j] * vote[j] for j in the five judges), 0, 1)
```

There is no intercept or weight normalization. Diagnostic axes explain visual differences but do not contribute to this overall score. The continuous preferences feed a soft Bradley?Terry model; intervals use 2,000 whole-page bootstrap replicates with fixed proxy predictions and coefficients.

Two deployment datasets are retained under the same frozen method:

| Dataset | GPT-5.6-Sol BT score | Role |
|---|---:|---|
| `original_single` | 154.34 | Original single-vote deployment |
| `candidate_refresh` | 154.70 | **Current paper results** |

The refresh accepted 54 model-page replacements affecting 98 comparisons. Of these, 21 belong to the earlier human subset; old labels do not validate the updated candidates. All 10,000 deployment comparisons are proxy-scored, so the leaderboard does not measure full-deployment human accuracy, recall, or F1.

### Validate and replay recorded judgments

The repository includes frozen code, configuration, prompts, aggregate evidence, and both leaderboard releases. Full judgment records, candidate/reference images, and participant annotations are external artifacts. The `artifacts/` paths below are examples of separately supplied files.

Validate the paper's recorded results:

```bash
python -m benchmark.frozen_proxy_cli validate --results artifacts/candidate_refresh/single_results.jsonl --config configs/subjective_ols5_frozen.json --artifact-manifest evidence/subjective/dataset_versions.json --dataset candidate_refresh
```

Replay the leaderboard and bootstrap intervals:

```bash
python -m benchmark.frozen_proxy_cli replay --results artifacts/candidate_refresh/single_results.jsonl --config configs/subjective_ols5_frozen.json --artifact-manifest evidence/subjective/dataset_versions.json --dataset candidate_refresh --out-dir outputs/candidate_refresh_replay --bootstrap 2000
```

Replay and calibration require a new output directory and refuse to overwrite an existing one. Without an artifact manifest, validation checks structure but cannot authenticate a paper release. `--fixture` is for small synthetic tests and must not be described as paper reproduction.

Use `calibrate` to reproduce OLS fitting and leave-one-annotation-block-out evaluation from recorded calibration judgments and human targets. See [Frozen OLS5 evaluation](docs/FROZEN_PROXY.md) for the command and detailed scoring semantics.

The repository does not provide a new image-to-render-to-API collection pipeline. Historical inference files do not contain the 54 refreshed candidates, and objective GT v5 is not interchangeable with the frozen subjective GT snapshot. See [version and provenance notes](docs/SUBJECTIVE_SYNC_PROVENANCE.md) before preparing external inputs.

## Repository layout

```text
benchmark/
  cli.py                    # Objective evaluation CLI
  objective_v3.py           # Alignment, metrics, reports, and visualizations
  objective_reports.py      # Merge existing objective score CSVs
  html_extractor.py         # Visible block extraction with Chromium
  schema.py                 # Page and element data structures
  scorer.py                 # Legacy objective Python API
  frozen_proxy_cli.py       # Offline validate, replay, and calibrate commands
  frozen_proxy.py           # Frozen scoring, soft Bradley?Terry, and bootstrap
  frozen_proxy_io.py        # Record validation and artifact identity checks
  frozen_calibration.py     # OLS fitting and annotation-block evaluation
  prompts/                  # Frozen English OLS5 prompt
configs/                    # Frozen OLS5 coefficients
scripts/                    # Report and dataset preparation utilities
tests/                      # Unit, integration, and regression tests
evidence/subjective/        # Aggregate evidence, result versions, and digests
docs/                       # Architecture, protocol, and provenance
main.py                     # Objective CLI entry point
```

See [Architecture](docs/ARCHITECTURE.md) for module responsibilities and [Subjective evidence](evidence/subjective/README.md) for the included experiment records.
