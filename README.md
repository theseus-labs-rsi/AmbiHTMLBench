# AmbiHTMLBench

AmbiHTMLBench evaluates how well models reconstruct page images as executable HTML:

- **Objective evaluation:** measures text, table, and layout fidelity against annotated ground truth.
- **Subjective evaluation:** combines five judges using the paper's frozen OLS5 preference proxy and produces a soft Bradley-Terry ranking.

## Installation

Requires Python 3.10 or later.

```bash
python -m pip install -e .
python -m playwright install chromium
```
Chromium is needed only for objective evaluation. Subjective replay requires no browser or API key.

To get benchmark images for model inference， visit https://huggingface.co/datasets/Theseus-Lab/AmbiHTMLBench.

## Objective evaluation

Prepare HTML predictions locally. Model directory names identify models, and the first four characters of each filename identify the ground-truth page.

```text
all_model_infer/
  model_a/html/0001_example.html
  model_a/html/0002_example.html
  model_b/html/0001_example.html
```

Run with the included ground-truth annotations:

```bash
python main.py eval-objective-v3 --gt-json AmbiHTMLBench_GT_v5_1115.json --infer-dir all_model_infer --out-dir outputs/objective_v3
```

Useful options:

- `--max-pages 20`: limit pages per model for a quick check.
- `--models model_a,model_b`: select models.
- `--page-ids 0001,0002`: select pages.
- `--no-visualizations`: skip visual reports.

Results include `page_scores.csv`, `leaderboard.csv`, and HTML match visualizations. Metrics cover table content and structure, text edit distance, and layout graph similarity. Partial CSV files support resuming interrupted runs; use a new output directory when changing the evaluation scope.

Model predictions under `all_model_infer/` and generated results under `outputs/` are ignored by Git and are not included in the current main branch.

## Subjective evaluation

The paper uses the frozen OLS5 protocol with the `candidate_refresh` dataset: 10,000 comparisons, five recorded judge votes per comparison, and 2,000 page-level bootstrap replicates. The tools validate and replay recorded judgments without calling model APIs.

Supply the external judgment file at the example path below, then reproduce the leaderboard:

```bash
python -m benchmark.frozen_proxy_cli replay --results artifacts/candidate_refresh/single_results.jsonl --config configs/subjective_ols5_frozen.json --artifact-manifest evidence/subjective/dataset_versions.json --dataset candidate_refresh --out-dir outputs/subjective_replay --bootstrap 2000
```

The repository includes the frozen configuration, English prompt, aggregate evidence, and published leaderboard. Raw judgments, images, and human annotations must be supplied separately. Replay requires a new output directory. Scores represent proxy preferences, not measured human accuracy.

Use `validate` to check recorded inputs and `calibrate` to reproduce calibration from saved judgments and human targets. See [the protocol guide](docs/FROZEN_PROXY.md) for commands and input requirements.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Frozen OLS5 protocol](docs/FROZEN_PROXY.md)
- [Data provenance and reproducibility](docs/SUBJECTIVE_SYNC_PROVENANCE.md)

Run `python main.py --help` or `python -m benchmark.frozen_proxy_cli --help` for command options.

## License

This repository is licensed under the [MIT License](LICENSE).
