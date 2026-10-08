# Predicting Bazel Build Times: Reproduction and Improvement

**Reproduction and improvement of the paper's Bazel build-time prediction pipeline**
(*"Predicting Bazel Build Times Using Machine Learning"*, Nadeem & Raza, Stanford CS229; data and scripts from their GitHub repo `CS229-Final-Project`).

UE24CS352A Machine Learning mini-project.

## Problem Statement
Bazel is a build system used on very large code bases. A build can take a long time and waste CPU. If we can predict how much CPU time the next build will need, developers can plan for it. We predict **build CPU time (ms)** from simple information about the files that changed.

## Objective
1. Reproduce the paper's approach (file prefix + file type + feature cross + linear regression) cleanly and evaluate it properly.
2. Find and document problems in the original implementation that stop it from being scientifically valid.
3. Apply **one** correction, **fix the mismatch between the commit-level features and the build target**, and measure whether it helps.

This is **not** a new algorithm. Both models use the same features and the same model class; only the pairing of features to targets changes.

## Dataset
Provided by the paper's repository (no new data collected): 500 builds of the open-source `bazelbuild/bazel` project.

| File | Content |
|---|---|
| `data/original/InputData.csv` | per build: dominant changed-file path prefix, dominant changed-file type |
| `data/original/CPUTimes.csv` | per build: CPU time in ms from the Build Event Protocol |

I checked that the CPU times in `CPUTimes.csv` match the 500 raw JSON files of the original repo exactly. The 55 MB of JSONs are therefore not included here.

CPU time is very skewed: median about 1.8 s, mean about 4.1 s, maximum 37 s (train + validation builds). About 30% of builds are no-op rebuilds of roughly 0.6 s.

## Features and target
* **Inputs (known before the build):** `prefix` (e.g. `src/main`, `site/`), `type` (e.g. `JAVA`, `C/C++`), and the paper's **feature cross** `prefix|type`. All three are one-hot encoded.
* **Target:** `cpu_time_ms`.
* **Never used as input:** CPU time, build duration, number of actions, or anything else from the Build Event Protocol (these are only known after the build).

## Methodology (pipeline)
```
data/original/*.csv
   -> data_preprocessing.py   builds 'paper_pairing' and 'aligned' datasets (+ split labels)
   -> train_baseline.py       Model A: fit on train, choose alpha on validation
   -> train_improved.py       Model B: same, on the aligned data
   -> audit_original.py       evidence for the original-code issues (train+val only)
   -> evaluate.py             the ONLY script that reads the test split: metrics, bootstrap, plots
   -> demo.py                 live predictions with Model B
```
**Split** (identical for both models, contiguous blocks in build order, no overlap): train = builds 2-300 (299), validation = 301-400 (100), test = 401-500 (100). Build 1 is the cold-start build (44.9 s, nothing built before it) and is dropped from both pipelines.

**Model** (identical for both): one-hot encoding of `prefix`, `type`, `prefix|type`, then Ridge regression (L2-regularised linear regression). `alpha` is chosen on **validation RMSE** from `[0.001 ... 1000]`. The test split is never touched until `evaluate.py`.

**Metrics:** RMSE, MAE, R², all in original milliseconds.

## Original implementation: issues found
Evidence is in `results/audit_original_issues.txt` and `results/alignment_check.csv`.

| # | Issue in original repo | Effect |
|---|---|---|
| 1 | `test_model()` creates a **new, untrained** regressor and never calls `train` | It predicts 0. RMSE of predicting 0 on the test block is **7,459 ms**, which matches the flat ~7,500 line in the paper's test plots. The paper's "test results" do not evaluate a trained model. |
| 2 | Validation uses `.loc[301:400]` (inclusive) | Row 400 is in both validation and test; row 300 is never used. |
| 3 | **Features and target are misaligned.** `extractBuildInfo.py` runs `git checkout HEAD~1`, records the diff of the checked-out commit, then runs an *incremental* build. The files rebuilt are those of the **previous** row's change set. | With the original pairing, features explain no more than chance (see below). |
| 4 | Paper describes grid-search CV (lr 0.01, batch 16); not in the code. `learning_rate` is unused. | Reported hyperparameters cannot be traced to code. |
| 5 | The paper's train/validation RMSE (about 5.6k ms) is about the same as predicting the average (about 5.4-5.6k ms) | The paper's models barely beat a constant. |
| 6 | Extractor counts substrings (`.c` also matches `.css`; `.h` matches `.html`) | 20 of 60 `site/` rows are labelled C/C++, none HTML/CSS/JS. **Not fixable**: the raw git diffs are not in the repo. Kept as a limitation. |

On issue 3, the *mechanism* is inferred from reading the extraction script plus the data; I cannot re-run the original build machine. The data supports it clearly (train+val builds only):

| Features taken from | Val RMSE | Val R² | Variance explained (chance level, 95th pct) |
|---|---|---|---|
| same build (original pairing) | 5764 | -0.09 | 0.036 (0.084) |
| **previous build (corrected)** | **4389** | **0.37** | **0.280 (0.088)** |
| two builds earlier / next / two later | 5680 / 5684 / 5763 | about -0.06 | within chance |

## Models
* **Model A - clean paper baseline.** Paper features, paper pairing, linear model with feature cross. Issues 1 and 2 are fixed by proper train / validation / test handling. Only `alpha` is tuned.
* **Model B - corrected pipeline.** Identical, except each build's features are taken from the change set that was actually rebuilt (previous row).

**Not an exact reproduction.** The paper used TensorFlow's `LinearRegressor` (SGD) with hashed crossed columns. I use scikit-learn Ridge on explicit one-hot crosses. The paper's own SGD setup is not recoverable from its repo. On validation, Model A prefers the strongest regularisation (`alpha = 1000`), i.e. it behaves almost like a constant predictor. That is a *favourable* baseline: with weak regularisation it does worse than the mean (`results/validation_baseline.csv`).

## Results (test set, 100 builds, run by `evaluate.py`)

| Model | Val RMSE | Test RMSE (ms) | Test MAE (ms) | Test R² |
|---|---|---|---|---|
| Untrained model (predicts 0), what the paper's `test_model()` effectively evaluates | | 7459 | 4707 | -0.662 |
| Mean predictor (training mean) | | 5821 | 4299 | -0.012 |
| **Model A: paper baseline (original pairing)** | 5518 | 5818 | 4299 | -0.011 |
| **Model B: improved (corrected alignment)** | 4264 | **5354** | **3645** | **0.144** |

Paired bootstrap over the 100 test builds (2000 resamples, gain of B over A): RMSE **+464 ms** (95% CI 185 to 698), MAE **+657 ms** (258 to 1050), R² **+0.156** (0.066 to 0.228).

Plots: `results/actual_vs_predicted.png`, `results/model_comparison.png`, `results/target_distribution.png`.

**How to read this honestly**
* Model B beats Model A on all three metrics, and the bootstrap interval excludes zero. But the gain is **modest on the test set** (R² 0.14), much smaller than on validation (R² 0.40). Only 100 test builds, with a heavy tail, so expect a noisy estimate.
* The gain comes from **cheap builds** (< 1.5 s, 38 test builds): MAE 1312 ms (B) vs 3230 ms (A). B recognises that changes to `site/` or `src/test` lead to near no-op rebuilds. On the 62 expensive builds, B is not better (MAE 5074 vs 4953 ms). The features cannot tell *how* expensive a `src/main` or `tools/` rebuild will be.
* A log-transformed target was tried briefly during exploration (validation only) and was worse, so it was not used.

## How to Run
Python 3.9+.
```bash
git clone <your-repo-url> && cd <repo>
pip install -r requirements.txt

python src/data_preprocessing.py   # 1. build processed datasets
python src/train_baseline.py       # 2. Model A (train + validation only)
python src/train_improved.py       # 3. Model B (train + validation only)
python src/audit_original.py       # 4. evidence for original-code issues (optional)
python src/evaluate.py             # 5. final test evaluation, tables, plots

python src/demo.py                                  # replay the last 8 test builds
python src/demo.py --prefix src/main --type JAVA    # one custom prediction
python src/demo.py --list                           # valid values
```
Everything is deterministic: a clean re-run reproduced `final_results.csv` and `test_predictions.csv` byte-for-byte. It takes a few seconds.

## Integrity checks performed
* Leakage test: I scrambled the test targets and re-ran both training scripts; the saved models and chosen `alpha` were **identical**, so nothing in training or tuning depends on test data.
* Only `prefix`, `type`, `cross` enter the model (asserted in preprocessing).
* Disclosure: during my initial exploration of the data, before building the pipeline, I computed a quick variance-explained diagnostic over all 500 rows (test rows included) when I first spotted the alignment problem. All model selection, and the audit script shipped here, use train + validation only.

## Limitations
* Only 500 builds from one project on one machine (macOS), 100 test builds.
* Features are coarse (dominant prefix and type): no file counts, no dependency information. They cannot separate small from large rebuilds within `src/main` / `tools/`.
* Incremental builds depend on cache state, which is not modelled.
* The labelling artefact (issue 6) and the 3-of-11 unused prefixes cannot be repaired without the raw diffs.
* The alignment mechanism is inferred from code and data, not re-measured.

## Future work
Count of changed files, dependency-graph features (reverse-dependency count of changed targets), more builds and projects, a tree-based model, and a distribution-aware loss for the heavy tail.

## Repository layout
```
data/original/    InputData.csv, CPUTimes.csv (from the paper's repo)
data/processed/   paper_pairing.csv, aligned.csv
src/              common.py, data_preprocessing.py, train_baseline.py, train_improved.py,
                  audit_original.py, evaluate.py, demo.py
results/          metrics, validation tables, bootstrap, predictions, plots
models/           saved models (baseline.joblib, improved.joblib)
docs/             report.pdf, slides.pptx, viva_and_demo.md
```
