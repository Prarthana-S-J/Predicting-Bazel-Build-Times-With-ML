# Demo procedure and viva preparation

## Demo procedure (about 3 minutes)
Have a terminal open in the repo root, and `results/` plots ready.

1. **Show the data and the problem (20 s).** `head data/original/InputData.csv data/original/CPUTimes.csv`. "Each row is one Bazel build: dominant changed-file prefix and type, and the CPU time."
2. **Run the pipeline (60 s).** `python src/data_preprocessing.py`, `python src/train_baseline.py`, `python src/train_improved.py`. Point out: "The test split is not read in these two scripts. Alpha is chosen on validation."
3. **Final evaluation (30 s).** `python src/evaluate.py`. Show the table: Model B test RMSE 5354 / MAE 3645 / R² 0.144 vs Model A 5818 / 4299 / -0.011. Open `results/actual_vs_predicted.png`.
4. **Live predictions (60 s).**
   * `python src/demo.py --list`
   * `python src/demo.py --prefix src/main --type JAVA` (about 4.4 s)
   * `python src/demo.py --prefix site/ --type JAVA` (about 0.7 s)
   * `python src/demo.py` (replay of unseen test builds, predicted vs actual)
5. **Say the limits yourself (10 s).** "The gain is modest. It comes from separating cheap builds from real rebuilds. It cannot rank expensive builds."

Fallback if something fails: `results/final_results.md` and the plots already exist in the repo.

## Viva questions and short answers

**Why predict Bazel build time?** Builds are run millions of times and waste CPU and developer time. A prediction lets developers plan or catch expensive changes early.

**What is Bazel?** An open-source build and test tool from Google. It builds only what changed (incremental builds) and supports many languages.

**What is the target variable?** The CPU time of one build in milliseconds (`cpu_time_ms`), from the Build Event Protocol.

**What features are used? Why these?** The dominant path prefix of the changed files, the dominant file type, and their cross. The paper argues that where a file lives (`src/main` vs `site/`) shows how many other files depend on it, and C/C++ vs Python compile differently. In our data, `site/` changes lead to near no-op builds (about 0.6 s) while `src/main` and `tools/` changes lead to long ones.

**Why linear regression?** It is simple, fast, and it is what the paper used, so our baseline matches the paper's approach. With 500 samples a complex model would overfit. We used Ridge (L2 regularisation) to keep the weights stable.

**What is feature crossing?** Combining two categorical features into one, e.g. `src/main|JAVA`. A linear model can then give each combination its own weight, so it can learn interactions that two separate features cannot.

**Why did you choose this improvement?** While checking the paper's code we found the features were paired with the wrong builds. The script checks out the previous commit and builds incrementally, so each build rebuilds the previous row's change set. Fixing the pairing needs no new data, uses only information available before the build, and the validation data clearly supported it. It is a correction, not a new algorithm.

**How sure are you about that cause?** It is inferred from the script and the data; I cannot re-run the original machine. The data supports it: only the previous row's features explain CPU time, and offsets of -2, 0, +1, +2 are at chance level.

**Why RMSE? Why MAE?** RMSE is the paper's metric and punishes big errors, which matter for slow builds. MAE is the average absolute error in ms and is easier to read and less dominated by outliers. We report both, in original milliseconds.

**What does R² mean?** The share of variance in CPU time the model explains compared with always predicting the average. 0 = no better than the mean, 1 = perfect, negative = worse than the mean. Model B has 0.144 on test; Model A is -0.011.

**How did you prevent data leakage?** Inputs are only prefix, type and their cross, all known before the build. We never use CPU time, duration or action counts of the same build. The test split is read only in `evaluate.py`. We also scrambled the test targets and re-ran training: the saved models were identical, so training and tuning do not depend on the test data.

**How did you split the data?** Contiguous blocks in build order: train 299, validation 100, test 100, no overlap (the original code had one overlapping row). Build 1 (cold start) is dropped. Order matters because each build depends on the previous build's state, so we do not shuffle.

**What did the original code get wrong?** (1) `test_model()` evaluates an untrained model that predicts 0 (RMSE 7,459 ms matches the paper's flat line). (2) Validation/test overlap. (3) Features paired with the wrong build. (4) Grid search not in code. (5) Substring bug in feature extraction (not fixable).

**Is your baseline an exact copy of the paper?** No. The paper used TensorFlow SGD with hashed crosses; we use scikit-learn Ridge with explicit one-hot crosses. It follows the same idea and features. We say so openly.

**Your baseline is almost a constant predictor. Is that fair?** Yes, it is a generous baseline. On validation the original pairing carries no signal, so tuning picks the strongest regularisation. With weak regularisation it does worse than the mean (see `results/validation_baseline.csv`). The paper's own RMSE is also close to a constant.

**What are the limitations?** 500 builds, one project, one machine, only 100 test builds; coarse features; cache state not modelled; gain mostly on cheap builds; cannot repair the labelling artefact; cause of misalignment inferred. Validation R² was 0.40 but test R² only 0.14, so the real gain is modest.

**Why is build time hard to predict?** It depends on the dependency graph (how many targets are affected), cache state, machine load, and the size of the change. Prefix and type capture only a little of that. CPU time is also very skewed (median 1.8 s, max 37 s).

**Why not log-transform the target?** We tried on validation only; it was worse (R² 0.09 vs 0.37 at the same alpha), so we did not use it.

**What would you do with more data?** Collect more builds from several projects, add the number of changed files and dependency-graph features (how many targets depend on the changed files), try tree-based models, and cross-validate over time.

**What is the bootstrap you used?** We resampled the 100 test builds 2000 times and recomputed the metric difference between the two models each time. The RMSE gain is 464 ms with a 95% interval of 185 to 698, which excludes zero, so the improvement is unlikely to be luck. It does not remove the limits of such a small test set.
