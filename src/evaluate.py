"""Step 5: FINAL evaluation. This is the ONLY script that reads the test split.

Models were already fitted on TRAIN and tuned on VALIDATION by train_baseline.py / train_improved.py.
Reports RMSE, MAE, R2 in original CPU-time units (milliseconds) on the test builds, plus a paired
bootstrap over test builds to show how much the small sample (100 builds) limits the conclusion.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MODELS, RES, get_split, load_processed, metrics

paper = load_processed("paper_pairing")
aligned = load_processed("aligned")
Xtr_p, ytr = get_split(paper, "train")
Xte_p, yte = get_split(paper, "test")
Xte_a, yte_a = get_split(aligned, "test")
assert np.allclose(yte, yte_a), "both pipelines must be evaluated on the SAME test builds"

base = joblib.load(MODELS / "baseline.joblib")
impr = joblib.load(MODELS / "improved.joblib")
cb = json.loads((RES / "baseline_config.json").read_text())
ci = json.loads((RES / "improved_config.json").read_text())

pred = {
    "Untrained model (predicts 0) - what the paper's test_model() effectively evaluates": np.zeros_like(yte),
    "Mean predictor (training mean)": np.full_like(yte, ytr.mean()),
    "Model A: Paper baseline (original pairing)": base.predict(Xte_p),
    "Model B: Improved (corrected alignment)": impr.predict(Xte_a),
}
val = {"Model A: Paper baseline (original pairing)": cb, "Model B: Improved (corrected alignment)": ci}

rows = []
for name, p in pred.items():
    m = metrics(yte, p)
    v = val.get(name)
    rows.append({"Model": name, "Val RMSE": v["val_RMSE"] if v else np.nan, "Test RMSE": m["RMSE"],
                 "Test MAE": m["MAE"], "Test R2": m["R2"]})
res = pd.DataFrame(rows)
res.to_csv(RES / "final_results.csv", index=False)

# ---- paired bootstrap over the 100 test builds ------------------------------------------------
rng = np.random.default_rng(42)
pa, pb = pred["Model A: Paper baseline (original pairing)"], pred["Model B: Improved (corrected alignment)"]
diffs = {"RMSE": [], "MAE": [], "R2": []}
for _ in range(2000):
    i = rng.integers(0, len(yte), len(yte))
    ma, mb = metrics(yte[i], pa[i]), metrics(yte[i], pb[i])
    diffs["RMSE"].append(ma["RMSE"] - mb["RMSE"])     # positive = improved is better
    diffs["MAE"].append(ma["MAE"] - mb["MAE"])
    diffs["R2"].append(mb["R2"] - ma["R2"])           # positive = improved is better
boot = pd.DataFrame({k: {"mean_gain": np.mean(v), "ci95_low": np.percentile(v, 2.5), "ci95_high": np.percentile(v, 97.5),
                         "share_resamples_improved_better": float(np.mean(np.array(v) > 0))} for k, v in diffs.items()}).T
boot.to_csv(RES / "bootstrap_improvement.csv")

# ---- save predictions + markdown table ---------------------------------------------------------
out = paper[paper["split"] == "test"][["build_no"]].copy()
out["actual_ms"] = yte
out["baseline_pred_ms"] = pa
out["improved_pred_ms"] = pb
out["improved_features_from_build"] = aligned[aligned["split"] == "test"]["feature_source_build"].to_numpy()
out["improved_prefix"] = aligned[aligned["split"] == "test"]["prefix"].to_numpy()
out["improved_type"] = aligned[aligned["split"] == "test"]["type"].to_numpy()
out.to_csv(RES / "test_predictions.csv", index=False)

md = ["| Model | Val RMSE | Test RMSE (ms) | Test MAE (ms) | Test R² |", "|---|---|---|---|---|"]
for _, r in res.iterrows():
    v = "" if np.isnan(r["Val RMSE"]) else f"{r['Val RMSE']:.0f}"
    md.append(f"| {r['Model']} | {v} | {r['Test RMSE']:.0f} | {r['Test MAE']:.0f} | {r['Test R2']:.3f} |")
(RES / "final_results.md").write_text("\n".join(md))
print("\n".join(md))
print("\nPaired bootstrap over test builds (positive = Model B better), 2000 resamples:")
print(boot.round(3).to_string())

# ---- plots -------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10})
# 1. target distribution (train+val builds only)
tv = paper[paper["split"] != "test"]["cpu_time_ms"]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(tv, bins=np.logspace(np.log10(tv.min()), np.log10(tv.max()), 30), color="#4C78A8")
ax.set_xscale("log"); ax.set_xlabel("Build CPU time (ms, log scale)"); ax.set_ylabel("Number of builds")
ax.set_title("Target distribution (train + validation builds)")
ax.axvline(tv.median(), color="k", ls="--", lw=1); ax.text(tv.median() * 1.05, ax.get_ylim()[1] * 0.9, f"median {tv.median():.0f} ms", fontsize=9)
fig.tight_layout(); fig.savefig(RES / "target_distribution.png", dpi=160); plt.close(fig)

# 2. actual vs predicted (test)
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for ax, (t, p, c) in zip(axs, [("Model A: paper baseline", pa, "#E45756"), ("Model B: corrected alignment", pb, "#54A24B")]):
    m = metrics(yte, p)
    ax.scatter(yte, np.clip(p, 100, None), s=16, alpha=0.7, color=c)
    ax.plot([300, 40000], [300, 40000], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{t}\nRMSE {m['RMSE']:.0f} ms, R² {m['R2']:.2f}", fontsize=10)
    ax.set_xlabel("Actual CPU time (ms)")
axs[0].set_ylabel("Predicted CPU time (ms)")
fig.suptitle("Test builds: actual vs predicted (log axes; dashed = perfect)", fontsize=10)
fig.tight_layout(); fig.savefig(RES / "actual_vs_predicted.png", dpi=160); plt.close(fig)

# 3. model comparison
sel = res.iloc[1:]
labels = ["Mean\npredictor", "Model A\nbaseline", "Model B\nimproved"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, col in zip(axs, ["Test RMSE", "Test MAE"]):
    b = ax.bar(labels, sel[col], color=["#9D9D9D", "#E45756", "#54A24B"])
    ax.set_title(col + " (ms, lower is better)"); ax.bar_label(b, fmt="%.0f", fontsize=9)
fig.tight_layout(); fig.savefig(RES / "model_comparison.png", dpi=160); plt.close(fig)
print("\nSaved results/final_results.{csv,md}, bootstrap_improvement.csv, test_predictions.csv and 3 plots.")
