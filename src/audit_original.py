"""Step 4 (diagnostics): evidence for the issues found in the ORIGINAL repository.

Uses ONLY train + validation builds (build 2..400). The test builds are not touched.
Writes results/audit_original_issues.txt and results/alignment_check.csv.

Checks
  1. Split bug  : pandas .loc[301:400] is inclusive -> row 400 is in validation AND test, row 300 is unused.
  2. Alignment  : for each offset k, pair target of build i with features of build i+k.
                  k=0 is the paper/original pairing, k=-1 is 'previous row' (our corrected pairing).
                  Reported as out-of-sample validation RMSE/R2 and as in-sample variance explained (eta^2)
                  against a shuffled-label chance level.
  3. Feature labelling: the original extractor counts substrings, so '.c' also matches '.css', '.h' matches '.html'.
                  Visible as 'site/' rows labelled C/C++ (the raw git diffs are not in the repo, so this cannot be re-computed).
"""
import numpy as np
import pandas as pd

from common import ORIG, RES, SPLITS, make_model, metrics

lines = []


def say(s=""):
    print(s)
    lines.append(s)


X = pd.read_csv(ORIG / "InputData.csv")
y = pd.read_csv(ORIG / "CPUTimes.csv")["CPUTime"].to_numpy(float)
X["cross"] = X["prefix"] + "|" + X["type"]

# ---- 1. split bug in buildTimePredictor.py -------------------------------------------------
tr_rows = set(range(0, 300))               # head(300)
va_rows = set(range(301, 401))             # .loc[301:400]  (label based, INCLUSIVE)
te_rows = set(range(400, 500))             # tail(100)
say("1) Original split (0-based rows)")
say(f"   validation/test overlap : {sorted(va_rows & te_rows)}")
say(f"   rows never used         : {sorted(set(range(500)) - tr_rows - va_rows - te_rows)}")
say("   -> our split uses non-overlapping contiguous blocks in build order.\n")

# ---- 2. alignment evidence (train+val builds only) -----------------------------------------
train_idx = np.arange(SPLITS["train"][0] - 1, SPLITS["train"][1] - 1)   # 0-based rows of builds 2..300
val_idx = np.arange(SPLITS["val"][0] - 1, SPLITS["val"][1] - 1)         # builds 301..400
last_ok = SPLITS["val"][1] - 2                                           # last 0-based row we may use (build 400)
cols = ["prefix", "type", "cross"]


def paired(idx, k):
    src = idx + k
    keep = (src >= 0) & (src <= last_ok)           # feature source must lie in train/val region
    return idx[keep], X.iloc[src[keep]][cols].reset_index(drop=True)


say("2) Which row's features explain build i's CPU time?  (train+val builds only)")
rows = []
rng = np.random.default_rng(42)
for k in (-2, -1, 0, 1, 2):
    ti, Ft = paired(train_idx, k)
    vi, Fv = paired(val_idx, k)
    m = make_model(1.0, True).fit(Ft, y[ti])
    mv = metrics(y[vi], m.predict(Fv))
    # in-sample variance explained by group means over train+val rows, vs shuffled-label chance
    ai, Fa = paired(np.concatenate([train_idx, val_idx]), k)
    g = Fa["cross"].to_numpy()
    ya = y[ai]
    def eta2(gg):
        gm = pd.Series(ya).groupby(gg).transform("mean").to_numpy()
        return 1 - ((ya - gm) ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
    e = eta2(g)
    chance95 = np.percentile([eta2(rng.permutation(g)) for _ in range(300)], 95)
    rows.append({"feature_row_offset": k, "meaning": {-2: "two builds earlier", -1: "previous build (OUR alignment)",
                 0: "same build (paper/original)", 1: "next build", 2: "two builds later"}[k],
                 "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"], "val_R2": mv["R2"],
                 "eta2_in_sample": e, "eta2_chance_95pct": chance95})
al = pd.DataFrame(rows)
al.to_csv(RES / "alignment_check.csv", index=False)
say(al.round(3).to_string(index=False))
mu = y[train_idx].mean()
say(f"   reference: predicting the training mean gives val RMSE {np.sqrt(((y[val_idx]-mu)**2).mean()):.0f}\n")

# ---- 3. labelling artefact -----------------------------------------------------------------
say("3) 'site/' rows by labelled file type (features only):")
say(pd.crosstab(X["prefix"], X["type"]).loc[["site/"]].to_string())
say("   'site/' holds web pages (.html/.css/.js), yet many rows are labelled C/C++ and none HTML/CSS/JS:")
say("   substring matching in extractBuildInfo.py (e.g. '.c' inside '.css'). Documented as a limitation.\n")
say("   Also: only 8 of the paper's 11 prefixes and 4 of its 5 types ever occur in the data.")
(RES / "audit_original_issues.txt").write_text("\n".join(lines))
