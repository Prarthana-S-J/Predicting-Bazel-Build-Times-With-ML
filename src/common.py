"""Shared helpers: paths, split definition, model pipeline, metrics.

Everything that must be IDENTICAL between the baseline and the improved
pipeline lives here, so the two can differ only in the data pairing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "data" / "original"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
MODELS = ROOT / "models"

# Contiguous split in BUILD ORDER (build_no = row number in the original CSV, 1-based).
# Build 1 is the cold start build (no previous build state) and is dropped from both pipelines.
# Half-open ranges on build_no.  Same sizes as the paper (~300 / 100 / 100) but with NO overlap.
SPLITS = {"train": (2, 301), "val": (301, 401), "test": (401, 501)}

ALPHAS = [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]   # ridge strengths tried on validation
SEED = 42


def split_of(build_no: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= build_no < hi:
            return name
    return "dropped"


def load_processed(name: str) -> pd.DataFrame:
    """name = 'paper_pairing' or 'aligned'."""
    return pd.read_csv(PROC / f"{name}.csv")


def make_model(alpha: float, use_cross: bool = True) -> Pipeline:
    """One-hot(prefix, type [, prefix x type cross]) -> Ridge (regularised linear regression).

    The prefix x type cross is the paper's 'feature cross' idea, written as an
    explicit one-hot of the combined category instead of TensorFlow's hashed crossed_column.
    """
    cols = ["prefix", "type"] + (["cross"] if use_cross else [])
    enc = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cols)])
    return Pipeline([("enc", enc), ("reg", Ridge(alpha=alpha))])


def metrics(y_true, y_pred) -> dict:
    """All metrics are in ORIGINAL units (milliseconds of CPU time)."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def get_split(df: pd.DataFrame, split: str):
    d = df[df["split"] == split]
    return d[["prefix", "type", "cross"]], d["cpu_time_ms"].to_numpy(dtype=float)


def train_and_validate(dataset: str, tag: str) -> None:
    """Fit on TRAIN, choose ridge alpha on VALIDATION. The test split is never read here."""
    import json
    import joblib

    df = load_processed(dataset)
    Xtr, ytr = get_split(df, "train")
    Xva, yva = get_split(df, "val")

    rows = []
    for use_cross in (False, True):
        for a in ALPHAS:
            m = make_model(a, use_cross).fit(Xtr, ytr)
            rows.append({"dataset": dataset, "features": "prefix+type+cross" if use_cross else "prefix+type",
                         "alpha": a, **{f"val_{k}": v for k, v in metrics(yva, m.predict(Xva)).items()}})
    table = pd.DataFrame(rows)
    table.to_csv(RES / f"validation_{tag}.csv", index=False)

    # Protocol (fixed in advance): the final model ALWAYS uses the paper's feature cross;
    # only alpha is tuned, on validation RMSE.
    cross_rows = table[table["features"] == "prefix+type+cross"]
    best = cross_rows.loc[cross_rows["val_RMSE"].idxmin()]
    model = make_model(float(best["alpha"]), True).fit(Xtr, ytr)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{tag}.joblib")
    cfg = {"tag": tag, "dataset": dataset, "features": "prefix+type+cross", "alpha": float(best["alpha"]),
           "val_RMSE": float(best["val_RMSE"]), "val_MAE": float(best["val_MAE"]), "val_R2": float(best["val_R2"]),
           "n_train": int(len(ytr)), "n_val": int(len(yva))}
    (RES / f"{tag}_config.json").write_text(json.dumps(cfg, indent=2))

    print(f"[{tag}] dataset={dataset}  train={len(ytr)}  val={len(yva)}")
    print(table.round(4).to_string(index=False))
    print(f"[{tag}] chosen alpha={cfg['alpha']}  VAL RMSE={cfg['val_RMSE']:.1f}  MAE={cfg['val_MAE']:.1f}  R2={cfg['val_R2']:.3f}")
    print(f"[{tag}] model saved to models/{tag}.joblib (test split NOT used)")
