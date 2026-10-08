"""Demo: predict the CPU time of the NEXT incremental Bazel build with the improved model (Model B).

Input = the change set that is about to be (re)built: its dominant path prefix and dominant file type.

  python src/demo.py                                  # replay the last 8 test builds: predicted vs actual
  python src/demo.py --prefix src/main --type JAVA    # one custom prediction
  python src/demo.py --list                           # valid prefix / type values
"""
import argparse

import joblib
import pandas as pd

from common import MODELS, RES, load_processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix"); ap.add_argument("--type")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n", type=int, default=8, help="number of test builds to replay")
    a = ap.parse_args()

    df = load_processed("aligned")
    model = joblib.load(MODELS / "improved.joblib")
    prefixes, types = sorted(df["prefix"].unique()), sorted(df["type"].unique())

    if a.list:
        print("prefixes:", prefixes); print("types   :", types); return
    if a.prefix and a.type:
        row = pd.DataFrame({"prefix": [a.prefix], "type": [a.type], "cross": [f"{a.prefix}|{a.type}"]})
        seen = (df["cross"] == row["cross"][0]).any()
        print(f"Change set: prefix={a.prefix}, type={a.type}")
        print(f"Predicted CPU time of the build: {model.predict(row)[0]:.0f} ms" + ("" if seen else "   (unseen prefix/type combination: weaker estimate)"))
        return

    t = pd.read_csv(RES / "test_predictions.csv").tail(a.n)
    print("Replay of test builds (never used for training or tuning):\n")
    print(f"{'build':>5} {'change set (prefix | type)':<28} {'actual ms':>10} {'predicted ms':>13}")
    for _, r in t.iterrows():
        print(f"{int(r.build_no):>5} {r.improved_prefix + ' | ' + r.improved_type:<28} {r.actual_ms:>10.0f} {r.improved_pred_ms:>13.0f}")


if __name__ == "__main__":
    main()
