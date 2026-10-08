"""Step 1: build the two processed datasets from the ORIGINAL CSVs (data/original).

paper_pairing : row i features  ->  row i CPU time        (exactly how the original repo pairs them)
aligned       : row i-1 features -> row i CPU time        (corrected pairing, see README)

Why 'aligned' is the right pairing
----------------------------------
The original extraction script (extractBuildInfo.py) loops: `git checkout HEAD~1`, record
`git diff HEAD~1..HEAD` (the change introduced by the commit just checked out), then run an
INCREMENTAL `bazel build`. That build starts from the previous iteration's source state, so the
files Bazel actually has to rebuild are the ones in the change set recorded in the PREVIOUS row,
not the current one. Same inputs/targets, same data: only the pairing changes.
Both datasets use the same build numbers and the same train/val/test split.
"""
import pandas as pd

from common import ORIG, PROC, split_of


def main():
    X = pd.read_csv(ORIG / "InputData.csv")
    y = pd.read_csv(ORIG / "CPUTimes.csv")
    assert len(X) == len(y) == 500, "expected 500 builds"
    assert (y["commitID"].to_numpy() == range(1, 501)).all(), "CPUTimes rows must be in build order 1..500"
    assert X.isna().sum().sum() == 0 and y.isna().sum().sum() == 0, "missing values"

    X["build_no"] = range(1, 501)
    base = pd.DataFrame({"build_no": X["build_no"], "prefix": X["prefix"], "type": X["type"],
                         "cpu_time_ms": y["CPUTime"].astype(float)})

    # 1) paper pairing: features of build i with CPU time of build i
    paper = base.copy()
    paper["feature_source_build"] = paper["build_no"]

    # 2) aligned pairing: features of build i-1 with CPU time of build i
    aligned = base[["build_no", "cpu_time_ms"]].copy()
    aligned["prefix"] = base["prefix"].shift(1)
    aligned["type"] = base["type"].shift(1)
    aligned["feature_source_build"] = aligned["build_no"] - 1

    for name, d in (("paper_pairing", paper), ("aligned", aligned)):
        d["split"] = d["build_no"].map(split_of)
        d = d[d["split"] != "dropped"].copy()          # drops build 1 (cold start, nothing before it)
        d["cross"] = d["prefix"] + "|" + d["type"]     # prefix x type feature cross (paper idea)
        d = d[["build_no", "feature_source_build", "prefix", "type", "cross", "cpu_time_ms", "split"]]
        d.to_csv(PROC / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} rows ->", d["split"].value_counts().to_dict())

    # leakage guard: CPU time must never appear among the inputs
    assert "cpu_time_ms" not in ["prefix", "type", "cross"]
    print("OK. Inputs = prefix, type, cross (all known BEFORE the build). Target = cpu_time_ms.")


if __name__ == "__main__":
    main()
