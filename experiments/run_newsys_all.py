"""
run_newsys_all.py - run recompute_newsys.py for all 9 conditions (3 within-dataset + 6 cross-dataset) and merge the results
==================================================================================
Run from the working folder (the one containing runs/ and the v1 per-participant CSVs):
    python run_newsys_all.py
If the labels are elsewhere:  python run_newsys_all.py --label-dir "../real data"
Output: newsys_recompute/newsys_participant_ALL.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

LABEL_NAME = {"pure": "PURE_label.npy", "ubfc": "UBFC_label.npy", "cohface": "COHFACE_label.npy"}
DS = ["pure", "ubfc", "cohface"]


def conditions():
    for t in DS:                                   # within-dataset
        yield t, t, f"runs/{t}_cv5", f"results_eval_noSRE/rpfi_participant_{t}_geometric.csv"
    for s in DS:                                   # cross-dataset
        for t in DS:
            if s != t:
                yield (f"{s}_to_{t}", t, f"runs/cross_{s}_to_{t}",
                       f"{s}_to_{t}/rpfi_participant_{t}_geometric.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="newsys_recompute")
    args = ap.parse_args()

    missing = [p for _, t, r, c in conditions()
               for p in (Path(r), Path(c), Path(args.label_dir) / LABEL_NAME[t]) if not p.exists()]
    if missing:
        sys.exit("Paths not found:\n  " + "\n  ".join(sorted(set(map(str, missing)))))

    for tag, tgt, runs, csv in conditions():
        print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "recompute_newsys.py"), "--dataset", tgt, "--runs", runs,
                        "--label", str(Path(args.label_dir) / LABEL_NAME[tgt]),
                        "--rpfi-csv", csv, "--tag", tag, "--out", args.out], check=True)

    frames = [pd.read_csv(Path(args.out) / f"newsys_participant_{tag}.csv") for tag, *_ in conditions()]
    allf = Path(args.out) / "newsys_participant_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    print(f"\nDone: {allf}")


if __name__ == "__main__":
    main()
