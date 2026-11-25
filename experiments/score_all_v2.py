"""
score_all_v2.py - re-score all 9 conditions from scratch with the release scorer rpfi_v2.py (final result file)
==========================================================================================
Purpose
  1) reproduce the paper's final numbers (Table 1 etc.) with a single release code path
  2) compute the polarity flag (polarity_suspect) on all real data
Agreement with the values produced by the analysis scripts (newsys / sensitivity / wcr_capped / recompute_hr)
was checked separately (max |difference| <= 3.6e-15).

Usage (from the working folder):  python score_all_v2.py      (use --label-dir if the labels are elsewhere)
Output: rpfi_v2_scores/rpfi_v2_ALL.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import rpfi_eval as R
import rpfi_v2 as RV
from recompute_newsys import load_chunks
from sensitivity_v2 import LABEL_NAME, conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="rpfi_v2_scores")
    a = ap.parse_args()
    miss = [p for _, t, r in conditions() for p in (Path(r), Path(a.label_dir) / LABEL_NAME[t]) if not p.exists()]
    if miss:
        sys.exit("Paths not found: " + ", ".join(sorted(set(map(str, miss)))))
    out = Path(a.out); out.mkdir(exist_ok=True); frames = []
    print(f"rpfi_v2 {RV.__version__}")
    for tag, ds, runs in conditions():
        fs = R.FS_BY_DATASET[ds]; lab = np.load(Path(a.label_dir) / LABEL_NAME[ds]); rows = []
        for m in R.discover_models(runs):
            t0 = time.time()
            for subj, (Lr, Pr, L, P) in sorted(load_chunks(runs, lab, m, fs, 10 * fs, 100).items()):
                r = RV.score_chunks(list(Lr), list(Pr), fs)
                rows.append(dict(condition=tag, dataset=ds, model=m, participant=subj, **r))
            print(f"[{tag}] {m:14s} ({time.time() - t0:.1f}s)", flush=True)
        df = pd.DataFrame(rows); df.to_csv(out / f"rpfi_v2_{tag}.csv", index=False); frames.append(df)
    allf = out / "rpfi_v2_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    print(f"\nDone: {allf}")


if __name__ == "__main__":
    main()
