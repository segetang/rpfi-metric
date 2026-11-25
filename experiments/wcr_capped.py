"""
wcr_capped.py - recompute the lag-aligned WCR with the alignment capped at a quarter cycle
==========================================================================================
In the sensitivity_v2 results, units with |lag| > 0.35 RR (10%) had WCR -0.49 (anti-correlated) before alignment
and +0.52 after. A half-cycle shift cannot be distinguished from a polarity inversion, so uncapped alignment makes
phase-inverted predictions look good. The alignment is therefore capped at +/- 0.25 x (reference median RR):
d_cap = clip(lag_s, -0.25 RR, +0.25 RR).
lag_s and rr_med_s are taken from sensitivity_v2/sensitivity_ALL.csv as they are (no rhythm recomputation).

Usage (from the working folder):  python wcr_capped.py
Output: sensitivity_v2/wcr_cap_ALL.csv
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
from recompute_newsys import load_chunks
from sensitivity_v2 import LABEL_NAME, conditions, shift_trim

CAP_FRAC = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--sens", default="sensitivity_v2/sensitivity_ALL.csv")
    a = ap.parse_args()
    if not Path(a.sens).exists():
        sys.exit(f"not found: {a.sens}")
    S = pd.read_csv(a.sens).set_index(["condition", "model", "participant"])
    rows = []
    for tag, ds, runs in conditions():
        fs = R.FS_BY_DATASET[ds]; seg_len = 10 * fs
        lab = np.load(Path(a.label_dir) / LABEL_NAME[ds])
        for m in R.discover_models(runs):
            t0 = time.time()
            for subj, (Lr, Pr, L, P) in sorted(load_chunks(runs, lab, m, fs, seg_len, 100).items()):
                r = S.loc[(tag, m, subj)]
                cap = CAP_FRAC * float(r.rr_med_s)
                d = float(np.clip(r.lag_s, -cap, cap))
                rows.append(dict(condition=tag, dataset=ds, model=m, participant=subj, lag_s=float(r.lag_s),
                                 lag_used_s=d, capped=abs(r.lag_s) > cap, WCR_aligned_cap=shift_trim(L, P, fs, d)))
            print(f"[{tag}] {m:14s} ({time.time() - t0:.1f}s)", flush=True)
    out = Path("sensitivity_v2/wcr_cap_ALL.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
