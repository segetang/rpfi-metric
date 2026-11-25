"""
exp1_refeature.py - add the lag-aligned WCR and the lag to the Experiment 1 cases (secondary re-analysis)
==================================================================================
The frozen cases (distortion type and strength) are left unchanged; the predictions are regenerated identically
from the same base segments and random numbers, and the following are computed:
  lag_s        : constant-lag estimate of the rhythm component (s)
  WCR_aligned  : WCR after aligning the prediction by lag_s (same as sensitivity_v2.shift_trim)
  WCR_check    : the original WCR recomputed from the regenerated prediction - must match the WCR in the CSV (reproduction check)
No classification or verdict is made here.

Usage (from the working folder):  python exp1_refeature.py     (use --label-dir if the labels are elsewhere)
Output: exp1_refeature/exp1_v2feat_{ubfc,pure,cohface,pure_holdout,cohface_holdout}.csv
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
import exp1_distortion_id as E
import rpfi_eval as R
from rhythm import rhythm_participant
from sensitivity_v2 import shift_trim

DS = {"ubfc": ("UBFC_label.npy", "mapping_UBFC.csv", 30),
      "pure": ("PURE_label.npy", "mapping_PURE.csv", 30),
      "cohface": ("COHFACE_label.npy", "mapping_COHFACE.csv", 20)}
SEED, N_BASE, N_CHUNKS = 2026, 40, 4


def run(ds, split, label_dir, out, n_base=N_BASE):
    lab_f, mp, fs = DS[ds]
    csv = Path(f"exp1_cases_{ds}.csv" if split == "explore" else f"exp1_cases_{ds}_holdout.csv")
    if not csv.exists():
        print(f"  {csv} not found - skipped"); return
    cases = pd.read_csv(csv)
    E.N_CHUNKS = N_CHUNKS
    dur = 10.0 * N_CHUNKS; dur_total = max(dur * 1.3 + 1.0, 0.0)
    bases = E.real_bases(str(Path(label_dir) / lab_f), mp, fs, dur_total, n_base, np.random.default_rng(SEED), split)
    tag = ds if split == "explore" else f"{ds}_holdout"
    rows, t0 = [], time.time()
    for bi, (rec, subj, yfull) in enumerate(bases):
        bid = f"{tag}_{bi}" if split == "explore" else f"{tag}_r{rec}"
        sub = cases[cases.base == bid]
        if sub.empty:
            continue
        n = int(dur * fs)
        ctx = E.make_ctx(yfull, fs, np.random.default_rng(SEED + bi))
        ref = E.Ref(yfull[:n], fs)
        for _, c in sub.iterrows():
            fn, _ = E.DIST[c.distortion]
            pred = fn(yfull, fs, float(c.strength), ctx)[:n]
            L, P = ref.L, ref.prep(pred)
            d = float(rhythm_participant(L, P, fs)["offset_s"])
            rows.append(dict(base=bid, distortion=c.distortion, target=c.target, lag_s=d,
                             WCR_aligned=shift_trim(L, P, fs, d), WCR_check=R.compute_wcr(L.ravel(), P.ravel())[0],
                             WCR_csv=c.WCR))
        if (bi + 1) % 10 == 0:
            print(f"  [{tag}] base {bi + 1}/{len(bases)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    dev = (df.WCR_check - df.WCR_csv).abs().max() if len(df) else np.nan
    f = out / f"exp1_v2feat_{tag}.csv"; df.to_csv(f, index=False)
    print(f"  [{tag}] cases {len(df)}/{len(cases)}  reproduction check max|dWCR| = {dev:.2e}  -> {f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="exp1_refeature")
    ap.add_argument("--n-base", type=int, default=N_BASE, help="for testing; the real run uses the default 40 (same as exploration)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    for ds in DS:
        run(ds, "explore", a.label_dir, out, a.n_base)
        if ds != "ubfc":
            run(ds, "holdout", a.label_dir, out, a.n_base)
    print("\nDone: 5 CSVs in exp1_refeature/ (reproduction-check values above 1e-6 indicate a problem)")


if __name__ == "__main__":
    main()
