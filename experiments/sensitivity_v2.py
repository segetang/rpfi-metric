"""
sensitivity_v2.py - RPFI v2 lag handling and sensitivity analyses in one pass
==================================================================================
For each of the 9 conditions x models x participants, computes the following and saves them to sensitivity_v2/sensitivity_ALL.csv.

  [lag / synchronization]
    lag_s          : constant lag estimated by the rhythm component (s, positive = prediction lags)
    WCR_raw        : current definition as is (chunks concatenated)
    WCR_raw_trim   : only the same edge trimming as below (reference for comparison)
    WCR_aligned    : computed after shifting the prediction earlier by lag_s (|lag| trimmed from both ends of each chunk)
  [beat-matching tolerance]   RHY_tol10, RHY_tol20 (default 0.15 RR), F1_tol10, F1_tol20
  [chunk vs. participant unit] WCR_chunkmean, RHY_chunkmean (computed per chunk, then averaged)
  [detrending]                WCR_noDT, RHY_noDT (band-pass only, no Tarvainen)

Usage (from the working folder):  python sensitivity_v2.py      (use --label-dir if the labels are elsewhere)
Output: sensitivity_v2/sensitivity_ALL.csv
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
from rhythm import rhythm_participant
from recompute_newsys import load_chunks

LABEL_NAME = {"pure": "PURE_label.npy", "ubfc": "UBFC_label.npy", "cohface": "COHFACE_label.npy"}
DS = ["pure", "ubfc", "cohface"]


def conditions():
    for t in DS:
        yield t, t, f"runs/{t}_cv5"
    for s in DS:
        for t in DS:
            if s != t:
                yield f"{s}_to_{t}", t, f"runs/cross_{s}_to_{t}"


def shift_trim(L, P, fs, d):
    """Shift the prediction earlier by d s (P(t+d)), trim |d| from both ends of each chunk, and concatenate. d = 0: trimming only."""
    m = int(np.ceil(abs(d) * fs)) + 1
    t = np.arange(L.shape[1]) / fs
    ys, qs = [], []
    for l, p in zip(L, P):
        q = np.interp(t + d, t, p)
        if 2 * m < len(t) - 10:
            ys.append(l[m:-m]); qs.append(q[m:-m])
    if not ys:
        return np.nan
    return R.compute_wcr(np.concatenate(ys), np.concatenate(qs))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="sensitivity_v2")
    a = ap.parse_args()
    miss = [p for _, t, r in conditions() for p in (Path(r), Path(a.label_dir) / LABEL_NAME[t]) if not p.exists()]
    if miss:
        sys.exit("Paths not found: " + ", ".join(sorted(set(map(str, miss)))))
    out = Path(a.out); out.mkdir(exist_ok=True)
    frames = []
    for tag, ds, runs in conditions():
        fs = R.FS_BY_DATASET[ds]; seg_len = 10 * fs
        lab = np.load(Path(a.label_dir) / LABEL_NAME[ds])
        rows = []
        for m in R.discover_models(runs):
            t0 = time.time()
            data = load_chunks(runs, lab, m, fs, seg_len, 100)
            for subj, (Lr, Pr, L, P) in sorted(data.items()):
                rh = rhythm_participant(L, P, fs)
                d = float(rh["offset_s"])
                r10 = rhythm_participant(L, P, fs, tol_frac=0.10)
                r20 = rhythm_participant(L, P, fs, tol_frac=0.20)
                Ln = np.array([R.preprocess_chunk(c, fs, False, True) for c in Lr])
                Pn = np.array([R.preprocess_chunk(c, fs, False, True) for c in Pr])
                rn = rhythm_participant(Ln, Pn, fs)
                rows.append(dict(
                    condition=tag, dataset=ds, model=m, participant=subj, n_chunks=len(L),
                    lag_s=d, rr_med_s=rh["rr_med_s"], RHY=rh["RHY"],
                    WCR_raw=R.compute_wcr(L.ravel(), P.ravel())[0],
                    WCR_raw_trim=_trim_only(L, P, fs, d),
                    WCR_aligned=shift_trim(L, P, fs, d),
                    RHY_tol10=r10["RHY"], RHY_tol20=r20["RHY"], F1_tol10=r10["F1"], F1_tol20=r20["F1"],
                    WCR_chunkmean=float(np.mean([R.compute_wcr(L[c], P[c])[0] for c in range(len(L))])),
                    RHY_chunkmean=float(np.mean([rhythm_participant(L[c:c + 1], P[c:c + 1], fs)["RHY"]
                                                 for c in range(len(L))])),
                    WCR_noDT=R.compute_wcr(Ln.ravel(), Pn.ravel())[0], RHY_noDT=rn["RHY"]))
            print(f"[{tag}] {m:14s} participants={len(data):3d} ({time.time() - t0:.1f}s)", flush=True)
        df = pd.DataFrame(rows); df.to_csv(out / f"sensitivity_{tag}.csv", index=False); frames.append(df)
    allf = out / "sensitivity_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    print(f"\nDone: {allf}")


def _trim_only(L, P, fs, d):
    """WCR with the same trimming as WCR_aligned but no lag correction (isolates the effect of trimming)."""
    m = int(np.ceil(abs(d) * fs)) + 1
    ys, qs = [], []
    for l, p in zip(L, P):
        if 2 * m < len(l) - 10:
            ys.append(l[m:-m]); qs.append(p[m:-m])
    return R.compute_wcr(np.concatenate(ys), np.concatenate(qs))[0] if ys else np.nan


if __name__ == "__main__":
    main()
