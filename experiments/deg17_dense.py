"""
deg17_dense.py - dense degradation battery: fine levels near 0, bootstrap confidence intervals, out-of-band contamination
==========================================================================================
Scores the 17 degradations of deg17_v2.py plus 2 out-of-band contaminations at 10 levels (0, .01, .02, .05, .1, .2, .4, .6, .8, 1).
  hf_noise    : high-frequency noise from 8 Hz to 0.48 fs (outside SHAPE 0.5-8 Hz and WCR/rhythm 0.75-3 Hz)
  band3to8    : 3-8 Hz noise (inside the SHAPE band, outside the WCR/rhythm band)
Summary (trajectory = fixed fs x repetition, only the level changes; 95% CI from 2,000 trajectory-level bootstrap resamples)
  rev      : fraction of adjacent steps where the score 'improves' by more than 0.01
  drop     : score drop from level 0 to level 1 (median)
  d01/d02  : drop at levels 0.01 and 0.02 (continuity near 0)
  improve  : fraction of trajectories that score more than 0.01 above level 0 at any level (does contamination 'improve' the score?)
Outputs: deg17_dense_raw.csv, deg17_dense_report.txt
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import deg17_v2 as D
import rpfi_v2 as RV

LEVELS = np.array([0, .01, .02, .05, .1, .2, .4, .6, .8, 1.0])
COMP = ["s_SHAPE", "s_WCR", "s_RHY", "RPFI"]


def _bandnoise(n, fs, lo, hi, rng):
    w = rng.standard_normal(n); b, a = butter(4, [lo / (fs / 2), min(hi, 0.48 * fs) / (fs / 2)], btype="band")
    z = filtfilt(b, a, w); return z / (z.std() + 1e-12)


def d_hf(y, fs, lv, rng, hr): return y + 2.0 * lv * y.std() * _bandnoise(y.size, fs, 8.0, 0.48 * fs, rng)
def d_b38(y, fs, lv, rng, hr): return y + 2.0 * lv * y.std() * _bandnoise(y.size, fs, 3.0, 8.0, rng)


DEG = dict(D.DEG); DEG["hf_noise_8Hz+"] = d_hf; DEG["band_noise_3to8Hz"] = d_b38


def run():
    rows = []; t0 = time.time()
    for fs in D.FS_LIST:
        seg = int(10 * fs)
        for rep in range(D.REPS):
            yb, hr = D.base(fs, np.random.default_rng(10_000 * fs + rep)); ref = D.win(yb, fs)
            for di, (name, fn) in enumerate(DEG.items()):
                for lv in LEVELS:
                    rng = np.random.default_rng(1_000_000 + 10_000 * fs + 100 * rep + di)
                    pred = D.win(fn(yb.copy(), fs, float(lv), rng, hr), fs)
                    r = RV.score_chunks(ref.reshape(-1, seg), pred.reshape(-1, seg), fs)
                    rows.append(dict(fs=fs, rep=rep, degradation=name, level=float(lv), **{k: r[k] for k in COMP + ["lag_s"]}))
            print(f"  fs={fs} rep={rep + 1}/{D.REPS} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def traj_stats(tr):
    """Statistics of one trajectory (levels in ascending order)."""
    out = {}
    for c in COMP:
        v = tr[c].values; dv = np.diff(v)
        out[f"{c}_rev"] = float((dv > 0.01).mean()); out[f"{c}_drop"] = float(v[0] - v[-1])
        out[f"{c}_d01"] = float(v[0] - v[1]); out[f"{c}_d02"] = float(v[0] - v[2])
        out[f"{c}_improve"] = float((v[1:] > v[0] + 0.01).any())
    return out


def summarize(d, B=2000):
    res = []
    for name, g in d.groupby("degradation", sort=False):
        T = pd.DataFrame([traj_stats(tr.sort_values("level")) for _, tr in g.groupby(["fs", "rep"])])
        rng = np.random.default_rng(0); row = dict(degradation=name)
        for col in T.columns:
            agg = np.mean if col.endswith(("_rev", "_improve")) else np.median
            bs = [agg(T[col].values[rng.integers(0, len(T), len(T))]) for _ in range(B)]
            row[col] = agg(T[col].values); row[col + "_lo"], row[col + "_hi"] = np.percentile(bs, [2.5, 97.5])
        res.append(row)
    return pd.DataFrame(res)


if __name__ == "__main__":
    d = run(); d.to_csv("deg17_dense_raw.csv", index=False)
    S = summarize(d); S.to_csv("deg17_dense_summary.csv", index=False)
    pd.set_option("display.width", 260)
    L = [f"{len(DEG)} types x {len(LEVELS)} levels x {D.REPS} repetitions x {len(D.FS_LIST)} fs = {len(d)} cases, {len(d) // len(LEVELS)} trajectories"]
    for c in COMP:
        cols = ["degradation", f"{c}_drop", f"{c}_d01", f"{c}_d02", f"{c}_rev", f"{c}_rev_hi", f"{c}_improve", f"{c}_improve_hi"]
        L += ["", f"[{c}]  drop/d01/d02 = median drop at level 1/0.01/0.02, rev = reversal rate (mean, upper CI), improve = fraction of trajectories better than level 0",
              S[cols].round(3).to_string(index=False)]
    open("deg17_dense_report.txt", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
