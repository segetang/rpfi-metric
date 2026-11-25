"""
exp2c_extract.py - Experiment 2c: instantaneous heart-rate tracking error (pyPPG beat fiducials)
==================================================================================
Run in the dedicated pyPPG environment.

Each recording: pyPPG systolic peaks (same peak_times as exp2b) -> IBI cleaning (same rule as exp2b: 0.33-2.0 s, median +/- 20%)
  -> beat-wise instantaneous HR_k = 60 / IBI_k (placed at the end time of that interval)
  -> linear interpolation onto a 4 Hz grid; grid points more than 2 s from a retained beat are left empty
  -> on grid points where both reference and prediction are defined, the mean absolute difference (bpm) of the
     mean-centred curves = HR 'variation' tracking error (mean-centring separates it from the constant HR error, HR MAE)
  A recording is missing if the common span is shorter than 20 s. Participant value = common-span-weighted mean over recordings.

Usage (from the working folder):  ppgenv\\Scripts\\python exp2c_extract.py
Output: exp2c_ihr/exp2c_ihr_ALL.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _a, _b in (("NaN", np.nan), ("Inf", np.inf)):
    if not hasattr(np, _a):
        setattr(np, _a, _b)

from exp2_extract import CONDITIONS, FS, LABEL_NAME
from exp2b_extract import DEV_MAX, IBI_MAX, IBI_MIN, peak_times

GRID_HZ, GAP_S, MIN_COMMON_S = 4.0, 2.0, 20.0


def ihr_curve(t, dur):
    """Peak times -> (grid times, instantaneous HR; missing = nan)."""
    g = np.arange(0, dur, 1 / GRID_HZ)
    h = np.full(g.size, np.nan)
    if t.size < 3:
        return g, h
    ibi = np.diff(t); te = t[1:]
    ok = (ibi >= IBI_MIN) & (ibi <= IBI_MAX)
    if ok.sum() >= 3:
        med = np.median(ibi[ok]); ok &= np.abs(ibi - med) <= DEV_MAX * med
    if ok.sum() < 3:
        return g, h
    tb, hb = te[ok], 60.0 / ibi[ok]
    inside = (g >= tb[0]) & (g <= tb[-1])
    h[inside] = np.interp(g[inside], tb, hb)
    near = np.min(np.abs(g[:, None] - tb[None, :]), axis=1) <= GAP_S
    h[~near] = np.nan
    return g, h


def _task(args):
    key, y, fs = args
    return key, peak_times(y, fs), len(y) / fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="exp2c_ihr")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    import rpfi_eval as R
    out = Path(a.out); out.mkdir(exist_ok=True)
    conds = [c for c in CONDITIONS if not a.only or c[0] in a.only.split(",")]
    miss = [p for c in conds for p in (Path(c[2]), Path(a.label_dir) / LABEL_NAME[c[1]]) if not p.exists()]
    if miss:
        sys.exit("Paths not found: " + ", ".join(sorted(set(map(str, miss)))))
    labels = {ds: np.load(Path(a.label_dir) / LABEL_NAME[ds]).astype(float).ravel() for ds in {c[1] for c in conds}}
    ref_cache, frames = {}, []
    for tag, ds, runs, primary in conds:
        fs = FS[ds]; lab = labels[ds]; jobs, meta = [], []
        for m in R.discover_models(runs):
            for inf_p in sorted(Path(runs).glob(f"{m}_f*_info.json")):
                info = json.load(open(inf_p))
                preds = np.load(Path(runs) / f"{m}_f{info['fold']}_test_pred.npy", allow_pickle=True)
                for i, ((s, e), subj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
                    p = np.asarray(preds[i], float).ravel(); n = min(len(p), e - s)
                    if n < 10 * fs:
                        continue
                    meta.append((m, subj, s, e, n)); jobs.append((("pred", m, subj, s), p[:n], fs))
                    if (ds, s, e) not in ref_cache:
                        ref_cache[(ds, s, e)] = None; jobs.append((("ref", ds, s, e), lab[s:e], fs))
        t0 = time.time(); res = {}
        print(f"\n[{tag}] recording jobs {len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, (key, tp, dur) in enumerate(ex.map(_task, jobs, chunksize=4)):
                res[key] = tp
                if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                    print(f"  pyPPG {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        for key, v in res.items():
            if key[0] == "ref":
                ref_cache[key[1:]] = v
        per = defaultdict(list)
        for m, subj, s, e, n in meta:
            dur = n / fs
            _, hr = ihr_curve(ref_cache[(ds, s, e)], dur); _, hp = ihr_curve(res[("pred", m, subj, s)], dur)
            both = np.isfinite(hr) & np.isfinite(hp)
            cov = both.sum() / GRID_HZ
            if cov < MIN_COMMON_S:
                per[(m, subj)].append((np.nan, 0.0, np.nan)); continue
            a_, b_ = hr[both], hp[both]
            ite = float(np.mean(np.abs((b_ - b_.mean()) - (a_ - a_.mean()))))
            per[(m, subj)].append((ite, cov, float(np.std(a_))))
        rows = []
        for (m, subj), v in sorted(per.items()):
            w = np.array([c for _, c, _ in v]); e_ = np.array([x for x, _, _ in v])
            ok = np.isfinite(e_) & (w > 0)
            rows.append(dict(condition=tag, dataset=ds, primary=primary, model=m, participant=subj,
                             ITE_bpm=float(np.average(e_[ok], weights=w[ok])) if ok.any() else np.nan,
                             common_s=float(w.sum()), n_rec=len(v), n_rec_ok=int(ok.sum()),
                             ref_HR_sd_bpm=float(np.nanmean([sd for _, _, sd in v])) if ok.any() else np.nan))
        df = pd.DataFrame(rows); df.to_csv(out / f"exp2c_ihr_{tag}.csv", index=False); frames.append(df)
        print(f"  saved: exp2c_ihr_{tag}.csv (rows {len(df)}, missing {int(df.ITE_bpm.isna().sum())})", flush=True)
    allf = out / "exp2c_ihr_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    print(f"\nDone: {allf}")


if __name__ == "__main__":
    main()
