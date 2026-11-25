"""
exp2b_extract.py - Experiment 2b: pulse rate variability (PRV) from the beat fiducials of a third-party tool (pyPPG)
==================================================================================
Run in the dedicated pyPPG environment (see requirements-pyppg.txt), like exp2_extract.py.

For each condition, model and participant, the whole reference and predicted recordings are upsampled to 120 Hz and
pyPPG fiducials (get_fiducials, correct=True) are computed; inter-beat intervals (IBI) are built from the systolic-peak
(sp) times. Our own beat detection is not used.
  IBI cleaning (fixed in advance): remove IBIs outside 0.33-2.0 s and IBIs deviating more than 20% from the recording median.
  RMSSD: only differences of consecutive retained IBI pairs within a recording. SDNN: all retained IBIs.
  Participant level: IBIs pooled over recordings; missing if fewer than 20 valid IBIs.

Usage (from the working folder):  ppgenv\\Scripts\\python exp2b_extract.py
Output: exp2b_prv/exp2b_prv_ALL.csv
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

from exp2_extract import CONDITIONS, FS, FS_UP, LABEL_NAME

IBI_MIN, IBI_MAX, DEV_MAX, MIN_IBI = 0.33, 2.0, 0.20, 20


def peak_times(y, fs):
    """One recording -> pyPPG systolic-peak times (s). Empty array on failure."""
    from dotmap import DotMap
    from scipy.signal import resample_poly
    from pyPPG import PPG
    import pyPPG.fiducials as FP
    from pyPPG.preproc import Preprocessing
    y = np.asarray(y, float)
    if y.size < fs * 10 or not np.all(np.isfinite(y)) or np.std(y) < 1e-12:
        return np.array([])
    try:
        v = resample_poly(y - np.mean(y), FS_UP // fs, 1)
        s = DotMap(); s.v = v; s.fs = FS_UP; s.start = 0; s.end = len(v); s.name = "rec"
        s.filt_sig, s.filt_d1, s.filt_d2, s.filt_d3 = Preprocessing(s, True)
        p = PPG(s)
        fp = FP.FpCollection(p).get_fiducials(p, True)
        sp = pd.to_numeric(fp["sp"], errors="coerce").dropna().to_numpy(float)
        return np.unique(sp) / FS_UP
    except Exception:
        return np.array([])


def clean_ibi(t):
    """Peak times -> (cleaned IBIs, differences of consecutive pairs)."""
    if t.size < 3:
        return np.array([]), np.array([])
    ibi = np.diff(t)
    ok = (ibi >= IBI_MIN) & (ibi <= IBI_MAX)
    if ok.sum() >= 3:
        med = np.median(ibi[ok]); ok &= np.abs(ibi - med) <= DEV_MAX * med
    succ = np.diff(ibi)[ok[1:] & ok[:-1]]
    return ibi[ok], succ


def prv(parts):
    ibi = np.concatenate([p[0] for p in parts]) if parts else np.array([])
    succ = np.concatenate([p[1] for p in parts]) if parts else np.array([])
    if ibi.size < MIN_IBI or succ.size < MIN_IBI - 1:
        return np.nan, np.nan, int(ibi.size)
    return float(np.sqrt(np.mean(succ ** 2)) * 1000), float(np.std(ibi, ddof=1) * 1000), int(ibi.size)


def _task(args):
    key, y, fs = args
    return key, clean_ibi(peak_times(y, fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="exp2b_prv")
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
        fs = FS[ds]; lab = labels[ds]
        jobs, meta = [], []
        for m in R.discover_models(runs):
            for inf_p in sorted(Path(runs).glob(f"{m}_f*_info.json")):
                info = json.load(open(inf_p))
                preds = np.load(Path(runs) / f"{m}_f{info['fold']}_test_pred.npy", allow_pickle=True)
                for i, ((s, e), subj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
                    p = np.asarray(preds[i], float).ravel(); n = min(len(p), e - s)
                    if n < 10 * fs:
                        continue
                    meta.append((m, subj, s, e)); jobs.append((("pred", m, subj, s), p[:n], fs))
                    if (ds, s, e) not in ref_cache:
                        ref_cache[(ds, s, e)] = None; jobs.append((("ref", ds, s, e), lab[s:e], fs))
        t0 = time.time(); res = {}
        print(f"\n[{tag}] recording jobs {len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, (key, v) in enumerate(ex.map(_task, jobs, chunksize=4)):
                res[key] = v
                if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                    print(f"  pyPPG {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        for key, v in res.items():
            if key[0] == "ref":
                ref_cache[key[1:]] = v
        per = defaultdict(lambda: dict(ref=[], pred=[]))
        for m, subj, s, e in meta:
            per[(m, subj)]["ref"].append(ref_cache[(ds, s, e)]); per[(m, subj)]["pred"].append(res[("pred", m, subj, s)])
        rows = []
        for (m, subj), d in sorted(per.items()):
            rr, rs, rn = prv(d["ref"]); pr, ps, pn = prv(d["pred"])
            rows.append(dict(condition=tag, dataset=ds, primary=primary, model=m, participant=subj,
                             ref_RMSSD_ms=rr, ref_SDNN_ms=rs, ref_nIBI=rn, pred_RMSSD_ms=pr, pred_SDNN_ms=ps, pred_nIBI=pn))
        df = pd.DataFrame(rows); df.to_csv(out / f"exp2b_prv_{tag}.csv", index=False); frames.append(df)
        print(f"  saved: exp2b_prv_{tag}.csv (rows {len(df)}, predicted PRV missing {int(df.pred_RMSSD_ms.isna().sum())})", flush=True)
    allf = out / "exp2b_prv_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    print(f"\nDone: {allf}")


if __name__ == "__main__":
    main()
