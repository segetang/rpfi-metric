"""
exp2_extract.py - Experiment 2: morphology features of reference and predicted pulses from a third-party tool (pyPPG)
================================================================================
pyPPG is not compatible with pandas 3 / NumPy 2, so this runs in a dedicated virtual environment (requirements-pyppg.txt).

For each condition, model and participant
  - the whole reference (label) and predicted recordings are upsampled to 120 Hz (resample_poly) -> default pyPPG pipeline
    (Preprocessing(filtering=True) -> FpCollection.get_fiducials(correct=True) -> BmCollection.get_biomarkers)
  - beat-wise candidate features -> per participant, the median over the beats of all recordings
  - for the reference, medians over odd and even beats are stored too (split-half reliability for the feature-selection rule)
  - conventional metrics Pearson r and RMSE (same 10 s chunks and preprocessing as rpfi_eval)
Feature selection, error computation and statistics are done in exp2_analyze.py (normal environment).

Usage (from the working folder, with the dedicated environment's python):
    ppgenv\\Scripts\\python exp2_extract.py            (use --label-dir if the labels are elsewhere)
Output: exp2_features/exp2_features_ALL.csv
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
for _a, _b in (("NaN", np.nan), ("Inf", np.inf)):   # pyPPG uses NumPy 1.x aliases
    if not hasattr(np, _a):
        setattr(np, _a, _b)

FS_UP = 120
# candidate features (from the literature; amplitude-free ratios only). IPA failed the upsampling-invariance check (relative error 16-43%).
CAND = ["Tsp/Tpi", "Tpw50/Tpi", "Ab/Aa", "AGI", "Tsys/Tdia", "deltaT/Tpi", "Tdw50/Tsw50"]
LABEL_NAME = {"pure": "PURE_label.npy", "ubfc": "UBFC_label.npy"}
FS = {"pure": 30, "ubfc": 30, "cohface": 20}
# only targets where SHAPE is defined (UBFC, PURE). COHFACE excluded in advance (SHAPE gated in 85-92%).
CONDITIONS = [  # (condition, target dataset, runs folder, primary?)
    ("ubfc", "ubfc", "runs/ubfc_cv5", True),
    ("pure", "pure", "runs/pure_cv5", True),
    ("pure_to_ubfc", "ubfc", "runs/cross_pure_to_ubfc", False),
    ("cohface_to_ubfc", "ubfc", "runs/cross_cohface_to_ubfc", False),
    ("ubfc_to_pure", "pure", "runs/cross_ubfc_to_pure", False),
    ("cohface_to_pure", "pure", "runs/cross_cohface_to_pure", False),
]


def ppg_beats(y, fs):
    """One recording -> DataFrame of beat-wise candidate features (empty DataFrame on failure)."""
    from dotmap import DotMap
    from scipy.signal import resample_poly
    from pyPPG import PPG, Fiducials
    import pyPPG.fiducials as FP
    import pyPPG.biomarkers as BM
    from pyPPG.preproc import Preprocessing
    y = np.asarray(y, float)
    if y.size < fs * 10 or not np.all(np.isfinite(y)) or np.std(y) < 1e-12:
        return pd.DataFrame(columns=CAND)
    try:
        v = resample_poly(y - np.mean(y), FS_UP // fs, 1)
        s = DotMap(); s.v = v; s.fs = FS_UP; s.start = 0; s.end = len(v); s.name = "rec"
        s.filt_sig, s.filt_d1, s.filt_d2, s.filt_d3 = Preprocessing(s, True)
        p = PPG(s)
        fp = FP.FpCollection(p).get_fiducials(p, True)
        _, vals, _ = BM.BmCollection(p, Fiducials(fp)).get_biomarkers()
        df = pd.concat(vals.values(), axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df["deltaT/Tpi"] = df["deltaT"] / df["Tpi"]
        out = df[CAND].astype(float).replace([np.inf, -np.inf], np.nan)
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=CAND)


def _task(args):
    key, y, fs = args
    t0 = time.time()
    return key, ppg_beats(y, fs), time.time() - t0


def chunks_pp(x, fs, R, seg_len, Lambda=100):
    c = R.chunkify(np.asarray(x, float), seg_len)
    return [R.preprocess_chunk(cc, fs, True, True, Lambda) for cc in c]


def summarize(beats, prefix, split=False):
    out = {f"{prefix}_nbeats": len(beats)}
    for f in CAND:
        col = beats[f] if f in beats else pd.Series(dtype=float)
        out[f"{prefix}_med_{f}"] = float(np.nanmedian(col)) if col.notna().any() else np.nan
        out[f"{prefix}_valid_{f}"] = int(col.notna().sum())
        if split:
            odd, even = col.iloc[0::2], col.iloc[1::2]
            out[f"{prefix}_odd_{f}"] = float(np.nanmedian(odd)) if odd.notna().any() else np.nan
            out[f"{prefix}_even_{f}"] = float(np.nanmedian(even)) if even.notna().any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="exp2_features")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", default=None, help="for testing: comma-separated conditions")
    a = ap.parse_args()
    import rpfi_eval as R
    out = Path(a.out); out.mkdir(exist_ok=True)
    conds = [c for c in CONDITIONS if not a.only or c[0] in a.only.split(",")]
    miss = [p for c in conds for p in (Path(c[2]), Path(a.label_dir) / LABEL_NAME[c[1]]) if not p.exists()]
    if miss:
        sys.exit("Paths not found: " + ", ".join(sorted(set(map(str, miss)))))

    labels = {ds: np.load(Path(a.label_dir) / LABEL_NAME[ds]).astype(float).ravel() for ds in {c[1] for c in conds}}
    ref_cache = {}   # (dataset, s, e) -> reference beat features (independent of model/condition, computed once)
    frames = []
    for tag, ds, runs, primary in conds:
        fs = FS[ds]; seg_len = 10 * fs; lab = labels[ds]
        models = R.discover_models(runs)
        print(f"\n[{tag}] fs={fs} models={models} workers={a.workers}", flush=True)
        jobs, meta = [], []   # meta: (model, subj, s, e, pred)
        for m in models:
            for inf_p in sorted(Path(runs).glob(f"{m}_f*_info.json")):
                info = json.load(open(inf_p))
                preds = np.load(Path(runs) / f"{m}_f{info['fold']}_test_pred.npy", allow_pickle=True)
                for i, ((s, e), subj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
                    p = np.asarray(preds[i], float).ravel(); n = min(len(p), e - s)
                    if n < seg_len:
                        continue
                    meta.append((m, subj, s, e, p[:n]))
                    jobs.append((("pred", m, subj, s), p[:n], fs))
                    if (ds, s, e) not in ref_cache:
                        ref_cache[(ds, s, e)] = None
                        jobs.append((("ref", ds, s, e), lab[s:e], fs))
        t0 = time.time(); res = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, (key, df, dt) in enumerate(ex.map(_task, jobs, chunksize=4)):
                res[key] = df
                if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                    print(f"  pyPPG {k + 1}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)
        for key, df in res.items():
            if key[0] == "ref":
                ref_cache[key[1:]] = df
        # participant-level aggregation
        per = defaultdict(lambda: dict(ref=[], pred=[], L=[], P=[], n_rec=0))
        for m, subj, s, e, p in meta:
            d = per[(m, subj)]
            d["ref"].append(ref_cache[(ds, s, e)]); d["pred"].append(res[("pred", m, subj, s)]); d["n_rec"] += 1
            l = lab[s:s + len(p)]
            d["L"] += chunks_pp(l, fs, R, seg_len); d["P"] += chunks_pp(p, fs, R, seg_len)
        rows = []
        for (m, subj), d in sorted(per.items()):
            rb = pd.concat(d["ref"], ignore_index=True) if d["ref"] else pd.DataFrame(columns=CAND)
            pb = pd.concat(d["pred"], ignore_index=True) if d["pred"] else pd.DataFrame(columns=CAND)
            y, q = np.concatenate(d["L"]), np.concatenate(d["P"])
            rows.append(dict(condition=tag, dataset=ds, primary=primary, model=m, participant=subj, n_rec=d["n_rec"],
                             Pearson=float(np.corrcoef(y, q)[0, 1]), RMSE=R.compute_rmse(y, q),
                             **summarize(rb, "ref", split=True), **summarize(pb, "pred")))
        df = pd.DataFrame(rows); df.to_csv(out / f"exp2_features_{tag}.csv", index=False); frames.append(df)
        print(f"  saved: {out / f'exp2_features_{tag}.csv'}  (rows {len(df)}, median reference beats {df.ref_nbeats.median():.0f}, "
              f"median predicted beats {df.pred_nbeats.median():.0f}, predicted pyPPG failures {int((df.pred_nbeats == 0).sum())})", flush=True)
    allf = out / "exp2_features_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    print(f"\nDone: {allf}")


if __name__ == "__main__":
    main()
