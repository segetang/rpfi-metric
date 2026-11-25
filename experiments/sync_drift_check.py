"""
sync_drift_check.py - synchronization check without the raw data
==================================================================================
Without the original timestamps, the lag between the preprocessed label and the POS output is measured in every
10 s chunk within a recording, and the following are summarized per dataset:
  - constant lag : median of the chunk-wise lags of a recording (ms)
  - drift        : slope of a straight-line fit of the chunk-wise lag against time (ms/min); non-zero under a frame-rate
                   mismatch or index-interpolation misalignment
  - step changes : number of times the lag between neighbouring chunks jumps by more than a quarter cycle
                   (a trace of dropped frames or shifted segments)
POS is an untrained extractor, so it reflects the video-label timing as it is.
A within-dataset learned model (bimamba) is measured as well for comparison.

Usage (from the working folder):  python sync_drift_check.py
Output: sync_drift/sync_drift.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import rpfi_eval as R
from rhythm import rhythm_participant

LABEL = {"pure": "PURE_label.npy", "ubfc": "UBFC_label.npy", "cohface": "COHFACE_label.npy"}
MODELS = ("pos", "bimamba")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    a = ap.parse_args()
    out = Path("sync_drift"); out.mkdir(exist_ok=True); rows = []
    for ds in ("ubfc", "pure", "cohface"):
        fs = R.FS_BY_DATASET[ds]; seg = 10 * fs; lab = np.load(Path(a.label_dir) / LABEL[ds]).astype(float)
        runs = Path(f"runs/{ds}_cv5")
        for m in MODELS:
            for inf in sorted(runs.glob(f"{m}_f*_info.json")):
                info = json.load(open(inf)); preds = np.load(runs / f"{m}_f{info['fold']}_test_pred.npy", allow_pickle=True)
                for i, ((s, e), subj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
                    p = np.asarray(preds[i], float).ravel(); n = min(len(p), e - s); k = n // seg
                    if k < 3:
                        continue
                    lags, rrs, f1s = [], [], []
                    for c in range(k):
                        L = R.preprocess_chunk(lab[s + c * seg:s + (c + 1) * seg], fs)[None]
                        P = R.preprocess_chunk(p[c * seg:(c + 1) * seg], fs)[None]
                        r = rhythm_participant(L, P, fs); lags.append(r["offset_s"]); rrs.append(r["rr_med_s"]); f1s.append(r["F1"])
                    lags, rrs, f1s = map(np.asarray, (lags, rrs, f1s)); tmin = (np.arange(k) + 0.5) * 10 / 60
                    ok = f1s >= 0.5                                   # trust the lag only in chunks with a valid beat correspondence
                    slope = np.polyfit(tmin[ok], lags[ok], 1)[0] * 1000 if ok.sum() >= 3 else np.nan
                    jumps = int((np.abs(np.diff(lags[ok])) > 0.25 * np.median(rrs)).sum()) if ok.sum() >= 2 else np.nan
                    rows.append(dict(dataset=ds, model=m, participant=subj, recording_start=s, n_chunks=k, n_chunks_ok=int(ok.sum()),
                                     lag_median_ms=1000 * np.median(lags[ok]) if ok.any() else np.nan,
                                     drift_ms_per_min=slope, n_jumps=jumps, rr_med_s=float(np.median(rrs)),
                                     lag_by_chunk_ms=";".join(f"{1000 * x:.0f}" for x in lags)))
            print(f"[{ds}] {m} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "sync_drift.csv", index=False)
    g = df.groupby(["dataset", "model"])
    print(pd.concat({"recordings": g.size(), "const_lag_median_ms": g.lag_median_ms.median(), "drift_median_ms/min": g.drift_ms_per_min.median(),
                     "|drift|_p90_ms/min": g.drift_ms_per_min.apply(lambda x: x.abs().quantile(.9)),
                     "step_change_rec_frac": g.n_jumps.apply(lambda x: (x > 0).mean())}, axis=1).round(1).to_string())
    print(f"\nDone: {out / 'sync_drift.csv'}")


if __name__ == "__main__":
    main()
