"""
score_unsup_v2.py - score the rPPG-Toolbox unsupervised extractors (POS, CHROM, GREEN, ICA, LGI, PBV, OMIT) with rpfi_v2.py
==================================================================================
Applies RPFI v2 to unsupervised extractors. Every requested method found in the predictions folder is scored.

- prediction files: {dataset}_{METHOD}_prediction.npy in the predictions folder (full-length 1D, case-insensitive)
- recording -> participant mapping: test_ranges / test_subject_of_range in runs/{ds}_cv5/pos_f*_info.json
  (exactly the same recordings and participants as for the learned models)
- per participant, the 10 s chunks are pooled and scored once with rpfi_v2.score_chunks (as for Table 1)
- two variants: raw (as is) and flip (one sign flip for the whole method - uses no reference, so deployable)
- if a POS file is present it is scored too, to check against the POS scores from runs/
- SNR (de Haan & Jeanne 2013) is computed as well

Usage (from the working folder):
    python score_unsup_v2.py --pred-dir "../predictions"
    (use --label-dir if the labels are elsewhere; default "../real data")
Output: unsup_v2/unsup_v2_ALL.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import rpfi_eval as R
import rpfi_v2 as RV

TOOLBOX = ["POS", "CHROM", "GREEN", "ICA", "LGI", "PBV", "OMIT"]  # rPPG-Toolbox unsupervised methods
LABEL_NAME = {"pure": "PURE_label.npy", "ubfc": "UBFC_label.npy", "cohface": "COHFACE_label.npy"}
DS = ["pure", "ubfc", "cohface"]


def snr_chunk(L, P, fs, nfft=8192):
    f0 = R.hr_from_psd(L, fs) / 60.0
    if not np.isfinite(f0) or f0 <= 0 or np.std(P) < 1e-12:
        return np.nan
    x = (P - P.mean()) * np.hanning(P.size)
    S = np.abs(np.fft.rfft(x, n=max(nfft, P.size))) ** 2
    f = np.fft.rfftfreq(max(nfft, P.size), 1 / fs)
    band = (f >= 0.5) & (f <= 4.0)
    sig = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2))
    num, den = S[sig].sum(), S[band & ~sig].sum()
    return float(10 * np.log10(num / den)) if num > 0 and den > 0 else np.nan


def recordings(runs_dir):
    recs = []
    for p in sorted(Path(runs_dir).glob("pos_f*_info.json")):
        info = json.load(open(p))
        recs += [(int(s), int(e), subj) for (s, e), subj in zip(info["test_ranges"], info["test_subject_of_range"])]
    if not recs:
        sys.exit(f"no pos_f*_info.json in {runs_dir}")
    recs.sort()
    assert len({r[0] for r in recs}) == len(recs), "the same recording is in the test set of two folds"
    return recs


def find_preds(pred_dir, ds):
    out = {}
    for p in Path(pred_dir).glob("*.npy"):
        m = re.fullmatch(rf"{ds}_(.+)_prediction", p.stem, flags=re.I)
        if m:
            out[m.group(1).upper()] = p
    return dict(sorted(out.items()))


def score_method(label, pred, recs, fs):
    seg = 10 * fs
    per = defaultdict(lambda: ([], []))
    for s, e, subj in recs:
        l, p = label[s:e], pred[s:e]
        n = min(len(l), len(p))
        if n < seg:
            continue
        lc, pc = R.chunkify(l[:n], seg), R.chunkify(p[:n], seg)
        k = min(len(lc), len(pc))
        per[subj][0].extend(np.asarray(c, float) for c in lc[:k])
        per[subj][1].extend(np.asarray(c, float) for c in pc[:k])
    rows = []
    for subj, (Lr, Pr) in sorted(per.items(), key=lambda kv: str(kv[0])):
        r = RV.score_chunks(Lr, Pr, fs)
        L = [R.preprocess_chunk(c, fs, True, True, 100) for c in Lr]
        P = [R.preprocess_chunk(c, fs, True, True, 100) for c in Pr]
        v = [snr_chunk(a, b, fs) for a, b in zip(L, P)]
        v = [x for x in v if np.isfinite(x)]
        rows.append(dict(participant=subj, **r, SNR_dB=float(np.mean(v)) if v else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="../predictions")
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--out", default="unsup_v2")
    ap.add_argument("--methods", default=",".join(TOOLBOX), help="methods to score (comma separated)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    print(f"rpfi_v2 {RV.__version__}")
    frames, info = [], []
    for ds in DS:
        fs = R.FS_BY_DATASET[ds]
        lab_p = Path(a.label_dir) / LABEL_NAME[ds]
        if not lab_p.exists():
            print(f"[{ds}] label not found: {lab_p} -> skipped"); continue
        label = np.load(lab_p).astype(float).ravel()
        recs = recordings(f"runs/{ds}_cv5")
        want = [m.strip().upper() for m in a.methods.split(",") if m.strip()]
        found = find_preds(a.pred_dir, ds)
        preds = {m: found[m] for m in want if m in found}
        print(f"[{ds}] scoring: {list(preds)}  | no file: {[m for m in want if m not in found]}")
        for meth, path in preds.items():
            pred = np.load(path, allow_pickle=True).astype(float).ravel()
            fin = np.isfinite(pred)
            zero_frac = float(np.mean(pred[fin] == 0)) if fin.any() else 1.0
            info.append(dict(dataset=ds, method=meth, file=path.name, n_pred=len(pred), n_label=len(label),
                             nonfinite_frac=float(1 - fin.mean()), zero_frac=zero_frac))
            if zero_frac > 0.5:
                print(f"  warning: {zero_frac:.0%} of the {meth} prediction values are exactly 0 (output may be empty)")
            if len(pred) != len(label):
                print(f"  {meth}: length {len(pred)} != label {len(label)} -> skipped"); continue
            for var, x in (("raw", pred), ("flip", -pred)):
                t0 = time.time()
                rows = score_method(label, x, recs, fs)
                df = pd.DataFrame(rows)
                df.insert(0, "variant", var); df.insert(0, "method", meth); df.insert(0, "dataset", ds)
                df["zero_frac"] = zero_frac
                frames.append(df)
                print(f"  {meth:6s} {var:4s} N={len(df):2d} RPFI={df.RPFI.mean():.3f} "
                      f"cov={df.SHAPE_defined.mean():.2f} HR-MAE={df.HR_MAE_bpm.mean():5.2f} "
                      f"flag={df.polarity_suspect.mean():.2f} zero={zero_frac:.2f} ({time.time() - t0:.0f}s)", flush=True)
    if not frames:
        sys.exit("nothing was scored; check --pred-dir / --label-dir.")
    allf = out / "unsup_v2_ALL.csv"
    pd.concat(frames, ignore_index=True).to_csv(allf, index=False)
    pd.DataFrame(info).to_csv(out / "unsup_v2_files.csv", index=False)
    print(f"\nDone: {allf}  (+ unsup_v2_files.csv)")


if __name__ == "__main__":
    main()
