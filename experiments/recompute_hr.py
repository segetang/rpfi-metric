"""
recompute_hr.py - recompute only the HR metrics from existing predictions (fast, no DTW)
=====================================================================
The HR estimator in rpfi_eval.py was replaced (Welch-bin 7/6 bpm quantization -> zero-padded periodogram +
parabolic interpolation). BWMD/WCR/EDD/RPFI do not depend on HR, so only the HR columns are recomputed
from the existing predictions (runs/*_test_pred.npy).

Columns saved per participant x model
  HR_MAE_bpm         : new spectral HR, mean |label - pred| over 10 s chunks (same chunks as rpfi_eval)
  HR_MAE_peak_bpm    : peak-interval HR, per chunk (reference only)
  HR_MAE_legacy_bpm  : previous estimator - must match HR_MAE_bpm in --rpfi-csv (checked automatically)
  HR_MAE_rec_bpm     : spectral HR of each whole recording (~60 s) preprocessed at once, |label - pred|,
                       averaged over recordings (the video-level HR MAE convention of the rPPG literature)
  HR_label_bpm, HR_pred_bpm : chunk means of the new spectral HR

Usage (from the working folder)
  python recompute_hr.py --dataset ubfc --runs runs/ubfc_cv5 \
      --label "../real data/UBFC_label.npy" \
      --rpfi-csv results_eval_noSRE/rpfi_participant_ubfc_geometric.csv \
      --tag ubfc --out hr_recompute
  # cross-dataset: --runs runs/cross_pure_to_ubfc --tag pure_to_ubfc
  #                --rpfi-csv pure_to_ubfc/rpfi_participant_ubfc_geometric.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    import rpfi_eval as R
except ImportError as e:
    sys.exit(f"[error] cannot import rpfi_eval.py ({e}); it must be importable (repository rpfi/ folder).")

if not hasattr(R, "hr_from_psd_legacy"):
    sys.exit("[error] rpfi_eval.py predates the HR-estimator change; use the repository version.")


def detrend_sparse(x, Lambda=100):
    """Mathematically identical to R.detrend_tarvainen (sparse linear solve). For long recordings."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return x
    e = np.ones(n)
    D = sparse.spdiags([e, -2 * e, e], [0, 1, 2], n - 2, n)
    A = sparse.identity(n) + (Lambda ** 2) * (D.T @ D)
    return x - spsolve(A.tocsc(), x)


def preprocess_recording(x, fs, Lambda):
    y = detrend_sparse(x, Lambda)
    if len(y) > 18:
        y = R.bandpass(y, fs)
    return R.znorm(y)


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def run_model(runs_dir, label_full, model, fs, seg_len, Lambda, rec_label_cache):
    acc = defaultdict(lambda: defaultdict(list))
    for inf_p in sorted(Path(runs_dir).glob(f"{model}_f*_info.json")):
        info = json.load(open(inf_p))
        fold = info.get("fold")
        preds = np.load(Path(runs_dir) / f"{model}_f{fold}_test_pred.npy", allow_pickle=True)
        for rec_i, ((s, e), subj) in enumerate(zip(info["test_ranges"],
                                                   info["test_subject_of_range"])):
            p_rec = np.asarray(preds[rec_i], dtype=float).ravel()
            l_rec = np.asarray(label_full[s:e], dtype=float).ravel()
            L = min(len(p_rec), len(l_rec))
            if L < seg_len:
                continue
            # -- per chunk: same chunks/preprocessing as rpfi_eval.load_participant_data --
            p_ch, l_ch = R.chunkify(p_rec[:L], seg_len), R.chunkify(l_rec[:L], seg_len)
            for c in range(min(len(p_ch), len(l_ch))):
                lc = R.preprocess_chunk(l_ch[c], fs, True, True, Lambda)
                pc = R.preprocess_chunk(p_ch[c], fs, True, True, Lambda)
                a = acc[subj]
                a["l_sp"].append(R.hr_from_psd(lc, fs));        a["p_sp"].append(R.hr_from_psd(pc, fs))
                a["l_pk"].append(R.hr_from_peaks(lc, fs));      a["p_pk"].append(R.hr_from_peaks(pc, fs))
                a["l_lg"].append(R.hr_from_psd_legacy(lc, fs)); a["p_lg"].append(R.hr_from_psd_legacy(pc, fs))
            # -- per recording: whole recording preprocessed at once --
            key = (s, s + L)
            if key not in rec_label_cache:
                rec_label_cache[key] = R.hr_from_psd(preprocess_recording(l_rec[:L], fs, Lambda), fs)
            hr_lr = rec_label_cache[key]
            hr_pr = R.hr_from_psd(preprocess_recording(p_rec[:L], fs, Lambda), fs)
            acc[subj]["rec_err"].append(abs(hr_lr - hr_pr))
    rows = []
    for subj, a in sorted(acc.items()):
        rows.append(dict(model=model, participant=subj, n_chunks=len(a["l_sp"]),
                         n_recordings=len(a["rec_err"]),
                         HR_MAE_bpm=mae(a["l_sp"], a["p_sp"]),
                         HR_MAE_peak_bpm=mae(a["l_pk"], a["p_pk"]),
                         HR_MAE_legacy_bpm=mae(a["l_lg"], a["p_lg"]),
                         HR_MAE_rec_bpm=float(np.nanmean(a["rec_err"])),
                         HR_label_bpm=float(np.nanmean(a["l_sp"])),
                         HR_pred_bpm=float(np.nanmean(a["p_sp"]))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"],
                    help="evaluation (target) dataset")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rpfi-csv", default=None, help="v1 rpfi_participant CSV (for checking and merging)")
    ap.add_argument("--models", default=None)
    ap.add_argument("--tag", default=None, help="output file tag (default: dataset)")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--out", default="hr_recompute")
    args = ap.parse_args()

    fs = R.FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    tag = args.tag or args.dataset
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    label_full = np.load(args.label)
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else R.discover_models(args.runs))
    print(f"[{tag}] fs={fs} seg_len={seg_len} models={models}")

    rows, cache = [], {}
    for m in models:
        t0 = time.time()
        r = run_model(args.runs, label_full, m, fs, seg_len, args.lambda_detrend, cache)
        rows += r
        print(f"  {m:14s} participants={len(r):3d}  ({time.time() - t0:.1f}s)")
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", args.dataset)
    df.insert(1, "condition", tag)

    if args.rpfi_csv:
        old = pd.read_csv(args.rpfi_csv)
        keep = [c for c in ["model", "participant", "RPFI", "BWMD", "WCR", "EDD",
                            "s_BWMD", "s_WCR", "s_EDD", "RMSE", "CCC", "Spearman",
                            "HR_MAE_bpm"] if c in old.columns]
        old = old[keep].rename(columns={"HR_MAE_bpm": "HR_MAE_bpm_in_csv"})
        df = df.merge(old, on=["model", "participant"], how="left")
        if "HR_MAE_bpm_in_csv" in df:
            d = (df["HR_MAE_legacy_bpm"] - df["HR_MAE_bpm_in_csv"]).abs()
            print(f"\n[check] previous estimator recomputed vs. HR_MAE_bpm in the CSV: max difference {d.max():.2e} "
                  f"({'match - same loading path as rpfi_eval' if d.max() < 1e-6 else '[!] mismatch - check paths/options'})")

    f = out / f"hr_participant_{tag}.csv"
    df.to_csv(f, index=False)

    print(f"\n{'model':14s} {'legacy':>8s} {'spectral':>9s} {'peak':>7s} {'rec-lvl':>8s}  "
          f"{'legacy==0':>9s} {'new<0.5':>8s}")
    for m, g in df.groupby("model"):
        print(f"{m:14s} {g.HR_MAE_legacy_bpm.mean():8.2f} {g.HR_MAE_bpm.mean():9.2f} "
              f"{g.HR_MAE_peak_bpm.mean():7.2f} {g.HR_MAE_rec_bpm.mean():8.2f}  "
              f"{(g.HR_MAE_legacy_bpm == 0).sum():9d} {(g.HR_MAE_bpm < 0.5).sum():8d}")
    print(f"\nsaved: {f}")


if __name__ == "__main__":
    main()
