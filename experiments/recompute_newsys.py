"""
recompute_newsys.py - compute the raw SHAPE and rhythm values from existing predictions (no DTW)
==================================================================================
Reflects the 2026-11-24 changes:
  - shape_template.py : phase normalization (64 points per beat) + fundamental-phase alignment + coherence gate + r clipped to [0, 1]
  - rhythm.py         : minimum beat spacing adapted to the signal's own period (fixes double detection at low HR)
SHAPE and rhythm are computed per participant on the same chunks and preprocessing as rpfi_eval.

Usage (from the working folder)
  python recompute_newsys.py --dataset ubfc --runs runs/ubfc_cv5 \
      --label "../real data/UBFC_label.npy" \
      --rpfi-csv results_eval_noSRE/rpfi_participant_ubfc_geometric.csv --tag ubfc
All 9 conditions at once: run_newsys_all.py.
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

try:
    import rpfi_eval as R
    from rhythm import rhythm_participant
    from shape_template import shape_participant
except ImportError as e:
    sys.exit(f"[error] {e} - rpfi_eval.py, rhythm.py and shape_template.py must be importable (repository rpfi/ folder).")


def load_chunks(runs_dir, label_full, model, fs, seg_len, Lambda):
    """Per participant: (raw reference chunks, raw prediction chunks, preprocessed reference, preprocessed prediction)."""
    D = defaultdict(lambda: ([], [], [], []))
    for inf_p in sorted(Path(runs_dir).glob(f"{model}_f*_info.json")):
        info = json.load(open(inf_p))
        preds = np.load(Path(runs_dir) / f"{model}_f{info['fold']}_test_pred.npy", allow_pickle=True)
        for rec_i, ((s, e), subj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
            p_rec = np.asarray(preds[rec_i], float).ravel()
            l_rec = np.asarray(label_full[s:e], float).ravel()
            n = min(len(p_rec), len(l_rec))
            if n < seg_len:
                continue
            pc, lc = R.chunkify(p_rec[:n], seg_len), R.chunkify(l_rec[:n], seg_len)
            for c in range(min(len(pc), len(lc))):
                lr, pr, ls, ps = D[subj]
                lr.append(np.asarray(lc[c], float)); pr.append(np.asarray(pc[c], float))
                ls.append(R.preprocess_chunk(lc[c], fs, True, True, Lambda))
                ps.append(R.preprocess_chunk(pc[c], fs, True, True, Lambda))
    return {s: tuple(np.array(x) for x in v) for s, v in D.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rpfi-csv", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--out", default="newsys_recompute")
    a = ap.parse_args()

    fs = R.FS_BY_DATASET[a.dataset]
    seg_len = int(round(fs * a.eval_seconds))
    tag = a.tag or a.dataset
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    label_full = np.load(a.label)
    models = [m.strip() for m in a.models.split(",")] if a.models else R.discover_models(a.runs)
    print(f"[{tag}] fs={fs} models={models}")

    rows = []
    for m in models:
        t0 = time.time()
        data = load_chunks(a.runs, label_full, m, fs, seg_len, a.lambda_detrend)
        for subj, (Lr, Pr, L, P) in sorted(data.items()):
            rh = rhythm_participant(L, P, fs)
            sh = shape_participant(list(Lr), list(Pr), fs, L, P)
            y, q = L.ravel(), P.ravel()
            wcr, ccc, _ = R.compute_wcr(y, q)
            rows.append(dict(model=m, participant=subj, n_chunks=len(L),
                             WCR_new=wcr, CCC_new=ccc,
                             **{f"sh_{k}": v for k, v in sh.items()}, **rh))
        print(f"  {m:14s} participants={len(data):3d} ({time.time() - t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", a.dataset); df.insert(1, "condition", tag)
    if a.rpfi_csv:
        old = pd.read_csv(a.rpfi_csv)
        keep = [c for c in ["model", "participant", "RPFI", "BWMD", "WCR", "EDD", "s_BWMD", "s_WCR", "s_EDD",
                            "CCC", "RMSE", "HR_MAE_bpm", "HR_MAE_peak_bpm"] if c in old.columns]
        df = df.merge(old[keep], on=["model", "participant"], how="left")
    f = out / f"newsys_participant_{tag}.csv"
    df.to_csv(f, index=False)
    s = df.groupby("model")[["sh_SHAPE", "sh_gated", "sh_coh_pred", "WCR_new", "RHY", "F1", "nIBIE"]].median()
    print(s.round(3).to_string()); print(f"saved: {f}")


if __name__ == "__main__":
    main()
