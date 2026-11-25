"""
export_pos_baseline.py — export the POS baseline in the same format as the learned models
==============================================================================
POS needs no training (*_POS_prediction.npy is the prediction itself).
For a fair comparison, however, the evaluation pipeline must consume it **in the same way** as the learned-model
outputs, so it is exported with the same split, the same windowing and the same normalization.

Why this is needed
------------------------------
In the earlier analysis, the MAE of POS was almost equal to the label's DC offset:
    PURE    : POS MAE 40.1010  vs  label mean absolute value 40.1008  (0.0% difference)
    COHFACE : POS MAE 37.2213  vs  label mean absolute value 37.0364  (0.5% difference)
The POS output has zero mean while the PURE/COHFACE labels carry a DC offset (40.1 / 37.0), so the
"error" was an artifact of missing calibration, not of algorithm performance.

Per-window z-normalization removes this artifact (Pearson unchanged):
    PURE    RMSE 43.7804 -> 1.1271   Pearson +0.3649 (unchanged)
    UBFC    RMSE 29.1194 -> 1.3095   Pearson +0.1426 (unchanged)
    COHFACE RMSE 39.4890 -> 1.4145   Pearson -0.0004 (unchanged)

Usage
------
    # single split
    python export_pos_baseline.py --pred UBFC_POS_prediction.npy \
        --split splits/ubfc_split.json --out runs/ubfc

    # K-fold CV (exported with the same fold structure as the learned models)
    python export_pos_baseline.py --pred UBFC_POS_prediction.npy \
        --cv-split splits/ubfc_cv5.json --out runs/ubfc_cv5
"""

import argparse
import json
from pathlib import Path

import numpy as np

from subject_split_v2 import (load_split, load_cv, extract_by_ranges,
                              verify_split_matches)


def znorm_windows(a):
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    return (a - m) / np.maximum(s, 1e-8)


def pos_predict(x_list, seq_len):
    """POS prediction = the input signal itself. As for the learned models, it is cut into
    non-overlapping windows and z-normalized per window."""
    out = []
    for xr in x_list:
        n = (len(xr) // seq_len) * seq_len
        if n == 0:
            out.append(np.array([], dtype=np.float32)); continue
        w = znorm_windows(np.asarray(xr[:n], dtype=np.float32).reshape(-1, seq_len))
        out.append(w.reshape(-1).astype(np.float32))
    return out


def emit(pred, split, out_dir, tag=""):
    seq_len = split["seg_len"]
    te_ranges = split["test_ranges"]
    te_x = extract_by_ranges(pred, te_ranges)
    preds = pos_predict(te_x, seq_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"pos{tag}_test_pred.npy",
            np.array(preds, dtype=object), allow_pickle=True)
    info = dict(model="pos", fold=split.get("fold"), run_seed=None,
                n_params=0, best_val=None, epochs_run=0,
                convergence={"stop_reason": "not_trained",
                             "status": "baseline_no_training",
                             "tail_rel_gain": None, "final_lr": None},
                train_seconds=0.0,
                seq_len=int(seq_len), train_windows=0, val_windows=0,
                train_stride=None, val_subjects=[],
                train_subjects=split.get("train_subjects"),
                test_subjects=split["test_subjects"],
                test_ranges=[list(r) for r in te_ranges],
                test_subject_of_range=split["test_subject_of_range"],
                test_rec_lengths=[int(len(p)) for p in preds],
                pred_space="window_znorm",
                config=dict(note="POS baseline: no training. The input signal is "
                                 "processed with the same windowing and z-normalization as the learned models"))
    json.dump(info, open(out_dir / f"pos{tag}_info.json", "w"), indent=2, default=str)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="*_POS_prediction.npy")
    ap.add_argument("--split", help="single split JSON")
    ap.add_argument("--cv-split", help="K-fold CV JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.split and not args.cv_split:
        ap.error("--split or --cv-split is required")

    pred = np.load(args.pred)
    ds_guess = None
    for d in ("pure", "ubfc", "cohface"):
        if d in Path(args.pred).name.lower():
            ds_guess = d
    out_dir = Path(args.out)
    infos = []

    if args.cv_split:
        cv = load_cv(args.cv_split)
        verify_split_matches(cv, pred, ds_guess, "prediction array")
        for f in cv["folds"]:
            fs_ = dict(f, seg_len=cv["seg_len"], seg_seconds=cv["seg_seconds"])
            infos.append(emit(pred, fs_, out_dir, tag=f"_f{f['fold']}"))
            print(f"  pos fold {f['fold']}: {sum(infos[-1]['test_rec_lengths']):,} samples")
    else:
        sp = load_split(args.split)
        verify_split_matches(sp, pred, ds_guess, "prediction array")
        infos.append(emit(pred, sp, out_dir))
        print(f"  pos: {sum(infos[-1]['test_rec_lengths']):,} samples")

    print(f"done -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
