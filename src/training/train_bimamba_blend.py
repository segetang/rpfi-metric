"""
train_bimamba_blend.py — training script for the BiMamba ensemble (blend)
====================================================================
Reproduces the idea of the original bimamba.py (blend version), "train K models and fit blending weights
by least squares", while respecting the subject-wise principle.


Leakage removed relative to the original (three leaks found in the original)
-------------------------------------------------
| item | original | this script |
|------|------|------|
| base-model K-fold | `KFold(shuffle=True)`, stride-1 sliding windows | split by **subject** (GroupKFold-like) |
| normalization | `MinMaxScaler` fit on the whole array (train+val+blend) | per-window z-norm (as in train_models.py; no global-statistics leakage) |
| blending-weight fit | last 20% of the whole array (already seen by the base models because of the shuffled KFold) | a **separate subject group** never used to train any base model |
| final evaluation | whole array (incl. train+val+blend) | outer test subjects only, non-overlapping |
| post-processing | Butterworth low-pass (3.0 Hz) | none (same conditions as the other 7 models - fair comparison) |
| training protocol | 70 epochs, patience 5, CosineAnnealingWarmRestarts | same as the common harness: at most 200 epochs, patience 15,
  SmoothL1, ReduceLROnPlateau - deliberately unified so that the "common protocol for all models" claim holds |
| K (number of base models) | 5 | K_outer - 2 = **3 for all three datasets** (explained below) |

The 5 groups of the outer folds are reused as they are (v2, blend-frac removed)
------------------------------------------------------------------
When subject_split_v2.py builds the outer CV it already divides all subjects into 5 non-overlapping
groups (each fold's test_subjects is one such group).
By the definition of K-fold, the train_subjects of outer fold f are always "the union of the other 4 groups".
These 4 groups are used as they are, without reshuffling:

  1) 1 of the 4 groups (fixed rotation: (f+1) % K) -> blend_subjects.
     No base model is trained on this group; it is used only for the blending weights (least squares).
  2) the other 3 groups -> the "held-out group" (= early-stopping validation set) of each of K_inner = 3 base models.
     Model i is trained on these 3 groups minus its own group.

No separate --blend-frac hyperparameter is needed, and K_inner follows mechanically from the outer K (= 5)
as K - 2 = 3. For UBFC/COHFACE (8 subjects per group) and PURE (2 per group) alike this is
"the same rule applied one level deeper", so the procedure is the same regardless of dataset size.
(Group *size* follows the number of subjects in the dataset, but the *number* of groups is always 3.)

Usage
------
    python train_bimamba_blend.py --dataset ubfc \
        --pred "../real data/UBFC_POS_prediction.npy" \
        --label "../real data/UBFC_label.npy" \
        --cv-split splits/ubfc_cv5.json --out runs

    python train_bimamba_blend.py --dataset pure \
        --pred "../real data/PURE_POS_prediction.npy" \
        --label "../real data/PURE_label.npy" \
        --cv-split splits/pure_cv5.json --out runs

The command is the same for all three datasets (K_inner need not be given).
Outputs go to the same directory as train_models.py (runs/{dataset}_cv{K}/)
under model = "bimamba_blend", so the evaluation pipeline consumes them unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models import build_model
from subject_split_v2 import load_cv, extract_by_ranges, verify_split_matches
from train_models import (
    znorm_windows, windows_from_recordings, make_loader, train_one,
    predict_test, eval_metrics, _stable_hash,
)


# ══════════════════════════════════════════════════════════════
# deterministic seed / subject split
# ══════════════════════════════════════════════════════════════

def _seed_for(base_seed, fold, tag):
    """Same principle as run_seed in train_models.py (md5 based, reproducible across processes)."""
    return base_seed + 1000 * fold + (_stable_hash(tag) % 997)


def partition_subjects(all_folds, fold):
    """Reuse the 5 groups of the outer CV as they are (no new RNG, fully deterministic).

    all_folds: all of cv["folds"] (the test_subjects of the K non-overlapping groups covering all subjects,
               built by subject_split_v2.make_cv_folds).
    fold: index of the fold currently used as the outer test.

    Returns (blend_subjects, [inner_group_0, inner_group_1, ...])
          The number of groups is always len(all_folds) - 2 (excluding one outer test and one blend group).
    """
    k_outer = len(all_folds)
    other_idx = [k for k in range(k_outer) if k != fold]
    blend_idx = other_idx[(fold + 1) % len(other_idx)]          # fixed rotation, no RNG needed
    inner_idx = [k for k in other_idx if k != blend_idx]

    blend_subs = sorted(all_folds[blend_idx]["test_subjects"])
    groups = [sorted(all_folds[k]["test_subjects"]) for k in inner_idx]
    return blend_subs, groups


def windows_nonoverlap_znorm(x_list, y_list, seq_len):
    """Non-overlapping windows + z-norm (the preprocessing of predict_test applied to the label side).
    Returns a list of 1D arrays, one per recording."""
    yw = []
    for yr in y_list:
        n = (len(yr) // seq_len) * seq_len
        if n:
            yw.append(znorm_windows(
                np.asarray(yr[:n], dtype=np.float32).reshape(-1, seq_len)).reshape(-1))
        else:
            yw.append(np.array([], dtype=np.float32))
    return yw


# ══════════════════════════════════════════════════════════════
# train one inner model
# ══════════════════════════════════════════════════════════════

def train_inner_model(tag, fit_subs, val_subs, subj_x, subj_y, seq_len,
                      args, device, out_dir, fold):
    run_seed = _seed_for(args.seed, fold, f"bimamba_blend_{tag}")
    torch.manual_seed(run_seed); np.random.seed(run_seed % (2 ** 31))

    def gather(subs):
        xs, ys = [], []
        for s in subs:
            xs.extend(subj_x[s]); ys.extend(subj_y[s])
        return xs, ys

    fit_x, fit_y = gather(fit_subs)
    val_x, val_y = gather(val_subs)

    Xf, Yf = windows_from_recordings(fit_x, fit_y, seq_len, args.train_stride)
    val_stride = args.val_stride or max(1, seq_len // 4)
    Xv, Yv = windows_from_recordings(val_x, val_y, seq_len, val_stride)
    Xf, Yf = znorm_windows(Xf), znorm_windows(Yf)
    Xv, Yv = znorm_windows(Xv), znorm_windows(Yv)

    # same as the bimamba DEFAULTS (lr = 3e-4, AdamW) + common protocol (epochs/patience/scheduler)
    cfg = dict(lr=3e-4, epochs=args.epoch_cap, batch=32, loss=args.unify_loss, opt="adamw")
    tr_ld = make_loader(Xf, Yf, cfg["batch"], True)
    va_ld = make_loader(Xv, Yv, cfg["batch"], False)

    model = build_model("bimamba").to(device)
    n_par = sum(p.numel() for p in model.parameters())
    ck = out_dir / f"bimamba_blend_f{fold}_{tag}.pth"
    print(f"    [{tag}] params {n_par:,} | train {Xf.shape[0]} win / val {Xv.shape[0]} win "
          f"(fit={len(fit_subs)} subjects / val={len(val_subs)} subjects)")

    hist, best, diag = train_one(model, tr_ld, va_ld, device, cfg, ck,
                                 patience=args.patience, sched_kind="plateau",
                                 smooth_beta=args.smooth_beta, verbose=not args.quiet)
    model.load_state_dict(torch.load(ck, map_location=device))

    return model, dict(tag=tag, run_seed=int(run_seed), n_params=int(n_par),
                       best_val=float(best), epochs_run=len(hist), convergence=diag,
                       fit_subjects=fit_subs, val_subjects=val_subs, ckpt=str(ck))


# ══════════════════════════════════════════════════════════════
# run one fold (train K_inner models + blending + outer test evaluation)
# ══════════════════════════════════════════════════════════════

def run_fold(fold_split, all_folds, pred, label, args, device, out_dir):
    fold = fold_split["fold"]
    seq_len = fold_split["seg_len"]
    train_subjects = fold_split["train_subjects"]
    test_subjects = fold_split["test_subjects"]

    tr_ranges = fold_split["train_ranges"]
    tr_subj_of = fold_split["train_subject_of_range"]
    subj_x = {s: [] for s in train_subjects}
    subj_y = {s: [] for s in train_subjects}
    for (s, e), subj in zip(tr_ranges, tr_subj_of):
        subj_x[subj].append(np.asarray(pred[s:e], dtype=np.float32))
        subj_y[subj].append(np.asarray(label[s:e], dtype=np.float32))

    blend_subs, groups = partition_subjects(all_folds, fold)
    # integrity: outer train_subjects must equal exactly the union of blend + groups
    assert set(train_subjects) == set(blend_subs) | {s for g in groups for s in g}, \
        "outer train_subjects != blend_subjects | inner_groups - group reuse is broken"
    print(f"  -- bimamba_blend fold {fold}/{fold_split.get('K', '?')} : "
          f"blend_subjects={len(blend_subs)} subjects, K_inner={len(groups)}, "
          f"group sizes={[len(g) for g in groups]}")

    models, infos = [], []
    for i, val_grp in enumerate(groups):
        fit_subs = sorted(set(train_subjects) - set(blend_subs) - set(val_grp))
        m, info = train_inner_model(f"inner{i}", fit_subs, val_grp, subj_x, subj_y,
                                    seq_len, args, device, out_dir, fold)
        models.append(m); infos.append(info)

    # -- blending weights: least squares on blend_subjects only --
    blend_x, blend_y = [], []
    for s in blend_subs:
        blend_x.extend(subj_x[s]); blend_y.extend(subj_y[s])

    blend_preds_per_model = [predict_test(m, blend_x, seq_len, device) for m in models]
    blend_y_w = windows_nonoverlap_znorm(blend_x, blend_y, seq_len)

    flat_preds = [np.concatenate(bp) for bp in blend_preds_per_model]   # K_inner arrays, each (N,)
    P = np.stack(flat_preds, axis=1)                                   # (N, K_inner)
    y_flat = np.concatenate(blend_y_w)
    n = min(len(P), len(y_flat)); P, y_flat = P[:n], y_flat[:n]

    w, *_ = np.linalg.lstsq(P, y_flat, rcond=None)
    w = w / w.sum()
    print(f"    blend weights: {np.round(w, 4).tolist()}")

    # -- apply the weighted ensemble to the outer test subjects --
    te_ranges = fold_split["test_ranges"]
    te_subj_of = fold_split["test_subject_of_range"]
    te_x = extract_by_ranges(pred, te_ranges)
    te_y = extract_by_ranges(label, te_ranges)

    test_preds_per_model = [predict_test(m, te_x, seq_len, device) for m in models]
    final_preds = []
    for rec_i in range(len(te_x)):
        acc = np.zeros_like(test_preds_per_model[0][rec_i])
        for wk, mp in zip(w, test_preds_per_model):
            acc = acc + wk * mp[rec_i]
        final_preds.append(acc.astype(np.float32))

    te_y_w = windows_nonoverlap_znorm(te_x, te_y, seq_len)
    metrics = eval_metrics(te_y_w, final_preds)
    print(f"    [bimamba_blend_f{fold}] test  Pearson {metrics['pearson_mean']:+.4f}  "
          f"PSNR {metrics['psnr_db']:.2f}dB  RMSE {metrics['rmse']:.4f}  "
          f"MAE {metrics['mae']:.4f}")

    np.save(out_dir / f"bimamba_blend_f{fold}_test_pred.npy",
            np.array(final_preds, dtype=object), allow_pickle=True)

    info = dict(
        model="bimamba_blend", fold=fold,
        k_inner=len(groups),
        blend_group_source="outer_cv_fold_reuse",  # marks that the blend/inner groups reuse the outer 5-way split
        blend_subjects=blend_subs, inner_groups=groups,
        blend_weights=[float(x) for x in w],
        inner_models=infos,
        train_subjects=train_subjects, test_subjects=test_subjects,
        test_ranges=[list(r) for r in te_ranges],
        test_subject_of_range=te_subj_of,
        test_rec_lengths=[int(len(p)) for p in final_preds],
        pred_space="window_znorm",
        test_metrics=metrics,
        config=dict(lr=3e-4, opt="adamw", loss=args.unify_loss,
                   epoch_cap=args.epoch_cap, patience=args.patience,
                   smooth_beta=args.smooth_beta, train_stride=args.train_stride,
                   scheduler="plateau"),
    )
    json.dump(info, open(out_dir / f"bimamba_blend_f{fold}_info.json", "w"),
              indent=2, default=str)
    return info


# ══════════════════════════════════════════════════════════════
# entry point
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cv-split", required=True, help="K-fold CV JSON (subject_split_v2.py --cv K)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--folds", default=None, help="run only these folds (comma separated), e.g. --folds 0,1")
    ap.add_argument("--train-stride", type=int, default=10)
    ap.add_argument("--val-stride", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--epoch-cap", type=int, default=200)
    ap.add_argument("--unify-loss", choices=["mse", "smoothl1"], default="smoothl1")
    ap.add_argument("--smooth-beta", type=float, default=1.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = np.load(args.pred); label = np.load(args.label)

    cv = load_cv(args.cv_split)
    verify_split_matches(cv, pred, args.dataset, "prediction array")
    verify_split_matches(cv, label, args.dataset, "label array")

    out_dir = Path(args.out) / f"{args.dataset}_cv{cv['K']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_folds = None
    if args.folds:
        sel_folds = {int(x) for x in args.folds.split(",") if x.strip()}

    print(f"dataset={args.dataset}  bimamba-blend  {cv['K']}-fold subject CV  "
          f"K_inner={cv['K'] - 2} (outer groups reused: 1 blend + {cv['K'] - 2} inner)  "
          f"device={device}")

    summary = []
    for f in cv["folds"]:
        if sel_folds is not None and f["fold"] not in sel_folds:
            continue
        done = out_dir / f"bimamba_blend_f{f['fold']}_info.json"
        if args.skip_existing and done.exists():
            print(f"  -- fold {f['fold']}: already exists, skipped")
            summary.append(json.load(open(done)))
            continue
        fold_split = dict(f, seg_len=cv["seg_len"], seg_seconds=cv["seg_seconds"], K=cv["K"])
        t0 = time.time()
        info = run_fold(fold_split, cv["folds"], pred, label, args, device, out_dir)
        print(f"    ({time.time() - t0:.1f}s)")
        summary.append(info)

    pr = [r["test_metrics"].get("pearson_mean", np.nan) for r in summary]
    ps = [r["test_metrics"].get("psnr_db", np.nan) for r in summary]
    print(f"\n=== bimamba_blend fold summary ===  n_fold={len(summary)}  "
          f"Pearson {np.nanmean(pr):+.4f}±{np.nanstd(pr):.4f}  PSNR {np.nanmean(ps):.2f}dB")

    bad = [r["fold"] for r in summary
           if any(im["convergence"]["status"] == "capped_still_improving"
                  for im in r.get("inner_models", []))]
    if bad:
        print(f"  [!] folds suspected of under-training: {bad} (some inner models ended as capped_still_improving)")
    else:
        print("  [ok] all inner models stopped on the convergence criterion (no under-training)")

    # merge summary.json (accumulated in the same file as the other models of train_models.py)
    sp = out_dir / "summary.json"
    merged = {}
    if sp.exists():
        for r in json.load(open(sp)):
            merged[(r["model"], r.get("fold"))] = r
    for r in summary:
        merged[(r["model"], r.get("fold"))] = r
    json.dump(list(merged.values()), open(sp, "w"), indent=2, default=str)
    print(f"done -> {out_dir.resolve()}  ({len(merged)} summary entries)")


if __name__ == "__main__":
    main()
