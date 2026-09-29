"""
score_npy_rpfi.py - score full-length 1D prediction arrays (*_prediction.npy) with RPFI.
Used for fold-independent predictions such as unsupervised extractors (GREEN, LGI, PCA, SSR, POS).

Same procedure as the main experiments (rpfi_eval.py):
  - recording -> participant mapping from test_ranges / test_subject_of_range of every fold in
    splits/{ds}_cv5.json (each recording appears exactly once as a test recording)
  - each recording is cut into 10 s non-overlapping chunks; trailing samples are discarded
  - per chunk: preprocess_chunk (detrend lambda=100 -> 0.75-3.0 Hz BPF -> z-norm); EDD without BPF
  - per participant: BWMD = mean over chunks; WCR and EDD once on the concatenated signal
  - fixed anchors + weights 0.375/0.375/0.25, weighted geometric mean, participant cluster bootstrap CI

Usage:
  python score_npy_rpfi.py --dataset pure --label PURE_label.npy --split splits/pure_cv5.json \
      --pred POS=PURE_POS_prediction.npy GREEN=PURE_GREEN_prediction.npy ... --out results_npy
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import rpfi_eval as R

ANCHORS = {"BWMD": (0.0006209498075351768, 0.902266222602872), "WCR": (0.0, 1.0), "EDD": (0.0, 1.0)}
WEIGHTS = {"BWMD": 0.375, "WCR": 0.375, "EDD": 0.25}
DIRECTION = {"BWMD": "lower", "WCR": "higher", "EDD": "lower"}


def normalize(name, v):
    lo, hi = ANCHORS[name]
    t = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    return 1.0 - t if DIRECTION[name] == "lower" else t


def rpfi_geometric(s, eps=1e-6):
    w = sum(WEIGHTS.values())
    return float(np.exp(sum(WEIGHTS[c] * np.log(max(s[c], eps)) for c in WEIGHTS) / w) * 100.0)


def recordings(split):
    recs = []
    for f in split["folds"]:
        recs += [(int(s), int(e), subj) for (s, e), subj in zip(f["test_ranges"], f["test_subject_of_range"])]
    recs.sort()
    starts = [r[0] for r in recs]
    assert len(starts) == len(set(starts)), "a recording appears in the test set of more than one fold"
    return recs


def score(label, pred, recs, fs, n_boot=2000, seed=42):
    seg = int(round(fs * 10.0))
    per = defaultdict(lambda: {"L": [], "P": [], "Le": [], "Pe": []})
    n_rec = defaultdict(int)
    for s, e, subj in recs:
        l, p = np.asarray(label[s:e], float), np.asarray(pred[s:e], float)
        n = (min(len(l), len(p)) // seg) * seg
        for c in range(0, n, seg):
            per[subj]["L"].append(R.preprocess_chunk(l[c:c + seg], fs, True, True, 100))
            per[subj]["P"].append(R.preprocess_chunk(p[c:c + seg], fs, True, True, 100))
            per[subj]["Le"].append(R.preprocess_chunk_edd(l[c:c + seg], fs, True, 100))
            per[subj]["Pe"].append(R.preprocess_chunk_edd(p[c:c + seg], fs, True, 100))
        n_rec[subj] += 1
    rows = []
    for subj, d in sorted(per.items()):
        L, P, Le, Pe = (np.array(d[k]) for k in ("L", "P", "Le", "Pe"))
        bwmd = float(np.mean([R.compute_bwmd(L[i], P[i], fs)[0] for i in range(len(L))]))
        wcr = R.compute_wcr(L.ravel(), P.ravel())[0]
        edd = R.compute_edd(Le.ravel(), Pe.ravel(), fs)
        comp = {"BWMD": bwmd, "WCR": wcr, "EDD": edd}
        s = {c: normalize(c, comp[c]) for c in comp}
        hl = np.array([R.hr_from_psd(x, fs) for x in L]); hp = np.array([R.hr_from_psd(x, fs) for x in P])
        m = ~(np.isnan(hl) | np.isnan(hp))
        rows.append(dict(participant=subj, n_recordings=n_rec[subj], n_chunks=len(L),
                         BWMD=bwmd, WCR=wcr, EDD=edd, RPFI=rpfi_geometric(s),
                         HR_MAE_bpm=float(np.mean(np.abs(hl[m] - hp[m]))) if m.any() else np.nan,
                         RMSE=float(np.sqrt(np.mean((P.ravel() - L.ravel()) ** 2)))))
    mean, lo, hi = R.cluster_bootstrap_ci([r["RPFI"] for r in rows], n_boot, seed=seed)
    summ = dict(n_participants=len(rows), n_chunks=sum(r["n_chunks"] for r in rows),
                RPFI=mean, RPFI_lo=lo, RPFI_hi=hi)
    for k in ("BWMD", "WCR", "EDD", "HR_MAE_bpm", "RMSE"):
        summ[k] = float(np.nanmean([r[k] for r in rows]))
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="NAME=path.npy ...")
    ap.add_argument("--out", default="results_npy")
    args = ap.parse_args()

    split = json.load(open(args.split))
    assert split["dataset"] == args.dataset, "dataset in split JSON does not match --dataset"
    fs = int(split["fs"])
    label = np.load(args.label).astype(float).ravel()
    assert len(label) == split["n_frames_total"], f"label length {len(label)} != split {split['n_frames_total']}"
    recs = recordings(split)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summary = []
    for item in args.pred:
        name, path = item.split("=", 1)
        pred = np.load(path).astype(float).ravel()
        assert len(pred) == len(label), f"{name}: length {len(pred)} != label {len(label)}"
        assert np.all(np.isfinite(pred)), f"{name}: contains NaN/inf"
        rows, summ = score(label, pred, recs, fs)
        summ = dict(dataset=args.dataset, method=name, **summ)
        summary.append(summ)
        with open(out / f"participant_{args.dataset}_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"{args.dataset:8s} {name:8s} N={summ['n_participants']:2d} chunks={summ['n_chunks']:4d} "
              f"RPFI={summ['RPFI']:6.2f} [{summ['RPFI_lo']:6.2f}, {summ['RPFI_hi']:6.2f}] "
              f"BWMD={summ['BWMD']:.3f} WCR={summ['WCR']:.3f} EDD={summ['EDD']:.3f} "
              f"HR-MAE={summ['HR_MAE_bpm']:.2f} RMSE={summ['RMSE']:.3f}", flush=True)
    with open(out / f"summary_{args.dataset}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)


if __name__ == "__main__":
    main()
