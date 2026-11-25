"""Figure 9 - Real examples with similar HR error but different RPFI (UBFC-rPPG; needs the labels and model outputs).

Run from the working folder that holds runs/ (see experiments/README.md):
    python <repo>/figures/fig9_examples.py --label-dir "../real data"
Each row: one participant-model pair. Left: a 10 s excerpt of the preprocessed reference (black) and prediction
(red), as scored. Right: the lag-corrected mean beat templates used by SHAPE.
Scores are taken from results/scores/rpfi_v2_ALL.csv (release scorer).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from figstyle import MODEL_NAME, PRED2, REF, ROOT, save  # sets the import paths
import rpfi_eval as R
import shape_template as ST
from rhythm import detect_beats

PAIRS = [("ubfc", "ubfc_S08", "bilstm", "physdiff"), ("ubfc", "ubfc_S25", "bimamba", "lstm")]
LABEL = {"ubfc": "UBFC_label.npy", "pure": "PURE_label.npy", "cohface": "COHFACE_label.npy"}


def recordings(runs, model, subj, label):
    out = []
    for inf in sorted(Path(runs).glob(f"{model}_f*_info.json")):
        info = json.load(open(inf))
        pr = np.load(Path(runs) / f"{model}_f{info['fold']}_test_pred.npy", allow_pickle=True)
        for i, ((s, e), sj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
            if sj == subj:
                p = np.asarray(pr[i], float).ravel(); l = np.asarray(label[s:e], float).ravel()
                n = min(len(p), len(l)); out.append((l[:n], p[:n]))
    if not out:
        raise SystemExit(f"{model}/{subj} not found in {runs}")
    return out


def chunks(recs, fs):
    seg = 10 * fs; Lr, Pr = [], []
    for l, p in recs:
        k = len(l) // seg
        Lr += [l[i * seg:(i + 1) * seg] for i in range(k)]; Pr += [p[i * seg:(i + 1) * seg] for i in range(k)]
    return Lr, Pr


def templates(Lr, Pr, fs):
    """Mean beat templates exactly as SHAPE builds them (lag from shape_participant)."""
    L = np.array([R.preprocess_chunk(c, fs) for c in Lr]); P = np.array([R.preprocess_chunk(c, fs) for c in Pr])
    sh = ST.shape_participant(Lr, Pr, fs, L, P)
    d = sh["offset_s"]
    br = [detect_beats(x, fs) for x in L]
    rr = float(np.median(np.concatenate([np.diff(b) for b in br if b.size > 1])))
    k = np.arange(ST.N_PHASE) * (rr / ST.N_PHASE)
    Tr, Tp = [], []
    for lr_, pr_, bb in zip(Lr, Pr, br):
        lw, pw = ST.prep_wide(lr_, fs), ST.prep_wide(pr_, fs)
        tt = np.arange(lw.size) / fs
        cr, cp = CubicSpline(tt, lw), CubicSpline(tt, pw)
        for tb in bb:
            a = tb - 0.3 * rr + k
            if a[0] < 0 or a[0] + d < 0 or a[-1] > tt[-1] or a[-1] + d > tt[-1]:
                continue
            wr, wp = cr(a), cp(a + d)
            if wr.std() > 1e-8 and wp.std() > 1e-8:
                Tr.append((wr - wr.mean()) / wr.std()); Tp.append((wp - wp.mean()) / wp.std())
    return np.array(Tr).mean(0), np.array(Tp).mean(0), sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--scores", default=str(ROOT / "results" / "scores" / "rpfi_v2_ALL.csv"))
    ap.add_argument("--chunk", type=int, default=2, help="which 10 s chunk to show")
    a = ap.parse_args()
    V = pd.read_csv(a.scores).set_index(["condition", "model", "participant"])
    rows = [(ds, subj, m) for ds, subj, g, p in PAIRS for m in (g, p)]
    fig = plt.figure(figsize=(9.5, 2.3 * len(rows)))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[3.2, 1], hspace=0.75, wspace=0.12)
    for k, (ds, subj, m) in enumerate(rows):
        fs = R.FS_BY_DATASET[ds]
        lab = np.load(Path(a.label_dir) / LABEL[ds])
        Lr, Pr = chunks(recordings(Path(a.runs_dir) / f"{ds}_cv5", m, subj, lab), fs)
        r = V.loc[(ds, m, subj)]
        c = min(a.chunk, len(Lr) - 1)
        t = np.arange(10 * fs) / fs
        ax = fig.add_subplot(gs[k, 0])
        ax.plot(t, R.preprocess_chunk(Lr[c], fs), color=REF, lw=1.5, label="Reference")
        ax.plot(t, R.preprocess_chunk(Pr[c], fs), color=PRED2, lw=1.3, label="Prediction")
        ax.set_xlim(0, 10); ax.set_yticks([])
        ax.set_title(f"({'ABCD'[k]})", fontsize=15,
                     loc="left", fontweight="bold")
        ax.set_title(f"HR MAE {r.HR_MAE_bpm:.2f} bpm   RPFI {r.RPFI:.2f}", fontsize=11, loc="right")
        if k == len(rows) - 1:
            ax.set_xlabel("Time (s)")
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.16), ncol=2, frameon=False)
        tr, tp, sh = templates(Lr, Pr, fs)
        ax = fig.add_subplot(gs[k, 1])
        ph = np.arange(ST.N_PHASE) / ST.N_PHASE
        ax.plot(ph, tr, color=REF, lw=1.8); ax.plot(ph, tp, color=PRED2, lw=1.6)
        ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
        ax.set_title(f"S {r.s_SHAPE:.2f}  W {r.s_WCR:.2f}  R {r.s_RHY:.2f}", fontsize=10, loc="right")
        if k == len(rows) - 1:
            ax.set_xlabel("Beat phase")
    save(fig, "fig9_examples")


if __name__ == "__main__":
    main()
