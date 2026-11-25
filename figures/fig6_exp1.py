"""Figure 6 - Experiment 1: what went wrong at the same CCC.

(A) three distortions from different families matched to CCC = 0.95 on one synthetic base (exp1_distortion_id.py)
(B) hold-out cases (P1 + P2): component deficits by distortion family (log scale)
(C) the same cases in conventional metrics
(D) balanced accuracy of the pre-registered classifier on the hold-out sets P1 and P2, with
    subject-cluster bootstrap 95% CIs (chance = 1/3)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score as BA

from figstyle import FAMILY, PRED, REF, panel_label, save  # sets the import paths
import exp1_confirm as EC
import exp1_distortion_id as E
import exp1_secondary as S
import rpfi_eval as R

# ---------------------------------------------------------------- (A) examples
E.N_CHUNKS = 4
fs = 30
n = 40 * fs
yfull = E.synth_base(np.random.default_rng(7), fs, 53.0)
ctx = E.make_ctx(yfull, fs, np.random.default_rng(8))
ref = E.Ref(yfull[:n], fs)
EXAMPLES = [("peak_broadening", "shape: systolic-peak broadening"),
            ("missed_beats", "local: missed beats"),
            ("hr_error", "drift: heart-rate error")]
ex = []
for name, title in EXAMPLES:
    fn, smax = E.DIST[name]
    s, c = E.match_strength(ref, yfull, fs, lambda yy, ff, ss, cc, fn=fn: fn(yy, ff, ss, cc)[:n], smax, ctx, 0.95)
    pred = fn(yfull, fs, s, ctx)[:n]
    f = E.features(ref, pred, yfull[:n])
    ex.append((title, pred, c, f))
    print(f"{name}: CCC={c:.3f} SHAPE={f['SHAPE']:.2f} WCR={f['WCR']:.2f} RHY={f['RHY']:.2f} HR_MAE={f['HR_MAE']:.2f}")

# ---------------------------------------------------------------- data for (B)-(D)
exd = {ds: S.load(ds, False) for ds in ("pure", "cohface")}
ho = {ds: S.load(ds, True) for ds in ("pure", "cohface")}
hc = ho["cohface"]
p1 = hc[~hc.subject.isin(set(exd["cohface"].subject)) & ~hc.subject.str.startswith("unknown")].reset_index(drop=True)
EVALS = {"P1 COHFACE\n(subject-disjoint)": (exd["cohface"], p1), "P2 PURE\n(recording-disjoint)": (exd["pure"], ho["pure"])}
SETS = {"RPFI profile": EC.RPFI, "conventional": EC.CONV, "SNR": ["SNR_dB"], "conventional + SNR": EC.CONV + ["SNR_dB"]}
SET_COLOR = {"RPFI profile": "#333333", "conventional": "#9e9e9e", "SNR": "#c7c7c7", "conventional + SNR": "#6f6f6f"}


def boot_ci(te, y, pred, seed=0, B=2000):
    rng = np.random.default_rng(seed)
    subj = te.subject.values; us = np.unique(subj); pos = {s: np.where(subj == s)[0] for s in us}
    v = []
    for _ in range(B):
        ii = np.concatenate([pos[s] for s in rng.choice(us, len(us))])
        if len(np.unique(y[ii])) > 1:
            v.append(BA(y[ii], pred[ii]))
    return np.percentile(v, [2.5, 97.5])


acc = []
for ename, (tr, te) in EVALS.items():
    y = te.g.values
    for sname, cols in SETS.items():
        p = EC.fit(tr, cols).predict(te[cols].values)
        lo, hi = boot_ci(te, y, p)
        acc.append(dict(eval=ename, set=sname, BA=BA(y, p), lo=lo, hi=hi))
acc = pd.DataFrame(acc)
print(acc.round(3).to_string())

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(9.5, 10.6))
gs = fig.add_gridspec(3, 6, height_ratios=[0.8, 1.15, 1.1], hspace=0.7, wspace=0.9)
tt = np.arange(n) / fs
m = (tt >= 10) & (tt <= 14)
for k, (title, pred, c, f) in enumerate(ex):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
    fam, nm = title.split(": ")
    ax.plot(tt[m] - 10, R.preprocess_chunk(yfull[10 * fs:20 * fs], fs)[:int(4 * fs) + 1], color=REF, lw=1.3)
    ax.plot(tt[m] - 10, R.preprocess_chunk(pred[10 * fs:20 * fs], fs)[:int(4 * fs) + 1], color=FAMILY[fam], lw=1.3)
    ax.set_yticks([]); ax.set_xlim(0, 4); ax.set_xlabel("Time (s)")
    print(f"{fam}: {nm}  CCC {c:.2f}; S/W/R {1 - f['SHAPE']:.2f} / {f['WCR']:.2f} / {1 - f['RHY']:.2f}")
    if k == 0:
        panel_label(ax, "A")

te = pd.concat([p1, ho["pure"]], ignore_index=True)
FLOOR = 1e-4
from matplotlib.patches import Patch
for j, (cols, labels, letter, title, ylab) in enumerate([
        ([lambda d: d.SHAPE, lambda d: d.RHY, lambda d: 1 - d.WCR], ["SHAPE", "RHY", "WCR"], "B",
         "RPFI profile", "deficit $1-s$ (log)"),
        ([lambda d: 1 - d.Pearson, lambda d: d.RMSE, lambda d: d.HR_MAE], ["$1-r$", "RMSE", "HR MAE"], "C",
         "Conventional metrics", "value (log)")]):
    ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
    fams = list(FAMILY)
    for gi, f in enumerate(cols):
        for fi, fam in enumerate(fams):
            v = np.clip(f(te[te.g == fam]).values, FLOOR, None)
            bp = ax.boxplot(v, positions=[gi + (fi - 1) * 0.27], widths=0.23, patch_artist=True, showfliers=False,
                            medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
            bp["boxes"][0].set(facecolor=FAMILY[fam], alpha=0.75, lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels); ax.set_ylabel(ylab)
    ax.set_xlim(-0.55, len(cols) - 0.45)
    if j == 0:
        ax.legend([Patch(color=FAMILY[f], alpha=0.75) for f in fams], [f"{f} family" for f in fams],
                  loc="lower left", bbox_to_anchor=(0.0, -0.32), frameon=False, fontsize=9, ncol=3)
    panel_label(ax, letter, title)

ax = fig.add_subplot(gs[2, :])
evals = list(EVALS); sets = list(SETS); w = 0.19
for i, sname in enumerate(sets):
    d = acc[acc.set == sname].set_index("eval").loc[evals]
    x = np.arange(len(evals)) + (i - 1.5) * w
    ax.bar(x, d.BA, width=w * 0.92, color=SET_COLOR[sname], label=sname, edgecolor="#333333", lw=0.6)
    ax.errorbar(x, d.BA, yerr=[d.BA - d.lo, d.hi - d.BA], fmt="none", ecolor="#111111", lw=1.0, capsize=2.5)
ax.axhline(1 / 3, color="#D9534F", lw=1.0, ls="--", label="chance")
ax.set_xticks(np.arange(len(evals))); ax.set_xticklabels(evals)
ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("Balanced accuracy")
ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=9, columnspacing=1.0, handlelength=1.5)
panel_label(ax, "D", "Family identification on hold-out recordings")

save(fig, "fig6_exp1")
