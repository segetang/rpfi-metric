"""Figure 5 - Targeted degradations (synthetic battery, results/robustness/deg17_dense_raw.csv).

Left: reference (black) and the degraded signal at two levels, one 10 s excerpt.
Right: median component scores and RPFI across the 16 paired trajectories (2 frame rates x 8 repetitions),
with the interquartile range shaded. One row per component: each degradation mainly lowers the highlighted component.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figstyle import COMP, COMP_LABEL, REF, ROOT, panel_label, save  # sets the import paths
import deg17_dense as DD
import deg17_v2 as D
import rpfi_eval as R

# (name, title, highlighted component)
ROWS = [("peak_broadening", "Systolic-peak broadening", "SHAPE"),
        ("phase_shift", "Constant lag (up to 0.4 s)", "WCR"),
        ("spike_dispersed", "Dispersed spikes", "RHY")]
LEVELS = (0.5, 1.0)
raw = pd.read_csv(ROOT / "results" / "robustness" / "deg17_dense_raw.csv")

fs, rep = 30, 0
yb, hr = D.base(fs, np.random.default_rng(10_000 * fs + rep))
ref = D.win(yb, fs)
seg = slice(20 * fs, 30 * fs)
tt = np.arange(10 * fs) / fs

fig = plt.figure(figsize=(9.5, 8.2))
gs = fig.add_gridspec(len(ROWS), 2, width_ratios=[1.35, 1], hspace=0.62, wspace=0.25)
for k, (name, title, target) in enumerate(ROWS):
    di = list(DD.DEG).index(name)
    col = COMP[target]
    ax = fig.add_subplot(gs[k, 0])
    ax.plot(tt, R.preprocess_chunk(ref[seg], fs), color=REF, lw=1.5, label="Reference")
    for lv, alpha, lw in ((LEVELS[0], 0.45, 1.1), (LEVELS[1], 1.0, 1.3)):
        rng = np.random.default_rng(1_000_000 + 10_000 * fs + 100 * rep + di)
        pred = D.win(DD.DEG[name](yb.copy(), fs, lv, rng, hr), fs)
        ax.plot(tt, R.preprocess_chunk(pred[seg], fs), color=col, lw=lw, alpha=alpha, label=f"level {lv:g}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (a.u.)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.5 * (hi - lo))
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9, columnspacing=1.0, handlelength=1.6)
    panel_label(ax, "ABC"[k], title)

    ax = fig.add_subplot(gs[k, 1])
    g = raw[raw.degradation == name]
    for c in ("SHAPE", "WCR", "RHY", "RPFI"):
        colname = "RPFI" if c == "RPFI" else f"s_{c}"
        q = g.groupby("level")[colname].quantile([0.25, 0.5, 0.75]).unstack()
        lw = 2.4 if c == target else 1.4
        ax.plot(q.index, q[0.5], "o-", color=COMP[c], lw=lw, ms=3.5, label=COMP_LABEL[c],
                ls="--" if c == "RPFI" else "-")
        ax.fill_between(q.index, q[0.25], q[0.75], color=COMP[c], alpha=0.12, lw=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Degradation level"); ax.set_ylabel("Score")
    if k == 0:
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9.5)

save(fig, "fig5_degradation")
