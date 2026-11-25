"""Figure 5 - Targeted degradations (synthetic battery, results/robustness/deg17_dense_raw.csv).

Left: reference (black) and the degraded signal at two levels, one 10 s excerpt.
Right: median component scores and RPFI across the 16 paired trajectories (2 frame rates x 8 repetitions),
with the interquartile range shaded.
Row D shows the heart-rate bias, which compresses time by 1 + 0.5 x level from the window start. Its left panel shows
the timing offset that accumulates over the 60 s window, against the Rhythm tolerance and the WCR lag cap of this
example's RR interval; its right panel uses the heart-rate error in percent (levels 0-0.1) as the x-axis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figstyle import COMP, COMP_LABEL, REF, ROOT, panel_label, save  # sets the import paths
import deg17_dense as DD
import deg17_v2 as D
import rpfi_eval as R

# (name, title, highlighted component, left-panel levels, last level shown on the right, x scale for the right panel)
ROWS = [("peak_broadening", "Systolic-peak broadening", "SHAPE", (0.5, 1.0), 1.0, 1.0),
        ("phase_shift", "Constant lag (up to 0.4 s)", "WCR", (0.5, 1.0), 1.0, 1.0),
        ("spike_dispersed", "Dispersed spikes", "RHY", (0.5, 1.0), 1.0, 1.0),
        ("hr_bias", "Heart-rate bias", "RPFI", (0.01, 0.02), 0.1, 50.0)]
raw = pd.read_csv(ROOT / "results" / "robustness" / "deg17_dense_raw.csv")

fs, rep = 30, 0
yb, hr = D.base(fs, np.random.default_rng(10_000 * fs + rep))
ref = D.win(yb, fs)
seg = slice(20 * fs, 30 * fs)
tt = np.arange(10 * fs) / fs

fig = plt.figure(figsize=(9.5, 11.0))
gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1], hspace=0.62, wspace=0.25)
for k, (name, title, target, lvs, lv_max, xscale) in enumerate(ROWS):
    di = list(DD.DEG).index(name)
    col = COMP[target]
    ax = fig.add_subplot(gs[k, 0])
    if name == "hr_bias":
        # the waveforms only look like a lag here, so show how the beat-timing offset accumulates instead
        rr = 60.0 / hr
        tw = np.linspace(0, 60, 601)
        for lv, alpha, lw in ((lvs[0], 0.55, 1.6), (lvs[1], 1.0, 2.0)):
            ax.plot(tw, 0.5 * lv * tw, color="#555555", alpha=alpha, lw=lw, label=f"HR +{50 * lv:g}%")
        for frac, txt, ls in ((0.25, "WCR lag cap (0.25 RR)", "--"), (0.15, "Rhythm tolerance (0.15 RR)", ":")):
            ax.axhline(frac * rr, color="#999999", lw=1.2, ls=ls, label=txt)
        ax.set_xlim(0, 60); ax.set_ylim(0, 1.0)
        ax.set_xlabel("Time in the 60 s window (s)"); ax.set_ylabel("Timing offset (s)")
        ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9, columnspacing=1.2, handlelength=1.8)
        panel_label(ax, "ABCD"[k], title)
    else:
        ax.plot(tt, R.preprocess_chunk(ref[seg], fs), color=REF, lw=1.5, label="Reference")
        for lv, alpha, lw in ((lvs[0], 0.45, 1.1), (lvs[1], 1.0, 1.3)):
            rng = np.random.default_rng(1_000_000 + 10_000 * fs + 100 * rep + di)
            pred = D.win(DD.DEG[name](yb.copy(), fs, lv, rng, hr), fs)
            ax.plot(tt, R.preprocess_chunk(pred[seg], fs), color=col, lw=lw, alpha=alpha, label=f"level {lv:g}")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (a.u.)")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.5 * (hi - lo))
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9, columnspacing=1.0, handlelength=1.6)
        panel_label(ax, "ABCD"[k], title)

    ax = fig.add_subplot(gs[k, 1])
    g = raw[(raw.degradation == name) & (raw.level <= lv_max + 1e-9)]
    for c in ("SHAPE", "WCR", "RHY", "RPFI"):
        colname = "RPFI" if c == "RPFI" else f"s_{c}"
        q = g.groupby("level")[colname].quantile([0.25, 0.5, 0.75]).unstack()
        lw = 2.4 if c == target else 1.4
        xs = q.index * xscale
        ax.plot(xs, q[0.5], "o-", color=COMP[c], lw=lw, ms=3.5, label=COMP_LABEL[c],
                ls="--" if c == "RPFI" else "-")
        ax.fill_between(xs, q[0.25], q[0.75], color=COMP[c], alpha=0.12, lw=0)
    ax.set_xlim(-0.02 * lv_max * xscale, 1.02 * lv_max * xscale); ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Heart-rate error (%)" if name == "hr_bias" else "Degradation level"); ax.set_ylabel("Score")
    if k == 0:
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9.5)

save(fig, "fig5_degradation")
