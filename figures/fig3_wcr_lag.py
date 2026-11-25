"""Figure 3 - WCR with lag alignment capped at a quarter cycle (synthetic pulse, release code).

(A) small constant lag (within the cap): aligned WCR recovers the agreement that the raw WCR loses
(B) half-cycle lag: looks like a polarity inversion; uncapped alignment would reward it, the cap does not,
    and the polarity flag is raised
(C) WCR against an imposed lag from 0 to one cardiac cycle
"""
import numpy as np
import matplotlib.pyplot as plt

from figstyle import COMP, PRED, REF, panel_label, save  # sets the import paths
import rpfi_eval as R
import rpfi_v2 as RV
from rhythm import rhythm_participant

FS, SEG = 30, 300
rng = np.random.default_rng(5)
t = np.arange(60 * FS) / FS
HR = 72.0
phase = np.cumsum(HR / 60 * (1 + 0.04 * np.sin(2 * np.pi * 0.25 * t))) / FS


def pulse(ph):
    x = ph % 1.0
    return np.exp(-0.5 * ((x - 0.18) / 0.06) ** 2) + 0.45 * np.exp(-0.5 * ((x - 0.47) / 0.09) ** 2)


ref = pulse(phase)
noise = 0.10 * ref.std() * rng.standard_normal(t.size)
RR = 60 / HR


def shifted(lag_s):
    return np.interp(t - lag_s, t, ref, period=None) + noise


def scores(pred):
    Lr, Pr = ref.reshape(-1, SEG), pred.reshape(-1, SEG)
    r = RV.score_chunks(list(Lr), list(Pr), FS)
    L = np.array([R.preprocess_chunk(c, FS) for c in Lr]); P = np.array([R.preprocess_chunk(c, FS) for c in Pr])
    lag = rhythm_participant(L, P, FS)["offset_s"]
    r["WCR_uncapped"] = RV._shift_trim_wcr(L, P, FS, lag)
    return r


fig = plt.figure(figsize=(9.5, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45, wspace=0.18)
tt = np.arange(SEG) / FS
m = (tt >= 2) & (tt <= 7)

for k, (lag, title) in enumerate([(0.10, "Lag of 100 ms (within the cap)"),
                                  (RR / 2, f"Lag of half a cycle ({RR / 2 * 1000:.0f} ms)")]):
    pred = shifted(lag)
    r = scores(pred)
    ax = fig.add_subplot(gs[0, k])
    ax.plot(tt[m], R.preprocess_chunk(ref[SEG:2 * SEG], FS)[m], color=REF, lw=1.6, label="Reference")
    ax.plot(tt[m], R.preprocess_chunk(pred[SEG:2 * SEG], FS)[m], color=PRED, lw=1.4, label="Prediction")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (a.u.)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.6 * (hi - lo))
    txt = (f"WCR raw {r['WCR_raw']:.2f} | uncapped {r['WCR_uncapped']:.2f} | capped {r['WCR_aligned']:.2f}\n"
           f"$\\Delta t$ = {r['lag_s'] * 1000:.0f} ms, polarity flag: {'yes' if r['polarity_suspect'] else 'no'}").replace("-", "\u2212")
    ax.text(0.01, 0.97, txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#999999", lw=0.6))
    if k == 1:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=2, frameon=False, fontsize=9)
    panel_label(ax, "AB"[k], title)

# (C) sweep of the imposed lag
lags = np.linspace(0, RR, 41)
rows = [scores(shifted(l)) for l in lags]
x = lags / RR
ax = fig.add_subplot(gs[1, :])
ax.axvspan(0.25, 0.75, color="#f0f0f0", zorder=0, lw=0)
ax.text(0.5, 0.62, "lag beyond the\nquarter-cycle cap", ha="center", va="top", fontsize=9.5,
        color="#555555")
ax.plot(x, [r["WCR_raw"] for r in rows], color="#999999", lw=1.8, ls="--", label="WCR raw (no alignment)")
ax.plot(x, [r["WCR_uncapped"] for r in rows], color=COMP["RHY"], lw=1.8, ls=":", label="aligned, uncapped")
ax.plot(x, [r["WCR_aligned"] for r in rows], color=COMP["WCR"], lw=2.4, label="aligned, capped at 0.25 RR (RPFI)")
flag = np.array([r["polarity_suspect"] for r in rows])
ax.plot(x[flag], np.full(flag.sum(), -0.95), "v", color="#D9534F", ms=6, label="polarity flag raised")
ax.axhline(0, color=GRID if (GRID := "#bbbbbb") else None, lw=0.8)
ax.set_xlim(0, 1); ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("Imposed lag (fraction of the cardiac cycle)"); ax.set_ylabel("WCR")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9.5)
panel_label(ax, "C", "WCR against an imposed lag")

save(fig, "fig3_wcr_lag")
