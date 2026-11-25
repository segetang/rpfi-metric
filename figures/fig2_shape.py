"""Figure 2 - SHAPE (schematic on a synthetic pulse; all numbers computed with the release code).

(A) beat windows cut at the reference beats, after removing the constant lag
(B) phase-normalized beats and their mean templates
(C) first-derivative (VPG) templates, whose correlation gives s_SHAPE
Reference = two-Gaussian pulse (systolic peak, dicrotic notch, diastolic peak) with respiratory HR variation.
Prediction = the same beats with a blunted systolic peak and a flattened diastolic wave, additive noise and an 80 ms lag.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from figstyle import PRED, REF, panel_label, save  # sets the import paths
import rpfi_eval as R
import rpfi_v2 as RV
import shape_template as ST
from rhythm import detect_beats

FS, SEG = 30, 300
rng = np.random.default_rng(3)
t = np.arange(60 * FS) / FS
phase = np.cumsum(68 / 60 * (1 + 0.05 * np.sin(2 * np.pi * 0.25 * t))) / FS


def pulse(ph, w_sys=0.06, a_dia=0.45):
    x = ph % 1.0
    return np.exp(-0.5 * ((x - 0.18) / w_sys) ** 2) + a_dia * np.exp(-0.5 * ((x - 0.47) / 0.09) ** 2)


lag = 0.08
y = pulse(phase)
pred = np.interp(t - lag, t, pulse(phase, w_sys=0.10, a_dia=0.18))
pred = pred + 0.08 * pred.std() * rng.standard_normal(t.size)

Lr, Pr = y.reshape(-1, SEG), pred.reshape(-1, SEG)
res = RV.score_chunks(list(Lr), list(Pr), FS)
L = np.array([R.preprocess_chunk(c, FS) for c in Lr]); P = np.array([R.preprocess_chunk(c, FS) for c in Pr])
sh = ST.shape_participant(list(Lr), list(Pr), FS, L, P)
d = sh["offset_s"]
print(f"lag_s={res['lag_s']*1000:.0f} ms  s_SHAPE={res['s_SHAPE']:.3f}  r_vpg={sh['r_vpg']:.3f}  lag={d * 1000:.0f} ms  beats={sh['n_beats']}")

# re-extract the beat templates exactly as shape_template does (for plotting)
br = [detect_beats(x, FS) for x in L]
rr = float(np.median(np.concatenate([np.diff(q) for q in br if q.size > 1])))
k = np.arange(ST.N_PHASE) * (rr / ST.N_PHASE)
Tr, Tp = [], []
for lr_, pr_, bb in zip(Lr, Pr, br):
    lw, pw = ST.prep_wide(lr_, FS), ST.prep_wide(pr_, FS)
    tt = np.arange(lw.size) / FS
    cr, cp = CubicSpline(tt, lw), CubicSpline(tt, pw)
    for tb in bb:
        a_ = tb - 0.3 * rr + k
        if a_[0] < 0 or a_[0] + d < 0 or a_[-1] > tt[-1] or a_[-1] + d > tt[-1]:
            continue
        wr, wp = cr(a_), cp(a_ + d)
        Tr.append((wr - wr.mean()) / wr.std()); Tp.append((wp - wp.mean()) / wp.std())
Tr, Tp = np.array(Tr), np.array(Tp)
ph = np.arange(ST.N_PHASE) / ST.N_PHASE

fig = plt.figure(figsize=(4.6, 6.6))
gs = fig.add_gridspec(3, 1, hspace=0.85)

# (A) beat windows on a 6 s excerpt
ax = fig.add_subplot(gs[0, 0])
lw0, pw0 = ST.prep_wide(Lr[1], FS), ST.prep_wide(Pr[1], FS)
tt = np.arange(SEG) / FS
m = tt <= 6
ax.plot(tt[m], lw0[m], color=REF, lw=1.6, label="Reference")
ax.plot(tt[m], pw0[m], color=PRED, lw=1.4, label="Prediction")
for tb in br[1]:
    if tb - 0.3 * rr >= 0 and tb + 0.7 * rr <= 6:
        ax.axvspan(tb - 0.3 * rr, tb + 0.7 * rr, color="#e8e8e8", zorder=0, lw=0)
        ax.axvline(tb - 0.3 * rr, color="#bbbbbb", lw=0.7, zorder=0)
ax.set_xlabel("Time (s)"); ax.set_yticks([])
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9, borderaxespad=0.1, handlelength=1.4, columnspacing=1.0)
panel_label(ax, "A", "Beat windows")

# (B) phase-normalized beats and mean templates
ax = fig.add_subplot(gs[1, 0])
for w in Tr[::2]:
    ax.plot(ph, w, color=REF, lw=0.5, alpha=0.12)
for w in Tp[::2]:
    ax.plot(ph, w, color=PRED, lw=0.5, alpha=0.12)
ax.plot(ph, Tr.mean(0), color=REF, lw=2.0, label="Reference template")
ax.plot(ph, Tp.mean(0), color=PRED, lw=2.0, label="Prediction template")
ax.set_xlabel("Beat phase ($-0.3$ to $+0.7$ RR around the peak)")
ax.set_yticks([]); ax.set_xlim(0, 1)
ax.text(0.01, 0.96, f"{len(Tr)} beats, coherence {sh['coh_pred']:.2f}", transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#999999", lw=0.6))
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.3 * (hi - lo))
panel_label(ax, "B", "Beat templates")

# (C) VPG templates
ax = fig.add_subplot(gs[2, 0])
vr, vp = np.diff(Tr.mean(0)), np.diff(Tp.mean(0))
ax.axhline(0, color="#bbbbbb", lw=0.8)
ax.plot(ph[1:], vr / np.abs(vr).max(), color=REF, lw=2.0, label="Reference VPG")
ax.plot(ph[1:], vp / np.abs(vp).max(), color=PRED, lw=2.0, label="Prediction VPG")
ax.set_xlabel("Beat phase"); ax.set_yticks([]); ax.set_xlim(0, 1)
ax.text(0.01, 0.96, f"$r_{{\\mathrm{{VPG}}}}$ = {sh['r_vpg']:.2f}   $\\rightarrow$   $s_{{\\mathrm{{SHAPE}}}}$ = {res['s_SHAPE']:.2f}",
        transform=ax.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round", fc="white", ec="#999999", lw=0.6))
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.3 * (hi - lo))
panel_label(ax, "C", "VPG templates")

save(fig, "fig2_shape", width="col")
