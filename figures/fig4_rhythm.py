"""Figure 4 - Rhythm component (synthetic pulse; matching done with the release code).

(A) beats detected on reference and prediction (one 10 s chunk), 1:1 matching within +/- 0.15 RR:
    matched pairs, a missed beat (FN) and a spurious beat (FP)
(B) inter-beat intervals of matched consecutive pairs (reference vs. prediction) over the recording
Prediction = reference with beat-to-beat timing jitter, one missed beat and one spurious beat per chunk.
"""
import numpy as np
import matplotlib.pyplot as plt

from figstyle import COMP, PRED, REF, panel_label, save  # sets the import paths
import rpfi_eval as R
import rpfi_v2 as RV
import rhythm as RH

FS, SEG = 30, 300
rng = np.random.default_rng(11)
t = np.arange(60 * FS) / FS
HR = 70.0
inst = HR / 60 * (1 + 0.05 * np.sin(2 * np.pi * 0.25 * t))
phase = np.cumsum(inst) / FS


def pulse(ph):
    x = ph % 1.0
    return np.exp(-0.5 * ((x - 0.18) / 0.06) ** 2) + 0.45 * np.exp(-0.5 * ((x - 0.47) / 0.09) ** 2)


ref = pulse(phase)
# prediction: edits on the clean pulse (one missed and one spurious beat per 10 s chunk), then beat-wise timing jitter
base = ref.copy()
starts = np.where(np.diff(np.floor(phase)) > 0)[0] + 1          # beat boundaries (pulse at baseline)
for c in range(6):
    sb = starts[(t[starts] > c * 10 + 2.5) & (t[starts] < c * 10 + 9)]
    a, b = sb[0], sb[1]
    base[a:b] = np.linspace(base[a], base[b], b - a)            # missed beat
    tb = t[sb[3]] + 0.62 * 60 / HR
    base += 0.9 * np.exp(-0.5 * ((t - tb) / 0.05) ** 2)          # spurious beat
nb = int(phase[-1]) + 2
jit = rng.normal(0, 0.03, nb)                                    # seconds, one offset per beat
pred = np.interp(t - jit[np.floor(phase).astype(int)], t, base)
pred += 0.05 * rng.standard_normal(t.size)

Lr, Pr = ref.reshape(-1, SEG), pred.reshape(-1, SEG)
res = RV.score_chunks(list(Lr), list(Pr), FS)
L = np.array([R.preprocess_chunk(c, FS) for c in Lr]); P = np.array([R.preprocess_chunk(c, FS) for c in Pr])
rh = RH.rhythm_participant(L, P, FS)
d, rr = rh["offset_s"], rh["rr_med_s"]
tol = RH.TOL_FRAC * rr
print(f"F1={res['F1']:.3f} nIBIE={res['nIBIE']:.3f} s_RHY={res['s_RHY']:.3f} TP={rh['TP']} FP={rh['FP']} FN={rh['FN']}")

fig = plt.figure(figsize=(9.5, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.45)

# (A) one chunk
edge = lambda x: (x >= tol) & (x <= 10 - tol)                    # beats near chunk edges are not scored


def chunk_match(c):
    r_b, p_b = RH.detect_beats(L[c], FS), RH.detect_beats(P[c], FS)
    pairs = RH._greedy_match(r_b, p_b - d, tol)
    mr, mp = {i for i, _ in pairs}, {j for _, j in pairs}
    fn = [i for i in range(len(r_b)) if i not in mr and edge(r_b[i])]
    fp = [j for j in range(len(p_b)) if j not in mp and edge(p_b[j] - d)]
    return r_b, p_b, pairs, mr, mp, fn, fp


c = max(range(len(L)), key=lambda c: (min(len(chunk_match(c)[5]), 1) + min(len(chunk_match(c)[6]), 1)))
r_b, p_b, pairs, mr, mp, fn, fp = chunk_match(c)
print("chunk", c, "FN", len(fn), "FP", len(fp))
tt = np.arange(SEG) / FS
off = 3.2
ax = fig.add_subplot(gs[0, 0])
for tb in r_b:
    ax.axvspan(tb - tol, tb + tol, color="#ececec", lw=0, zorder=0)
ax.plot(tt, L[c] + off, color=REF, lw=1.5, label="Reference")
ax.plot(tt, P[c], color=PRED, lw=1.4, label="Prediction")
yr, yp = np.interp(r_b, tt, L[c]) + off, np.interp(p_b, tt, P[c])
for i, j in pairs:
    ax.plot([r_b[i], p_b[j]], [yr[i], yp[j]], color="#888888", lw=0.9, zorder=1)
ax.plot(r_b[list(mr)], yr[list(mr)], "o", color=REF, ms=5, zorder=3)
ax.plot(p_b[list(mp)], yp[list(mp)], "o", color=PRED, ms=5, zorder=3)
ax.plot(r_b[fn], yr[fn], "x", color="#D9534F", ms=10, mew=2.2, zorder=4, label="missed beat (FN)")
ax.plot(p_b[fp], yp[fp], "o", mfc="none", mec="#D9534F", ms=11, mew=2.0, zorder=4, label="spurious beat (FP)")
ax.set_yticks([]); ax.set_xlabel("Time (s)"); ax.set_xlim(0, 10)
from matplotlib.patches import Patch
h, l = ax.get_legend_handles_labels()
h.append(Patch(color="#ececec")); l.append("$\\pm$0.15 RR tolerance")
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.38 * (hi - lo))
ax.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9.5)
panel_label(ax, "A", "Beat detection and 1:1 matching")

# (B) IBIs of matched consecutive pairs, whole recording
ri, pi_ = [], []
tcur = 0.0
for cc in range(len(L)):
    rb, pb = RH.detect_beats(L[cc], FS), RH.detect_beats(P[cc], FS) - d
    pm = dict(RH._greedy_match(rb, pb, tol))
    for i in range(len(rb) - 1):
        if i in pm and (i + 1) in pm and pm[i + 1] == pm[i] + 1:
            ri.append((cc * 10 + rb[i + 1], rb[i + 1] - rb[i], pb[pm[i + 1]] - pb[pm[i]]))
ri = np.array(ri)
ax = fig.add_subplot(gs[1, 0])
ax.plot(ri[:, 0], ri[:, 1] * 1000, "o-", color=REF, ms=3.5, lw=1.2, label="Reference IBI")
ax.plot(ri[:, 0], ri[:, 2] * 1000, "o-", color=PRED, ms=3.5, lw=1.2, label="Prediction IBI")
for x0 in range(10, 60, 10):
    ax.axvline(x0, color="#dddddd", lw=0.8, zorder=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Inter-beat interval (ms)"); ax.set_xlim(0, 60)
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.35 * (hi - lo))
ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9.5)
ax.text(0.01, 0.97, f"F1 = {res['F1']:.2f}, nIBIE = {res['nIBIE']:.3f}  $\\rightarrow$  $s_{{\\mathrm{{RHY}}}}$ = {res['s_RHY']:.2f}",
        transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="#999999", lw=0.6))
panel_label(ax, "B", "Inter-beat intervals of matched consecutive beats")

save(fig, "fig4_rhythm")
