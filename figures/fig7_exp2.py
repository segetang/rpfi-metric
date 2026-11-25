"""Figure 7 - External validity against a third-party fiducial tool (pyPPG), UBFC and PURE within-dataset units.

(A) SHAPE deficit vs. pyPPG morphology error (Experiment 2)
(B) Spearman correlation of each metric with the morphology error, and the pre-registered partial correlations
(C) rhythm deficit vs. instantaneous-HR tracking error (Experiment 2c)
(D) the same for the tracking error
Error bars: participant-cluster bootstrap 95% CI (2,000 resamples). Metrics are oriented so that larger = worse.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from figstyle import COMP, DATASET, DS_NAME, panel_label, save  # sets the import paths
import common as C
import exp2_analyze as A
import exp2c_analyze as AC
import exp2_snr as X

snr = __import__("pandas").read_csv(C.RES / "scores" / "snr_real.csv")[C.KEY + ["SNR_dB"]]
mu, feats = X.morph_units()
mu = mu.merge(snr, on=C.KEY, validate="1:1")
mu = mu.merge(__import__("pandas").read_csv(C.RES / "scores" / "newsys_participant_ALL.csv")[C.KEY + ["RHY", "WCR_new"]],
              on=C.KEY, validate="1:1")
iu, ctrl = X.ihr_units()
iu = iu.merge(snr, on=C.KEY, validate="1:1")


def corr_bars(d, err, metrics, boot, partial):
    out = []
    for lab, f, col in metrics:
        fn = lambda x, f=f: spearmanr(f(x), x[err])[0]
        ci = boot(d, fn)
        out.append((lab, fn(d), ci, col, False))
    for lab, fn in partial:
        ci = boot(d, fn)
        out.append((lab, fn(d), ci, "#333333", True))
    return out


def draw_bars(ax, rows):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, col, part) in zip(y, rows):
        ax.barh(yi, v, color=col, alpha=0.85 if not part else 0.35, edgecolor="#333333", lw=0.6,
                hatch="///" if part else None, height=0.7)
        ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#111111", lw=1.0, capsize=2.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Spearman $\\rho$ with the external error")


GRAY = "#9e9e9e"
fig = plt.figure(figsize=(9.5, 8.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.38, wspace=0.55)

# (A)
ax = fig.add_subplot(gs[0, 0])
for ds in ("ubfc", "pure"):
    d = mu[mu.dataset == ds]
    ax.scatter(d.sh_SHAPE, d.MORPH_ERR, s=12, alpha=0.55, lw=0, color=DATASET[ds], label=DS_NAME[ds])
r = spearmanr(mu.sh_SHAPE, mu.MORPH_ERR)[0]
ax.set_xlabel("$1-s_{\\mathrm{SHAPE}}$ (gated units = 1)"); ax.set_ylabel("Morphology error (pyPPG, SD units)")
ax.legend(loc="upper left", frameon=False, markerscale=1.8)
print(f"Exp2: {len(mu)} units, {mu.participant_key.nunique()} participants, rho {r:.3f}")
panel_label(ax, "A", f"Morphology, $\\rho$ = {r:.2f}")

# (B)
ax = fig.add_subplot(gs[0, 1])
rows = corr_bars(mu, "MORPH_ERR", [
    ("$1-s_{\\mathrm{SHAPE}}$", lambda x: x.sh_SHAPE, COMP["SHAPE"]),
    ("$1-$WCR", lambda x: 1 - x.WCR_new, COMP["WCR"]),
    ("$1-s_{\\mathrm{RHY}}$", lambda x: x.RHY, COMP["RHY"]),
    ("$1-$Pearson $r$", lambda x: 1 - x.Pearson, GRAY),
    ("RMSE", lambda x: x.RMSE, GRAY),
    ("HR MAE", lambda x: x.HR_MAE_bpm, GRAY),
    ("$-$SNR", lambda x: -x.SNR_dB, GRAY)], A.cboot, [
    ("SHAPE | conv.", lambda d: A.partial_spearman(d.sh_SHAPE.values, d.MORPH_ERR.values, d[A.CONV].values)),
    ("SHAPE | conv. + SNR", lambda d: A.partial_spearman(d.sh_SHAPE.values, d.MORPH_ERR.values,
                                                       d[A.CONV + ["SNR_dB"]].values))])
draw_bars(ax, rows)
panel_label(ax, "B", "Morphology error")

# (C)
ax = fig.add_subplot(gs[1, 0])
for ds in ("ubfc", "pure"):
    d = iu[iu.dataset == ds]
    ax.scatter(d.RHY, d.PRV_ERR, s=12, alpha=0.55, lw=0, color=DATASET[ds], label=DS_NAME[ds])
r = spearmanr(iu.RHY, iu.PRV_ERR)[0]
ax.set_xlabel("$1-s_{\\mathrm{RHY}}$"); ax.set_ylabel("Instantaneous-HR tracking error (bpm)")
ax.legend(loc="upper left", frameon=False, markerscale=1.8)
print(f"Exp2c: {len(iu)} units, {iu.key.nunique()} participants, rho {r:.3f}")
panel_label(ax, "C", f"Beat timing, $\\rho$ = {r:.2f}")

# (D)
ax = fig.add_subplot(gs[1, 1])
rows = corr_bars(iu, "PRV_ERR", [
    ("$1-s_{\\mathrm{RHY}}$", lambda x: x.RHY, COMP["RHY"]),
    ("$1-$F1", lambda x: 1 - x.F1, COMP["RHY"]),
    ("nIBIE", lambda x: x.nIBIE, COMP["RHY"]),
    ("$1-$WCR (aligned)", lambda x: x.bWCRa, COMP["WCR"]),
    ("$1-s_{\\mathrm{SHAPE}}$", lambda x: x.sh_SHAPE, COMP["SHAPE"]),
    ("RMSE", lambda x: x.RMSE, GRAY),
    ("HR MAE", lambda x: x.HR_MAE, GRAY),
    ("$-$SNR", lambda x: -x.SNR_dB, GRAY)], AC.cboot, [
    ("RHY | ctrl.", lambda d: AC.pcor(d.RHY.values, d.PRV_ERR.values, d[ctrl].values)),
    ("RHY | ctrl. + SNR", lambda d: AC.pcor(d.RHY.values, d.PRV_ERR.values, d[ctrl + ["SNR_dB"]].values))])
draw_bars(ax, rows)
panel_label(ax, "D", "HR tracking error")

save(fig, "fig7_exp2")
