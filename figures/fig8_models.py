"""Figure 8 - Model comparison within each dataset (results/tables/table1.csv).

For each dataset, models are ordered by RPFI (participant mean, bootstrap 95% CI, black). Coloured markers give the
component profile (s_SHAPE over participants with a defined SHAPE; s_WCR; s_RHY); the right-hand strip gives HR MAE.
Coverage (fraction of participants with a defined SHAPE) is printed next to each model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figstyle import COMP, COMP_LABEL, DS_NAME, MODEL_NAME, ROOT, save  # sets the import paths

t1 = pd.read_csv(ROOT / "results" / "tables" / "table1.csv")
fig = plt.figure(figsize=(8.6, 11.0))
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.42, wspace=0.05)
letters = "ABC"
for k, ds in enumerate(("ubfc", "pure", "cohface")):
    d = t1[t1.condition == ds].sort_values("RPFI").reset_index(drop=True)
    y = np.arange(len(d))
    ax = fig.add_subplot(gs[k, 0])
    for c, col, off, mk in (("s_SHAPE_covered", COMP["SHAPE"], 0.22, "s"), ("s_WCR", COMP["WCR"], 0.0, "D"),
                            ("s_RHY", COMP["RHY"], -0.22, "^")):
        ax.plot(d[c], y + off, mk, color=col, ms=5, alpha=0.9, lw=0,
                label=COMP_LABEL[{"s_SHAPE_covered": "SHAPE", "s_WCR": "WCR", "s_RHY": "RHY"}[c]])
    ax.errorbar(d.RPFI, y, xerr=[d.RPFI - d.RPFI_lo, d.RPFI_hi - d.RPFI], fmt="o", color="#111111", ms=6.5,
                capsize=2.5, lw=1.2, label="RPFI [95% CI]")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{MODEL_NAME.get(m, m)} ({100 * cov:g}%)" for m, cov in zip(d.model, d.coverage)], fontsize=9.5)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Score" if k == 2 else ""); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
    ax.set_title(f"({letters[k]})", fontsize=15, fontweight="bold", loc="left")
    if k == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=4, frameon=False, fontsize=9.5, columnspacing=1.0)
    axh = fig.add_subplot(gs[k, 1], sharey=ax)
    axh.barh(y, d.HR_MAE, color="#bdbdbd", edgecolor="#555555", lw=0.6, height=0.55)
    axh.tick_params(labelleft=False); axh.set_xlabel("HR MAE (bpm)" if k == 2 else "")
    axh.set_xlim(0, max(16, d.HR_MAE.max() * 1.1)); axh.set_xticks([5, 10, 15])
    axh.spines["left"].set_visible(False); axh.tick_params(axis="y", length=0)
save(fig, "fig8_models")
