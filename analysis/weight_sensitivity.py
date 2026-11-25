"""Sensitivity of model rankings to the composite's weights and to internal parameters.

For each setting, RPFI is recomputed per participant (weighted geometric mean, exact zero kept),
averaged per model as in Table 1, and the model ranking is compared with the equal-weight ranking
(Kendall tau, and whether the top-ranked model is unchanged), separately for each of the 9 conditions.

Settings: Dirichlet-resampled weights (alpha = 1 and 10, 1,000 draws each), three single-component
up-weightings, beat-matching tolerance 0.10/0.20 x RR (default 0.15), and SHAPE gate threshold
0.05/0.20 (default 0.10).

Output: results/tables/weight_sensitivity.csv
Usage:  python analysis/weight_sensitivity.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import common as C

warnings.filterwarnings("ignore")
COMP = ["s_SHAPE", "s_WCR", "s_RHY"]


def main():
    v = pd.read_csv(C.RES / "scores" / "rpfi_v2_ALL.csv")
    s = pd.read_csv(C.RES / "robustness" / "sensitivity_ALL.csv")
    n = pd.read_csv(C.RES / "scores" / "newsys_participant_ALL.csv")
    v = v.merge(s[C.KEY + ["RHY_tol10", "RHY_tol20"]], on=C.KEY, validate="1:1")
    v = v.merge(n[C.KEY + ["sh_coh_pred", "sh_n_beats", "sh_r_vpg"]], on=C.KEY, validate="1:1")
    assert len(v) == 2250
    S = np.log(np.clip(v[COMP].values, 1e-300, None))
    Z = v[COMP].values <= 0

    def rpfi_w(w):
        r = np.exp(S @ w)
        r[(Z & (w > 0)).any(1)] = 0.0
        return r

    def geo(s3):
        return np.where((s3 > 0).all(1), np.exp(np.log(np.clip(s3, 1e-300, None)).mean(1)), 0.0)

    def model_mean(score):
        return v.assign(x=score).groupby(["condition", "model"]).x.mean()

    base = model_mean(rpfi_w(np.ones(3) / 3))

    def compare(score):
        m = model_mean(score)
        return {c: (kendalltau(base[c], m[c])[0], base[c].idxmax() == m[c].idxmax()) for c in C.ORDER}

    rows = []
    for alpha in (1.0, 10.0):
        W = np.random.default_rng(0).dirichlet(alpha * np.ones(3), size=1000)
        res = [compare(rpfi_w(w)) for w in W]
        for c in C.ORDER:
            t = np.array([r[c][0] for r in res])
            rows.append(dict(analysis=f"Dirichlet alpha={alpha:g}, 1000 draws", condition=c,
                             tau_median=np.median(t), tau_p05=np.percentile(t, 5),
                             top1_same=np.mean([r[c][1] for r in res])))
    for name, w in {"(0.5,0.25,0.25)": [.5, .25, .25], "(0.25,0.5,0.25)": [.25, .5, .25],
                    "(0.25,0.25,0.5)": [.25, .25, .5]}.items():
        r = compare(rpfi_w(np.array(w)))
        rows += [dict(analysis=f"weights {name} (SHAPE,WCR,RHY)", condition=c, tau_median=r[c][0],
                      top1_same=float(r[c][1])) for c in C.ORDER]
    for tol in ("RHY_tol10", "RHY_tol20"):
        r = compare(geo(v[["s_SHAPE", "s_WCR"]].assign(r=(1 - v[tol]).clip(0, 1)).values))
        rows += [dict(analysis=f"beat-matching tolerance 0.15 -> 0.{tol[-2:]} x RR", condition=c,
                      tau_median=r[c][0], top1_same=float(r[c][1])) for c in C.ORDER]
    for th in (0.05, 0.20):
        gated = (v.sh_coh_pred < np.maximum(th, 5 / v.sh_n_beats.clip(lower=1))) | (v.sh_n_beats < 3)
        r = compare(geo(np.column_stack([np.where(gated, 0.0, v.sh_r_vpg.clip(0, 1)), v.s_WCR, v.s_RHY])))
        rows += [dict(analysis=f"SHAPE gate threshold 0.10 -> {th:.2f}", condition=c, tau_median=r[c][0],
                      top1_same=float(r[c][1])) for c in C.ORDER]
    D = pd.DataFrame(rows)
    C.save(D, "weight_sensitivity.csv")
    pd.set_option("display.width", 250)
    print(D.pivot_table(index="analysis", columns="condition", values="tau_median", sort=False)[C.ORDER]
          .round(2).to_string())


if __name__ == "__main__":
    main()
