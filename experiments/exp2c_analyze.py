"""
exp2c_analyze.py - Experiment 2c analysis (pre-specified secondary analysis)
=========================================================================
Inputs
  exp2c_ihr/exp2c_ihr_ALL.csv                 (exp2c_extract.py, instantaneous-HR tracking error from pyPPG beats)
  newsys_recompute/newsys_participant_ALL.csv  (SHAPE, RHY, F1, nIBIE, WCR, RMSE)
  sensitivity_v2/sensitivity_ALL.csv           (WCR_aligned)
  --hr : participant CSV with the corrected HR MAE (columns: condition, model, participant, HR_MAE_bpm)
Criterion (C1): the lower bound of the participant-cluster bootstrap (B = 2000) 95% CI of the partial Spearman
          rho(RHY, HR-variation tracking error | 1-WCR_raw, 1-WCR_aligned, SHAPE, HR MAE, RMSE) > 0. Primary set = UBFC and PURE, within-dataset.
Output: exp2c_report.txt
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")
B, SEED = 2000, 0
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


def pcor(x, y, Z):
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    res = lambda v: rankdata(v) - Zr @ np.linalg.lstsq(Zr, rankdata(v), rcond=None)[0]
    return float(np.corrcoef(res(x), res(y))[0, 1])


def cboot(d, fn):
    rng = np.random.default_rng(SEED); g = d.key.values; us = np.unique(g); idx = {u: np.where(g == u)[0] for u in us}
    v = np.array([fn(d.iloc[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(B)])
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if v.size else np.array([np.nan, np.nan])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prv", default="exp2c_ihr/exp2c_ihr_ALL.csv")
    ap.add_argument("--newsys", default="newsys_recompute/newsys_participant_ALL.csv")
    ap.add_argument("--sens", default="sensitivity_v2/sensitivity_ALL.csv")
    ap.add_argument("--hr", default="hr_recompute/hr_participant_ALL.csv")
    a = ap.parse_args()
    for f in (a.prv, a.newsys, a.sens, a.hr):
        if not Path(f).exists():
            sys.exit(f"not found: {f}")
    k = ["condition", "model", "participant"]
    X = pd.read_csv(a.prv)
    N = pd.read_csv(a.newsys)[k + ["sh_SHAPE", "RHY", "F1", "nIBIE", "WCR_new", "RMSE"]]
    S = pd.read_csv(a.sens)[k + ["WCR_aligned"]]
    H = pd.read_csv(a.hr)[k + ["HR_MAE_bpm"]].rename(columns={"HR_MAE_bpm": "HR_MAE"})
    D = X.merge(N, on=k).merge(S, on=k).merge(H, on=k)
    D["key"] = D.dataset + ":" + D.participant.astype(str)
    D["PRV_ERR"] = D.ITE_bpm   # variable name shared with the 2b code (content = HR-variation tracking error, bpm)
    D["bWCR"], D["bWCRa"] = 1 - D.WCR_new, 1 - D.WCR_aligned
    ctrl = ["bWCR", "bWCRa", "sh_SHAPE", "HR_MAE", "RMSE"]
    pri_all = D[D.primary]
    pri = pri_all.dropna(subset=["PRV_ERR"] + ctrl).reset_index(drop=True)

    P("=" * 70 + "\nExperiment 2c - primary criterion C1 (UBFC and PURE, within-dataset)\n" + "=" * 70)
    P(f"  units {len(pri)} / {len(pri_all)} (excluded, e.g. common span < 20 s: {len(pri_all) - len(pri)}), participants {pri.key.nunique()}")
    b1 = lambda d: pcor(d.RHY.values, d.PRV_ERR.values, d[ctrl].values)
    r, ci = b1(pri), cboot(pri, b1)
    P(f"  C1 partial rho(RHY, HR tracking error | 1-WCR, 1-WCR_aligned, SHAPE, HR MAE, RMSE) = {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
      f"  -> {'met' if ci[0] > 0 else 'not met'}")

    P("\n" + "=" * 70 + "\nSecondary (reported only)\n" + "=" * 70)
    P("  [Spearman rho with the HR tracking error, primary set]")
    for c in ["RHY", "F1", "nIBIE", "bWCR", "bWCRa", "sh_SHAPE", "HR_MAE", "RMSE"]:
        f = lambda d, c=c: spearmanr(d[c], d.PRV_ERR)[0]
        ci_ = cboot(pri, f); P(f"    {c:9s} ρ = {f(pri):+.3f} [{ci_[0]:+.3f}, {ci_[1]:+.3f}]")
    for c in ["F1", "nIBIE"]:
        f = lambda d, c=c: pcor(d[c].values, d.PRV_ERR.values, d[ctrl].values)
        ci_ = cboot(pri, f); P(f"    sub-score {c} partial rho = {f(pri):+.3f} [{ci_[0]:+.3f}, {ci_[1]:+.3f}]")
    for name, sub in [("UBFC within", pri[pri.dataset == "ubfc"]), ("PURE within", pri[pri.dataset == "pure"]),
                      ("cross", D[~D.primary].dropna(subset=["PRV_ERR"] + ctrl).reset_index(drop=True))]:
        if len(sub) > 10:
            ci_ = cboot(sub.reset_index(drop=True), b1); P(f"    {name:8s} n={len(sub):4d}  C1 partial rho = {b1(sub):.3f} [{ci_[0]:.3f}, {ci_[1]:.3f}]")
    # sensitivity: units whose predicted value cannot be computed are included with the worst error
    w = pri_all.dropna(subset=ctrl).copy()
    worst = w.PRV_ERR.max()
    w["PRV_ERR"] = w.PRV_ERR.fillna(worst); w = w.reset_index(drop=True)
    P(f"  [sensitivity: {int(pri_all.PRV_ERR.isna().sum())} missing included as worst error] C1 partial rho = {b1(w):.3f}")
    P("  [missing rate by model]  " + str(pri_all.groupby("model").ITE_bpm.apply(lambda s: round(s.isna().mean(), 2)).to_dict()))
    rs = pri_all.drop_duplicates(["dataset", "participant"])
    P(f"  [median SD of the reference instantaneous HR] {rs.groupby('dataset').ref_HR_sd_bpm.median().round(2).to_dict()} bpm  | median tracking error {pri.PRV_ERR.median():.2f} bpm")
    Path("exp2c_report.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("\nsaved: exp2c_report.txt")


if __name__ == "__main__":
    main()
