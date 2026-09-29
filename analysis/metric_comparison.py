"""
metric_comparison.py - analyses on the released participant-level CSVs (Supplementary Sec. S6, Table S7 bands).

Outputs (in --out):
  component_ci.csv          cluster-bootstrap 95% CI (2,000 resamples, seed 42) of BWMD/WCR/EDD/RPFI per model (Table S10)
  metric_sig_pairs.csv      Friedman + Holm-corrected Wilcoxon: significant model pairs per metric and dataset (Table S11)
  hrmae_assoc_comparison.csv pooled Spearman correlation with HR MAE and participant-cluster bootstrap CI of
                            |rho_RPFI| - |rho_metric| (Table S11)
  rpfi_bands.csv            HR MAE and component medians per 20-point RPFI band (Table S7)

Usage (from the repository root):
  python analysis/metric_comparison.py --results results --out results/metric_comparison
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import rpfi_eval as R  # noqa: E402

METRICS = ["RPFI", "RMSE", "CCC", "Spearman", "PSNR_dB", "HR_MAE_bpm"]


def holm_count(g, col, alpha=0.05):
    W = g.pivot(index="participant", columns="model", values=col).dropna()
    fr = stats.friedmanchisquare(*[W[c] for c in W.columns]).pvalue
    ps = []
    for a, b in itertools.combinations(W.columns, 2):
        ps.append(stats.wilcoxon(W[a], W[b]).pvalue if np.any(W[a] != W[b]) else 1.0)
    ps = np.array(ps)
    n_sig = 0
    for k, i in enumerate(np.argsort(ps)):
        if ps[i] <= alpha / (len(ps) - k):
            n_sig += 1
        else:
            break
    return fr, (n_sig if fr < alpha else 0), len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/metric_comparison")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(Path(args.results) / f"rpfi_participant_{d}_geometric.csv")
                    for d in ("pure", "ubfc", "cohface")], ignore_index=True)

    rows = []
    for (ds, m), g in df.groupby(["dataset", "model"]):
        r = dict(dataset=ds, model=m, N=len(g))
        for c in ("BWMD", "WCR", "EDD", "RPFI"):
            r[c], r[c + "_lo"], r[c + "_hi"] = R.cluster_bootstrap_ci(g[c].values, args.n_boot, seed=args.seed)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "component_ci.csv", index=False)

    sc = []
    for ds, g in df.groupby("dataset"):
        for c in METRICS:
            fr, n, m = holm_count(g, c)
            sc.append(dict(dataset=ds, metric=c, friedman_p=fr, sig_pairs=n, pairs=m))
    pd.DataFrame(sc).to_csv(out / "metric_sig_pairs.csv", index=False)

    rho = lambda x, y: stats.spearmanr(x, y)[0]
    df["cluster"] = df.dataset + "_" + df.participant.astype(str)
    others = ["RMSE", "CCC", "Spearman", "PSNR_dB"]
    base = {c: rho(df[c], df.HR_MAE_bpm) for c in ["RPFI"] + others}
    rng = np.random.default_rng(args.seed)
    cl = df.cluster.unique(); idx = {c: np.where(df.cluster.values == c)[0] for c in cl}
    diffs = {c: [] for c in others}
    for _ in range(args.n_boot):
        s = df.iloc[np.concatenate([idx[c] for c in rng.choice(cl, len(cl), replace=True)])]
        r0 = abs(rho(s.RPFI, s.HR_MAE_bpm))
        for c in others:
            diffs[c].append(r0 - abs(rho(s[c], s.HR_MAE_bpm)))
    pd.DataFrame([dict(vs=c, rho_RPFI=base["RPFI"], rho_other=base[c],
                       diff_abs=abs(base["RPFI"]) - abs(base[c]),
                       lo=np.percentile(diffs[c], 2.5), hi=np.percentile(diffs[c], 97.5))
                  for c in others]).to_csv(out / "hrmae_assoc_comparison.csv", index=False)

    band = pd.cut(df.RPFI, [0, 20, 40, 60, 80, 100], right=False)
    df.groupby(band, observed=True).agg(
        n=("RPFI", "size"), HRMAE_mean=("HR_MAE_bpm", "mean"),
        pct_HRMAE_lt5=("HR_MAE_bpm", lambda x: 100 * (x < 5).mean()),
        WCR_median=("WCR", "median"), BWMD_median=("BWMD", "median"), EDD_median=("EDD", "median"),
    ).to_csv(out / "rpfi_bands.csv")
    print("written to", out)


if __name__ == "__main__":
    main()
