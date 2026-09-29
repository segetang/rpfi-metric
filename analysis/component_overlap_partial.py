"""
component_overlap_partial.py - overlap between RPFI components beyond the pooled Spearman matrix.

For each dataset and component pair (BWMD, WCR, EDD) it reports:
  pooled               Spearman correlation over all participant x model rows (Table 4 of the paper)
  adj_model            correlation of rank residuals after removing model main effects
  adj_both             after removing model and participant main effects (two-way demeaning of ranks),
                       with a t-test p-value using N - (models + participants - 1) - 2 degrees of freedom
  within_model_*       median / min / max of per-model Spearman correlations across participants
  between_model        Spearman correlation of model means
  partial_given_third  classical partial Spearman correlation given the third component

Usage (from the repository root):
  python analysis/component_overlap_partial.py --results results --out results/component_overlap_partial.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

COMPS = ("BWMD", "WCR", "EDD")
PAIRS = [("BWMD", "WCR"), ("BWMD", "EDD"), ("WCR", "EDD")]


def rank_residual(df, col, by):
    x = df[col].rank()
    x = x - x.mean()
    for _ in range(50):  # iterative demeaning; exact after one pass for a balanced design
        for b in by:
            x = x - x.groupby(df[b]).transform("mean")
    return x


def analyse(df, dataset):
    n, nm, npart = len(df), df.model.nunique(), df.participant.nunique()
    rows = []
    for a, b in PAIRS:
        out = dict(dataset=dataset, pair=f"{a}-{b}", N=n, models=nm, participants=npart,
                   pooled=stats.spearmanr(df[a], df[b])[0])
        for tag, by, k in (("adj_model", ["model"], nm),
                           ("adj_participant", ["participant"], npart),
                           ("adj_both", ["model", "participant"], nm + npart - 1)):
            r = float(np.corrcoef(rank_residual(df, a, by), rank_residual(df, b, by))[0, 1])
            dof = n - k - 2
            t = r * np.sqrt(dof / (1 - r * r))
            out[tag], out[tag + "_p"] = r, 2 * stats.t.sf(abs(t), dof)
        w = [stats.spearmanr(g[a], g[b])[0] for _, g in df.groupby("model")]
        out.update(within_model_median=np.median(w), within_model_min=min(w), within_model_max=max(w))
        m = df.groupby("model")[[a, b]].mean()
        out["between_model"] = stats.spearmanr(m[a], m[b])[0]
        c = [x for x in COMPS if x not in (a, b)][0]
        R = df[[a, b, c]].rank()
        ra = R[a] - np.polyval(np.polyfit(R[c], R[a], 1), R[c])
        rb = R[b] - np.polyval(np.polyfit(R[c], R[b], 1), R[c])
        out["partial_given_third"] = float(np.corrcoef(ra, rb)[0, 1])
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="folder with rpfi_participant_{ds}_geometric.csv")
    ap.add_argument("--out", default="results/component_overlap_partial.csv")
    args = ap.parse_args()
    rows = []
    for ds in ("pure", "ubfc", "cohface"):
        df = pd.read_csv(Path(args.results) / f"rpfi_participant_{ds}_geometric.csv")
        rows += analyse(df, ds)
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    cols = ["dataset", "pair", "pooled", "adj_model", "adj_both", "adj_both_p", "partial_given_third"]
    print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
