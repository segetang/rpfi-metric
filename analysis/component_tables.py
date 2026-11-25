"""Component-level analyses built from participant-level scores.

Outputs (results/tables/):
  overlap_decomposition.csv  rhythm vs. WCR overlap: raw Spearman and rank-based partial correlation after
                             removing model and participant fixed effects, within-dataset conditions,
                             with participant-cluster bootstrap 95% CI
  reliability_icc.csv        split-half (odd vs. even 10 s chunks) ICC(3,1), participant-cluster bootstrap CI,
                             and the Spearman-Brown full-length value
  example_pairs.csv          candidate participant-matched model pairs with similar HR MAE but different RPFI
                             (poor member must have a defined SHAPE; used to pick the example figure)

Usage:  python analysis/component_tables.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C


# ---------------------------------------------------------------- overlap
def overlap(d):
    comps = {"RHY (overall), 1 - s_RHY": lambda x: 1 - x.s_RHY,
             "beat detection, 1 - F1": lambda x: 1 - x.F1,
             "beat interval, nIBIE": lambda x: x.nIBIE}
    rows = []
    for ds in C.WITHIN:
        g = d[d.condition == ds].reset_index(drop=True)
        g = g.assign(y=1 - g.s_WCR)
        parts = g.participant.unique()
        rng = np.random.default_rng(C.SEED)
        boot_idx = [rng.choice(parts, len(parts)) for _ in range(C.N_BOOT)]
        pos = {p: np.where(g.participant.values == p)[0] for p in parts}
        for name, f in comps.items():
            g["x"] = f(g)
            rs = spearmanr(g.x, g.y)[0]
            pr = np.corrcoef(C.fe_residual(g, "x"), C.fe_residual(g, "y"))[0, 1]
            bs = []
            for pick in boot_idx:
                ii = np.concatenate([pos[p] for p in pick])
                lab = np.concatenate([[f"{p}#{i}"] * len(pos[p]) for i, p in enumerate(pick)])
                b = g.iloc[ii].assign(participant=lab)
                bs.append(np.corrcoef(C.fe_residual(b, "x"), C.fe_residual(b, "y"))[0, 1])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append(dict(dataset=ds, component=name, raw_spearman=rs, FE_partial=pr, lo=lo, hi=hi))
            print(f"  {ds:8s} {name:26s} raw {rs:.3f}  FE-partial {pr:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reliability
def icc31(x, y):
    """ICC(3,1), two-way mixed, consistency, single measurement (Shrout & Fleiss 1979)."""
    Y = np.column_stack([x, y]).astype(float)
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    if n < 3:
        return np.nan
    gm = Y.mean()
    ss_r = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)


def reliability():
    s = pd.read_csv(C.RES / "scores" / "split_half.csv")
    s = s[s.condition == s.dataset]
    metrics = {"RPFI": "RPFI", "s_SHAPE": "s_SHAPE", "s_WCR": "s_WCR", "s_RHY": "s_RHY", "F1": "F1",
               "nIBIE": "nIBIE", "HR_MAE_bpm": "HR_MAE_bpm", "SNR": "SNR"}
    rows = []
    for ds in C.WITHIN:
        g = s[s.dataset == ds].reset_index(drop=True)
        parts = g.participant.unique()
        rng = np.random.default_rng(C.SEED)
        picks = [rng.choice(parts, len(parts)) for _ in range(C.N_BOOT)]
        pos = {p: np.where(g.participant.values == p)[0] for p in parts}
        for name, col in metrics.items():
            a, b = g[f"{col}_odd"].values, g[f"{col}_even"].values
            icc = icc31(a, b)
            bs = [icc31(a[ii], b[ii]) for ii in (np.concatenate([pos[p] for p in pk]) for pk in picks)]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append(dict(dataset=ds, metric=name, n=len(g), ICC=icc, lo=lo, hi=hi,
                             SB_full=2 * icc / (1 + icc)))
        da = g.SHAPE_defined_odd.astype(str).str.lower().values == g.SHAPE_defined_even.astype(str).str.lower().values
        rows.append(dict(dataset=ds, metric="SHAPE_defined agreement", n=len(g), ICC=da.mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- example pairs
def example_pairs(d, hr_max=2.0, dhr_max=0.3):
    """Same participant, two models, both HR MAE <= hr_max bpm and |dHR| <= dhr_max bpm; sorted by RPFI gap."""
    rows = []
    for ds in ("ubfc", "pure"):
        g = d[d.condition == ds]
        for p, h in g.groupby("participant"):
            h = h.set_index("model")
            for a in h.index:
                for b in h.index:
                    A, B = h.loc[a], h.loc[b]
                    if a == b or A.RPFI < B.RPFI or not bool(B.SHAPE_defined):
                        continue
                    if max(A.HR_MAE_bpm, B.HR_MAE_bpm) > hr_max or abs(A.HR_MAE_bpm - B.HR_MAE_bpm) > dhr_max:
                        continue
                    rows.append(dict(dataset=ds, participant=p, good=a, poor=b,
                                     HR_good=A.HR_MAE_bpm, HR_poor=B.HR_MAE_bpm, RPFI_good=A.RPFI,
                                     RPFI_poor=B.RPFI, dR=A.RPFI - B.RPFI,
                                     prof_good=f"S{A.s_SHAPE:.2f}/W{A.s_WCR:.2f}/R{A.s_RHY:.2f}",
                                     prof_poor=f"S{B.s_SHAPE:.2f}/W{B.s_WCR:.2f}/R{B.s_RHY:.2f}",
                                     poor_defined=bool(B.SHAPE_defined)))
    return pd.DataFrame(rows).sort_values("dR", ascending=False).reset_index(drop=True)


def main():
    d = C.load_scores()
    C.save(overlap(d), "overlap_decomposition.csv")
    C.save(reliability(), "reliability_icc.csv")
    C.save(example_pairs(d), "example_pairs.csv")


if __name__ == "__main__":
    main()
