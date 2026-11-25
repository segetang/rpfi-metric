"""
exp2_analyze.py - Experiment 2 analysis (criteria fixed in the pre-registration)
==================================================================
Inputs: exp2_features/exp2_features_ALL.csv (exp2_extract.py), newsys_recompute/newsys_participant_ALL.csv
Order (enforced in code):
  A. feature selection - uses reference (label) columns only; predictions and metric values are not passed to this step.
  B. error computation - selected features only; per participant |prediction - reference| / (between-participant SD of the
     reference feature within the dataset)
  C. primary criteria E1, E2 (primary set = UBFC and PURE within-dataset conditions)
  D. secondary (reported only)
Outputs: exp2_report.txt, exp2_merged.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")
CAND = ["Tsp/Tpi", "Tpw50/Tpi", "Ab/Aa", "AGI", "Tsys/Tdia", "deltaT/Tpi", "Tdw50/Tsw50"]
VALID_BEAT_FRAC, VALID_PART_FRAC, REL_MIN, REL_FALLBACK, MAX_FEAT = 0.80, 0.90, 0.75, 0.50, 4
PRED_MIN_BEATS = 5
B, SEED = 2000, 0
CONV = ["HR_MAE_bpm", "RMSE", "Pearson"]
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


# -- A. feature selection (reference side only) ---------------------------------
def select_features(ref):
    """ref: one row per dataset x participant, 'ref_' columns only. Returns the selected features and a table."""
    assert all(c.startswith("ref_") or c in ("dataset", "participant") for c in ref.columns)
    rows = []
    for f in CAND:
        frac = ref[f"ref_valid_{f}"] / ref["ref_nbeats"].clip(lower=1)
        ok_ds = {ds: float((frac[ref.dataset == ds] >= VALID_BEAT_FRAC).mean()) for ds in ("ubfc", "pure")}
        o, e = ref[f"ref_odd_{f}"], ref[f"ref_even_{f}"]
        m = o.notna() & e.notna()
        r = spearmanr(o[m], e[m])[0] if m.sum() >= 10 else np.nan
        rel = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan     # Spearman-Brown
        rows.append(dict(feature=f, valid_ubfc=ok_ds["ubfc"], valid_pure=ok_ds["pure"], split_half_rel=rel,
                         pass_valid=min(ok_ds.values()) >= VALID_PART_FRAC))
    t = pd.DataFrame(rows)
    chosen = t[t.pass_valid & (t.split_half_rel >= REL_MIN)]
    fallback = False
    if len(chosen) < 2:
        chosen = t[t.pass_valid & (t.split_half_rel >= REL_FALLBACK)]; fallback = True
    chosen = chosen.sort_values("split_half_rel", ascending=False).head(MAX_FEAT)
    return list(chosen.feature), t, fallback


# -- statistics -------------------------------------------------------------------
def partial_spearman(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    res = lambda v: v - Zr @ np.linalg.lstsq(Zr, v, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def rmcorr(x, y, g):
    """Repeated-measures correlation (Bakdash & Marusich 2017): correlation after removing participant means."""
    d = pd.DataFrame(dict(x=x, y=y, g=g))
    d["xc"] = d.x - d.groupby("g").x.transform("mean"); d["yc"] = d.y - d.groupby("g").y.transform("mean")
    return float(np.corrcoef(d.xc, d.yc)[0, 1])


def cboot(d, fn, seed=SEED):
    rng = np.random.default_rng(seed)
    g = d.participant_key.values; us = np.unique(g); idx = {u: np.where(g == u)[0] for u in us}
    v = []
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        try:
            v.append(fn(d.iloc[ii]))
        except Exception:
            pass
    v = np.array(v, float); v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if v.size else np.array([np.nan, np.nan])


def main():
    fa, fn_ = Path("exp2_features/exp2_features_ALL.csv"), Path("newsys_recompute/newsys_participant_ALL.csv")
    for f in (fa, fn_):
        if not f.exists():
            sys.exit(f"not found: {f}")
    X = pd.read_csv(fa)

    # A ------------------------------------------------------------------------
    ref = X[X.primary].drop_duplicates(["dataset", "participant"])
    ref = ref[["dataset", "participant"] + [c for c in X.columns if c.startswith("ref_")]].reset_index(drop=True)
    feats, tab, fallback = select_features(ref)
    P("=" * 70 + "\nA. Feature selection (reference signal only)\n" + "=" * 70)
    P(f"  rule: participants with valid-beat rate >= {VALID_BEAT_FRAC:.0%} are >= {VALID_PART_FRAC:.0%} in both UBFC and PURE, "
      f"split-half reliability (Spearman-Brown) >= {REL_MIN} -> up to {MAX_FEAT} by reliability (relaxed to {REL_FALLBACK} if fewer than 2)")
    P("  " + tab.round(3).to_string(index=False).replace("\n", "\n  "))
    P(f"  selected: {feats}" + ("   (relaxed criterion applied)" if fallback else ""))
    if len(feats) == 0:
        P("  no feature selected; analysis stopped."); Path("exp2_report.txt").write_text("\n".join(OUT), encoding="utf-8"); return

    # B ------------------------------------------------------------------------
    sd = ref.groupby("dataset")[[f"ref_med_{f}" for f in feats]].std()
    for f in feats:
        err = (X[f"pred_med_{f}"] - X[f"ref_med_{f}"]).abs() / X.dataset.map(sd[f"ref_med_{f}"])
        ok = (X[f"pred_valid_{f}"] >= PRED_MIN_BEATS) & X[f"ref_med_{f}"].notna()
        X[f"err_{f}"] = err.where(ok)
    E = X[[f"err_{f}" for f in feats]]
    X["n_feat_ok"] = E.notna().sum(1)
    X["MORPH_ERR"] = E.mean(1).where(X.n_feat_ok >= int(np.ceil(len(feats) / 2)))

    N = pd.read_csv(fn_)
    keep = ["condition", "model", "participant", "sh_SHAPE", "sh_gated", "WCR_new", "RHY", "F1", "nIBIE", "HR_MAE_bpm"]
    D = X.merge(N[keep], on=["condition", "model", "participant"], how="inner")
    D["participant_key"] = D.dataset + ":" + D.participant.astype(str)
    D["1-WCR"] = 1 - D.WCR_new
    D.to_csv("exp2_merged.csv", index=False)

    # C ------------------------------------------------------------------------
    pri = D[D.primary & D.MORPH_ERR.notna()].reset_index(drop=True)
    drop = int((D.primary & D.MORPH_ERR.isna()).sum())
    P("\n" + "=" * 70 + "\nC. Primary criteria (UBFC and PURE within-dataset, participant-cluster bootstrap B=2000)\n" + "=" * 70)
    P(f"  units {len(pri)} (model x participant), participants {pri.participant_key.nunique()}, excluded (morphology error not computable) {drop}")
    e1 = lambda d: spearmanr(d.sh_SHAPE, d.MORPH_ERR)[0]
    e2 = lambda d: partial_spearman(d.sh_SHAPE.values, d.MORPH_ERR.values, d[CONV].values)
    r1, ci1 = e1(pri), cboot(pri, e1)
    r2, ci2 = e2(pri), cboot(pri, e2)
    s1, s2 = (r1 >= 0.30 and ci1[0] > 0), ci2[0] > 0
    P(f"  E1  Spearman rho(SHAPE, morphology error) = {r1:.3f}  [{ci1[0]:.3f}, {ci1[1]:.3f}]  -> {'met' if s1 else 'not met'} (criterion rho >= 0.30, CI lower bound > 0)")
    P(f"  E2  partial rho | HR MAE, RMSE, Pearson = {r2:.3f}  [{ci2[0]:.3f}, {ci2[1]:.3f}]  -> {'met' if s2 else 'not met'} (criterion CI lower bound > 0)")
    P(f"  verdict: {'confirmed' if s1 and s2 else ('partially confirmed' if s1 or s2 else 'not confirmed')}")

    # D ------------------------------------------------------------------------
    P("\n" + "=" * 70 + "\nD. Secondary (reported only, not used for the verdict)\n" + "=" * 70)
    P("  [correlation of each metric with the morphology error, primary set]  Spearman rho [95% CI] / repeated-measures correlation (within participant, across models)")
    for c in ["sh_SHAPE", "HR_MAE_bpm", "RMSE", "Pearson", "1-WCR", "RHY"]:
        f = lambda d, c=c: spearmanr(d[c], d.MORPH_ERR)[0]
        g = lambda d, c=c: rmcorr(d[c].values, d.MORPH_ERR.values, d.participant_key.values)
        ci, cg = cboot(pri, f), cboot(pri, g)
        P(f"    {c:10s} rho = {f(pri):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]   rm = {g(pri):+.3f} [{cg[0]:+.3f}, {cg[1]:+.3f}]")
    P("  [per feature, Spearman rho with SHAPE]")
    for f in feats:
        m = pri[f"err_{f}"].notna()
        P(f"    {f:12s} rho = {spearmanr(pri.sh_SHAPE[m], pri[f'err_{f}'][m])[0]:+.3f} (n={int(m.sum())})")
    P("  [per dataset / cross conditions]")
    for name, sub in [("UBFC within", pri[pri.dataset == "ubfc"]), ("PURE within", pri[pri.dataset == "pure"]),
                      ("all cross-dataset conditions", D[~D.primary & D.MORPH_ERR.notna()].reset_index(drop=True))]:
        if len(sub) > 10:
            ci = cboot(sub, e1); cp = cboot(sub, e2)
            P(f"    {name:20s} n={len(sub):4d}  E1 rho = {e1(sub):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]   E2 partial rho = {e2(sub):.3f} [{cp[0]:.3f}, {cp[1]:.3f}]")
    ng = pri[~pri.sh_gated.astype(bool)].reset_index(drop=True)
    P(f"  [sensitivity: {len(pri) - len(ng)} SHAPE-gated units excluded]  E1 rho = {e1(ng):.3f}, E2 partial rho = {e2(ng):.3f}")
    Path("exp2_report.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("\nsaved: exp2_report.txt, exp2_merged.csv")


if __name__ == "__main__":
    main()
