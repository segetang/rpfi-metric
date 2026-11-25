"""
hr_matched_pairs.py - exploratory analysis added after the confirmatory experiments
=====================================================================================
Question: among real model outputs that heart-rate error rates alike, does the RPFI profile show a difference, is that
difference reproducible, and does the component that carries it agree with an independent error measure?

Design (fixed before the analysis was run; exploratory, not part of the pre-specified criteria)
  units   : within-dataset UBFC-rPPG and PURE, every participant, every unordered pair of the nine models.
            COHFACE is not used because no model reaches HR MAE <= 2 bpm there.
  matched : both models HR MAE <= 2.0 bpm and |difference in HR MAE| <= 0.3 bpm (the selection rule of Fig. 9).
  gap     : |difference in RPFI| >= 0.10.
  outcomes
    1. share of matched pairs with a gap.
    2. reproducibility: share of gap pairs whose RPFI difference has the same sign in the odd and the even chunks.
    3. dominant component: SHAPE, WCR or Rhythm, whichever score differs most; "coverage" when SHAPE is defined for
       only one of the two models.
    4. external check (double dissociation): for SHAPE-dominant pairs, share in which the model with the lower SHAPE has
       the larger pyPPG morphology error (Experiment 2) and, separately, the larger instantaneous-HR tracking error
       (Experiment 2c); the same two shares for Rhythm-dominant pairs.
  uncertainty: participant-cluster bootstrap (B = 2000) 95% CIs of the shares.

Inputs : results/scores/rpfi_v2_ALL.csv, results/scores/split_half.csv, results/exp2/exp2_features_ALL.csv,
         results/exp2/exp2c_ihr_ALL.csv, results/scores/snr_real.csv
Outputs: results/tables/hr_matched_pairs.csv (one row per matched pair), results/tables/hr_matched_pairs_report.txt
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
HR_MAX, DHR_MAX, GAP = 2.0, 0.3, 0.10
COMPS = ("SHAPE", "WCR", "RHY")
B, SEED = 2000, 0
FEATS = ["Tpw50/Tpi", "deltaT/Tpi", "Tsp/Tpi", "Tsys/Tdia"]   # selected in Experiment 2 (exp2_report.txt)
PRED_MIN_BEATS = 5


def morph_error():
    """Morphology error exactly as in experiments/exp2_analyze.py, step B."""
    X = pd.read_csv(RES / "exp2" / "exp2_features_ALL.csv")
    ref = X[X.primary].drop_duplicates(["dataset", "participant"])
    sd = ref.groupby("dataset")[[f"ref_med_{f}" for f in FEATS]].std()
    for f in FEATS:
        err = (X[f"pred_med_{f}"] - X[f"ref_med_{f}"]).abs() / X.dataset.map(sd[f"ref_med_{f}"])
        ok = (X[f"pred_valid_{f}"] >= PRED_MIN_BEATS) & X[f"ref_med_{f}"].notna()
        X[f"err_{f}"] = err.where(ok)
    E = X[[f"err_{f}" for f in FEATS]]
    X["MORPH_ERR"] = E.mean(1).where(E.notna().sum(1) >= int(np.ceil(len(FEATS) / 2)))
    return X[["condition", "model", "participant", "MORPH_ERR"]]


def load():
    s = pd.read_csv(RES / "scores" / "rpfi_v2_ALL.csv")
    s = s[s.condition.isin(["ubfc", "pure"])].copy()
    s["SHAPE_defined"] = s.SHAPE_defined.astype(str).str.lower() == "true"
    h = pd.read_csv(RES / "scores" / "split_half.csv")
    h = h[h.condition.isin(["ubfc", "pure"])][["condition", "model", "participant", "RPFI_odd", "RPFI_even"]]
    m = morph_error()
    t = pd.read_csv(RES / "exp2" / "exp2c_ihr_ALL.csv")[["condition", "model", "participant", "ITE_bpm"]]
    n = pd.read_csv(RES / "scores" / "snr_real.csv")[["condition", "model", "participant", "SNR_dB"]]
    key = ["condition", "model", "participant"]
    d = s.merge(h, on=key, how="left").merge(m, on=key, how="left").merge(t, on=key, how="left").merge(n, on=key, how="left")
    return d


def pairs(d):
    rows = []
    for (cond, p), g in d.groupby(["condition", "participant"]):
        g = g.set_index("model")
        for a, b in itertools.combinations(sorted(g.index), 2):
            A, Bm = g.loc[a], g.loc[b]
            if max(A.HR_MAE_bpm, Bm.HR_MAE_bpm) > HR_MAX or abs(A.HR_MAE_bpm - Bm.HR_MAE_bpm) > DHR_MAX:
                continue
            if A.RPFI < Bm.RPFI:                      # orient: A is the higher-RPFI model
                A, Bm, a, b = Bm, A, b, a
            r = dict(condition=cond, participant=p, high=a, low=b, HR_high=A.HR_MAE_bpm, HR_low=Bm.HR_MAE_bpm,
                     RPFI_high=A.RPFI, RPFI_low=Bm.RPFI, dRPFI=A.RPFI - Bm.RPFI,
                     d_odd=A.RPFI_odd - Bm.RPFI_odd, d_even=A.RPFI_even - Bm.RPFI_even,
                     dSNR=A.SNR_dB - Bm.SNR_dB)
            for c in COMPS:
                r[f"{c}_high"], r[f"{c}_low"] = A[f"s_{c}"], Bm[f"s_{c}"]
            if A.SHAPE_defined != Bm.SHAPE_defined:
                dom = "coverage"
            else:
                dom = max(COMPS, key=lambda c: abs(A[f"s_{c}"] - Bm[f"s_{c}"]))
            r["dominant"] = dom
            if dom in COMPS:                           # model that is worse on the dominant component
                worse_is_low = Bm[f"s_{dom}"] < A[f"s_{dom}"]
                W, O = (Bm, A) if worse_is_low else (A, Bm)
                r["worse_has_larger_morph"] = (np.nan if pd.isna(W.MORPH_ERR) or pd.isna(O.MORPH_ERR)
                                               else float(W.MORPH_ERR > O.MORPH_ERR))
                r["worse_has_larger_ite"] = (np.nan if pd.isna(W.ITE_bpm) or pd.isna(O.ITE_bpm)
                                             else float(W.ITE_bpm > O.ITE_bpm))
            rows.append(r)
    return pd.DataFrame(rows)


def share_ci(df, col, rng):
    v = df[["participant_key", col]].dropna()
    if len(v) == 0:
        return np.nan, (np.nan, np.nan), 0
    est = v[col].mean()
    groups = {k: g[col].values for k, g in v.groupby("participant_key")}
    keys = list(groups)
    bs = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        x = np.concatenate([groups[keys[i]] for i in pick])
        bs.append(x.mean())
    return est, tuple(np.percentile(bs, [2.5, 97.5])), len(v)


def main():
    d = load()
    P = pairs(d)
    P["participant_key"] = P.condition + ":" + P.participant.astype(str)
    P["gap"] = P.dRPFI >= GAP
    P["same_sign_halves"] = (np.sign(P.d_odd) == np.sign(P.d_even)).astype(float).where(P.d_odd.notna() & P.d_even.notna())
    out_dir = RES / "tables"
    P.drop(columns="participant_key").to_csv(out_dir / "hr_matched_pairs.csv", index=False)

    rng = np.random.default_rng(SEED)
    L = []
    L.append("HR-matched model pairs (exploratory; UBFC-rPPG and PURE within-dataset)")
    L.append(f"rule: both HR MAE <= {HR_MAX} bpm, |dHR MAE| <= {DHR_MAX} bpm; gap: |dRPFI| >= {GAP}")
    for name, sub in [("all", P)] + [(c, P[P.condition == c]) for c in ("ubfc", "pure")]:
        L.append(f"  [{name}] matched pairs {len(sub)}, participants {sub.participant_key.nunique()}, "
                 f"median |dHR MAE| {(sub.HR_high - sub.HR_low).abs().median():.3f} bpm, "
                 f"median dRPFI {sub.dRPFI.median():.3f}, gap pairs {int(sub.gap.sum())} ({sub.gap.mean():.1%})")
    P["is_gap"] = P.gap.astype(float)
    e, ci, n = share_ci(P, "is_gap", rng)
    L.append(f"share of matched pairs with a gap: {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={n})")
    G = P[P.gap].copy()
    e, ci, n = share_ci(G, "same_sign_halves", rng)
    L.append(f"gap pairs with the same sign in odd and even chunks: {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={n}); chance 0.5")
    S = P[~P.gap].copy()
    e2, ci2, n2 = share_ci(S, "same_sign_halves", rng)
    L.append(f"  (pairs without a gap: {e2:.3f} [{ci2[0]:.3f}, {ci2[1]:.3f}], n={n2})")
    L.append("dominant component in gap pairs: " + ", ".join(
        f"{k} {v} ({v / len(G):.1%})" for k, v in G.dominant.value_counts().items()))
    for dom in ("SHAPE", "RHY", "WCR"):
        sub = G[G.dominant == dom]
        if len(sub) == 0:
            continue
        a, ca, na = share_ci(sub, "worse_has_larger_morph", rng)
        b, cb, nb = share_ci(sub, "worse_has_larger_ite", rng)
        L.append(f"  {dom}-dominant gap pairs (n={len(sub)}): model worse on {dom} has the larger morphology error in "
                 f"{a:.3f} [{ca[0]:.3f}, {ca[1]:.3f}] (n={na}), the larger tracking error in {b:.3f} [{cb[0]:.3f}, {cb[1]:.3f}] (n={nb})")
    sub = G[(G.dominant == "SHAPE") & G.dSNR.notna()].copy()
    key = ["condition", "model", "participant"]
    me = d.set_index(key).MORPH_ERR
    lo_m = np.array([me.get((r.condition, r.low, r.participant), np.nan) for r in sub.itertuples()])
    hi_m = np.array([me.get((r.condition, r.high, r.participant), np.nan) for r in sub.itertuples()])
    snr_worse_low = (sub.dSNR > 0).values               # the lower-RPFI model also has the lower SNR
    sub["snr_agrees_morph"] = np.where(snr_worse_low, lo_m > hi_m, hi_m > lo_m).astype(float)
    e, ci, n = share_ci(sub, "snr_agrees_morph", rng)
    L.append(f"  SHAPE-dominant gap pairs: model with the lower SNR has the larger morphology error in "
             f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={n})")
    ok = P.dSNR.notna()
    r, _ = spearmanr(P.dRPFI[ok], P.dSNR[ok].abs())
    L.append(f"Spearman between dRPFI and |dSNR| over matched pairs: {r:.3f}")
    G2 = G[G.dSNR.notna()]
    L.append(f"gap pairs in which the higher-RPFI model also has the higher SNR: {(G2.dSNR > 0).mean():.3f} (n={len(G2)})")
    txt = "\n".join(L)
    (out_dir / "hr_matched_pairs_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
