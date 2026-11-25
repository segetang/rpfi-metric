"""Experiment 1, secondary analyses on the frozen hold-out evaluation sets (P1, P2).

Same cases, families, classifier and evaluation sets as experiments/exp1_confirm.py
(the pre-registered primary analysis); only the feature sets differ:
  - SNR baselines: SNR alone, conventional + SNR, conventional + SNR + RPFI profile
  - component ablation: the frozen RPFI profile with one component removed
Bootstrap: participant (subject) clusters, 2,000 resamples, seed 0 (as in exp1_confirm.py).

  - lag-aligned WCR (the released definition, results/exp1/exp1_v2feat_*.csv) in place of the frozen unaligned WCR,
    alone and with |dt| as a fourth feature (added after the primary results)
Outputs (results/tables/): exp1_snr_baselines.csv, exp1_ablation.csv, exp1_aligned.csv
Usage:  python analysis/exp1_secondary.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA

import common as C
import exp1_confirm as E
from exp1_distortion_id import split_recordings, subject_col

E1 = C.RES / "exp1"
MAP = C.ROOT / "mappings"
KEYS = ["base", "distortion", "target"]


def load(ds, holdout):
    d = pd.read_csv(E1 / f"exp1_cases_{ds}{'_holdout' if holdout else ''}.csv")
    if not holdout:
        mp_path, fs = E.DS[ds]
        mp, ex, _ = split_recordings(MAP / mp_path, fs, E.DUR_TOTAL, E.N_BASE, np.random.default_rng(E.SEED))
        sc = subject_col(mp)
        idx = d.base.str.rsplit("_", n=1).str[1].astype(int)
        d["rec"] = [ex[i] for i in idx]
        d["subject"] = [str(mp.loc[r, sc]) if sc else str(r) for r in d.rec]
    snr = pd.read_csv(E1 / f"snr_exp1_{ds}{'_holdout' if holdout else ''}.csv")[KEYS + ["SNR_dB"]]
    d = d.merge(snr, on=KEYS, how="left", validate="1:1")
    assert d.SNR_dB.notna().all(), f"{ds}: SNR missing for some cases"
    v2 = pd.read_csv(E1 / f"exp1_v2feat_{ds}{'_holdout' if holdout else ''}.csv")[KEYS + ["WCR_aligned", "lag_s"]]
    d = d.merge(v2, on=KEYS, how="left", validate="1:1")
    d["lag_abs"] = d.lag_s.abs()
    return E.prep(d)


def boot(te, y, preds, pairs, seed=0):
    rng = np.random.default_rng(seed)
    subj = te.subject.values
    us = np.unique(subj)
    pos = {s: np.where(subj == s)[0] for s in us}
    out = {p: [] for p in pairs}
    for _ in range(C.N_BOOT):
        ii = np.concatenate([pos[s] for s in rng.choice(us, len(us))])
        if len(np.unique(y[ii])) < 2:
            continue
        ba = {k: BA(y[ii], v[ii]) for k, v in preds.items()}
        for a, b in pairs:
            out[(a, b)].append(ba[a] - ba[b])
    return {p: np.percentile(v, [2.5, 97.5]) for p, v in out.items()}


def boot_ba(te, y, preds, seed=0):
    """Subject-cluster bootstrap CI of each balanced accuracy (as in exp1_confirm.cluster_boot)."""
    rng = np.random.default_rng(seed)
    subj = te.subject.values
    us = np.unique(subj)
    pos = {s: np.where(subj == s)[0] for s in us}
    out = {k: [] for k in preds}
    for _ in range(C.N_BOOT):
        ii = np.concatenate([pos[s] for s in rng.choice(us, len(us))])
        if len(np.unique(y[ii])) < 2:
            continue
        for k, v in preds.items():
            out[k].append(BA(y[ii], v[ii]))
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def main():
    ex = {ds: load(ds, False) for ds in ("pure", "cohface")}
    ho = {ds: load(ds, True) for ds in ("pure", "cohface")}
    hc = ho["cohface"]
    p1 = hc[~hc.subject.isin(set(ex["cohface"].subject)) & ~hc.subject.str.startswith("unknown")].reset_index(drop=True)
    evals = {"P1 COHFACE (subject-disjoint hold-out)": (ex["cohface"], p1), "P2 PURE (hold-out)": (ex["pure"], ho["pure"])}

    sets = {"RPFI profile": E.RPFI, "conventional": E.CONV, "SNR": ["SNR_dB"],
            "conventional + SNR": E.CONV + ["SNR_dB"], "conventional + SNR + RPFI": E.CONV + ["SNR_dB"] + E.RPFI}
    abl = {f"without {c}": [x for x in E.RPFI if x != c] for c in E.RPFI}

    snr_rows, abl_rows, al_rows = [], [], []
    for name, (tr, te) in evals.items():
        y = te.g.values
        pr = {k: E.fit(tr, cols).predict(te[cols].values) for k, cols in {**sets, **abl}.items()}
        ba = {k: BA(y, v) for k, v in pr.items()}
        pairs = [("RPFI profile", "conventional"), ("RPFI profile", "conventional + SNR")] + \
                [(k, "RPFI profile") for k in abl]
        ci = boot(te, y, pr, pairs)
        for k in sets:
            snr_rows.append(dict(eval=name, features=k, BA=ba[k]))
        for a, b in pairs[:2]:
            snr_rows.append(dict(eval=name, features=f"delta: {a} - {b}", BA=ba[a] - ba[b],
                                 lo=ci[(a, b)][0], hi=ci[(a, b)][1]))
        abl_rows.append(dict(eval=name, removed="(none)", BA=ba["RPFI profile"], delta=0.0))
        for k in abl:
            abl_rows.append(dict(eval=name, removed=k.replace("without ", ""), BA=ba[k],
                                 delta=ba[k] - ba["RPFI profile"], lo=ci[(k, "RPFI profile")][0],
                                 hi=ci[(k, "RPFI profile")][1]))
        print(f"  {name}: " + ", ".join(f"{k} {ba[k]:.3f}" for k in sets))
        al = {"RPFI profile (aligned WCR)": ["SHAPE", "WCR_aligned", "RHY"],
              "RPFI profile (aligned WCR) + |dt|": ["SHAPE", "WCR_aligned", "RHY", "lag_abs"]}
        pa = {k: E.fit(tr.dropna(subset=v), v).predict(te[v].values) for k, v in al.items()}
        pa.update({k: pr[k] for k in ("conventional", "conventional + SNR")})
        pairs_a = [("RPFI profile (aligned WCR)", "conventional"), ("RPFI profile (aligned WCR)", "conventional + SNR")]
        cia = boot(te, y, pa, pairs_a)
        rng_ci = boot_ba(te, y, pa)
        for k in al:
            al_rows.append(dict(eval=name, features=k, BA=BA(y, pa[k]), lo=rng_ci[k][0], hi=rng_ci[k][1]))
        for a, b in pairs_a:
            al_rows.append(dict(eval=name, features=f"delta: {a} - {b}", BA=BA(y, pa[a]) - BA(y, pa[b]),
                                lo=cia[(a, b)][0], hi=cia[(a, b)][1]))
    C.save(pd.DataFrame(al_rows), "exp1_aligned.csv")
    C.save(pd.DataFrame(snr_rows), "exp1_snr_baselines.csv")
    C.save(pd.DataFrame(abl_rows), "exp1_ablation.csv")


if __name__ == "__main__":
    main()
