"""
exp1_confirm.py - Experiment 1 confirmation (hold-out recordings, pre-registered criteria)
=================================================================================
The classifier is trained on the exploration-stage cases (exp1_cases_{ds}.csv) and evaluated on the cases of hold-out
recordings never seen before (exp1_cases_{ds}_holdout.csv). Nothing is tuned on the hold-out cases.

Primary evaluation (fixed in advance)
  P1: COHFACE hold-out recordings of subjects who never appeared in the exploration stage (subject-disjoint; recordings
      with unknown subject excluded)
  P2: all PURE hold-out recordings (all 10 PURE subjects were used in exploration, so no subject-disjoint set exists
      -> recording-disjoint)
  families (fixed): shape  = peak_broadening, peak_sharpening, notch_attenuation
                    local  = white_noise, ibi_jitter, extra_beats, missed_beats      (beat-level local errors)
                    drift  = constant_lag, hr_error, rsa_removal                    (slow phase changes)
  features (fixed): RPFI = SHAPE, WCR, RHY / conventional = RMSE, Pearson, HR_MAE, PSNR
  classifier (fixed): standardization + multinomial logistic regression (C=1, class_weight=balanced), trained on all
                      exploration cases of the same dataset
  criteria: S1 = lower bound of the subject-cluster bootstrap (B=2000) 95% CI of BA(RPFI) - BA(conventional) > 0
            S2 = BA(RPFI) >= 0.70 (point estimate)
  verdict: S1 and S2 met in both P1 and P2 -> confirmed / only some met -> partially confirmed (report which) /
           S1 failed -> not confirmed

Secondary (reported only, not used for the verdict)
  all COHFACE hold-out recordings, a classifier trained on the pooled exploration cases of the 3 datasets, the original
  3 families (offset/shape/timing), per-distortion prediction distribution, SHAPE-only shape vs. non-shape.

Usage (from the working folder):  python exp1_confirm.py      -> exp1_confirm_report.txt
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from exp1_distortion_id import split_recordings, subject_col

SEED, N_BASE, DUR_TOTAL, B = 2026, 40, 10.0 * 4 * 1.3 + 1.0, 2000
DS = {"ubfc": ("mapping_UBFC.csv", 30), "pure": ("mapping_PURE.csv", 30), "cohface": ("mapping_COHFACE.csv", 20)}
G = {"peak_broadening": "shape", "peak_sharpening": "shape", "notch_attenuation": "shape",
     "white_noise": "local", "ibi_jitter": "local", "extra_beats": "local", "missed_beats": "local",
     "constant_lag": "drift", "hr_error": "drift", "rsa_removal": "drift"}
RPFI = ["SHAPE", "WCR", "RHY"]
CONV = ["RMSE", "Pearson", "HR_MAE", "PSNR"]
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


def load_explore(ds):
    f = Path(f"exp1_cases_{ds}.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)
    mp_path, fs = DS[ds]
    mp, ex, _ = split_recordings(mp_path, fs, DUR_TOTAL, N_BASE, np.random.default_rng(SEED))
    sc = subject_col(mp)
    idx = d.base.str.rsplit("_", n=1).str[1].astype(int)
    d["rec"] = [ex[i] for i in idx]
    d["subject"] = [str(mp.loc[r, sc]) if sc else str(r) for r in d.rec]
    return d


def prep(d):
    d = d.dropna(subset=RPFI + CONV).reset_index(drop=True)
    d["g"] = d.distortion.map(G)
    # recordings without subject information (COHFACE status=unknown): each recording is its own cluster; excluded from the
    # subject-disjoint set (P1). (Depending on the pandas version, missing values are read as the string 'nan' or as NaN;
    # both are handled and everything is converted to Python strings.)
    raw = d.subject.astype(object)
    unk = raw.isna() | raw.map(lambda x: str(x).strip().lower() in ("nan", "none", ""))
    d["subject"] = [("unknown_r" + str(int(float(r)))) if u else str(x) for x, u, r in zip(raw, unk, d.rec)]
    return d


def fit(tr, cols, lab="g"):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")).fit(
        tr[cols].values, tr[lab].values)


def cluster_boot(te, y, preds, seed=0):
    """Subject-level cluster bootstrap. preds: {name: predictions}. Returns {name: (lo, hi)}, delta (lo, hi)."""
    rng = np.random.default_rng(seed)
    subj = te.subject.values; us = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in us}
    res = {k: [] for k in preds}; dl = []
    for _ in range(B):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        if len(np.unique(y[ii])) < 2:
            continue
        v = {k: BA(y[ii], p[ii]) for k, p in preds.items()}
        for k in v: res[k].append(v[k])
        dl.append(v["RPFI"] - v["conv"])
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in res.items()}
    return ci, np.percentile(dl, [2.5, 97.5])


def evaluate(name, tr, te, lab="g", primary=False):
    y = te[lab].values
    pr = {"RPFI": fit(tr, RPFI, lab).predict(te[RPFI].values), "conv": fit(tr, CONV, lab).predict(te[CONV].values)}
    ci, dci = cluster_boot(te, y, pr)
    ba = {k: BA(y, p) for k, p in pr.items()}
    P(f"  {name}  (cases {len(te)}, recordings {te.base.nunique()}, subjects {te.subject.nunique()})")
    for k in pr:
        P(f"    {k:5s} BA = {ba[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    d = ba["RPFI"] - ba["conv"]
    P(f"    ΔBA = {d:+.3f}  [{dci[0]:+.3f}, {dci[1]:+.3f}]")
    s1, s2 = dci[0] > 0, ba["RPFI"] >= 0.70
    if primary:
        P(f"    -> S1 {'met' if s1 else 'not met'}, S2 {'met' if s2 else 'not met'}")
    return s1, s2, pr["RPFI"]


def main():
    global N_BASE
    if "--n-base" in sys.argv:                 # for testing; the real confirmation uses the default 40 (same as exploration)
        N_BASE = int(sys.argv[sys.argv.index("--n-base") + 1])
    ex = {ds: load_explore(ds) for ds in DS}
    ho = {ds: pd.read_csv(f"exp1_cases_{ds}_holdout.csv") for ds in ("pure", "cohface")
          if Path(f"exp1_cases_{ds}_holdout.csv").exists()}
    miss = [f"exp1_cases_{d}.csv" for d in ("pure", "cohface") if ex[d] is None] + \
           [f"exp1_cases_{d}_holdout.csv" for d in ("pure", "cohface") if d not in ho]
    if miss:
        sys.exit("files not found: " + ", ".join(miss))
    ex = {k: prep(v) for k, v in ex.items() if v is not None}
    ho = {k: prep(v) for k, v in ho.items()}

    # leakage check: hold-out recordings must not overlap exploration recordings
    for ds in ho:
        ov = set(ho[ds].rec) & set(ex[ds].rec)
        if ov:
            sys.exit(f"[stop] {ds}: hold-out recordings overlap exploration recordings: {sorted(ov)[:10]}")
    P("leakage check passed: hold-out recordings do not overlap exploration recordings")

    hc = ho["cohface"]
    coh_disj = hc[~hc.subject.isin(set(ex["cohface"].subject)) & ~hc.subject.str.startswith("unknown")].reset_index(drop=True)

    P("\n" + "=" * 70 + "\nPrimary evaluation (pre-registered criteria, families = shape / local / drift)\n" + "=" * 70)
    r1 = evaluate("P1  COHFACE hold-out, subject-disjoint", ex["cohface"], coh_disj, primary=True)
    r2 = evaluate("P2  PURE hold-out, recording-disjoint", ex["pure"], ho["pure"], primary=True)
    ok = [r1[0], r1[1], r2[0], r2[1]]
    verdict = "confirmed" if all(ok) else ("not confirmed" if not (r1[0] and r2[0]) else "partially confirmed")
    P(f"\n  final verdict: {verdict}   (P1 S1/S2 = {r1[0]}/{r1[1]}, P2 S1/S2 = {r2[0]}/{r2[1]})")

    P("\n" + "=" * 70 + "\nSecondary (reported only, not used for the verdict)\n" + "=" * 70)
    evaluate("all COHFACE hold-out", ex["cohface"], ho["cohface"])
    pooled = pd.concat(ex.values(), ignore_index=True)
    for ds in ho:
        evaluate(f"pooled training (exploration {'+'.join(ex)}) -> {ds} hold-out", pooled, ho[ds])
    for ds in ho:
        evaluate(f"original 3 families (offset/shape/timing) - {ds} hold-out", ex[ds], ho[ds], lab="family")

    P("\n  SHAPE alone, shape vs. non-shape (logistic trained on exploration -> hold-out evaluation)")
    for ds in ho:
        tr, te = ex[ds].assign(sb=(ex[ds].g == "shape")), ho[ds].assign(sb=(ho[ds].g == "shape"))
        p = fit(tr, ["SHAPE"], "sb").predict(te[["SHAPE"]].values)
        rate = te.assign(p=p)[te.sb].groupby("distortion").p.mean().round(2).to_dict()
        P(f"    {ds}: BA {BA(te.sb, p):.3f} | detection rate per shape distortion {rate} | false-positive rate on non-shape {p[~te.sb.values].mean():.3f}")

    P("\n  prediction distribution per distortion (primary classifier, RPFI)")
    for ds, (te, pr) in (("cohface (subject-disjoint)", (coh_disj, r1[2])), ("pure", (ho["pure"], r2[2]))):
        P(f"   [{ds}]")
        P("    " + pd.crosstab([te.g, te.distortion], pr, normalize="index").round(2).to_string().replace("\n", "\n    "))

    Path("exp1_confirm_report.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("\nsaved: exp1_confirm_report.txt")


if __name__ == "__main__":
    main()
