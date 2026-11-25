"""
exp1_severity.py - Experiment 1 without CCC matching (exploratory; added after the confirmatory analysis)
==========================================================================================================
Question: does the RPFI profile still identify the error family when conventional metrics are not held constant, i.e.
when every case has its own severity and RMSE, Pearson correlation, CCC, HR MAE, PSNR and SNR vary with it?

Design (fixed before the analysis was run)
  - recordings, exploration/hold-out split, distortions and families: identical to Experiment 1
    (exp1_distortion_id.py, exp1_confirm.py; seed 2026, 40 exploration recordings per dataset, four 10 s chunks).
  - strength: instead of matching CCC, each (recording, distortion) gets two cases whose strength is drawn uniformly
    from the fixed range below (units of the distortion functions in exp1_distortion_id.py):
        peak_broadening   0.2 - 1.0     (low-pass cutoff 2.6 - 1.1 Hz)
        peak_sharpening   1.0 - 10.0    (unsharp-masking gain)
        notch_attenuation 0.3 - 1.0     (fraction of the second harmonic removed)
        ibi_jitter        0.5 - 2.0     (beat-time SD 20 - 80 ms)
        missed_beats      0.1 - 0.5     (5 - 25 % of beats removed)
        extra_beats       0.1 - 0.5     (spurious beats between 5 - 25 % of beat pairs)
        rsa_removal       0.5 - 1.0     (fraction of the interval variation removed)
        hr_error          1/60 - 1/6    (heart rate scaled by +0.5 % to +5 %)
        constant_lag      0.02 - 0.15   (s)
        white_noise       0.2 - 1.0     (noise SD relative to the signal SD)
  - features: RPFI profile = s_SHAPE, s_WCR, s_RHY of the release implementation (rpfi_v2.score_chunks, lag-aligned WCR);
              conventional = RMSE, Pearson r, CCC, HR MAE, PSNR; conventional + SNR; all features together.
  - added after the first run (post hoc): the profile with the removed lag, |Δt| (s), as a fourth feature ("RPFI+dt"),
    because RPFI reports Δt as part of the profile; cases with no matched beat (Δt undefined) get |Δt| = 1 s.
    The four original feature sets and their results are unchanged by this addition.
  - classifier, training and evaluation sets: as in Experiment 1 (standardization + multinomial logistic regression,
    C = 1, balanced classes; trained on the exploration cases of the dataset; P1 = COHFACE subject-disjoint hold-out,
    P2 = PURE hold-out). Subject-cluster bootstrap (B = 2000) for the CIs.

Usage (from the working folder that holds mapping_PURE.csv and mapping_COHFACE.csv, as for run_exp1_confirm.py):
    python exp1_severity.py --label-dir "../real data"
    python exp1_severity.py --synthetic            (smoke test on synthetic pulses; no real data needed)
    python exp1_severity.py --analyze-only         (re-run the evaluation from existing exp1sev_cases_*.csv files)
Outputs: exp1sev_cases_{dataset}_{explore|holdout}.csv, exp1_severity_report.txt
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rpfi_eval as R
import rpfi_v2 as RV
from exp1_distortion_id import DIST, make_ctx, real_bases, synth_base
from exp1_confirm import G, fit
from score_unsup_v2 import snr_chunk

SEED, N_BASE, N_CHUNKS, CASES_PER_PAIR, B = 2026, 40, 4, 2, 2000
DUR_TOTAL = 10.0 * N_CHUNKS * 1.3 + 1.0
RANGE = {"peak_broadening": (0.2, 1.0), "peak_sharpening": (1.0, 10.0), "notch_attenuation": (0.3, 1.0),
         "ibi_jitter": (0.5, 2.0), "missed_beats": (0.1, 0.5), "extra_beats": (0.1, 0.5), "rsa_removal": (0.5, 1.0),
         "hr_error": (1 / 60, 1 / 6), "constant_lag": (0.02, 0.15), "white_noise": (0.2, 1.0)}
DS = {"pure": ("PURE_label.npy", "mapping_PURE.csv", 30), "cohface": ("COHFACE_label.npy", "mapping_COHFACE.csv", 20)}
SETS = {"RPFI": ["s_SHAPE", "s_WCR", "s_RHY"],
        "conv": ["RMSE", "Pearson", "CCC", "HR_MAE", "PSNR"],
        "conv+SNR": ["RMSE", "Pearson", "CCC", "HR_MAE", "PSNR", "SNR"],
        "all": ["s_SHAPE", "s_WCR", "s_RHY", "RMSE", "Pearson", "CCC", "HR_MAE", "PSNR", "SNR"]}
POST_HOC = {"RPFI+dt": ["s_SHAPE", "s_WCR", "s_RHY", "lag_abs"]}
LAG_UNDEFINED_S = 1.0
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def features(yref, pred, fs):
    seg = 10 * fs
    Lr = [yref[i * seg:(i + 1) * seg] for i in range(N_CHUNKS)]
    Pr = [pred[i * seg:(i + 1) * seg] for i in range(N_CHUNKS)]
    r = RV.score_chunks(Lr, Pr, fs)
    L = np.array([R.preprocess_chunk(c, fs) for c in Lr]); Q = np.array([R.preprocess_chunk(c, fs) for c in Pr])
    y, q = L.ravel(), Q.ravel()
    snr = [snr_chunk(L[i], Q[i], fs) for i in range(N_CHUNKS)]
    snr = [v for v in snr if np.isfinite(v)]
    return dict(s_SHAPE=r["s_SHAPE"], s_WCR=r["s_WCR"], s_RHY=r["s_RHY"], RPFI=r["RPFI"],
                SHAPE_defined=r["SHAPE_defined"],
                lag_abs=float(abs(r["lag_s"])) if np.isfinite(r["lag_s"]) else LAG_UNDEFINED_S, RMSE=R.compute_rmse(y, q), Pearson=float(np.corrcoef(y, q)[0, 1]),
                CCC=R.concordance_ccc(y, q), HR_MAE=r["HR_MAE_bpm"], PSNR=R.compute_psnr(y, q),
                SNR=float(np.mean(snr)) if snr else np.nan)


def make_cases(bases, tag):
    rows = []
    for bi, (bid, fs, yfull, rec, subj) in enumerate(bases):
        n = int(10 * N_CHUNKS * fs)
        ctx = make_ctx(yfull, fs, np.random.default_rng(SEED + bi))
        srng = np.random.default_rng(SEED + 100_000 + bi)
        for dname, (fn, _) in DIST.items():
            lo, hi = RANGE[dname]
            for k in range(CASES_PER_PAIR):
                s = float(srng.uniform(lo, hi))
                pred = fn(yfull, fs, s, ctx)[:n]
                rows.append(dict(base=bid, rec=rec, subject=subj, fs=fs, distortion=dname, g=G[dname], strength=s,
                                 **features(yfull[:n], pred, fs)))
        print(f"  [{tag}] base {bi + 1}/{len(bases)} done ({len(rows)} cases)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"exp1sev_cases_{tag}.csv", index=False)
    return d


DELTAS = [("RPFI", "conv"), ("RPFI", "conv+SNR"), ("all", "conv+SNR"), ("RPFI+dt", "conv+SNR")]


def cluster_boot(te, y, preds, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    subj = te.subject.astype(str).values; us = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in us}
    res = {k: [] for k in preds}; dl = {d: [] for d in DELTAS}
    for _ in range(B):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        if len(np.unique(y[ii])) < 2:
            continue
        v = {k: BA(y[ii], p[ii]) for k, p in preds.items()}
        for k in v: res[k].append(v[k])
        for a, b in DELTAS:
            if a in v and b in v:
                dl[(a, b)].append(v[a] - v[b])
    return ({k: np.percentile(v, [2.5, 97.5]) for k, v in res.items()},
            {k: np.percentile(v, [2.5, 97.5]) for k, v in dl.items() if v})


def evaluate(name, tr, te):
    cols = sorted({c for v in SETS.values() for c in v})
    tr, te = tr.dropna(subset=cols).reset_index(drop=True), te.dropna(subset=cols).reset_index(drop=True)
    y = te.g.values
    pr = {k: fit(tr, v).predict(te[v].values) for k, v in SETS.items()}
    for k, v in POST_HOC.items():                      # only when the case files contain the extra feature
        if all(c in tr.columns and c in te.columns for c in v):
            pr[k] = fit(tr, v).predict(te[v].values)
    ci, dci = cluster_boot(te, y, pr)
    P(f"  {name}  (cases {len(te)}, recordings {te.base.nunique()}, subjects {te.subject.nunique()})")
    for k, p in pr.items():
        P(f"    {k:9s} BA = {BA(y, p):.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    for (a_, b_), c in dci.items():
        P(f"    {a_} - {b_:9s} = {BA(y, pr[a_]) - BA(y, pr[b_]):+.3f}  [{c[0]:+.3f}, {c[1]:+.3f}]")
    P("    CCC of the cases by family (median [IQR]): " + ", ".join(
        f"{g} {q[0.5]:.3f} [{q[0.25]:.3f}, {q[0.75]:.3f}]"
        for g, q in te.groupby("g").CCC.quantile([0.25, 0.5, 0.75]).unstack().iterrows()))
    P("    RPFI classifier, prediction share per distortion:")
    P("    " + pd.crosstab([te.g, te.distortion], pr["RPFI"], normalize="index").round(2).to_string().replace("\n", "\n    "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on synthetic pulses")
    ap.add_argument("--n-synth", type=int, default=6)
    ap.add_argument("--analyze-only", action="store_true", help="evaluate existing exp1sev_cases_*.csv files")
    a = ap.parse_args()
    data = {}
    if a.analyze_only:
        data = {ds: tuple(pd.read_csv(f"exp1sev_cases_{ds}_{sp}.csv") for sp in ("explore", "holdout"))
                for ds in ("pure", "cohface")}
        ex_c, ho_c = data["cohface"]
        s = ho_c.subject.astype(str)
        coh_disj = ho_c[~s.isin(set(ex_c.subject.astype(str))) & ~s.str.lower().isin(["nan", "none", ""])]
    elif a.synthetic:
        rng = np.random.default_rng(SEED)
        for ds, fs in (("pure", 30), ("cohface", 20)):
            bases = [(f"{ds}_s{i}", fs, synth_base(rng, fs, DUR_TOTAL + 20), i, f"subj{i}") for i in range(2 * a.n_synth)]
            data[ds] = (make_cases(bases[:a.n_synth], f"{ds}_explore"), make_cases(bases[a.n_synth:], f"{ds}_holdout"))
        ex_c, ho_c = data["cohface"]
        coh_disj = ho_c
    else:
        for ds, (lab, mp, fs) in DS.items():
            lab_p = Path(a.label_dir) / lab
            if not lab_p.exists() or not Path(mp).exists():
                sys.exit(f"not found: {lab_p} or {mp}")
            parts = []
            for split in ("explore", "holdout"):
                rng = np.random.default_rng(SEED)          # same first rng call as Experiment 1 -> same split
                bases = [(f"{ds}_{split}_r{rec}", fs, y, rec, subj)
                         for rec, subj, y in real_bases(lab_p, mp, fs, DUR_TOTAL, N_BASE, rng, split)]
                parts.append(make_cases(bases, f"{ds}_{split}"))
            data[ds] = tuple(parts)
        ex_c, ho_c = data["cohface"]
        s = ho_c.subject.astype(str)
        coh_disj = ho_c[~s.isin(set(ex_c.subject.astype(str))) & ~s.str.lower().isin(["nan", "none", ""])]
    P("Experiment 1 without CCC matching (exploratory)" + ("  [SYNTHETIC SMOKE TEST]" if a.synthetic else ""))
    ov = {ds: set(data[ds][0].rec) & set(data[ds][1].rec) for ds in data}
    P("leakage check: " + ("passed" if not any(ov.values()) else f"overlap {ov}"))
    evaluate("P1  COHFACE hold-out, subject-disjoint", data["cohface"][0], coh_disj.reset_index(drop=True))
    evaluate("P2  PURE hold-out, recording-disjoint", *data["pure"])
    Path("exp1_severity_report.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print("\nsaved: exp1_severity_report.txt")


if __name__ == "__main__":
    main()
