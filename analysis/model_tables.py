"""Model-level tables built from participant-level scores.

Outputs (results/tables/):
  table1.csv            per condition x model: RPFI [95% CI], coverage, RPFI given covered, profile
                        (s_SHAPE over covered participants, s_WCR, s_RHY, F1, nIBIE), median lag,
                        HR MAE, RMSE, CCC, SNR, v1 RPFI, polarity-flag rate
  table_unsupervised.csv  rPPG-Toolbox unsupervised extractors (raw and whole-method sign flip)
  pairwise_summary.csv  Friedman test + pairwise Wilcoxon signed-rank (Holm) per condition and metric
  pairwise_detail.csv   every model pair with Holm-adjusted p and mean difference

Usage:  python analysis/model_tables.py
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

import common as C


def table1(d):
    rows = []
    for c in C.ORDER:
        for m, g in d[d.condition == c].groupby("model", sort=True):
            mean, lo, hi = C.boot_ci(g.RPFI.values)
            cov = g.SHAPE_defined.values
            rows.append(dict(
                condition=c, model=m, n=len(g), RPFI=mean, RPFI_lo=lo, RPFI_hi=hi,
                coverage=cov.mean(), RPFI_given_covered=g.RPFI[cov].mean() if cov.any() else np.nan,
                s_SHAPE_covered=g.s_SHAPE[cov].mean() if cov.any() else np.nan,
                s_WCR=g.s_WCR.mean(), s_RHY=g.s_RHY.mean(), F1=g.F1.mean(), nIBIE=g.nIBIE.mean(),
                lag_ms_median=1000 * g.lag_s.median(), HR_MAE=g.HR_MAE_bpm.mean(), RMSE=g.RMSE.mean(),
                CCC=g.CCC.mean(), SNR_dB=g.SNR_dB.mean(), RPFI_v1=g.RPFI_v1.mean(),
                polarity_flag_rate=g.polarity_suspect.mean()))
    return pd.DataFrame(rows)


def table_unsupervised():
    u = pd.read_csv(C.RES / "scores" / "unsup_v2_ALL.csv")
    u["SHAPE_defined"] = u.SHAPE_defined.astype(str).str.lower().eq("true")
    u["polarity_suspect"] = u.polarity_suspect.astype(str).str.lower().eq("true")
    rows = []
    for (ds, meth, var), g in u.groupby(["dataset", "method", "variant"]):
        mean, lo, hi = C.boot_ci(g.RPFI.values)
        cov = g.SHAPE_defined.values
        rows.append(dict(dataset=ds, method=meth, variant=var, n=len(g), RPFI=mean, RPFI_lo=lo, RPFI_hi=hi,
                         coverage=cov.mean(), s_SHAPE_covered=g.s_SHAPE[cov].mean() if cov.any() else np.nan,
                         s_WCR=g.s_WCR.mean(), s_RHY=g.s_RHY.mean(), HR_MAE=g.HR_MAE_bpm.mean(),
                         SNR_dB=g.SNR_dB.mean(), WCR_raw=g.WCR_raw.mean(),
                         polarity_flag_rate=g.polarity_suspect.mean(), zero_frac=g.zero_frac.iloc[0]))
    t = pd.DataFrame(rows)
    # Polarity flag vs. an actual whole-method sign error at participant level:
    # "flip helps" = flipping the sign raises RPFI by more than 0.05.
    r = u[u.variant == "raw"].set_index(["dataset", "method", "participant"])
    f = u[u.variant == "flip"].set_index(["dataset", "method", "participant"])
    x = pd.DataFrame(dict(flag=r.polarity_suspect, helps=(f.RPFI - r.RPFI) > 0.05, gain=f.RPFI - r.RPFI))
    x = x[x.index.get_level_values("method") != "CHROM"]
    tp = int((x.flag & x.helps).sum())
    print(f"  polarity flag (unsupervised, CHROM excluded): flagged {int(x.flag.sum())}, "
          f"flip helps {int(x.helps.sum())}, both {tp}, precision {tp / max(x.flag.sum(), 1):.2f}, "
          f"recall {tp / max(x.helps.sum(), 1):.2f}")
    return t


METRICS = {"RPFI": "RPFI", "s_SHAPE": "s_SHAPE", "s_WCR": "s_WCR", "s_RHY": "s_RHY",
           "negHR": "HR_MAE_bpm", "negRMSE": "RMSE", "CCC": "CCC", "SNR": "SNR_dB"}
NEGATE = {"negHR", "negRMSE"}


def pairwise(d, alpha=0.05):
    summ, detail = [], []
    for c in C.ORDER:
        dc = d[d.condition == c]
        models = sorted(dc.model.unique())
        for name, col in METRICS.items():
            w = dc.pivot(index="participant", columns="model", values=col)[models].dropna()
            if name in NEGATE:
                w = -w
            try:
                fp = friedmanchisquare(*[w[m].values for m in models]).pvalue
            except ValueError:
                fp = np.nan
            res = R_posthoc({m: w[m].values for m in models}, models)
            # post hoc pairs are counted only when the Friedman test rejects (as stated in the paper)
            sig = sum(r[4] < alpha for r in res) if (np.isfinite(fp) and fp < alpha) else 0
            summ.append(dict(condition=c, metric=name, n_part=len(w), n_models=len(models),
                             pairs=len(res), sig_pairs=sig, friedman_p=fp))
            detail += [dict(condition=c, metric=name, a=a, b=b, W=st, p_raw=pr, p_holm=pj, mean_diff=md)
                       for a, b, st, pr, pj, md in res]
    return pd.DataFrame(summ), pd.DataFrame(detail)


def R_posthoc(scores, models):
    return C.R.posthoc_wilcoxon(scores, models)


def main():
    d = C.load_scores()
    t1 = table1(d)
    C.save(t1, "table1.csv")
    C.save(table_unsupervised(), "table_unsupervised.csv")
    s, det = pairwise(d)
    C.save(s, "pairwise_summary.csv")
    C.save(det, "pairwise_detail.csv")
    pd.set_option("display.width", 200)
    print(s.pivot_table(index="condition", columns="metric", values="sig_pairs", sort=False)
          .reindex(C.ORDER)[list(METRICS)].to_string())


if __name__ == "__main__":
    main()
