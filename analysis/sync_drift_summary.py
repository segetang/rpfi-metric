"""Summarize the within-recording lag drift (results/robustness/sync_drift.csv; Supplementary Sec. S1).

sync_drift_check.py measures the lag between reference and prediction in every 10 s chunk of a recording, for POS and
the within-dataset BiMamba. Recordings whose chunks all gave a valid beat correspondence (F1 >= 0.5) and that have at
least five chunks are used. For each recording, the slope of the chunk-wise lag against time is compared with the
slopes over all orderings of the same chunks (exact two-sided permutation test; at most 8 chunks, otherwise 20,000
random orderings).
"""
import itertools
import numpy as np
import pandas as pd
from common import RES, save

rng = np.random.default_rng(0)
d = pd.read_csv(RES / "robustness" / "sync_drift.csv")
d = d[(d.n_chunks_ok == d.n_chunks) & (d.n_chunks >= 5)].copy()


def slope(t, y):
    tc = t - t.mean()
    return float(tc @ (y - y.mean()) / (tc @ tc))


rows = []
for _, r in d.iterrows():
    lag = np.array([float(x) for x in r.lag_by_chunk_ms.split(";")])
    t = (np.arange(lag.size) + 0.5) * 10 / 60                       # minutes
    obs = slope(t, lag)
    if lag.size <= 8:
        null = np.array([slope(t, lag[list(p)]) for p in itertools.permutations(range(lag.size))])
    else:
        null = np.array([slope(t, rng.permutation(lag)) for _ in range(20000)])
    rows.append(dict(dataset=r.dataset, model=r.model, p=float(np.mean(np.abs(null) >= abs(obs) - 1e-9)),
                     drift=abs(obs), lag_range=lag.max() - lag.min()))
x = pd.DataFrame(rows)
out = x.groupby(["dataset", "model"]).agg(recordings=("p", "size"), frac_p05=("p", lambda v: (v < 0.05).mean()),
                                          drift_ms_per_min_median=("drift", "median"),
                                          lag_range_ms_median=("lag_range", "median")).reset_index()
print(out.round(3).to_string(index=False))
save(out, "sync_drift_summary.csv")

# Lag summary per condition (Supplementary Table S1, main text Sec. IV-C): median |lag| for learned models and POS,
# share of units beyond the quarter-cycle cap, polarity-flag rate, and the share of lag variance that is common to
# all models of the same participant.
v = pd.read_csv(RES / "scores" / "rpfi_v2_ALL.csv")
lag_rows = []
for c, g in v.groupby("condition", sort=False):
    pm = g.groupby("participant").lag_s.transform("mean")
    share = float(pm.var(ddof=0) / g.lag_s.var(ddof=0))
    for lab, sub in (("learned", g[g.model != "pos"]), ("POS", g[g.model == "pos"])):
        if len(sub):
            lag_rows.append(dict(condition=c, models=lab, abs_lag_ms_median=1000 * sub.lag_s.abs().median(),
                                 capped=sub.lag_capped.mean(), polarity_flag=sub.polarity_suspect.mean(),
                                 participant_share_of_lag_variance=share))
lag = pd.DataFrame(lag_rows)
print(lag.round(3).to_string(index=False))
save(lag, "lag_summary.csv")
