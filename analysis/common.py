"""Shared paths and loaders for the table-building scripts in analysis/.

All scripts here read only the CSV files under results/ (no raw data, no model outputs)
and write their outputs to results/tables/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "tables"
TAB.mkdir(parents=True, exist_ok=True)
for p in (ROOT / "rpfi", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rpfi_eval as R  # noqa: E402  (core utilities: cluster bootstrap, Holm, Wilcoxon post-hoc)

WITHIN = ["ubfc", "pure", "cohface"]
ORDER = ["ubfc", "pure", "cohface", "pure_to_ubfc", "cohface_to_ubfc", "ubfc_to_pure",
         "cohface_to_pure", "ubfc_to_cohface", "pure_to_cohface"]
KEY = ["condition", "model", "participant"]
N_BOOT, SEED = 2000, 42


def load_scores():
    """Participant-level table for all 9 conditions: release scores (rpfi_v2 2.0.2) +
    conventional metrics (RMSE, CCC, v1 RPFI) + spectral SNR."""
    v = pd.read_csv(RES / "scores" / "rpfi_v2_ALL.csv")
    n = pd.read_csv(RES / "scores" / "newsys_participant_ALL.csv")[KEY + ["RMSE", "CCC", "RPFI"]]
    n = n.rename(columns={"RPFI": "RPFI_v1"})
    s = pd.read_csv(RES / "scores" / "snr_real.csv")[KEY + ["SNR_dB"]]
    d = v.merge(n, on=KEY, how="left", validate="1:1").merge(s, on=KEY, how="left", validate="1:1")
    assert len(d) == len(v) == 2250, len(d)
    miss = d[["RMSE", "CCC", "SNR_dB"]].isna().sum()
    if miss.any():
        print(f"  note: missing values in conventional metrics: {miss[miss > 0].to_dict()}")
    d["SHAPE_defined"] = d["SHAPE_defined"].astype(str).str.lower().eq("true")
    d["polarity_suspect"] = d["polarity_suspect"].astype(str).str.lower().eq("true")
    return d


def boot_ci(values, seed=SEED):
    """Participant-level cluster bootstrap (percentile, 2,000 resamples) of the mean."""
    m, lo, hi = R.cluster_bootstrap_ci(np.asarray(values, float), N_BOOT, seed=seed)
    return m, lo, hi


def fe_residual(df, col):
    """Rank-transform `col`, then remove model and participant fixed effects (OLS residual)."""
    from scipy.stats import rankdata
    r = rankdata(df[col].values)
    X = pd.get_dummies(df[["model", "participant"]], drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    b, *_ = np.linalg.lstsq(X.values, r, rcond=None)
    return r - X.values @ b


def save(df, name):
    path = TAB / name
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path
