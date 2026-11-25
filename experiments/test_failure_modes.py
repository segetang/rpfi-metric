"""
test_failure_modes.py - RPFI v2 failure handling
====================================================
Builds failure and edge cases from one synthetic pulse (60 s, 6 chunks) and tabulates the profiles.
Checks
  - missing or adding one early beat does not break the matching of later beats (1:1 nearest matching, not order-based)
  - missed and added beats are penalized explicitly as FN/FP (F1)
  - peaks straddling chunk boundaries do not break the matching
  - rules for flat outputs, noise only, non-finite values, invalid references and too-short signals
Output: failure_modes.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import rpfi_v2 as RV
from rhythm import detect_beats
import rpfi_eval as R

FS = 30


def ppg(t, hr=72.0):
    x = (t * hr / 60.0) % 1.0
    return np.exp(-((x - 0.18) / 0.07) ** 2) + 0.45 * np.exp(-((x - 0.45) / 0.10) ** 2)


def beats_of(y):
    return np.round(detect_beats(R.preprocess_chunk(y, FS), FS) * FS).astype(int)


def remove_beat(y, k):
    """Remove one beat by replacing the k-th beat (trough to trough) with a straight line."""
    z = y.copy(); tr = np.where((z[1:-1] < z[:-2]) & (z[1:-1] <= z[2:]))[0] + 1
    a, b = tr[k], tr[k + 1]; z[a:b + 1] = np.linspace(z[a], z[b], b - a + 1); return z


def add_beat(y, t_s):
    t = np.arange(y.size) / FS; return y + 0.9 * np.exp(-((t - t_s) / 0.06) ** 2)


def main():
    t = np.arange(60 * FS) / FS; y = ppg(t); rng = np.random.default_rng(0)
    cases = {}
    cases["identical"] = (y, y)
    cases["one early beat missing (2nd beat)"] = (y, remove_beat(y, 1))
    cases["one middle beat missing (36th beat)"] = (y, remove_beat(y, 35))
    cases["one early extra beat (0.55 s)"] = (y, add_beat(y, 0.55))
    cases["one middle extra beat (30.55 s)"] = (y, add_beat(y, 30.55))
    # phase shift so that peaks fall on the chunk boundaries (10, 20, ... s)
    pk = beats_of(y[:10 * FS]); shift = (10 * FS - pk[-1]) / FS if len(pk) else 0.0
    ys = ppg(t - shift)
    cases["peaks on chunk boundaries (reference = prediction)"] = (ys, ys)
    cases["peaks on chunk boundaries + noise 0.3 SD"] = (ys, ys + 0.3 * ys.std() * rng.standard_normal(t.size))
    cases["flat output (0)"] = (y, np.zeros_like(y))
    cases["flat output (constant 5)"] = (y, np.full_like(y, 5.0))
    cases["amplitude x 1e-9 (scale invariance)"] = (y, 1e-9 * y)
    cases["noise only"] = (y, rng.standard_normal(t.size))
    z = y.copy(); z[12 * FS:18 * FS] = np.nan
    cases["prediction partly NaN (2nd chunk)"] = (y, z)
    cases["prediction all NaN"] = (y, np.full_like(y, np.nan))
    z = y.copy(); z[42 * FS:45 * FS] = np.nan
    cases["reference partly NaN (5th chunk dropped)"] = (z, y)
    z = y.copy(); z[50 * FS:] = 0.0
    cases["reference partly flat (6th chunk dropped)"] = (z, y)
    cases["length 25 s (2 chunks)"] = (y[:25 * FS], y[:25 * FS])
    rows = []
    for k, (r, p) in cases.items():
        try:
            o = RV.score(r, p, FS)
            rows.append(dict(case=k, RPFI=o["RPFI"], s_SHAPE=o["s_SHAPE"], s_WCR=o["s_WCR"], s_RHY=o["s_RHY"], F1=o["F1"],
                             SHAPE_defined=o["SHAPE_defined"], pred_flat_frac=o["pred_flat_frac"],
                             ref_dropped=o["n_ref_chunks_dropped"], pred_nonfinite=o["n_pred_chunks_nonfinite"],
                             n_chunks=o["n_chunks"], note=""))
        except ValueError as e:
            rows.append(dict(case=k, note=f"ValueError: {e}"))
    for k, (r, p) in {"length 9 s (0 chunks)": (y[:9 * FS], y[:9 * FS]), "reference all NaN": (np.full_like(y, np.nan), y)}.items():
        try:
            RV.score(r, p, FS); rows.append(dict(case=k, note="no error (unexpected)"))
        except ValueError as e:
            rows.append(dict(case=k, note=f"ValueError: {e}"))
    d = pd.DataFrame(rows); d.to_csv("failure_modes.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
