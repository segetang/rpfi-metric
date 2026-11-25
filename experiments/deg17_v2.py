"""
deg17_v2.py - 17-type synthetic degradation battery, rewritten for RPFI v2 (paired-trajectory monotonicity)
==================================================================================
Re-scores the same 17 degradation types as the v1 battery with the v2 components (rpfi_v2.score_chunks).
Changes from v1
  - paired trajectories: for each (fs, repetition, degradation) the base signal and random numbers are fixed and only
    the level changes -> monotonicity and reversal rate per trajectory.
  - edge artifacts removed: the base signal is generated longer at both ends and only the central 60 s is scored after
    degradation. (In v1, hr_bias clipped the end after time compression, leaving a flat last segment, and phase_shift
    used np.roll, wrapping the ends together.)
  - polarity inversion is graded by inverting a contiguous segment whose length is proportional to the level.
Outputs: deg17_v2_raw.csv (per case), deg17_v2_report.txt (summary)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import exp1_distortion_id as E
import rpfi_v2 as RV
from synth import synth_ppg

LEVELS = np.round(np.linspace(0, 1, 6), 2)
REPS, FS_LIST, WIN_S, PAD_S = 8, (30, 20), 60.0, 15.0
COMP = ["s_SHAPE", "s_WCR", "s_RHY", "RPFI"]


def base(fs, rng):
    n_tot = int((PAD_S + WIN_S * 1.6 + PAD_S) * fs)
    y, hr = synth_ppg(n_tot, fs, rng)
    return y, hr


def win(z, fs):
    a = int(PAD_S * fs); return z[a:a + int(WIN_S * fs)]


def warp(y, fs, src):
    t = np.arange(y.size) / fs
    return np.interp(np.clip(src(t), 0, t[-1]), t, y)


# -- 17 types: f(y_full, fs, lv, rng, hr) -> degraded y_full ---------------------------
def d_white(y, fs, lv, rng, hr): return y + 2.0 * lv * y.std() * rng.standard_normal(y.size)
def d_colored(y, fs, lv, rng, hr):
    w = rng.standard_normal(y.size); f = np.fft.rfftfreq(y.size, 1 / fs)
    c = np.fft.irfft(np.fft.rfft(w) / np.maximum(f, 1e-3), n=y.size); c /= c.std() + 1e-12
    return y + 2.0 * lv * y.std() * c
def d_wander(y, fs, lv, rng, hr):
    t = np.arange(y.size) / fs; return y + 3.0 * lv * y.std() * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
def d_amp(y, fs, lv, rng, hr): return y * (1 + 4 * lv)
def d_dc(y, fs, lv, rng, hr): return y + 5 * lv * y.std()
def _spk(y, idx, rng):
    z = y.copy(); z[idx] += 4 * y.std() * rng.choice([-1, 1], size=len(idx)); return z
def d_burst(y, fs, lv, rng, hr):
    a, n = int(PAD_S * fs), int(WIN_S * fs); k = max(1, int(0.05 * lv * n)); s = a + rng.integers(0, n - k)
    return _spk(y, np.arange(s, s + k), rng) if lv > 0 else y.copy()
def d_disp(y, fs, lv, rng, hr):
    a, n = int(PAD_S * fs), int(WIN_S * fs); k = max(1, int(0.05 * lv * n))
    return _spk(y, a + rng.choice(n, size=k, replace=False), rng) if lv > 0 else y.copy()
def d_dropout(y, fs, lv, rng, hr):
    a, n = int(PAD_S * fs), int(WIN_S * fs); z = y.copy(); k = int(0.3 * lv * n); s = a + rng.integers(0, n - k + 1)
    z[s:s + k] = 0.0; return z
def d_jitter(y, fs, lv, rng, hr):
    fj = rng.uniform(0.05, 0.2); return warp(y, fs, lambda t: t + 0.15 * lv * np.sin(2 * np.pi * fj * t))
def d_phase(y, fs, lv, rng, hr): return warp(y, fs, lambda t: t - 0.4 * lv)
def d_broad(y, fs, lv, rng, hr):
    if lv <= 0: return y.copy()
    return filtfilt(*butter(4, min(max(1.2, 3.0 - 1.7 * lv) / (fs / 2), 0.99), btype="low"), y)
def d_quant(y, fs, lv, rng, hr):
    if lv <= 0: return y.copy()
    L = max(3, int(round(64 * (1 - 0.95 * lv)))); lo, hi = y.min(), y.max()
    return np.round((y - lo) / (hi - lo) * (L - 1)) / (L - 1) * (hi - lo) + lo
def d_hrbias(y, fs, lv, rng, hr):
    t0 = PAD_S; return warp(y, fs, lambda t: t0 + (t - t0) * (1 + 0.5 * lv))   # compression anchored at the window start (no clipping)
def d_polarity(y, fs, lv, rng, hr):
    a, n = int(PAD_S * fs), int(WIN_S * fs); z = y.copy(); k = int(lv * n); s = a + rng.integers(0, n - k + 1)
    m = z[s:s + k].mean() if k else 0; z[s:s + k] = 2 * m - z[s:s + k]; return z
def d_notch(y, fs, lv, rng, hr):
    return E.d_notch_attenuation(y, fs, lv, dict(f0=hr / 60.0))
def d_extra(y, fs, lv, rng, hr):
    return E.d_extra_beats(y, fs, lv, E.make_ctx(y, fs, rng))
def d_sharp(y, fs, lv, rng, hr):
    return E.d_peak_sharpening(y, fs, lv, None)


DEG = {"white_noise": d_white, "colored_noise": d_colored, "baseline_wander": d_wander, "amplitude_scale": d_amp,
       "dc_offset": d_dc, "spike_burst": d_burst, "spike_dispersed": d_disp, "dropout": d_dropout,
       "temporal_jitter": d_jitter, "phase_shift": d_phase, "peak_broadening": d_broad, "quantization": d_quant,
       "hr_bias": d_hrbias, "polarity_inversion": d_polarity, "dicrotic_notch_atten": d_notch,
       "extra_beats": d_extra, "peak_sharpening": d_sharp}
BY_DESIGN = {"amplitude_scale": "z-normalization removes amplitude (no response by design)",
             "dc_offset": "z-normalization removes offset (no response by design)",
             "phase_shift": "lag up to a quarter cycle is separated as dt (SHAPE/RHY unresponsive by design; WCR responds beyond the cap)"}


def run():
    rows = []; t0 = time.time()
    for fs in FS_LIST:
        seg = int(10 * fs)
        for rep in range(REPS):
            yb, hr = base(fs, np.random.default_rng(10_000 * fs + rep))
            ref = win(yb, fs)
            for di, (name, fn) in enumerate(DEG.items()):
                for lv in LEVELS:
                    rng = np.random.default_rng(1_000_000 + 10_000 * fs + 100 * rep + di)   # fixed regardless of level
                    pred = win(fn(yb.copy(), fs, float(lv), rng, hr), fs)
                    r = RV.score_chunks(ref.reshape(-1, seg), pred.reshape(-1, seg), fs)
                    rows.append(dict(fs=fs, rep=rep, degradation=name, level=float(lv), HR=hr, **{k: r[k] for k in
                                     ("RPFI", "s_SHAPE", "s_WCR", "s_RHY", "lag_s", "SHAPE_defined", "F1", "nIBIE", "lag_capped")}))
            print(f"  fs={fs} rep={rep + 1}/{REPS} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(d):
    out = []
    for (name), g in d.groupby("degradation", sort=False):
        row = dict(degradation=name)
        for c in COMP:
            rhos, rev, tot = [], 0, 0
            for _, tr in g.groupby(["fs", "rep"]):
                tr = tr.sort_values("level"); v = tr[c].values
                rhos.append(-spearmanr(tr.level, v)[0] if np.std(v) > 1e-9 else 0.0)
                dv = np.diff(v); rev += int((dv > 0.01).sum()); tot += len(dv)
            m0 = g[g.level == 0][c].median(); m1 = g[g.level == 1][c].median()
            row[f"{c}_rho"] = float(np.nanmedian(rhos)); row[f"{c}_drop"] = float(m0 - m1); row[f"{c}_rev"] = rev / tot
        row["lag_ms_at_max"] = 1000 * g[g.level == 1].lag_s.median()
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = run(); d.to_csv("deg17_v2_raw.csv", index=False)
    S = summarize(d)
    lines = ["17 degradation types x 6 levels x 8 repetitions x 2 fs = %d cases (%d paired trajectories)" % (len(d), len(d) // 6),
             "rho: median per-trajectory Spearman(level, score) signed so that 'worse' = + / drop: median score difference between level 0 and 1",
             "rev: fraction of adjacent steps where the score 'improves' by more than 0.01 (reversal rate)", ""]
    pd.set_option("display.width", 250)
    lines.append(S.round(3).to_string(index=False))
    lines += ["", "No response by design:"] + [f"  {k}: {v}" for k, v in BY_DESIGN.items()]
    rv = S[[f"{c}_rev" for c in COMP]]
    lines.append(f"\nOverall reversal rate: max {rv.values.max():.3f}, mean {rv.values.mean():.4f}")
    open("deg17_v2_report.txt", "w", encoding="utf-8").write("\n".join(lines)); print("\n".join(lines))
