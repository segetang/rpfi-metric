"""
rpfi_eval.py - shared preprocessing, metrics and statistics (originally the RPFI v1 evaluation pipeline)
====================================================================================
RPFI v2 (rpfi_v2.py) uses this module for chunking, preprocessing (Tarvainen detrend -> 0.75-3.0 Hz zero-phase
band-pass -> z-normalization, identical for reference and prediction), WCR/CCC, spectral HR, and the
participant-level statistics (cluster bootstrap, Holm correction, Wilcoxon post-hoc).

The v1 components (BWMD, EDD), fixed v1 anchors and the v1 command-line evaluation (main) are kept unchanged so that
v1 numbers (the 'RPFI_v1' column and the conventional metrics RMSE/CCC reported in the v2 paper) can be reproduced.
They are not part of the v2 score.

Notes on choices that also apply to v2
  - Analysis unit = participant: predictions from all folds are regrouped by test_subject_of_range in
    runs/{ds}_cv{K}/*_info.json; with subject-wise K-fold every participant is a test subject exactly once.
  - Statistics: participant-level cluster bootstrap; Friedman test, then pairwise Wilcoxon signed-rank with Holm.
  - Spectral HR (hr_from_psd, changed 2026-11-23): zero-padded Hann periodogram + parabolic interpolation instead of
    the Welch maximum bin (7/6 bpm quantization); the old estimator is kept as hr_from_psd_legacy for comparison.
  - PSNR uses MAX = maximum absolute value of the z-normalized reference; PRV in ms.

Input format (outputs of train_models.py / train_bimamba_blend.py)
-------------------------------------------------------------
    runs/{dataset}_cv{K}/{model}_f{k}_test_pred.npy   object array, one 1D array per recording (z-space)
    runs/{dataset}_cv{K}/{model}_f{k}_info.json       test_ranges, test_subject_of_range, ...
    ../real data/{DS}_label.npy                       full-dataset label (raw)

Usage (v1 evaluation)
------
    python rpfi_eval.py --dataset ubfc \
        --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" \
        --out results_eval

    # subset of models / band-pass sensitivity check
    python rpfi_eval.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" \
        --models rnn,bilstm,bimamba --no-bpf --out results_eval_nobpf
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

try:
    from fastdtw import fastdtw as _fastdtw
    USE_FASTDTW = True
except ImportError:
    USE_FASTDTW = False

# numpy 2.0 renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ══════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════

FS_BY_DATASET = {"pure": 30, "ubfc": 30, "cohface": 20}

CARDIAC_BAND = (0.75, 3.0)          # Hz — 45~180 bpm
COMPONENTS = ["BWMD", "WCR", "EDD"]

# component direction: lower = smaller is better
DIRECTION = {"BWMD": "lower", "WCR": "higher", "EDD": "lower"}

# v1 fixed anchors (independent of the model pool).
#   WCR/EDD: bounded by definition - WCR is in [-1, 1], but any negative correlation is treated as worst (lower anchor 0);
#            EDD is a power ratio in [0, 1].
#   BWMD: has a theoretical upper bound (3.0) on the z-normalized path that is never reached, so the p1-p99 of the
#         synthetic degradation benchmark is used as the anchor.
#   Can be overridden with --anchors (JSON); a warning is printed if observed values fall outside the anchors.
DEFAULT_ANCHORS = {
    "BWMD": (0.009, 0.859),
    "WCR":  (0.0, 1.0),
    "EDD":  (0.0, 1.0),
}

# v1 weights (after removing the earlier SRE component, its weight 0.2 was redistributed in proportion
# to BWMD:WCR:EDD = 0.3:0.3:0.2 -> 0.375 / 0.375 / 0.25).
DEFAULT_WEIGHTS = {"BWMD": 0.375, "WCR": 0.375, "EDD": 0.25}

BWMD_LAMBDA_P = 1.0      # weight of the peak-timing term
BWMD_LAMBDA_A = 0.5      # weight of the amplitude-overlap term
WCR_BETA = 0.6           # CCC vs. Spearman mix
# EDD uses CARDIAC_BAND as its band (see compute_edd)


# ══════════════════════════════════════════════════════════════════
# Preprocessing - applied identically to label and prediction
# ══════════════════════════════════════════════════════════════════

def detrend_tarvainen(signal, Lambda=100):
    """Tarvainen et al. (2002) smoothness-priors detrending."""
    n = len(signal)
    if n < 3:
        return np.asarray(signal, dtype=float)
    H = np.identity(n)
    ones = np.ones(n)
    diags_data = np.array([ones, -2 * ones, ones])
    D = spdiags(diags_data, np.array([0, 1, 2]), (n - 2), n).toarray()
    return np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)


def bandpass(x, fs, low=CARDIAC_BAND[0], high=CARDIAC_BAND[1], order=6):
    nyq = 0.5 * fs
    hi = min(high / nyq, 0.99)
    lo = max(low / nyq, 1e-6)
    if lo >= hi:
        return np.asarray(x, dtype=float)
    b, a = butter(order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, np.asarray(x, dtype=np.double))


def znorm(x):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def preprocess_chunk(x, fs, do_detrend=True, do_bpf=True, Lambda=100):
    """Per-chunk preprocessing (standard path used by WCR, rhythm, SHAPE gating). Applied identically to label and prediction."""
    y = np.asarray(x, dtype=float).copy()
    if do_detrend and len(y) >= 3:
        y = detrend_tarvainen(y, Lambda)
    if do_bpf and len(y) > 18:      # room for the filtfilt padlen
        y = bandpass(y, fs)
    return znorm(y)


def preprocess_chunk_edd(x, fs, do_detrend=True, Lambda=100):
    """
    v1 EDD preprocessing - detrend + z-norm only, **never band-pass filtered**.

    Reason: the standard-path signals are band-passed to 0.75-3.0 Hz, so the residual's in-band power ratio
    measured on them would always be ~1.0 (the denominator is already confined to the band). EDD needs
    out-of-band residual content, so this function always skips the BPF regardless of --no-bpf.
    """
    y = np.asarray(x, dtype=float).copy()
    if do_detrend and len(y) >= 3:
        y = detrend_tarvainen(y, Lambda)
    return znorm(y)


def chunkify(sig, seg_len):
    """Cut a 1D signal into non-overlapping chunks of seg_len samples; the remainder is discarded."""
    n = (len(sig) // seg_len) * seg_len
    if n == 0:
        return np.empty((0, seg_len))
    return np.asarray(sig[:n], dtype=float).reshape(-1, seg_len)


# ══════════════════════════════════════════════════════════════════
# Data loading - regroup per-fold predictions by participant
# ══════════════════════════════════════════════════════════════════

def discover_models(runs_dir):
    models = set()
    for p in Path(runs_dir).glob("*_f*_info.json"):
        stem = p.name[: -len("_info.json")]
        model = stem.rsplit("_f", 1)[0]
        models.add(model)
    return sorted(models)


def load_participant_data(runs_dir, label_full, model, fs, seg_len,
                          do_detrend, do_bpf, Lambda):
    """
    Returns {participant_id: {"label": (n_chunk, seg_len),        # standard path (BPF)
                            "pred": (n_chunk, seg_len),
                            "label_edd": (n_chunk, seg_len),     # v1 EDD only (no BPF)
                            "pred_edd": (n_chunk, seg_len),
                            "n_rec": int, "folds": set}}
    With subject-wise K-fold, each participant appears as a test subject in exactly one fold.
    """
    runs_dir = Path(runs_dir)
    per_subj_label = defaultdict(list)
    per_subj_pred = defaultdict(list)
    per_subj_label_edd = defaultdict(list)
    per_subj_pred_edd = defaultdict(list)
    per_subj_nrec = defaultdict(int)
    per_subj_folds = defaultdict(set)

    info_files = sorted(runs_dir.glob(f"{model}_f*_info.json"))
    if not info_files:
        return {}

    for inf_p in info_files:
        info = json.load(open(inf_p))
        fold = info.get("fold")
        pred_p = runs_dir / f"{model}_f{fold}_test_pred.npy"
        if not pred_p.exists():
            raise FileNotFoundError(f"prediction file not found: {pred_p}")
        preds = np.load(pred_p, allow_pickle=True)

        ranges = info["test_ranges"]
        subj_of = info["test_subject_of_range"]
        if not (len(preds) == len(ranges) == len(subj_of)):
            raise ValueError(
                f"{model} fold{fold}: count mismatch "
                f"(pred {len(preds)} / ranges {len(ranges)} / subjects {len(subj_of)})")

        for rec_i, ((s, e), subj) in enumerate(zip(ranges, subj_of)):
            p_rec = np.asarray(preds[rec_i], dtype=float).ravel()
            l_rec = np.asarray(label_full[s:e], dtype=float).ravel()
            L = min(len(p_rec), len(l_rec))
            if L < seg_len:
                continue
            p_ch = chunkify(p_rec[:L], seg_len)
            l_ch = chunkify(l_rec[:L], seg_len)
            n = min(len(p_ch), len(l_ch))
            if n == 0:
                continue
            for c in range(n):
                per_subj_pred[subj].append(
                    preprocess_chunk(p_ch[c], fs, do_detrend, do_bpf, Lambda))
                per_subj_label[subj].append(
                    preprocess_chunk(l_ch[c], fs, do_detrend, do_bpf, Lambda))
                per_subj_pred_edd[subj].append(
                    preprocess_chunk_edd(p_ch[c], fs, do_detrend, Lambda))
                per_subj_label_edd[subj].append(
                    preprocess_chunk_edd(l_ch[c], fs, do_detrend, Lambda))
            per_subj_nrec[subj] += 1
            per_subj_folds[subj].add(fold)

    out = {}
    for subj in sorted(per_subj_label):
        out[subj] = {
            "label": np.array(per_subj_label[subj]),
            "pred": np.array(per_subj_pred[subj]),
            "label_edd": np.array(per_subj_label_edd[subj]),
            "pred_edd": np.array(per_subj_pred_edd[subj]),
            "n_rec": per_subj_nrec[subj],
            "folds": sorted(per_subj_folds[subj]),
        }
    return out


# ══════════════════════════════════════════════════════════════════
# v1 component 1: BWMD - morphology (DTW + peak timing) / amplitude overlap
# ══════════════════════════════════════════════════════════════════

def dtw_distance(a, b, sakoe_chiba_ratio=0.1):
    """
    DTW with a Sakoe-Chiba band constraint, made dimensionless:
      (i) amplitude: each signal divided by max|.| -> scale removed, in [-1, 1]
      (ii) path length: cumulative cost divided by (na + nb) -> length dependence removed
    With both normalizations dtw_norm is bounded in [0, 2].
    """
    na, nb = len(a), len(b)
    band = max(1, int(sakoe_chiba_ratio * max(na, nb)))

    if USE_FASTDTW:
        dist, _ = _fastdtw(a, b, radius=band, dist=lambda x, y: abs(x - y))
        return float(dist) / max(na + nb, 1)

    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        j_lo, j_hi = max(1, i - band), min(nb, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb]) / max(na + nb, 1)


def peak_times(sig, fs):
    """Systolic peak times (s). Minimum spacing 0.4 s = 150 bpm upper limit."""
    peaks, _ = find_peaks(sig, distance=max(1, int(0.4 * fs)))
    return peaks / float(fs)


def median_rr(pt):
    return float(np.median(np.diff(pt))) if len(pt) >= 2 else None


def compute_bwmd(y, yhat, fs):
    """
    BWMD = (DTW_norm + λp·PeakDiff_norm) / (1 + λa·AreaOverlap)

    PeakDiff_norm is divided by one RR period (dimensionless) and clamped to [0, 1];
    1.0 = peaks off by a full cardiac cycle = phase information completely lost.
    AreaOverlap is an IoU (0-1), so the denominator is in [1, 1 + lambda_a].
    """
    a = y / (np.max(np.abs(y)) + 1e-8)
    b = yhat / (np.max(np.abs(yhat)) + 1e-8)
    dtw_norm = dtw_distance(a, b)

    pt_y, pt_h = peak_times(y, fs), peak_times(yhat, fs)
    if len(pt_y) >= 2 and len(pt_h) >= 2:
        k = min(len(pt_y), len(pt_h))
        diff_mean = float(np.mean(np.abs(pt_y[:k] - pt_h[:k])))
        rr = median_rr(pt_y)
        rr_norm = rr if (rr and rr > 1e-6) else (len(y) / float(fs))
        peak_norm = diff_mean / rr_norm
    elif len(pt_y) >= 1 and len(pt_h) >= 1:
        peak_norm = abs(pt_y[0] - pt_h[0]) / (len(y) / float(fs))
    else:
        peak_norm = 0.5      # peak detection failed: neutral value
    peak_norm = float(np.clip(peak_norm, 0.0, 1.0))

    abs_y, abs_h = np.abs(y), np.abs(yhat)
    inter = float(_trapz(np.minimum(abs_y, abs_h)))
    union = float(_trapz(np.maximum(abs_y, abs_h)))
    area_overlap = inter / (union + 1e-8)

    bwmd = (dtw_norm + BWMD_LAMBDA_P * peak_norm) / (1.0 + BWMD_LAMBDA_A * area_overlap)
    return float(bwmd), float(dtw_norm), peak_norm, float(area_overlap)


# ══════════════════════════════════════════════════════════════════
# RMSE: conventional error metric, reported for reference only (not part of any composite).
# ══════════════════════════════════════════════════════════════════

def compute_rmse(y, yhat):
    """RMSE (reference only, not in the composite). On the z-normalized path RMSE is in [0, 2]."""
    e = np.ravel(yhat) - np.ravel(y)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 0.0


# ══════════════════════════════════════════════════════════════════
# WCR - agreement (CCC + Spearman); used by v1 and, lag-aligned, by v2
# ══════════════════════════════════════════════════════════════════

def concordance_ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return 0.0
    return float(np.clip(2.0 * cov / denom, -1.0, 1.0))


def compute_wcr(y, yhat, beta=WCR_BETA):
    x, z = np.ravel(y), np.ravel(yhat)
    ccc = concordance_ccc(x, z)
    rho, _ = spearmanr(x, z)
    if np.isnan(rho):
        rho = 0.0
    return float(np.clip(beta * ccc + (1 - beta) * rho, -1.0, 1.0)), ccc, float(rho)


# ══════════════════════════════════════════════════════════════════
# v1 component 3: EDD - in-band fraction of the residual power
# ══════════════════════════════════════════════════════════════════

def compute_edd(y_nobpf, yhat_nobpf, fs, band=CARDIAC_BAND):
    """
    EDD (v1) - fraction of the residual power inside the cardiac band.

    e = yhat - y (computed on detrended + z-normalized signals without band-pass)
    EDD = (cardiac-band power of e) / (total power of e)

    Lower is better. Residual energy spread outside the cardiac band means what the model missed is
    unstructured noise (sensor noise, timing jitter); residual energy concentrated inside the band means the
    prediction carries a coherent but wrong pulse (wrong phase or wrong rate), a 'ghost pulse'.

    Bounded in [0, 1] by definition; the anchors are the theoretical bounds.

    (An earlier histogram-based JSD version rewarded POS on UBFC: the Gaussianity of the residual is unrelated
    to fidelity, since the difference of two unrelated signals tends towards Gaussian.)

    y_nobpf / yhat_nobpf must come from preprocess_chunk_edd() (no band-pass); band-passed inputs make
    the ratio always ~1.0 and meaningless.
    """
    e = np.ravel(yhat_nobpf) - np.ravel(y_nobpf)
    nperseg = min(256, e.size)
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(e, fs=fs, nperseg=nperseg)
    m = (freqs >= band[0]) & (freqs <= band[1])
    band_power = _trapz(psd[m], freqs[m])
    total_power = _trapz(psd, freqs)
    if total_power < 1e-12:
        return 0.0                       # residual ~ 0 -> no deviation (degenerate case)
    return float(np.clip(band_power / total_power, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════
# Auxiliary metrics: HR, PRV
# ══════════════════════════════════════════════════════════════════

HR_NFFT_MIN = 8192


def hr_from_psd(sig, fs, band=CARDIAC_BAND, nfft_min=HR_NFFT_MIN):
    """
    Spectral HR (bpm) - zero-padded Hann periodogram + parabolic interpolation.

    The previous estimator (-> hr_from_psd_legacy) took the maximum bin of welch(nperseg=min(256, N)):
      - grid = fs/nperseg -> quantized to 7.03 bpm (30 Hz, 300 samples) and 6.00 bpm (20 Hz, 200 samples);
        if reference and prediction fall in the same bin, HR MAE is exactly 0
        (checked 2026-11: true for 97 participant-model combinations on UBFC).
      - at 30 Hz there was a single segment, discarding the last 44 samples (15%) of each chunk.
    Now: Hann window over the whole chunk -> zero-padding to nfft >= 8192 (grid <= 0.22 bpm) ->
         log-power parabolic interpolation around the in-band maximum. Synthetic check (validate_hr.py):
         pure sine MAE 0.00 bpm, PPG-like + noise 0.3-0.5 bpm (previously 1.5-1.9 bpm).
    Resolution (separating two components) is set by the window length and is not increased by zero-padding;
    what this fixes is the grid quantization of locating a single dominant frequency.
    """
    x = np.asarray(sig, dtype=float).ravel()
    n = x.size
    if n < max(16, int(2 * fs)) or not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
        return np.nan
    x = x - x.mean()
    nfft = max(int(nfft_min), 1 << int(np.ceil(np.log2(n))))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)      # periodic Hann
    spec = np.abs(np.fft.rfft(x * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if idx.size == 0:
        return np.nan
    k = int(idx[np.argmax(spec[idx])])
    f = freqs[k]
    if idx[0] < k < idx[-1]:            # no interpolation for a maximum at the band edge
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
        den = a - 2 * b + c
        if den < 0:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
            f = freqs[k] + delta * (freqs[1] - freqs[0])
    return float(60.0 * f)


def hr_from_peaks(sig, fs, band=CARDIAC_BAND, prominence=0.3):
    """Mean HR from peak spacing = 60 (n - 1) / (t_last - t_first). An auxiliary estimate independent of the
    spectrum (reference only); unstable at low SNR, so not used as a primary metric."""
    x = np.asarray(sig, dtype=float).ravel()
    if x.size < 2 * fs or np.std(x) < 1e-12:
        return np.nan
    x = (x - x.mean()) / x.std()
    pk, _ = find_peaks(x, distance=max(1, int(np.floor(fs / band[1]))),
                       prominence=prominence)
    if pk.size < 3:
        return np.nan
    span = (pk[-1] - pk[0]) / fs
    return float(60.0 * (pk.size - 1) / span) if span > 0 else np.nan


def hr_from_psd_legacy(sig, fs, band=CARDIAC_BAND):
    """Previous estimator (Welch bin, 7/6 bpm quantization) - kept only for reproduction and comparison."""
    nperseg = min(256, len(sig))
    if nperseg < 16:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m):
        return np.nan
    return float(freqs[m][np.argmax(psd[m])] * 60.0)


def _hr_mae(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def prv_from_chunks(chunks, fs):
    """
    PRV (pulse rate variability): pulse-wave based, hence PRV rather than ECG HRV.
    Units: ms (pNN50 in %).

    - RR intervals are computed only *within* each chunk; detecting peaks on concatenated chunks would create
      intervals across chunk boundaries that do not exist and contaminate SDNN/RMSSD.
    - A 10 s chunk holds only about 8-15 RR samples, so SDNN/RMSSD confidence intervals are wide.
      Reported as a diagnostic only; not part of any score.
    """
    rr_all, n_short = [], 0
    for ch in chunks:
        pt = peak_times(ch, fs)
        if len(pt) < 2:
            n_short += 1
            continue
        rr_all.append(np.diff(pt))
    if not rr_all:
        return {k: np.nan for k in
                ["mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "mean_hr_bpm", "n_rr"]}
    rr = np.concatenate(rr_all)
    drr = np.concatenate([np.diff(r) for r in rr_all if len(r) > 1]) \
        if any(len(r) > 1 for r in rr_all) else np.array([])
    mean_rr = float(np.mean(rr))
    return {
        "mean_rr_ms": mean_rr * 1000.0,
        "sdnn_ms": float(np.std(rr, ddof=1)) * 1000.0 if rr.size > 1 else 0.0,
        "rmssd_ms": float(np.sqrt(np.mean(drr ** 2))) * 1000.0 if drr.size else 0.0,
        "pnn50_pct": 100.0 * float(np.mean(np.abs(drr) > 0.05)) if drr.size else 0.0,
        "mean_hr_bpm": 60.0 / mean_rr if mean_rr > 0 else np.nan,
        "n_rr": int(rr.size),
    }


def compute_psnr(ref, pred):
    """
    PSNR (dB). MAX = maximum absolute value of the z-normalized reference
    (there is no fixed dynamic range such as 255 for images).
    """
    mse = float(np.mean((ref - pred) ** 2))
    if mse == 0:
        return float("inf")
    max_val = float(np.max(np.abs(ref)))
    return float(10.0 * np.log10((max_val ** 2) / mse))


# ══════════════════════════════════════════════════════════════════
# v1 normalization and aggregation (fixed anchors, non-compensatory)
# ══════════════════════════════════════════════════════════════════

def normalize_component(name, value, anchors):
    """Raw component value -> [0, 1] quality score (1 = best). Independent of the model pool."""
    lo, hi = anchors[name]
    if hi <= lo:
        return 0.0
    t = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    return 1.0 - t if DIRECTION[name] == "lower" else t


def aggregate(scores, weights, mode="geometric", eps=1e-6):
    """
    scores: {comp: [0, 1] quality score}
    mode:
      "arithmetic" - weighted sum. Compensatory (a zero on one axis is offset by the others).
      "geometric"  - weighted geometric mean. Non-compensatory: if one axis is near 0, the total is pulled to 0.
    """
    w_sum = sum(weights[c] for c in COMPONENTS)
    if mode == "arithmetic":
        agg = sum(weights[c] * scores[c] for c in COMPONENTS) / w_sum
    else:
        log_sum = sum(weights[c] * np.log(max(scores[c], eps)) for c in COMPONENTS)
        agg = float(np.exp(log_sum / w_sum))
    return float(np.clip(agg, 0.0, 1.0) * 100.0)


# ══════════════════════════════════════════════════════════════════
# Statistics - participant-level cluster bootstrap, post-hoc tests
# ══════════════════════════════════════════════════════════════════

def cluster_bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    """Resample participants as clusters. `values` holds one value per participant."""
    vals = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if vals.size < 2:
        m = float(vals.mean()) if vals.size else np.nan
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boots = vals[idx].mean(axis=1)
    a = (100 - ci) / 2
    return (float(vals.mean()),
            float(np.percentile(boots, a)),
            float(np.percentile(boots, 100 - a)))


def holm_adjust(pvals):
    """Holm-Bonferroni adjustment."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def posthoc_wilcoxon(model_scores, models):
    """
    Pairwise Wilcoxon signed-rank tests with Holm adjustment (after a significant Friedman test).
    model_scores: {model: list of per-participant scores} (same participant order assumed)
    """
    pairs, stats, praw = [], [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = np.asarray(model_scores[models[i]], dtype=float)
            b = np.asarray(model_scores[models[j]], dtype=float)
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 6 or np.allclose(a[m], b[m]):
                continue
            try:
                s, p = wilcoxon(a[m], b[m])
            except ValueError:
                continue
            pairs.append((models[i], models[j]))
            stats.append(float(s))
            praw.append(float(p))
    if not pairs:
        return []
    padj = holm_adjust(praw)
    return [(pa[0], pa[1], st, pr, pj, float(np.nanmean(model_scores[pa[0]]) -
                                             np.nanmean(model_scores[pa[1]])))
            for pa, st, pr, pj in zip(pairs, stats, praw, padj)]


# ══════════════════════════════════════════════════════════════════
# v1 sensitivity / redundancy
# ══════════════════════════════════════════════════════════════════

def weight_sensitivity(per_model_comp_means, anchors, base_weights,
                       mode, n_draw=500, seed=42):
    """
    Draw random weights from a Dirichlet and measure how much the model ranking moves.
    Returns (distribution of Spearman rho with the base ranking, top-1 retention rate)
    """
    rng = np.random.default_rng(seed)
    models = list(per_model_comp_means)

    def rank_with(w):
        sc = []
        for m in models:
            s = {c: normalize_component(c, per_model_comp_means[m][c], anchors)
                 for c in COMPONENTS}
            sc.append(aggregate(s, w, mode))
        return np.asarray(sc)

    base = rank_with(base_weights)
    base_top = models[int(np.argmax(base))]
    alpha = np.array([base_weights[c] for c in COMPONENTS]) * 20.0
    rhos, top_same = [], 0
    for _ in range(n_draw):
        draw = rng.dirichlet(alpha)
        w = {c: float(draw[i]) for i, c in enumerate(COMPONENTS)}
        cur = rank_with(w)
        r, _ = spearmanr(base, cur)
        if not np.isnan(r):
            rhos.append(r)
        if models[int(np.argmax(cur))] == base_top:
            top_same += 1
    return np.asarray(rhos), top_same / float(n_draw), base_top


# ══════════════════════════════════════════════════════════════════
# Main (v1 evaluation)
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True, help="runs/{dataset}_cv{K} directory")
    ap.add_argument("--label", required=True, help="full-dataset label .npy")
    ap.add_argument("--models", default=None, help="comma separated; auto-discovered if omitted")
    ap.add_argument("--out", default="results_eval")
    ap.add_argument("--fs", type=int, default=None, help="dataset default if omitted")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--agg", choices=["geometric", "arithmetic"], default="geometric",
                    help="geometric = non-compensatory (default), arithmetic = weighted sum")
    ap.add_argument("--anchors", default=None, help="override anchors with a JSON file")
    ap.add_argument("--weights", default=None, help="override weights with a JSON file")
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true", help="band-pass sensitivity check")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-multi-fold", action="store_true",
                    help="proceed with a warning if a participant appears in several folds. "
                        "Use only when scoring cross_dataset_eval.py outputs. "
                        "If a within-dataset run needs this, the --runs path is probably wrong.")
    args = ap.parse_args()

    fs = args.fs or FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    do_detrend = not args.no_detrend
    do_bpf = not args.no_bpf

    anchors = dict(DEFAULT_ANCHORS)
    if args.anchors:
        anchors.update({k: tuple(v) for k, v in json.load(open(args.anchors)).items()})
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.load(open(args.weights)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_full = np.load(args.label)
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else discover_models(args.runs))
    if not models:
        raise SystemExit(f"no models found in {args.runs}")

    print("=" * 78)
    print(f"RPFI (v1) evaluation - dataset={args.dataset}  fs={fs}Hz  "
          f"chunk={seg_len} samples ({args.eval_seconds}s)")
    print(f"  preprocessing: detrend={do_detrend}(lambda={args.lambda_detrend})  "
          f"bandpass={do_bpf}{CARDIAC_BAND if do_bpf else ''}  z-norm=True (identical for label/pred)")
    print(f"  aggregation: {args.agg}{'  (non-compensatory)' if args.agg == 'geometric' else '  (compensatory)'}"
          f"   weights: {weights}")
    print(f"  anchors (fixed): " + ", ".join(f"{c}{anchors[c]}" for c in COMPONENTS))
    print(f"  DTW backend: {'fastdtw' if USE_FASTDTW else 'naive+Sakoe-Chiba'}")
    print(f"  {len(models)} models: {', '.join(models)}")
    print("=" * 78)

    timing = defaultdict(list)
    rows = []                       # full participant x model record
    per_model_participant = defaultdict(dict)   # model -> {subj: rpfi}
    per_model_comp_vals = defaultdict(lambda: defaultdict(list))
    raw_component_pool = defaultdict(list)      # for the anchor check

    for model in models:
        data = load_participant_data(args.runs, label_full, model, fs, seg_len,
                                     do_detrend, do_bpf, args.lambda_detrend)
        if not data:
            print(f"  [skipped] {model}: no outputs")
            continue

        # -- safeguard: with within-dataset K-fold each participant must be a test subject in exactly one fold.
        # Two or more folds almost certainly means cross_dataset_eval.py outputs (the source's 5 fold checkpoints
        # applied to the whole target) are being scored through a wrong --runs path. --
        multi = {s: d["folds"] for s, d in data.items() if len(d["folds"]) > 1}
        if multi and not args.allow_multi_fold:
            sample = list(multi.items())[:3]
            raise SystemExit(
                f"\n[!] {model}: {len(multi)} participants appear in several folds "
                f"(e.g. {sample}).\n"
                f"    With within-dataset K-fold a participant must be a test subject in exactly one "
                f"fold.\n"
                f"    --runs='{args.runs}' is probably a cross_dataset_eval.py output"
                f" (e.g. runs/cross_X_to_{args.dataset}) - check the path.\n"
                f"    If you are deliberately scoring cross-dataset results, add --allow-multi-fold.")
        elif multi:
            print(f"  [i] {model}: {len(multi)} participants appear in several folds "
                  f"(allowed by --allow-multi-fold - make sure this is cross-dataset scoring)")

        for subj, d in sorted(data.items()):
            L, P = d["label"], d["pred"]
            L_e, P_e = d["label_edd"], d["pred_edd"]
            n_chunk = len(L)
            if n_chunk == 0:
                continue

            # -- per chunk: BWMD, HR --
            bwmds, hr_l, hr_p = [], [], []
            t0 = time.perf_counter()
            for c in range(n_chunk):
                b, _, _, _ = compute_bwmd(L[c], P[c], fs)
                bwmds.append(b)
            timing["BWMD"].append((time.perf_counter() - t0) / n_chunk)
            hr_lpk, hr_ppk, hr_lleg, hr_pleg = [], [], [], []
            for c in range(n_chunk):
                hr_l.append(hr_from_psd(L[c], fs))
                hr_p.append(hr_from_psd(P[c], fs))
                hr_lpk.append(hr_from_peaks(L[c], fs))
                hr_ppk.append(hr_from_peaks(P[c], fs))
                hr_lleg.append(hr_from_psd_legacy(L[c], fs))
                hr_pleg.append(hr_from_psd_legacy(P[c], fs))

            y_cat, p_cat = L.ravel(), P.ravel()
            y_cat_e, p_cat_e = L_e.ravel(), P_e.ravel()

            t0 = time.perf_counter(); wcr, ccc, rho = compute_wcr(y_cat, p_cat)
            timing["WCR"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); edd = compute_edd(y_cat_e, p_cat_e, fs)
            timing["EDD"].append(time.perf_counter() - t0)
            rmse = compute_rmse(y_cat, p_cat)      # diagnostic only, not in the composite

            bwmd = float(np.mean(bwmds))
            comp = {"BWMD": bwmd, "WCR": wcr, "EDD": edd}
            scores = {c: normalize_component(c, comp[c], anchors) for c in COMPONENTS}
            rpfi = aggregate(scores, weights, args.agg)
            rpfi_alt = aggregate(scores, weights,
                                 "arithmetic" if args.agg == "geometric" else "geometric")

            hr_l_a = np.asarray(hr_l, dtype=float)
            hr_p_a = np.asarray(hr_p, dtype=float)
            hm = ~(np.isnan(hr_l_a) | np.isnan(hr_p_a))
            hr_mae = float(np.mean(np.abs(hr_l_a[hm] - hr_p_a[hm]))) if hm.any() else np.nan

            prv_l = prv_from_chunks(L, fs)
            prv_p = prv_from_chunks(P, fs)

            for c in COMPONENTS:
                per_model_comp_vals[model][c].append(comp[c])
                raw_component_pool[c].append(comp[c])
            per_model_participant[model][subj] = rpfi

            rows.append(dict(
                dataset=args.dataset, model=model, participant=subj,
                n_recordings=d["n_rec"], n_chunks=n_chunk, folds="|".join(map(str, d["folds"])),
                BWMD=bwmd, WCR=wcr, EDD=edd,
                s_BWMD=scores["BWMD"],
                s_WCR=scores["WCR"], s_EDD=scores["EDD"],
                RPFI=rpfi, RPFI_alt_agg=rpfi_alt,
                RMSE=rmse, PSNR_dB=compute_psnr(y_cat, p_cat),
                CCC=ccc, Spearman=rho,
                HR_label_bpm=float(np.nanmean(hr_l_a)), HR_pred_bpm=float(np.nanmean(hr_p_a)),
                HR_MAE_bpm=hr_mae,
                HR_MAE_peak_bpm=_hr_mae(hr_lpk, hr_ppk),
                HR_MAE_legacy_bpm=_hr_mae(hr_lleg, hr_pleg),
                PRV_sdnn_label_ms=prv_l["sdnn_ms"], PRV_sdnn_pred_ms=prv_p["sdnn_ms"],
                PRV_rmssd_label_ms=prv_l["rmssd_ms"], PRV_rmssd_pred_ms=prv_p["rmssd_ms"],
                PRV_n_rr_label=prv_l["n_rr"], PRV_n_rr_pred=prv_p["n_rr"],
            ))

        print(f"  [done] {model:16s} participants={len(data)}  "
              f"chunks={sum(len(d['label']) for d in data.values())}")

    if not rows:
        raise SystemExit("no results were aggregated.")

    # -- anchor check --
    print("\n" + "=" * 78)
    print("Anchor check (observed values outside the fixed anchors are clipped = information loss)")
    print("=" * 78)
    print(f"  {'Comp':>6s} | {'anchor_lo':>10s} {'anchor_hi':>10s} | "
          f"{'obs_p1':>10s} {'obs_p99':>10s} | {'clipped':>10s}")
    any_clip = False
    for c in COMPONENTS:
        v = np.asarray(raw_component_pool[c])
        lo, hi = anchors[c]
        n_clip = int(((v < lo) | (v > hi)).sum())
        any_clip |= n_clip > 0
        flag = "" if n_clip == 0 else "  ⚠"
        print(f"  {c:>6s} | {lo:>10.4f} {hi:>10.4f} | "
              f"{np.percentile(v, 1):>10.4f} {np.percentile(v, 99):>10.4f} | "
              f"{n_clip:>5d}/{v.size}{flag}")
    if any_clip:
        print("\n  [!] Clipping occurred. Anchors must come from an external reference, not from the evaluated")
        print("    model pool, so do not refit them on this run. Re-run the synthetic degradation")
        print("    benchmark, take that component's p1-p99 again,")
        print("    and pass it with --anchors.")

    # -- model summary (participant cluster bootstrap CI) --
    print("\n" + "=" * 78)
    print(f"Model summary - participant-level cluster bootstrap 95% CI (n_boot={args.n_boot})")
    print("=" * 78)

    summary_rows, ranking = [], {}
    for model in models:
        if model not in per_model_participant:
            continue
        subs = sorted(per_model_participant[model])
        rpfis = [per_model_participant[model][s] for s in subs]
        r_m, r_lo, r_hi = cluster_bootstrap_ci(rpfis, args.n_boot, seed=args.seed)
        ranking[model] = r_m

        line = {"model": model, "n_participants": len(subs),
                "RPFI": r_m, "RPFI_lo": r_lo, "RPFI_hi": r_hi}
        print(f"\n>> {model}   (participants={len(subs)})")
        print(f"   RPFI : {r_m:6.2f}   95%CI [{r_lo:6.2f}, {r_hi:6.2f}]")
        for c in COMPONENTS:
            m_, lo_, hi_ = cluster_bootstrap_ci(per_model_comp_vals[model][c],
                                                args.n_boot, seed=args.seed)
            arrow = "↓" if DIRECTION[c] == "lower" else "↑"
            print(f"   {c:5s}: {m_:8.5f}  95%CI [{lo_:8.5f}, {hi_:8.5f}]  ({arrow})")
            line[c] = m_; line[f"{c}_lo"] = lo_; line[f"{c}_hi"] = hi_

        mrows = [r for r in rows if r["model"] == model]
        for key, lbl in [("HR_MAE_bpm", "HR MAE (bpm)"), ("RMSE", "RMSE"),
                         ("PSNR_dB", "PSNR (dB)")]:
            v = np.asarray([r[key] for r in mrows], dtype=float)
            line[key] = float(np.nanmean(v))
            print(f"   {lbl:16s}: {np.nanmean(v):8.4f}")
        summary_rows.append(line)

    # -- computational cost --
    print("\n" + "=" * 78)
    print("Computational cost (per component, per participant / BWMD per chunk)")
    print("=" * 78)
    print(f"  {'Component':>12s} | {'Mean(ms)':>10s} | {'Std(ms)':>10s} | {'Max(ms)':>10s}")
    for c in COMPONENTS:
        if timing[c]:
            a = np.asarray(timing[c]) * 1000
            print(f"  {c:>12s} | {a.mean():>10.3f} | {a.std():>10.3f} | {a.max():>10.3f}")

    # -- component redundancy --
    print("\n" + "=" * 78)
    print("Component intercorrelation (Spearman) - redundancy check")
    print("=" * 78)
    pool = [np.asarray(raw_component_pool[c]) for c in COMPONENTS]
    print(f"{'':>8s}" + "".join(f"{c:>9s}" for c in COMPONENTS))
    for i, ci_ in enumerate(COMPONENTS):
        cells = []
        for j in range(len(COMPONENTS)):
            if i == j:
                cells.append(f"{1.0:>9.3f}")
            else:
                r, _ = spearmanr(pool[i], pool[j])
                cells.append(f"{0.0 if np.isnan(r) else r:>9.3f}")
        print(f"{ci_:>8s}" + "".join(cells))
    print("\n  |rho|<0.3 independent | 0.3-0.6 partly shared | >0.6 possibly redundant")

    # -- weight sensitivity --
    comp_means = {m: {c: float(np.mean(per_model_comp_vals[m][c])) for c in COMPONENTS}
                  for m in per_model_participant}
    if len(comp_means) >= 3:
        rhos, top_rate, base_top = weight_sensitivity(
            comp_means, anchors, weights, args.agg, seed=args.seed)
        print("\n" + "=" * 78)
        print("Weight sensitivity (500 Dirichlet draws)")
        print("=" * 78)
        print(f"  top model with the base weights: {base_top}")
        print(f"  ranking Spearman rho: mean={rhos.mean():.4f}  min={rhos.min():.4f}  "
              f"p5={np.percentile(rhos, 5):.4f}")
        print(f"  top-1 retention: {100 * top_rate:.1f}%")
        print("  -> rho near 1 and high retention = ranking robust to the choice of weights.")

    # -- Friedman + post-hoc --
    print("\n" + "=" * 78)
    print("Statistical significance of model differences (unit = participant)")
    print("=" * 78)
    valid = [m for m in models if m in per_model_participant]
    common = sorted(set.intersection(*[set(per_model_participant[m]) for m in valid])) \
        if valid else []
    print(f"  {len(valid)} models, common participants N={len(common)}")
    print(f"  Friedman chi2 theoretical maximum = N x (k-1) = {len(common)} x {len(valid) - 1} "
          f"= {len(common) * (len(valid) - 1)}"
          if len(valid) >= 2 else "")

    posthoc = []
    if len(valid) >= 3 and len(common) >= 5:
        groups = [[per_model_participant[m][s] for s in common] for m in valid]
        stat, p = friedmanchisquare(*groups)
        print(f"  Friedman χ² = {stat:.4f},  p = {p:.6g}")
        if p < 0.05:
            print("  -> significant (p<0.05). Running post-hoc tests.\n")
            posthoc = posthoc_wilcoxon(
                {m: [per_model_participant[m][s] for s in common] for m in valid}, valid)
            print(f"  {'model A':>16s} {'model B':>16s} {'ΔRPFI':>9s} "
                  f"{'p_raw':>10s} {'p_Holm':>10s}  sig")
            for a, b, _s, pr, pj, dlt in sorted(posthoc, key=lambda x: x[4]):
                print(f"  {a:>16s} {b:>16s} {dlt:>+9.2f} {pr:>10.4g} {pj:>10.4g}"
                      f"  {'*' if pj < 0.05 else ''}")
            n_sig = sum(1 for x in posthoc if x[4] < 0.05)
            print(f"\n  significant pairs after Holm: {n_sig}/{len(posthoc)}")
            if n_sig == 0:
                print("  [!] The omnibus test is significant but no single pair is - "
                      "the practical differences between models are small.")
        else:
            print("  -> not significant (p>=0.05). No RPFI difference between models can be claimed.")
    else:
        print("  test skipped: too few models or participants")

    # -- final ranking --
    print("\n" + "=" * 78)
    print(f"Final RPFI (v1) ranking (0-100, higher = more faithful / aggregation={args.agg})")
    print("=" * 78)
    for i, (m, s) in enumerate(sorted(ranking.items(), key=lambda x: -x[1]), 1):
        row = next(r for r in summary_rows if r["model"] == m)
        print(f"  {i:2d}. {m:18s} RPFI={s:6.2f}  "
              f"[{row['RPFI_lo']:.2f}, {row['RPFI_hi']:.2f}]   "
              f"HR MAE={row['HR_MAE_bpm']:.2f} bpm")

    # -- save CSVs --
    tag = f"{args.dataset}_{args.agg}" + ("" if do_bpf else "_nobpf")
    f1 = out_dir / f"rpfi_participant_{tag}.csv"
    with open(f1, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    f2 = out_dir / f"rpfi_summary_{tag}.csv"
    with open(f2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    files = [f1, f2]
    if posthoc:
        f3 = out_dir / f"rpfi_posthoc_{tag}.csv"
        with open(f3, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model_A", "model_B", "wilcoxon_stat", "p_raw", "p_holm", "delta_RPFI"])
            for a, b, s_, pr, pj, dl in posthoc:
                w.writerow([a, b, s_, pr, pj, dl])
        files.append(f3)

    cfg = dict(dataset=args.dataset, fs=fs, seg_len=seg_len,
               eval_seconds=args.eval_seconds, detrend=do_detrend,
               lambda_detrend=args.lambda_detrend, bandpass=do_bpf,
               cardiac_band=CARDIAC_BAND, aggregation=args.agg,
               weights=weights, anchors={k: list(v) for k, v in anchors.items()},
               models=list(ranking), n_boot=args.n_boot, seed=args.seed,
               dtw_backend="fastdtw" if USE_FASTDTW else "naive")
    f4 = out_dir / f"rpfi_config_{tag}.json"
    json.dump(cfg, open(f4, "w"), indent=2, default=str)
    files.append(f4)

    print("\nSaved:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
