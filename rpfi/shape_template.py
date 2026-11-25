"""
shape_template.py - SHAPE (morphology) component: lag-corrected, beat-normalized template comparison
=========================================================================================
In v1, the response of BWMD came mostly from PeakDiff (timing), and its DTW part had weak shape
specificity (synthetic check). Because morphology lives in low-energy harmonics, SHAPE compares
'the shape of one beat' directly rather than sample-wise correlation or energy.

Procedure (per participant)
  1) Build a separate wide-band signal (default 0.5-8 Hz, capped below Nyquist): band-pass + z-norm.
     Beat times are detected on the standard-band (0.75-3 Hz) signal (stable); a constant lag is estimated
     with the same median method as rhythm and removed from the prediction, so that timing error does not
     leak into the shape comparison.
  2) For each reference beat, the window [b - 0.3*RR, b + 0.7*RR] is cut at the same position from the
     reference and the prediction, and each window is normalized by its own mean and SD (amplitude/offset removed).
  3) Mean templates of reference and prediction are formed, and the correlation r_v between their first-derivative
     (VPG) templates is computed. Differentiation multiplies the k-th harmonic by k, emphasizing shape (harmonic) content.
  4) SHAPE = 1 - clip(r_v, 0, 1) in [0, 1], lower is better.
  Template averaging suppresses noise that is random from beat to beat and keeps systematic shape changes common to all beats.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

import rpfi_eval as R
from rhythm import detect_beats

WIDE_BAND = (0.5, 8.0)
N_PHASE = 64          # resample one beat to 64 points (phase normalization: resolution independent of HR and fs)


COH_MIN = 0.10        # minimum phase coherence: unrelated-signal null p95 ~ 0.03, white noise 2 SD p05 ~ 0.23


def prep_wide(x, fs, band=WIDE_BAND):
    # Tarvainen detrending (lambda = 100) is not used: at fs = 30 it attenuates 0.75 Hz by a factor of 0.86
    # (damping the fundamental at low HR, fs dependent). The 0.5 Hz high-pass removes baseline wander.
    y = np.asarray(x, float) - np.mean(x)
    hi = min(band[1], 0.45 * fs)
    b, a = butter(4, [band[0] / (fs / 2), hi / (fs / 2)], btype="band")
    return R.znorm(filtfilt(b, a, y))


def _offset(beats_ref, beats_pred, rr):
    d = 0.0
    for _ in range(2):
        diffs = []
        for r, p in zip(beats_ref, beats_pred):
            q = p - d
            if r.size == 0 or q.size == 0:
                continue
            nn = q[np.argmin(np.abs(q[None, :] - r[:, None]), axis=1)]
            dd = nn - r
            diffs.append(dd[np.abs(dd) <= rr / 2])
        diffs = np.concatenate(diffs) if diffs else np.array([])
        if diffs.size == 0:
            break
        d += float(np.median(diffs))
    return float(np.clip(d, -rr / 2, rr / 2))


def shape_participant(ref_raw_chunks, pred_raw_chunks, fs, L_std=None, P_std=None):
    """ref/pred_raw_chunks: raw chunks before preprocessing. L_std/P_std: standard-band preprocessed chunks (computed if None)."""
    if L_std is None:
        L_std = np.array([R.preprocess_chunk(c, fs) for c in ref_raw_chunks])
        P_std = np.array([R.preprocess_chunk(c, fs) for c in pred_raw_chunks])
    Lw = [prep_wide(c, fs) for c in ref_raw_chunks]
    Pw = [prep_wide(c, fs) for c in pred_raw_chunks]
    br = [detect_beats(x, fs) for x in L_std]
    bp = [detect_beats(x, fs) for x in P_std]
    rr_all = np.concatenate([np.diff(b) for b in br if b.size > 1]) if br else np.array([])
    rr = float(np.median(rr_all)) if rr_all.size else 0.8
    d = _offset(br, bp, rr)          # step 1: peak-based lag on the standard band
    pre = 0.3 * rr
    k = np.arange(N_PHASE) * (rr / N_PHASE)          # one cycle sampled at N_PHASE points (phase normalization)
    splines = []
    for lw, pw in zip(Lw, Pw):
        tt = np.arange(lw.size) / fs
        splines.append((tt[-1], CubicSpline(tt, lw), CubicSpline(tt, pw)))  # cubic spline: avoids the harmonic damping of linear interpolation

    def extract(dd):
        T_r, T_p = [], []
        for (tend, cr, cp), b in zip(splines, br):
            for tb in b:
                t_r = tb - pre + k
                t_p = t_r + dd
                if t_r[0] < 0 or t_p[0] < 0 or t_r[-1] > tend or t_p[-1] > tend:
                    continue
                wr, wp = cr(t_r), cp(t_p)
                if wr.std() < 1e-8 or wp.std() < 1e-8:
                    continue
                T_r.append((wr - wr.mean()) / wr.std()); T_p.append((wp - wp.mean()) / wp.std())
        return np.array(T_r), np.array(T_p)

    # step 2: fundamental-phase alignment. Shape is defined as the waveform difference left after aligning the fundamental.
    # (Alignment that maximizes VPG correlation is not used: it inflates the correlation even between unrelated signals.)
    e1 = np.exp(-2j * np.pi * np.arange(N_PHASE) / N_PHASE)
    for _ in range(3):
        T_r, T_p = extract(d)
        if len(T_r) < 3:
            break
        R1, P1 = T_r.mean(0) @ e1, T_p.mean(0) @ e1
        if abs(R1) < 1e-9 or abs(P1) < 1e-9:
            break
        s = -np.angle(P1 / R1) / (2 * np.pi)          # prediction lags by s cycles
        d = float(np.clip(d + s * rr, -rr, rr))
        if abs(s) < 1e-3:
            break
    T_r, T_p = extract(d)
    if len(T_r) < 3:                                # failure rule: fewer than 3 usable beats -> no shape information
        return dict(SHAPE=1.0, r_vpg=0.0, r_ppg=0.0, r_apg=0.0, r_vpg_beat=0.0, n_beats=len(T_r), offset_s=d,
                    coh_pred=0.0, coh_ref=0.0, gated=True)
    tr, tp = T_r.mean(0), T_p.mean(0)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0
    r_v = corr(np.diff(tr), np.diff(tp))
    r_0 = corr(tr, tp)
    r_a = corr(np.diff(tr, 2), np.diff(tp, 2))
    r_vb = float(np.mean([corr(np.diff(a), np.diff(b)) for a, b in zip(T_r, T_p)]))   # mean per-beat VPG correlation (for reference)
    # phase coherence: mean-template power / mean per-beat power. 1 if the same shape repeats every beat, ~1/n for an unrelated signal
    coh = lambda T: float(np.sum(T.mean(0) ** 2) / np.mean(np.sum(T ** 2, 1)))
    c_p = coh(T_p)
    # failure rule 2: if the prediction is not phase-locked to the reference beats (unrelated-signal level), the mean
    # template is a noise residue and its correlation can be high by chance (unrelated-signal r_vpg p95 ~ 0.90).
    # SHAPE is then undefined and set to the worst value (1).
    gated = c_p < max(COH_MIN, 5.0 / len(T_r))
    shape = 1.0 if gated else 1.0 - min(max(r_v, 0.0), 1.0)   # clip r to [0, 1] to fix the theoretical end points
    return dict(SHAPE=shape, r_vpg=r_v, r_ppg=r_0, r_apg=r_a, r_vpg_beat=r_vb, n_beats=len(T_r), offset_s=d,
                coh_pred=c_p, coh_ref=coh(T_r), gated=bool(gated))
