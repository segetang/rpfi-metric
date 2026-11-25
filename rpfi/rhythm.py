"""
rhythm.py - RPFI rhythm (beat timing) component
========================================================
Purpose: define the 'plausible but physiologically wrong pulse' that v1's EDD targeted directly at the
beat level instead of through a spectral ratio. It avoids sample-wise comparison (which, after z-normalization,
converges to 1 - r, i.e. WCR) and serves beat-timing-dependent uses such as HRV/PRV and arrhythmia.

Procedure (per participant; chunks are rpfi_eval.preprocess_chunk outputs = detrend + BPF + z-norm)
  1) Beat detection: find_peaks on each chunk (minimum spacing from the 180 bpm band edge and the signal's own
     period, prominence >= PROM).
  2) Constant-lag (offset) correction: the measurement-site/device delay between facial rPPG and finger contact
     PPG is not an estimator error, so one constant lag d per participant is estimated as the median of
     the 'reference beat -> nearest predicted beat' time differences (within +/- RR/2), removed, and reported
     separately. (A 'maximize the number of matches' rule was discarded: because of chunk-boundary effects it
     can yield fewer matches at the true lag.)
  3) Matching: greedy 1:1 matching by distance within a tolerance of 0.15 x the reference median RR (equal to the
     conventional 150 ms at 60 bpm). Unmatched predicted beats = false beats (FP); unmatched reference beats =
     missed beats (FN). Beats within TOL of a chunk boundary are excluded, since their partner may lie outside the chunk.
  4) F1 = 2TP / (2TP + FP + FN)
  5) IBI error: an IBI pair is used only when two consecutive reference beats are both matched and their partners
     are also consecutive in the prediction. nIBIE = mean|IBI_ref - IBI_pred| / median(IBI_ref).
     With no IBI pair, nIBIE = 1 (no information).
  6) Raw score RHY = 1 - F1 * (1 - min(nIBIE, 1)) in [0, 1], lower is better
     (worse if all beats are found but intervals are wrong, or if intervals are right but beats are missed)
"""
import numpy as np
from scipy.signal import find_peaks

TOL_S = 0.150          # used only when tol_frac=None (fixed tolerance, not recommended)
TOL_FRAC = 0.15        # tolerance = 0.15 x reference median RR. Equals the conventional 150 ms at 60 bpm, and the
                       # chance-match level does not rise with HR (a fixed 150 ms gives null F1 0.34 -> 0.53 from 55 -> 125 bpm)
PROM = 0.3
HR_MAX = 180.0
RMSSD_FLOOR_S = 0.010   # floor of the RMSSD-normalized denominator (10 ms)


REFRACTORY_FRAC = 0.6   # minimum beat spacing = 0.6 x (period from the signal's own spectral HR)


def detect_beats(x, fs, prom=PROM):
    """Systolic peak detection. The minimum spacing follows the signal's own period rather than a fixed value
    (180 bpm). With a fixed spacing only, dicrotic/diastolic peaks of long (low-HR) beats are counted as separate
    beats (checked 2026-11: on the pure_05 reference, beats were over-counted 1.58x and the reference RMSSD was 386 ms)."""
    x = np.asarray(x, float)
    if x.size < 3 or np.std(x) < 1e-12:
        return np.array([], dtype=float)
    dist = fs * 60.0 / HR_MAX
    try:
        import rpfi_eval as _R
        hr = _R.hr_from_psd(x, fs)
        if np.isfinite(hr) and hr > 0:
            dist = max(dist, REFRACTORY_FRAC * fs * 60.0 / hr)
    except Exception:
        pass
    pk, _ = find_peaks(x, distance=max(1, int(np.floor(dist))), prominence=prom)
    # sub-sample peak location by parabolic interpolation (removes the 33 ms / 50 ms quantization at 30 / 20 Hz)
    t = pk.astype(float)
    inner = (pk > 0) & (pk < x.size - 1)
    k = pk[inner]
    a, b, c = x[k - 1], x[k], x[k + 1]
    den = a - 2 * b + c
    delta = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, -1.0), 0.0)
    t[inner] = k + np.clip(delta, -0.5, 0.5)
    return t / float(fs)


def _greedy_match(r, p, tol):
    """r, p: sorted beat times. Greedy 1:1 matching by distance. Returns [(i_ref, j_pred), ...]"""
    if r.size == 0 or p.size == 0:
        return []
    cand = []
    j0 = 0
    for i, t in enumerate(r):
        while j0 < p.size and p[j0] < t - tol:
            j0 += 1
        j = j0
        while j < p.size and p[j] <= t + tol:
            cand.append((abs(p[j] - t), i, j))
            j += 1
    cand.sort()
    ur, up, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in ur and j not in up:
            ur.add(i); up.add(j); pairs.append((i, j))
    return sorted(pairs)


def rhythm_participant(L, P, fs, tol=TOL_S, prom=PROM, lag_search=True, tol_frac=TOL_FRAC):
    """L, P: (n_chunk, seg_len) preprocessed reference/prediction. Returns dict(RHY, F1, nIBIE, offset_s, ...)"""
    seg_dur = L.shape[1] / float(fs)
    beats = [(detect_beats(L[c], fs, prom), detect_beats(P[c], fs, prom)) for c in range(len(L))]
    rr_all = np.concatenate([np.diff(r) for r, _ in beats if r.size > 1]) if beats else np.array([])
    rr_med = float(np.median(rr_all)) if rr_all.size else 0.8
    if tol_frac is not None:          # tolerance as a fraction of the cardiac period (chance-match level independent of HR)
        tol = tol_frac * rr_med

    def count(d, tol_):
        tp = fp = fn = 0
        dist = 0.0
        ibi_err = []
        for r, p in beats:
            q = p - d
            keep_r = (r >= tol_) & (r <= seg_dur - tol_)
            keep_p = (q >= tol_) & (q <= seg_dur - tol_)
            pairs = _greedy_match(r, q, tol_)
            mr = {i for i, _ in pairs}; mp = {j for _, j in pairs}
            tp += sum(1 for i, j in pairs if keep_r[i])
            dist += sum(abs(q[j] - r[i]) for i, j in pairs if keep_r[i])
            fn += sum(1 for i in range(r.size) if keep_r[i] and i not in mr)
            fp += sum(1 for j in range(q.size) if keep_p[j] and j not in mp)
            pm = dict(pairs)
            for i in range(r.size - 1):
                if i in pm and (i + 1) in pm and pm[i + 1] == pm[i] + 1:
                    ibi_err.append(abs((r[i + 1] - r[i]) - (q[pm[i + 1]] - q[pm[i]])))
        return tp, fp, fn, ibi_err, dist

    d_best = 0.0
    if lag_search:
        # constant lag = median signed time difference (each reference beat -> nearest predicted beat),
        # using only differences within +/- RR/2. Two passes (shift by the first estimate, then re-estimate) for stability.
        for _ in range(2):
            diffs = []
            for r, p in beats:
                q = p - d_best
                if r.size == 0 or q.size == 0:
                    continue
                k = np.clip(np.searchsorted(q, r), 1, q.size - 1) if q.size > 1 else np.zeros(r.size, int)
                cand = np.stack([q[np.maximum(k - 1, 0)], q[k]], 1) if q.size > 1 else q[k][:, None]
                nn = cand[np.arange(r.size), np.argmin(np.abs(cand - r[:, None]), 1)]
                dd = nn - r
                diffs.append(dd[np.abs(dd) <= rr_med / 2])
            diffs = np.concatenate(diffs) if diffs else np.array([])
            if diffs.size == 0:
                break
            d_best += float(np.median(diffs))
        d_best = float(np.clip(d_best, -rr_med / 2, rr_med / 2))

    tp, fp, fn, ibi_err, _ = count(d_best, tol)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    mae_ibi = float(np.mean(ibi_err)) if ibi_err else np.nan
    # nIBIE: IBI error normalized by the median RR (insensitive to HRV-scale errors; confirmed in synthetic checks)
    nibie = mae_ibi / rr_med if ibi_err else 1.0
    rhy = 1.0 - f1 * (1.0 - min(nibie, 1.0))
    # hIBIE (alternative, reported only): IBI error normalized by the reference's own beat-to-beat variation (RMSSD).
    #     If the error is as large as the reference's beat-to-beat variation, HRV information is lost -> 1.
    sd = [np.diff(np.diff(r)) for r, _ in beats if r.size > 2]
    sd = np.concatenate(sd) if sd else np.array([])
    rmssd_ref = float(np.sqrt(np.mean(sd ** 2))) if sd.size else np.nan
    hibie = mae_ibi / max(rmssd_ref, RMSSD_FLOOR_S) if ibi_err and np.isfinite(rmssd_ref) else 1.0
    rhy2 = 1.0 - f1 * (1.0 - min(hibie, 1.0))
    return dict(RHY=float(rhy), RHY2=float(rhy2), F1=float(f1), nIBIE=float(nibie), hIBIE=float(hibie),
                IBI_MAE_ms=mae_ibi * 1000 if ibi_err else np.nan, RMSSD_ref_ms=rmssd_ref * 1000,
                TP=tp, FP=fp, FN=fn, n_ibi=len(ibi_err), offset_s=d_best, rr_med_s=rr_med)
