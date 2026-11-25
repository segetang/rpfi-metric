"""
rpfi_v2.py - RPFI v2 scorer (rPPG Physiological Fidelity Index, version 2)
=============================================================================
Given a reference (contact) PPG and an rPPG prediction, returns the component profile and the composite score.
This is the same computation as the paper's analysis pipeline (recompute_newsys / sensitivity_v2 / wcr_capped / recompute_hr).

  Profile
    s_SHAPE  shape     : 1 - SHAPE = clip(beat-template VPG correlation, 0, 1).
                         If the phase-coherence gate fails, SHAPE is undefined -> s_SHAPE = 0
    s_WCR    agreement : WCR (beta = 0.6 CCC + 0.4 Spearman) after lag alignment capped at a quarter cycle, clipped to [0, 1]
    s_RHY    rhythm    : 1 - RHY,  RHY = 1 - F1 * (1 - min(nIBIE, 1)),  beat-matching tolerance 0.15 * RR
    lag_s    dt        : constant-lag estimate (s, positive = prediction lags) - reported separately, not scored
    polarity_suspect   : polarity-inversion flag (diagnostic, not scored).
                         The sign of the prediction is flipped and WCR is re-measured the same way (quarter-cycle-capped
                         lag alignment); True if the flipped WCR >= 0.3 and exceeds the original by >= 0.3.
                         In principle this cannot be distinguished from a half-cycle lag, so read it as
                         "suspected polarity inversion or half-cycle lag".
    F1, nIBIE          : rhythm sub-scores (beat detection / inter-beat-interval error)
    HR_MAE_bpm         : mean spectral HR error over 10 s chunks (reported alongside)
  Composite
    RPFI = (s_SHAPE * s_WCR * s_RHY)^(1/3)   (weights 1:1:1, no epsilon - if any component is 0, RPFI is 0)
  Model-level summary (aggregate): participant mean RPFI, coverage (fraction with SHAPE defined), RPFI given covered
  Input checks (2.0.2)
    - reference chunk with non-finite values (NaN/inf) or numerically flat -> dropped from scoring (invalid reference)
    - prediction chunk with non-finite values -> replaced by a flat output (0) (counted as a model failure, not hidden)
    - the fraction of numerically flat prediction chunks is reported as pred_flat_frac
      (flat = standard deviation <= 1e-9 * (max absolute value + 1e-12))
    - no valid chunk at all -> ValueError
    - amplitude itself is not evaluated (scale invariant): even a very small output is judged by its shape
  Known limitation: the minimum beat spacing (0.6 x the signal's own period) prevents double counting of dicrotic
    peaks, but a single spurious peak that is lower than the true beats and falls between them is merged and not
    counted as a false positive. A spurious peak large enough to displace a true beat, or one in the late part of
    the cycle (~0.75), is caught as FN + FP.

Example
    import rpfi_v2 as RV
    r = RV.score(ref, pred, fs=30)                                    # one recording
    r = RV.score_participant([(ref1, pred1), (ref2, pred2)], fs=30)   # one participant (several recordings)
    summary = RV.aggregate([r1, r2, ...])                             # model level
Command line
    python rpfi_v2.py --ref label.npy --pred pred.npy --fs 30

Requires: numpy, scipy, rpfi_eval.py, rhythm.py, shape_template.py (same folder)
"""
import argparse
import json

import numpy as np

import rpfi_eval as R
from rhythm import rhythm_participant
from shape_template import shape_participant

__version__ = "2.0.2"
SEG_S = 10.0
LAMBDA = 100
LAG_CAP_FRAC = 0.25
POL_MIN, POL_GAIN = 0.3, 0.3


def _shift_trim_wcr(L, P, fs, d):
    """Shift the prediction earlier by d seconds, trim |d| from both ends of each chunk, and compute WCR
    (same as sensitivity_v2.shift_trim)."""
    m = int(np.ceil(abs(d) * fs)) + 1
    t = np.arange(L.shape[1]) / fs
    ys, qs = [], []
    for l, p in zip(L, P):
        q = np.interp(t + d, t, p)
        if 2 * m < len(t) - 10:
            ys.append(l[m:-m]); qs.append(q[m:-m])
    if not ys:
        return np.nan
    return R.compute_wcr(np.concatenate(ys), np.concatenate(qs))[0]


def _hr_mae(L, P, fs):
    a = np.array([R.hr_from_psd(c, fs) for c in L], float)
    b = np.array([R.hr_from_psd(c, fs) for c in P], float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def _is_flat(x):
    x = np.asarray(x, float)
    return bool(np.std(x) <= 1e-9 * (np.max(np.abs(x)) + 1e-12))


def score_chunks(ref_chunks, pred_chunks, fs):
    """List of raw 10 s chunks -> profile + RPFI."""
    Lr0 = [np.asarray(c, float) for c in ref_chunks]
    Pr0 = [np.asarray(c, float) for c in pred_chunks]
    if len(Lr0) == 0 or len(Lr0) != len(Pr0):
        raise ValueError("Reference and prediction must have the same number of chunks (at least one).")
    Lr, Pr, n_ref_bad, n_pred_nonfinite = [], [], 0, 0
    for l, p in zip(Lr0, Pr0):
        if (not np.all(np.isfinite(l))) or _is_flat(l):
            n_ref_bad += 1; continue
        if not np.all(np.isfinite(p)):
            p = np.zeros_like(l); n_pred_nonfinite += 1
        Lr.append(l); Pr.append(p)
    if not Lr:
        raise ValueError("No valid reference chunk (all non-finite or flat).")
    pred_flat_frac = float(np.mean([_is_flat(p) for p in Pr]))
    L = np.array([R.preprocess_chunk(c, fs, True, True, LAMBDA) for c in Lr])
    P = np.array([R.preprocess_chunk(c, fs, True, True, LAMBDA) for c in Pr])
    rh = rhythm_participant(L, P, fs)
    sh = shape_participant(Lr, Pr, fs, L, P)
    lag = float(rh["offset_s"]); rr = float(rh["rr_med_s"])
    cap = LAG_CAP_FRAC * rr
    lag_used = float(np.clip(lag, -cap, cap))
    wcr_al = _shift_trim_wcr(L, P, fs, lag_used)
    # polarity flag: lag-aligned WCR of the sign-flipped prediction
    rf = rhythm_participant(L, -P, fs)
    capf = LAG_CAP_FRAC * float(rf["rr_med_s"])
    wcr_flip = _shift_trim_wcr(L, -P, fs, float(np.clip(rf["offset_s"], -capf, capf)))
    pol = bool(np.isfinite(wcr_flip) and wcr_flip >= POL_MIN and wcr_flip - (wcr_al if np.isfinite(wcr_al) else 0) >= POL_GAIN)
    defined = not bool(sh["gated"])
    s_shape = float(np.clip(1.0 - sh["SHAPE"], 0, 1)) if defined else 0.0
    s_wcr = float(np.clip(wcr_al, 0, 1)) if np.isfinite(wcr_al) else 0.0
    s_rhy = float(np.clip(1.0 - rh["RHY"], 0, 1))
    S = np.array([s_shape, s_wcr, s_rhy])
    rpfi = float(np.exp(np.mean(np.log(S)))) if np.all(S > 0) else 0.0
    return dict(RPFI=rpfi, s_SHAPE=s_shape, s_WCR=s_wcr, s_RHY=s_rhy, SHAPE_defined=defined,
                lag_s=lag, lag_used_s=lag_used, lag_capped=bool(abs(lag) > cap),
                F1=float(rh["F1"]), nIBIE=float(rh["nIBIE"]), HR_MAE_bpm=_hr_mae(L, P, fs),
                WCR_raw=float(R.compute_wcr(L.ravel(), P.ravel())[0]), WCR_aligned=float(wcr_al),
                WCR_flipped=float(wcr_flip), polarity_suspect=pol,
                r_vpg=float(sh["r_vpg"]), coh_pred=float(sh["coh_pred"]), rr_med_s=rr,
                n_chunks=len(L), n_ref_chunks_dropped=n_ref_bad, n_pred_chunks_nonfinite=n_pred_nonfinite,
                pred_flat_frac=pred_flat_frac, version=__version__)


def _chunks(x, fs):
    return list(R.chunkify(np.asarray(x, float).ravel(), int(round(SEG_S * fs))))


def score(ref, pred, fs):
    """One recording. If lengths differ, the shorter length is used; a trailing remainder under 10 s is discarded."""
    n = min(len(ref), len(pred))
    return score_chunks(_chunks(ref[:n], fs), _chunks(pred[:n], fs), fs)


def score_participant(recordings, fs):
    """Several recordings of one participant [(ref, pred), ...] - chunks are pooled and scored once
    (same as the paper pipeline)."""
    Lc, Pc = [], []
    for ref, pred in recordings:
        n = min(len(ref), len(pred))
        lc, pc = _chunks(ref[:n], fs), _chunks(pred[:n], fs)
        k = min(len(lc), len(pc)); Lc += lc[:k]; Pc += pc[:k]
    return score_chunks(Lc, Pc, fs)


def aggregate(results):
    """List of participant results -> model-level summary. Mean RPFI = coverage x RPFI given covered."""
    rp = np.array([r["RPFI"] for r in results], float)
    df = np.array([r["SHAPE_defined"] for r in results], bool)
    out = dict(n=len(results), RPFI_mean=float(rp.mean()), coverage=float(df.mean()),
               RPFI_given_covered=float(rp[df].mean()) if df.any() else float("nan"))
    for k in ("s_WCR", "s_RHY", "F1", "nIBIE", "HR_MAE_bpm"):
        out[f"{k}_mean"] = float(np.nanmean([r[k] for r in results]))
    out["s_SHAPE_mean_covered"] = float(np.mean([r["s_SHAPE"] for r, d in zip(results, df) if d])) if df.any() else float("nan")
    out["lag_s_median"] = float(np.median([r["lag_s"] for r in results]))
    out["polarity_suspect_rate"] = float(np.mean([r["polarity_suspect"] for r in results]))
    return out


def _load(path):
    if path.endswith(".npy"):
        return np.load(path).astype(float).ravel()
    return np.loadtxt(path, delimiter=",").astype(float).ravel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RPFI v2 scoring: reference and predicted PPG (.npy or single-column .csv)")
    ap.add_argument("--ref", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--fs", type=float, required=True)
    a = ap.parse_args()
    fs = int(a.fs) if float(a.fs).is_integer() else a.fs
    print(json.dumps(score(_load(a.ref), _load(a.pred), fs), ensure_ascii=False, indent=2))
