"""
exp1_distortion_id.py - Experiment 1: at the same CCC, can 'what went wrong' be told apart?
==================================================================================
Claim: predictions that look identical by CCC (a single score) differ in the *type* of error when seen through the
      RPFI component profile. Ground truth = the distortion we injected (not computed by any metric).

Design (as run in the exploration stage; the confirmation regroups the distortions into the pre-registered
shape / local / drift families, see exp1_confirm.py)
  - base segments: synthetic pulses (default) or real reference PPG (--label/--mapping).
  - 10 distortion types -> 3 families (the component each family targets)
      shape  (BWMD): peak_broadening, peak_sharpening, notch_attenuation
      timing (rhythm): ibi_jitter, missed_beats, extra_beats, rsa_removal, hr_error
      offset (WCR only): constant_lag, white_noise
    All beat manipulations are time-warps/edits based on beat times detected on the reference, so they apply to real signals as is.
  - strength matching: for each (base, distortion), bisection finds the strength at which the pipeline CCC (= the CCC term of
    WCR, same preprocessing and chunk concatenation) reaches the target (0.98, 0.95). (0.90/0.80 cannot be reached by shape
    distortions - shape errors live in harmonics and barely move energy or correlation, which is itself a result.)
    Cases with |CCC - target| > 0.02 are excluded as 'unreachable'.
  - feature sets
      RPFI      : BWMD, WCR, RHY                 (v1 BWMD = DTW 10% window + PeakDiff + area)
      RPFI_new  : SHAPE, WCR, RHY                (SHAPE = VPG correlation of lag-corrected beat templates, shape_template.py)
      conventional : RMSE, Pearson r, HR MAE, PSNR
      conventional + RPFI_v2 : incremental check
  - classification: standardization + multinomial logistic regression, GroupKFold by base (the same reference segment is never
    in both training and evaluation). Metric: balanced accuracy (3 families, and 10 types).
  - uncertainty: base-level cluster bootstrap (B = 2000) 95% CI of the balanced-accuracy difference.

Success criteria (fixed before running)
  S1. 3-family balanced accuracy of RPFI_new - that of the conventional metrics > 0, with the 95% CI lower bound > 0
  S2. RPFI_new balanced accuracy >= 0.70 (chance 0.33)
  Both are required to claim that the decomposition gives diagnostic information beyond the conventional metrics.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, sosfiltfilt

sys.path.insert(0, ".")
import rpfi_eval as R
from rhythm import rhythm_participant, detect_beats
from shape_template import shape_participant

N_CHUNKS = 6
FAMILY = {"peak_broadening": "shape", "peak_sharpening": "shape", "notch_attenuation": "shape",
          "ibi_jitter": "timing", "missed_beats": "timing", "extra_beats": "timing",
          "rsa_removal": "timing", "hr_error": "timing",
          "constant_lag": "offset", "white_noise": "offset"}
TARGETS = (0.98, 0.95)   # shape distortions cannot lower CCC much -> compare where CCC "looks very good"


# -- base segments ----------------------------------------------------------------
def synth_base(rng, fs, dur):
    from synth import Pulse
    p = Pulse(rng)
    t = np.arange(int(dur * fs)) / fs
    return p(t)


def subject_col(mp):
    for c in ("subject_id", "subject"):
        if c in mp.columns:
            return c
    return None


def split_recordings(mapping_path, fs, dur, n_max, rng):
    """Exploration/hold-out split. Eligible recordings (length >= dur) are shuffled with rng.permutation; the first n_max
    are exploration, the rest hold-out. Same seed and same first rng call as the exploration stage (2026-11-23/24),
    so the same split is reproduced."""
    mp = pd.read_csv(mapping_path)
    elig = [i for i, r in mp.iterrows() if int(r.frame_end) - int(r.frame_start) >= int(dur * fs)]
    perm = rng.permutation(len(elig))
    return mp, [elig[i] for i in perm[:n_max]], [elig[i] for i in perm[n_max:]]


def real_bases(label_path, mapping_path, fs, dur, n_max, rng, split="explore"):
    """Returns [(recording row, subject, signal)]. split = explore (default) | holdout."""
    lab = np.load(label_path).astype(float).ravel()
    mp, ex, ho = split_recordings(mapping_path, fs, dur, n_max, rng)
    sc = subject_col(mp)
    out = []
    for i in (ex if split == "explore" else ho):
        s = int(mp.loc[i, "frame_start"])
        out.append((i, str(mp.loc[i, sc]) if sc else str(i), lab[s:s + int(dur * fs)]))
    return out


# -- distortions (y: raw signal, fs, s: strength >= 0, ctx: fixed random numbers and beat info) --
def _bp(lo, hi, fs, order=3):
    return butter(order, [lo / (fs / 2), min(hi / (fs / 2), 0.99)], btype="band", output="sos")


def make_ctx(y, fs, rng):
    yb = R.znorm(filtfilt(*butter(3, [0.5 / (fs / 2), min(5.0 / (fs / 2), 0.99)], btype="band"), y))
    b = detect_beats(yb, fs, prom=0.3)
    troughs = np.array([b[k] + np.argmin(yb[int(b[k] * fs):int(b[k + 1] * fs)]) / fs
                        for k in range(len(b) - 1)]) if len(b) > 1 else np.array([])
    f0 = R.hr_from_psd(yb, fs) / 60.0
    return dict(beats=b, troughs=troughs, f0=f0, z=rng.standard_normal(len(b) + 5),
                perm=rng.permutation(10_000), w=rng.standard_normal(y.size), n=y.size)


def _warp(y, fs, src_of_t):
    """pred(t) = y(src_of_t(t)) - evaluates the raw signal continuously with a cubic spline."""
    t = np.arange(y.size) / fs
    cs = CubicSpline(t, y, extrapolate=True)
    return cs(np.clip(src_of_t(t), 0, t[-1]))


def _beat_warp(y, fs, b, b_new):
    """Monotone time-warp moving beat times b -> b_new."""
    t = np.arange(y.size) / fs
    knots_new = np.concatenate([[0], b_new, [t[-1]]]); knots_old = np.concatenate([[0], b, [t[-1]]])
    order = np.argsort(knots_new)
    return _warp(y, fs, lambda tt: np.interp(tt, knots_new[order], knots_old[order]))


def d_peak_broadening(y, fs, s, c):          # s in [0, 1]: low-pass cutoff 3.0 -> 1.1 Hz
    if s <= 0: return y.copy()
    fc = 3.0 - 1.9 * min(s, 1.0)
    return filtfilt(*butter(4, fc / (fs / 2), btype="low"), y)


def d_peak_sharpening(y, fs, s, c):          # unsharp masking: y + s·k·(y − LP(y))
    lp = filtfilt(*butter(4, 1.2 / (fs / 2), btype="low"), y)
    return y + 3.0 * s * (y - lp)


def d_notch_attenuation(y, fs, s, c):        # attenuate the 2nd harmonic (2 f0 +/- 0.3 Hz)
    f2 = 2 * c["f0"]
    if not np.isfinite(f2) or f2 + 0.3 >= fs / 2: return y.copy()
    return y - min(s, 1.0) * sosfiltfilt(_bp(max(f2 - 0.3, 0.2), f2 + 0.3, fs), y)


def d_ibi_jitter(y, fs, s, c):               # independent per-beat time offsets, SD = 40 ms x s
    b = c["beats"]
    return _beat_warp(y, fs, b, b + 0.04 * s * c["z"][:b.size])


def d_rsa_removal(y, fs, s, c):              # remove the variation of beat intervals (mean HR kept)
    b = c["beats"]
    if b.size < 3: return y.copy()
    k = np.arange(b.size); uni = np.polyval(np.polyfit(k, b, 1), k)
    return _beat_warp(y, fs, b, b + min(s, 1.0) * (uni - b))


def d_hr_error(y, fs, s, c):                 # scale the HR itself by (1 + 0.3 s)
    return _warp(y, fs, lambda t: t * (1 + 0.3 * s))


def d_missed_beats(y, fs, s, c):             # flatten a trough-to-trough segment (remove one beat), fraction 0.5 s
    tr = c["troughs"]; out = y.copy(); t = np.arange(y.size) / fs
    if tr.size < 2: return out
    n = int(round(min(s, 1.0) * 0.5 * (tr.size - 1)))
    for k in c["perm"][:n] % (tr.size - 1):
        m = (t >= tr[k]) & (t <= tr[k + 1])
        if m.sum() > 2: out[m] = np.linspace(out[m][0], out[m][-1], m.sum())
    return out


def d_extra_beats(y, fs, s, c):              # spurious beats between beats (width 0.12 s), fraction 0.5 s
    b = c["beats"]; t = np.arange(y.size) / fs
    if b.size < 2: return y.copy()
    mids = (b[:-1] + b[1:]) / 2
    n = int(round(min(s, 1.0) * 0.5 * mids.size))
    bump = sum(np.exp(-((t - mids[k]) / 0.12) ** 2) for k in c["perm"][:n] % mids.size) if n else 0
    return y + 1.0 * np.std(y) * bump


def d_constant_lag(y, fs, s, c):             # constant lag of s seconds
    return _warp(y, fs, lambda t: t - s)


def d_white_noise(y, fs, s, c):
    return y + s * np.std(y) * c["w"][:y.size]


DIST = {"peak_broadening": (d_peak_broadening, 1.0), "peak_sharpening": (d_peak_sharpening, 15.0),
        "notch_attenuation": (d_notch_attenuation, 1.0), "ibi_jitter": (d_ibi_jitter, 6.0),
        "missed_beats": (d_missed_beats, 1.0), "extra_beats": (d_extra_beats, 1.0),
        "rsa_removal": (d_rsa_removal, 1.0), "hr_error": (d_hr_error, 1.0),
        "constant_lag": (d_constant_lag, 0.45), "white_noise": (d_white_noise, 3.0)}


# -- scoring ----------------------------------------------------------------------
class Ref:
    def __init__(self, y, fs):
        self.fs, self.seg = fs, 10 * fs
        self.L = np.array([R.preprocess_chunk(y[i * self.seg:(i + 1) * self.seg], fs) for i in range(N_CHUNKS)])

    def prep(self, p):
        return np.array([R.preprocess_chunk(p[i * self.seg:(i + 1) * self.seg], self.fs) for i in range(N_CHUNKS)])

    def ccc(self, p):
        return R.concordance_ccc(self.L.ravel(), self.prep(p).ravel())


def bwmd_v2(a_, b_, fs, win_s=0.3, lam_a=0.5):
    a = a_ / (np.max(np.abs(a_)) + 1e-8); b = b_ / (np.max(np.abs(b_)) + 1e-8)
    dtw = R.dtw_distance(a, b, sakoe_chiba_ratio=win_s * fs / len(a))
    ay, ah = np.abs(a_), np.abs(b_)
    area = float(R._trapz(np.minimum(ay, ah))) / (float(R._trapz(np.maximum(ay, ah))) + 1e-8)
    return dtw / (1 + lam_a * area)


def features(ref, p, y_ref_raw):
    fs = ref.fs; L = ref.L; P = ref.prep(p); seg = ref.seg
    sh = shape_participant([y_ref_raw[i * seg:(i + 1) * seg] for i in range(N_CHUNKS)],
                           [p[i * seg:(i + 1) * seg] for i in range(N_CHUNKS)], fs, L, P)
    y, q = L.ravel(), P.ravel()
    wcr, ccc, rho = R.compute_wcr(y, q)
    rh = rhythm_participant(L, P, fs)
    hr = [abs(R.hr_from_psd(L[i], fs) - R.hr_from_psd(P[i], fs)) for i in range(N_CHUNKS)]
    return dict(BWMD=float(np.mean([R.compute_bwmd(L[i], P[i], fs)[0] for i in range(N_CHUNKS)])),
                SHAPE=sh["SHAPE"],
                WCR=wcr, CCC=ccc, RHY=rh["RHY"], F1=rh["F1"], nIBIE=rh["nIBIE"],
                RMSE=R.compute_rmse(y, q), Pearson=float(np.corrcoef(y, q)[0, 1]),
                HR_MAE=float(np.nanmean(hr)), PSNR=R.compute_psnr(y, q))


def match_strength(ref, y, fs, fn, smax, ctx, target, iters=16, tol=0.02):
    f = lambda s: ref.ccc(fn(y, fs, s, ctx))
    lo, hi = 0.0, smax
    c_hi = f(hi)
    if c_hi > target:                      # even the maximum strength cannot reach the target CCC
        return None, c_hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > target: lo = mid
        else: hi = mid
    s = (lo + hi) / 2; c = f(s)
    return (s, c) if abs(c - target) <= tol else (None, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=20, help="number of base segments per fs")
    ap.add_argument("--label", default=None); ap.add_argument("--mapping", default=None)
    ap.add_argument("--fs", type=int, default=None, help="dataset fs in real-signal mode")
    ap.add_argument("--tag", default="synth")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split", choices=["explore", "holdout"], default="explore",
                    help="real-signal mode: explore = n_base exploration recordings, holdout = all remaining hold-out recordings (confirmation)")
    ap.add_argument("--n-chunks", type=int, default=6,
                    help="number of 10 s chunks to score. 4 recommended for real signals (~60 s recordings), leaving room for hr_error (x1.3)")
    a = ap.parse_args()
    global N_CHUNKS
    N_CHUNKS = a.n_chunks
    rng = np.random.default_rng(a.seed)
    dur = 10.0 * N_CHUNKS
    dur_total = max(dur * 1.3 + 1.0, 80.0 if not a.label else 0.0)   # includes room for hr_error
    bases = []
    if a.label:
        if a.fs is None or a.mapping is None:
            sys.exit("real-signal mode requires --label, --mapping and --fs.")
        fs = a.fs
        for i, (rec, subj, y) in enumerate(real_bases(a.label, a.mapping, fs, dur_total, a.n_base, rng, a.split)):
            bid = f"{a.tag}_{i}" if a.split == "explore" else f"{a.tag}_r{rec}"
            bases.append((bid, fs, y, rec, subj))
    else:
        for fs in (30, 20):
            for i in range(a.n_base):
                bases.append((f"fs{fs}_{i}", fs, synth_base(rng, fs, dur_total), -1, f"fs{fs}_{i}"))

    rows, skipped = [], []
    for bi, (bid, fs, yfull, rec, subj) in enumerate(bases):
        n = int(dur * fs)
        ctx = make_ctx(yfull, fs, np.random.default_rng(a.seed + bi))
        ref = Ref(yfull[:n], fs)
        for dname, (fn, smax) in DIST.items():
            for tgt in TARGETS:
                s, c = match_strength(ref, yfull, fs, lambda yy, ff, ss, cc, fn=fn: fn(yy, ff, ss, cc)[:n] if True else None,
                                      smax, ctx, tgt)
                if s is None:
                    skipped.append(dict(base=bid, distortion=dname, target=tgt, ccc_at_max=c)); continue
                pred = fn(yfull, fs, s, ctx)[:n]
                rows.append(dict(base=bid, rec=rec, subject=subj, fs=fs, distortion=dname, family=FAMILY[dname], target=tgt,
                                 strength=s, **features(ref, pred, yfull[:n])))
        print(f"  base {bi + 1}/{len(bases)} done (cases so far {len(rows)}, unreachable {len(skipped)})", flush=True)
    pd.DataFrame(rows).to_csv(f"exp1_cases_{a.tag}.csv", index=False)
    pd.DataFrame(skipped).to_csv(f"exp1_unreachable_{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
