"""
final_snr_reliability.py - SNR comparator and split-half reliability
==================================================================================
  1) rPPG SNR comparator (de Haan & Jeanne 2013)
       SNR = 10 log10( power within +/-0.1 Hz of the reference HR fundamental and +/-0.2 Hz of its 2nd harmonic
                       / remaining power in 0.5-4 Hz )
       The prediction is preprocessed per 10 s chunk (as in rpfi_eval) -> Hann window, 8192-point FFT -> chunk mean (dB).
       The fundamental is located on the *reference* signal of the same chunk (rpfi_eval.hr_from_psd).
       a) real model outputs, all 9 conditions                       -> final2/snr_real.csv
       b) Experiment 1 cases (exploration + hold-out, regenerated with the same seeds)
                                                                     -> final2/snr_exp1_{ubfc,pure,cohface,pure_holdout,cohface_holdout}.csv
  2) split-half reliability: in the within-dataset conditions, each participant's chunks are split into odd and even
     chunks and scored separately with rpfi_v2

Requires: rpfi_v2.py (2.0.2) and the files used by exp1_refeature.py (exp1_distortion_id.py, exp1_cases_*.csv, mapping_*.csv)
Usage (from the working folder):  python final_snr_reliability.py
Output: the final2/ folder
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import rpfi_eval as R
import rpfi_v2 as RV
from recompute_newsys import load_chunks
from sensitivity_v2 import LABEL_NAME, conditions

OUT = Path("final2")


def snr_chunk(L, P, fs, nfft=8192):
    f0 = R.hr_from_psd(L, fs) / 60.0
    if not np.isfinite(f0) or f0 <= 0 or np.std(P) < 1e-12:
        return np.nan
    x = (P - P.mean()) * np.hanning(P.size)
    S = np.abs(np.fft.rfft(x, n=max(nfft, P.size))) ** 2; f = np.fft.rfftfreq(max(nfft, P.size), 1 / fs)
    band = (f >= 0.5) & (f <= 4.0)
    sig = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2))
    num, den = S[sig].sum(), S[band & ~sig].sum()
    return float(10 * np.log10(num / den)) if num > 0 and den > 0 else np.nan


def snr_mean(L, P, fs):
    v = [snr_chunk(l, p, fs) for l, p in zip(L, P)]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def part_real(label_dir):
    rows, rel = [], []
    for tag, ds, runs in conditions():
        fs = R.FS_BY_DATASET[ds]; lab = np.load(Path(label_dir) / LABEL_NAME[ds])
        for m in R.discover_models(runs):
            t0 = time.time()
            for subj, (Lr, Pr, L, P) in sorted(load_chunks(runs, lab, m, fs, 10 * fs, 100).items()):
                rows.append(dict(condition=tag, dataset=ds, model=m, participant=subj, SNR_dB=snr_mean(L, P, fs)))
                if tag == ds and len(Lr) >= 4:
                    row = dict(condition=tag, dataset=ds, model=m, participant=subj, n_chunks=len(Lr))
                    for half, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
                        r = RV.score_chunks(list(Lr[sl]), list(Pr[sl]), fs)
                        for k in ("RPFI", "s_SHAPE", "s_WCR", "s_RHY", "F1", "nIBIE", "SHAPE_defined", "lag_s", "HR_MAE_bpm"):
                            row[f"{k}_{half}"] = r[k]
                        row[f"SNR_{half}"] = snr_mean(L[sl], P[sl], fs)
                    rel.append(row)
            print(f"[{tag}] {m:14s} ({time.time() - t0:.1f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "snr_real.csv", index=False)
    pd.DataFrame(rel).to_csv(OUT / "split_half.csv", index=False)


def part_exp1(label_dir):
    import exp1_distortion_id as E
    DS = {"ubfc": ("UBFC_label.npy", "mapping_UBFC.csv", 30), "pure": ("PURE_label.npy", "mapping_PURE.csv", 30),
          "cohface": ("COHFACE_label.npy", "mapping_COHFACE.csv", 20)}
    SEED, N_BASE, N_CH = 2026, 40, 4
    for ds, (lab_f, mp, fs) in DS.items():
        for split in (("explore", "holdout") if ds != "ubfc" else ("explore",)):
            csv = Path(f"exp1_cases_{ds}.csv" if split == "explore" else f"exp1_cases_{ds}_holdout.csv")
            if not csv.exists():
                print(f"  {csv} not found - skipped"); continue
            cases = pd.read_csv(csv); E.N_CHUNKS = N_CH
            dur = 10.0 * N_CH; dur_total = max(dur * 1.3 + 1.0, 0.0)
            bases = E.real_bases(str(Path(label_dir) / lab_f), mp, fs, dur_total, N_BASE, np.random.default_rng(SEED), split)
            tag = ds if split == "explore" else f"{ds}_holdout"; rows = []; t0 = time.time()
            for bi, (rec, subj, yfull) in enumerate(bases):
                bid = f"{tag}_{bi}" if split == "explore" else f"{tag}_r{rec}"
                sub = cases[cases.base == bid]
                if sub.empty:
                    continue
                n = int(dur * fs); ctx = E.make_ctx(yfull, fs, np.random.default_rng(SEED + bi)); ref = E.Ref(yfull[:n], fs)
                for _, c in sub.iterrows():
                    fn, _ = E.DIST[c.distortion]
                    pred = fn(yfull, fs, float(c.strength), ctx)[:n]
                    P = ref.prep(pred)
                    rows.append(dict(base=bid, distortion=c.distortion, target=c.target, SNR_dB=snr_mean(ref.L, P, fs),
                                     WCR_check=R.compute_wcr(ref.L.ravel(), P.ravel())[0], WCR_csv=c.WCR))
            df = pd.DataFrame(rows); df.to_csv(OUT / f"snr_exp1_{tag}.csv", index=False)
            print(f"  [{tag}] cases {len(df)}  reproduction check max|dWCR| = {(df.WCR_check - df.WCR_csv).abs().max():.1e} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--skip-exp1", action="store_true")
    a = ap.parse_args()
    if RV.__version__ != "2.0.2":
        raise SystemExit(f"rpfi_v2.py version is {RV.__version__}; 2.0.2 is required.")
    OUT.mkdir(exist_ok=True)
    print("== 1) SNR of real outputs + 2) split-half reliability =="); part_real(a.label_dir)
    if not a.skip_exp1:
        print("\n== 1b) SNR of Experiment 1 cases =="); part_exp1(a.label_dir)
    print(f"\nDone: {OUT}")


if __name__ == "__main__":
    main()
