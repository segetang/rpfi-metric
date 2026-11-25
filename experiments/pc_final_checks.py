"""
pc_final_checks.py - final robustness checks and example figures
==================================================================================
  D1  sampling rate    : UBFC and PURE (30 Hz) downsampled to 20 Hz (resample_poly 2/3) and re-scored -> unit-wise RPFI and model ranking compared
  D2  recording length : each participant scored on the first 3 chunks (30 s) only -> compared with all chunks (3 within-dataset conditions)
  D3  lag-alignment cap: WCR recomputed with caps of 0.15 RR and 0.35 RR (default 0.25) - uses the lags in rpfi_v2_ALL.csv
  D4  harmonic-limited SHAPE: beat templates restricted to harmonics 1-3 (HR-dependence check, all 9 conditions)
  H   example pairs    : two real cases with similar HR error but different RPFI
Requires: rpfi_v2.py (2.0.2), shape_template_k.py, rpfi_v2_scores/rpfi_v2_ALL.csv, the runs folder and labels
Usage (from the working folder):  python pc_final_checks.py
Output: 4 CSVs and 2 PNGs in final_checks/
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")
import rpfi_eval as R
import rpfi_v2 as RV
import shape_template_k as SK
from recompute_newsys import load_chunks
from sensitivity_v2 import LABEL_NAME, conditions

OUT = Path("final_checks")
PAIRS = [("ubfc", "ubfc_S08", "bilstm", "physdiff"), ("ubfc", "ubfc_S25", "bimamba", "lstm")]


def within():
    for tag, ds, runs in conditions():
        if tag == ds:
            yield tag, ds, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    ap.add_argument("--scores", default="rpfi_v2_scores/rpfi_v2_ALL.csv")
    a = ap.parse_args()
    if RV.__version__ != "2.0.2":
        sys.exit(f"rpfi_v2.py version is {RV.__version__}; 2.0.2 is required.")
    V = pd.read_csv(a.scores).set_index(["condition", "model", "participant"])
    OUT.mkdir(exist_ok=True)
    labs = {ds: np.load(Path(a.label_dir) / LABEL_NAME[ds]) for ds in ("ubfc", "pure", "cohface")}

    d1, d2, d3, d4 = [], [], [], []
    for tag, ds, runs in conditions():
        fs = R.FS_BY_DATASET[ds]; lab = labs[ds]
        for m in R.discover_models(runs):
            t0 = time.time()
            for subj, (Lr, Pr, L, P) in sorted(load_chunks(runs, lab, m, fs, 10 * fs, 100).items()):
                key = (tag, m, subj); base = V.loc[key]
                # D4: SHAPE with harmonics 1-3 (all conditions)
                sk = SK.shape_participant(list(Lr), list(Pr), fs, L, P, K=3)
                d4.append(dict(condition=tag, model=m, participant=subj, r_vpg_K3=sk["r_vpg"], gated_K3=sk["gated"],
                               HR_ref=60.0 / base.rr_med_s))
                # D3: cap variants (all conditions)
                row = dict(condition=tag, model=m, participant=subj)
                for cap in (0.15, 0.35):
                    c = cap * base.rr_med_s; dd = float(np.clip(base.lag_s, -c, c))
                    row[f"WCR_cap{int(cap * 100)}"] = RV._shift_trim_wcr(L, P, fs, dd)
                d3.append(row)
                if tag != ds:
                    continue
                # D2: first 3 chunks only
                if len(Lr) > 3:
                    r3 = RV.score_chunks(list(Lr[:3]), list(Pr[:3]), fs)
                    d2.append(dict(condition=tag, model=m, participant=subj, n_chunks_all=len(Lr),
                                   **{f"{k}_30s": r3[k] for k in ("RPFI", "s_SHAPE", "s_WCR", "s_RHY", "SHAPE_defined")}))
                # D1: 30 → 20 Hz
                if fs == 30:
                    L20 = [resample_poly(c, 2, 3) for c in Lr]; P20 = [resample_poly(c, 2, 3) for c in Pr]
                    r20 = RV.score_chunks(L20, P20, 20)
                    d1.append(dict(condition=tag, model=m, participant=subj,
                                   **{f"{k}_20Hz": r20[k] for k in ("RPFI", "s_SHAPE", "s_WCR", "s_RHY", "SHAPE_defined", "lag_s")}))
            print(f"[{tag}] {m:14s} ({time.time() - t0:.1f}s)", flush=True)
    for name, rows in (("D1_fs20", d1), ("D2_first30s", d2), ("D3_cap", d3), ("D4_shapeK3", d4)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    print("CSVs saved")

    # H: example-pair figures
    try:
        import json
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from scipy.signal import butter, filtfilt
        for ds, subj, good, poor in PAIRS:
            fs = R.FS_BY_DATASET[ds]; runs = f"runs/{ds}_cv5"; sig = {}
            for m in (good, poor):
                for inf in sorted(Path(runs).glob(f"{m}_f*_info.json")):
                    info = json.load(open(inf)); pr = np.load(Path(runs) / f"{m}_f{info['fold']}_test_pred.npy", allow_pickle=True)
                    for i, ((s, e), sj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
                        if sj == subj:
                            sig[m] = np.asarray(pr[i], float); sig["label"] = labs[ds][s:e]
            b, a_ = butter(3, [0.5 / (fs / 2), 8 / (fs / 2)], btype="band")
            z = lambda x: (lambda y: (y - y.mean()) / y.std())(filtfilt(b, a_, x - np.mean(x)))
            seg = slice(20 * fs, 30 * fs); t = np.arange(10 * fs) / fs
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            for ax, m in zip(axes, (good, poor)):
                r = V.loc[(ds, m, subj)]
                ax.plot(t, z(sig["label"])[seg], "k", lw=2.0, label="reference (contact PPG)")
                ax.plot(t, z(sig[m][: len(sig["label"])])[seg], "tab:blue" if m == good else "tab:red", lw=1.4, label=m)
                ax.set_title(f"{subj} — {m}: HR MAE {r.HR_MAE_bpm:.2f} bpm | RPFI {r.RPFI:.2f} "
                             f"(SHAPE {r.s_SHAPE:.2f}, WCR {r.s_WCR:.2f}, RHY {r.s_RHY:.2f}, Δt {1000 * r.lag_s:+.0f} ms)", fontsize=9)
                ax.legend(fontsize=8, loc="upper right")
            axes[-1].set_xlabel("s (20–30 s window, 0.5–8 Hz, z-score)")
            fig.tight_layout(); fig.savefig(OUT / f"H_pair_{subj}_{good}_vs_{poor}.png", dpi=130); plt.close(fig)
        print("figures saved")
    except Exception as ex:
        print("figures skipped:", ex)
    print(f"\nDone: {OUT}")


if __name__ == "__main__":
    main()
