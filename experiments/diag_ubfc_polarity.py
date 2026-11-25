"""
diag_ubfc_polarity.py - diagnose why some UBFC recordings yield 'opposite-phase' outputs
==================================================================================
Hypotheses
  H1 reference (label) polarity inverted in some recordings -> the label's asymmetry (fast rise, slow fall) is reversed
  H2 video-label synchronization offset (~half a cycle)     -> label asymmetry normal, POS-label correlation peaks near half a cycle
  H3 the POS input itself is inverted in some recordings   -> label normal, POS asymmetry reversed, zero-lag correlation negative
For every UBFC recording, for the label / POS (input) / model trained on UBFC / models trained on PURE and COHFACE:
  - asymmetry index: skewness of the first derivative after 0.5-8 Hz band-pass (positive for normal PPG, fast rise)
  - cross-correlation with the label: lag of the maximum |correlation| within +/- 1.0 s, its sign, and the zero-lag correlation
are computed, and 10 s excerpts of suspect and control recordings are saved as figures.

Usage (from the working folder):  python diag_ubfc_polarity.py   (use --label-dir if the labels are elsewhere)
Output: diag_ubfc/diag_ubfc_polarity.csv, diag_ubfc/*.png
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew

FS = 30
SUSPECT = ["ubfc_S42", "ubfc_S44", "ubfc_S37", "ubfc_S11", "ubfc_S04", "ubfc_S15", "ubfc_S49"]
SOURCES = {  # name: (runs folder, model)
    "POS_input": ("runs/ubfc_cv5", "pos"),
    "bimamba_trainedUBFC": ("runs/ubfc_cv5", "bimamba"),
    "bimamba_trainedPURE": ("runs/cross_pure_to_ubfc", "bimamba"),
    "bimamba_trainedCOHFACE": ("runs/cross_cohface_to_ubfc", "bimamba"),
}


def bp(x, lo=0.5, hi=8.0):
    b, a = butter(3, [lo / (FS / 2), min(hi, 0.45 * FS) / (FS / 2)], btype="band")
    y = filtfilt(b, a, np.asarray(x, float) - np.mean(x)); return (y - y.mean()) / (y.std() + 1e-12)


def asym(x):
    return float(skew(np.diff(bp(x))))


def xcorr(ref, x, max_lag_s=1.0):
    a, b = bp(ref, 0.75, 3.0), bp(x, 0.75, 3.0); n = min(len(a), len(b)); a, b = a[:n], b[:n]
    best = (0.0, 0)
    for k in range(-int(max_lag_s * FS), int(max_lag_s * FS) + 1):
        if k >= 0:
            r = np.corrcoef(a[:n - k], b[k:])[0, 1]
        else:
            r = np.corrcoef(a[-k:], b[:n + k])[0, 1]
        if abs(r) > abs(best[0]):
            best = (r, k)
    return float(np.corrcoef(a, b)[0, 1]), best[0], best[1] / FS


def load_preds(runs, model):
    out = {}
    for inf in sorted(Path(runs).glob(f"{model}_f*_info.json")):
        info = json.load(open(inf))
        preds = np.load(Path(runs) / f"{model}_f{info['fold']}_test_pred.npy", allow_pickle=True)
        for i, ((s, e), subj) in enumerate(zip(info["test_ranges"], info["test_subject_of_range"])):
            out[subj] = (s, e, np.asarray(preds[i], float).ravel())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="../real data")
    a = ap.parse_args()
    lab = np.load(Path(a.label_dir) / "UBFC_label.npy").astype(float).ravel()
    out = Path("diag_ubfc"); out.mkdir(exist_ok=True)
    P = {k: load_preds(*v) for k, v in SOURCES.items()}
    subjects = sorted(P["POS_input"])
    rows = []
    for subj in subjects:
        s, e, _ = P["POS_input"][subj]; ref = lab[s:e]
        row = dict(participant=subj, suspect=subj in SUSPECT, label_asym=asym(ref), rr_s=np.nan)
        f = np.fft.rfftfreq(len(ref), 1 / FS); sp = np.abs(np.fft.rfft(bp(ref, 0.75, 3.0)))
        row["label_HR_bpm"] = float(60 * f[np.argmax(sp)])
        for k, d in P.items():
            if subj not in d:
                continue
            x = d[subj][2][: e - s]
            r0, rmax, lag = xcorr(ref[: len(x)], x)
            row[f"{k}_asym"] = asym(x); row[f"{k}_r0"] = r0; row[f"{k}_rmax"] = rmax; row[f"{k}_lag_s"] = lag
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out / "diag_ubfc_polarity.csv", index=False)
    print(df[["participant", "suspect", "label_asym", "POS_input_asym", "POS_input_r0", "POS_input_lag_s",
              "bimamba_trainedPURE_r0", "bimamba_trainedUBFC_r0"]].round(2).to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ctrl = [x for x in subjects if x not in SUSPECT][:3]
        for subj in SUSPECT + ctrl:
            if subj not in P["POS_input"]:
                continue
            s, e, _ = P["POS_input"][subj]; t0 = 20 * FS; seg = slice(t0, t0 + 10 * FS); t = np.arange(10 * FS) / FS
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(t, bp(lab[s:e])[seg], "k", lw=2.2, label="label")
            for (k, d), c in zip(P.items(), ["tab:orange", "tab:green", "tab:red", "tab:purple"]):
                if subj in d:
                    ax.plot(t, bp(d[subj][2][: e - s])[seg] - 0.0, c, lw=1.2, alpha=0.85, label=k)
            EN = {"POS_input": "POS (input)", "bimamba_trainedUBFC": "bimamba trained on UBFC", "bimamba_trainedPURE": "bimamba trained on PURE",
                  "bimamba_trainedCOHFACE": "bimamba trained on COHFACE"}
            for ln in ax.get_lines()[1:]:
                ln.set_label(EN.get(ln.get_label(), ln.get_label()))
            ax.set_title(f"{subj} ({'suspect' if subj in SUSPECT else 'control'}) 20-30 s, 0.5-8 Hz z-score"); ax.legend(fontsize=8, ncol=5)
            ax.set_xlabel("s"); fig.tight_layout(); fig.savefig(out / f"{subj}.png", dpi=110); plt.close(fig)
        print(f"figures saved: {out}/*.png")
    except Exception as ex:
        print("figures skipped:", ex)
    print(f"Done: {out / 'diag_ubfc_polarity.csv'}")


if __name__ == "__main__":
    main()
