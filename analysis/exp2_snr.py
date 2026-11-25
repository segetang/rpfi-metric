"""Experiment 2 / 2c with spectral SNR added as a control (pre-specified extension).

E2 + SNR : partial Spearman rho(SHAPE, morphology error | HR MAE, RMSE, Pearson, SNR)
C1 + SNR : partial Spearman rho(RHY, instantaneous-HR tracking error | 1-WCR, 1-WCR_aligned, SHAPE, HR MAE, RMSE, SNR)
plus the plain Spearman correlation of SNR with each external error.
Same units, merges and participant-cluster bootstrap (B = 2,000, seed 0) as experiments/exp2_analyze.py
and experiments/exp2c_analyze.py.

Output: results/tables/exp2_snr.csv
Usage:  python analysis/exp2_snr.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
import exp2_analyze as A
import exp2c_analyze as AC

K = C.KEY


def morph_units():
    """Rebuild the Experiment 2 primary analysis set exactly as exp2_analyze.py does."""
    X = pd.read_csv(C.RES / "exp2" / "exp2_features_ALL.csv")
    ref = X[X.primary].drop_duplicates(["dataset", "participant"])
    ref = ref[["dataset", "participant"] + [c for c in X.columns if c.startswith("ref_")]].reset_index(drop=True)
    feats, _, _ = A.select_features(ref)
    sd = ref.groupby("dataset")[[f"ref_med_{f}" for f in feats]].std()
    for f in feats:
        err = (X[f"pred_med_{f}"] - X[f"ref_med_{f}"]).abs() / X.dataset.map(sd[f"ref_med_{f}"])
        ok = (X[f"pred_valid_{f}"] >= A.PRED_MIN_BEATS) & X[f"ref_med_{f}"].notna()
        X[f"err_{f}"] = err.where(ok)
    E = X[[f"err_{f}" for f in feats]]
    X["MORPH_ERR"] = E.mean(1).where(E.notna().sum(1) >= int(np.ceil(len(feats) / 2)))
    N = pd.read_csv(C.RES / "scores" / "newsys_participant_ALL.csv")
    D = X.merge(N[K + ["sh_SHAPE", "HR_MAE_bpm"]], on=K, how="inner")
    D["participant_key"] = D.dataset + ":" + D.participant.astype(str)
    return D[D.primary & D.MORPH_ERR.notna()].reset_index(drop=True), feats


def ihr_units():
    """Rebuild the Experiment 2c primary analysis set exactly as exp2c_analyze.py does."""
    X = pd.read_csv(C.RES / "exp2" / "exp2c_ihr_ALL.csv")
    N = pd.read_csv(C.RES / "scores" / "newsys_participant_ALL.csv")[K + ["sh_SHAPE", "RHY", "F1", "nIBIE", "WCR_new", "RMSE"]]
    S = pd.read_csv(C.RES / "robustness" / "sensitivity_ALL.csv")[K + ["WCR_aligned"]]
    H = pd.read_csv(C.RES / "scores" / "hr_participant_ALL.csv")[K + ["HR_MAE_bpm"]].rename(columns={"HR_MAE_bpm": "HR_MAE"})
    D = X.merge(N, on=K).merge(S, on=K).merge(H, on=K)
    D["key"] = D.dataset + ":" + D.participant.astype(str)
    D["PRV_ERR"] = D.ITE_bpm
    D["bWCR"], D["bWCRa"] = 1 - D.WCR_new, 1 - D.WCR_aligned
    ctrl = ["bWCR", "bWCRa", "sh_SHAPE", "HR_MAE", "RMSE"]
    return D[D.primary].dropna(subset=["PRV_ERR"] + ctrl).reset_index(drop=True), ctrl


def main():
    snr = pd.read_csv(C.RES / "scores" / "snr_real.csv")[K + ["SNR_dB"]]
    rows = []

    pri, feats = morph_units()
    pri = pri.merge(snr, on=K, how="left", validate="1:1")
    assert pri.SNR_dB.notna().all()
    tests = {
        "E1 rho(SHAPE, morph error)": lambda d: spearmanr(d.sh_SHAPE, d.MORPH_ERR)[0],
        "E2 partial rho | HR MAE, RMSE, Pearson": lambda d: A.partial_spearman(d.sh_SHAPE.values, d.MORPH_ERR.values, d[A.CONV].values),
        "E2 + SNR partial rho | HR MAE, RMSE, Pearson, SNR": lambda d: A.partial_spearman(
            d.sh_SHAPE.values, d.MORPH_ERR.values, d[A.CONV + ["SNR_dB"]].values),
        "rho(SNR, morph error)": lambda d: spearmanr(-d.SNR_dB, d.MORPH_ERR)[0],
    }
    print(f"  Experiment 2: {len(pri)} units, {pri.participant_key.nunique()} participants, features {feats}")
    for name, fn in tests.items():
        ci = A.cboot(pri, fn)
        rows.append(dict(experiment="2", test=name, n=len(pri), rho=fn(pri), lo=ci[0], hi=ci[1]))
        print(f"    {name:52s} {rows[-1]['rho']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    pc, ctrl = ihr_units()
    pc = pc.merge(snr, on=K, how="left", validate="1:1")
    assert pc.SNR_dB.notna().all()
    tests = {
        "C1 partial rho(RHY, tracking error | ctrl)": lambda d: AC.pcor(d.RHY.values, d.PRV_ERR.values, d[ctrl].values),
        "C1 + SNR partial rho | ctrl, SNR": lambda d: AC.pcor(d.RHY.values, d.PRV_ERR.values, d[ctrl + ["SNR_dB"]].values),
        "rho(RHY, tracking error)": lambda d: spearmanr(d.RHY, d.PRV_ERR)[0],
        "rho(SNR, tracking error)": lambda d: spearmanr(-d.SNR_dB, d.PRV_ERR)[0],
    }
    print(f"  Experiment 2c: {len(pc)} units, {pc.key.nunique()} participants")
    for name, fn in tests.items():
        ci = AC.cboot(pc, fn)
        rows.append(dict(experiment="2c", test=name, n=len(pc), rho=fn(pc), lo=ci[0], hi=ci[1]))
        print(f"    {name:52s} {rows[-1]['rho']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    C.save(pd.DataFrame(rows), "exp2_snr.csv")


if __name__ == "__main__":
    main()
