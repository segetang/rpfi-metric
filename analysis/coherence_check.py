"""Synthetic check of the SHAPE coherence gate (Sec. III-A).

For 100 seeds: the phase-locking coherence c of (a) a pulse unrelated to the reference and (b) the reference plus white
noise with twice its standard deviation, both scored against a 60 s synthetic reference at 30 Hz with rpfi_v2.
Output: results/tables/coherence_check.csv (percentiles) and the per-seed values in results/robustness/coherence_check_raw.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rpfi"), str(ROOT / "experiments")]
import rpfi_v2 as RV
from synth import Pulse

FS, DUR, N = 30, 60, 100


def chunks(x):
    return [x[i * 10 * FS:(i + 1) * 10 * FS] for i in range(DUR // 10)]


def main():
    t = np.arange(DUR * FS) / FS
    rows = []
    for k in range(N):
        rng = np.random.default_rng(k)
        y = Pulse(rng)(t)
        z = Pulse(np.random.default_rng(1000 + k))(t)
        noisy = y + 2 * np.std(y) * rng.standard_normal(y.size)
        rows.append(dict(seed=k, coh_unrelated=RV.score_chunks(chunks(y), chunks(z), FS)["coh_pred"],
                         coh_noise2sd=RV.score_chunks(chunks(y), chunks(noisy), FS)["coh_pred"]))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "robustness" / "coherence_check_raw.csv", index=False)
    out = pd.DataFrame([dict(case="unrelated pulse", p05=d.coh_unrelated.quantile(0.05), p50=d.coh_unrelated.median(),
                             p95=d.coh_unrelated.quantile(0.95), max=d.coh_unrelated.max()),
                        dict(case="reference + noise (2 SD)", p05=d.coh_noise2sd.quantile(0.05),
                             p50=d.coh_noise2sd.median(), p95=d.coh_noise2sd.quantile(0.95), min=d.coh_noise2sd.min())])
    out.to_csv(ROOT / "results" / "tables" / "coherence_check.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
