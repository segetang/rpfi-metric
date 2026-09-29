"""
dtw_timing_check.py - per-chunk time of the naive Sakoe-Chiba DTW used by BWMD versus fastdtw
(same absolute-difference cost, radius = Sakoe-Chiba band) on real reference/prediction chunks.

Usage (from the repository root; requires `pip install fastdtw`):
  python analysis/dtw_timing_check.py --label data/PURE_label.npy --pred data/PURE_POS_prediction.npy --fs 30
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import rpfi_eval as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--fs", type=int, required=True)
    ap.add_argument("--n-chunks", type=int, default=60)
    args = ap.parse_args()
    y = np.load(args.label).astype(float).ravel(); p = np.load(args.pred).astype(float).ravel()
    seg = args.fs * 10; band = max(1, int(0.1 * seg))
    t_naive, t_fast = [], []
    for i in range(0, seg * args.n_chunks, seg):
        a = R.preprocess_chunk(y[i:i + seg], args.fs, True, True, 100)
        b = R.preprocess_chunk(p[i:i + seg], args.fs, True, True, 100)
        a = a / (np.max(np.abs(a)) + 1e-8); b = b / (np.max(np.abs(b)) + 1e-8)
        t = time.perf_counter(); R.dtw_distance(a, b) if not R.USE_FASTDTW else None; t_naive.append(time.perf_counter() - t)
        t = time.perf_counter(); fastdtw(a, b, radius=band, dist=lambda u, v: abs(u - v)); t_fast.append(time.perf_counter() - t)
    print(f"{seg}-sample chunks: naive {1e3 * np.mean(t_naive):.2f} ms, fastdtw {1e3 * np.mean(t_fast):.2f} ms "
          f"({np.mean(t_fast) / np.mean(t_naive):.1f}x)")
    if R.USE_FASTDTW:
        print("note: fastdtw is installed, so rpfi_eval.dtw_distance itself uses fastdtw; uninstall it to time the naive path")


if __name__ == "__main__":
    main()
