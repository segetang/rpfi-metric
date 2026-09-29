"""
peak_count_check.py - number of systolic peaks BWMD detects per bandpassed 10 s chunk.
BWMD's PeakDiff uses its normal path only when both signals have >= 2 peaks in a chunk
(otherwise: chunk-duration normalization, or the neutral value 0.5 when either has none).

Usage (from the repository root):
  python analysis/peak_count_check.py --label data/PURE_label.npy --split splits/pure_cv5.json \
      --pred POS=data/PURE_POS_prediction.npy GREEN=data/PURE_GREEN_prediction.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import rpfi_eval as R  # noqa: E402
from score_npy_rpfi import recordings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pred", nargs="*", default=[], help="NAME=path.npy ...")
    args = ap.parse_args()
    split = json.load(open(args.split)); fs = int(split["fs"]); seg = int(round(fs * 10.0))
    sigs = {"reference": np.load(args.label).astype(float).ravel()}
    for item in args.pred:
        name, path = item.split("=", 1)
        sigs[name] = np.load(path).astype(float).ravel()
    for name, x in sigs.items():
        counts = []
        for s, e, _ in recordings(split):
            n = ((e - s) // seg) * seg
            for c in range(s, s + n, seg):
                counts.append(len(R.peak_times(R.preprocess_chunk(x[c:c + seg], fs, True, True, 100), fs)))
        counts = np.array(counts)
        print(f"{name:10s} chunks={len(counts)} min={counts.min()} median={int(np.median(counts))} "
              f"<2 peaks={int((counts < 2).sum())} no peaks={int((counts == 0).sum())}")


if __name__ == "__main__":
    main()
