"""
make_polarity_variants.py - build sign variants of an unsupervised extractor's predictions to check
output-polarity convention differences.
  flip   : one fixed sign inversion for the whole method (-x); convention-level correction, no reference used.
  recsign: per-recording sign aligned to the sign of its correlation with the reference
           (uses the reference = oracle, not deployable).
Usage:  python make_polarity_variants.py LABEL.npy SPLIT.json PRED.npy NAME OUT_DIR
Output: <OUT_DIR>/<DATASET>_<NAME>_<variant>.npy
"""
import json
import sys
from pathlib import Path

import numpy as np

import rpfi_eval as R
from score_npy_rpfi import recordings

label_p, split_p, pred_p, name, out = sys.argv[1:6]
label = np.load(label_p).astype(float).ravel()
pred = np.load(pred_p).astype(float).ravel()
split = json.load(open(split_p)); fs = int(split["fs"]); seg = int(split["seg_len"])
out = Path(out); out.mkdir(parents=True, exist_ok=True)
ds = split["dataset"]

np.save(out / f"{ds}_{name}_flip.npy", -pred)

rs = pred.copy(); n_flipped = 0; recs = recordings(split)
for s, e, _ in recs:
    n = ((e - s) // seg) * seg
    if n == 0:
        continue
    L = np.concatenate([R.preprocess_chunk(label[c:c + seg], fs, True, True, 100) for c in range(s, s + n, seg)])
    P = np.concatenate([R.preprocess_chunk(pred[c:c + seg], fs, True, True, 100) for c in range(s, s + n, seg)])
    if np.corrcoef(L, P)[0, 1] < 0:
        rs[s:e] = -rs[s:e]; n_flipped += 1
np.save(out / f"{ds}_{name}_recsign.npy", rs)
print(f"{ds} {name}: recording-level sign flipped in {n_flipped}/{len(recs)} recordings")
