"""
run_exp1_confirm.py - run the Experiment 1 confirmation (pre-registered)
==========================================================
From the working folder:  python run_exp1_confirm.py          (use --label-dir if the labels are elsewhere)
  1) build distortion cases from the PURE and COHFACE hold-out recordings -> exp1_cases_{pure,cohface}_holdout.csv
     (the exploration/hold-out split is reproduced with the same seed; recordings used for exploration are excluded)
  2) train on the exploration cases (exp1_cases_{ubfc,pure,cohface}.csv, generated 2026-11-24) -> evaluate on the hold-out
Output: exp1_confirm_report.txt and exp1_cases_{pure,cohface}_holdout.csv
Note: never regenerate the exploration case files (the classifier must be trained on them as they are);
      copy them from results/exp1/ into the working folder. Their SHA-256 is checked below.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "rpfi"))  # core modules (repository rpfi/)
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = {"pure": ("PURE_label.npy", "mapping_PURE.csv", 30),
      "cohface": ("COHFACE_label.npy", "mapping_COHFACE.csv", 20)}

ap = argparse.ArgumentParser()
ap.add_argument("--label-dir", default="../real data")
ap.add_argument("--regenerate", action="store_true", help="regenerate hold-out cases even if they exist")
a = ap.parse_args()
need = [Path(a.label_dir) / lab for lab, _, _ in DS.values()] + [Path(mp) for _, mp, _ in DS.values()] + \
       [Path(f"exp1_cases_{d}.csv") for d in ("ubfc", "pure", "cohface")] + [Path("mapping_UBFC.csv")]
miss = [str(p) for p in need if not p.exists()]
if miss:
    sys.exit("not found: " + ", ".join(miss))
# frozen exploration cases (SHA-256 recorded at the pre-registration freeze)
FROZEN = {
    "exp1_cases_ubfc.csv": "347556cdef0feea2a0d7861f64b6c14a4ed84d4a78ab31f07459b4a18cd18f4c",
    "exp1_cases_pure.csv": "234694c94b79fe90cee1e3b088a875db1687e8ee28ff0725c4f428884cc1fad0",
    "exp1_cases_cohface.csv": "d28e6be845d589d2e78763414e0689ab751ad3176d39c30916f2424941dc802b",
}
bad = [f for f, h in FROZEN.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h]
print("freeze check:", "all match" if not bad else f"mismatch {bad}")
if bad:
    sys.exit("[stop] exploration case files differ from the frozen version; use the files in results/exp1/ unchanged.")

for ds, (lab, mp, fs) in DS.items():
    out = Path(f"exp1_cases_{ds}_holdout.csv")
    print(f"\n{'=' * 60}\n{ds} hold-out recordings\n{'=' * 60}", flush=True)
    if out.exists() and not a.regenerate:
        print(f"  {out} exists - skipped"); continue
    subprocess.run([sys.executable, str(HERE / "exp1_distortion_id.py"), "--label", str(Path(a.label_dir) / lab),
                    "--mapping", mp, "--fs", str(fs), "--n-chunks", "4", "--n-base", "40",
                    "--split", "holdout", "--tag", f"{ds}_holdout"], check=True)
subprocess.run([sys.executable, str(HERE / "exp1_confirm.py")], check=True)
print("\nDone: exp1_confirm_report.txt and exp1_cases_{pure,cohface}_holdout.csv")
