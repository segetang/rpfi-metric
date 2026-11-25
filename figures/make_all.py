"""Render every scripted paper figure (2-8, and 9 when raw data paths are given) at 300 dpi.

Run from the repository root:
    python figures/make_all.py
    python figures/make_all.py --label-dir <folder with *_label.npy> --runs-dir <folder with *_test_pred.npy>
Outputs go to docs/figures/<name>.pdf (vector, for LaTeX) and .png (300 dpi preview).
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig2_shape.py", "fig3_wcr_lag.py", "fig4_rhythm.py", "fig5_degradation.py",
           "fig6_exp1.py", "fig7_exp2.py", "fig8_models.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--label-dir"); ap.add_argument("--runs-dir")
a = ap.parse_args()

failed = []
for s in SCRIPTS:
    print(f"--- {s}")
    if subprocess.call([sys.executable, str(HERE / s)], cwd=HERE.parent) != 0:
        failed.append(s)
if a.label_dir and a.runs_dir:
    print("--- fig9_examples.py")
    if subprocess.call([sys.executable, str(HERE / "fig9_examples.py"), "--label-dir", a.label_dir,
                        "--runs-dir", a.runs_dir], cwd=HERE.parent) != 0:
        failed.append("fig9_examples.py")
else:
    print("--- fig9_examples.py skipped (needs --label-dir and --runs-dir)")
print("failed:", failed if failed else "none")
sys.exit(1 if failed else 0)
