"""Regenerate every paper table and report from the CSV files in results/ (no raw data needed).

Steps
  1. pre-registered analyses (experiments/): exp1_confirm.py, exp2_analyze.py, exp2b_analyze.py, exp2c_analyze.py
     run in a scratch folder laid out the way those scripts expect; their reports are copied to results/tables/
  2. table scripts (analysis/): model_tables.py, component_tables.py, exp1_secondary.py, exp2_snr.py,
     weight_sensitivity.py

Usage:  python analysis/reproduce_all.py            (about 3-5 minutes on a laptop)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, TAB, EXP = ROOT / "results", ROOT / "results" / "tables", ROOT / "experiments"


def run(cmd, cwd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "rpfi"), str(EXP)]), PYTHONIOENCODING="utf-8")
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=cwd, env=env, check=True)


def stage(work):
    """Lay out results/ the way the experiment-stage scripts expect (relative paths)."""
    links = {
        "exp2_features/exp2_features_ALL.csv": RES / "exp2" / "exp2_features_ALL.csv",
        "exp2b_prv/exp2b_prv_ALL.csv": RES / "exp2" / "exp2b_prv_ALL.csv",
        "exp2c_ihr/exp2c_ihr_ALL.csv": RES / "exp2" / "exp2c_ihr_ALL.csv",
        "newsys_recompute/newsys_participant_ALL.csv": RES / "scores" / "newsys_participant_ALL.csv",
        "hr_recompute/hr_participant_ALL.csv": RES / "scores" / "hr_participant_ALL.csv",
        "sensitivity_v2/sensitivity_ALL.csv": RES / "robustness" / "sensitivity_ALL.csv",
    }
    for f in (RES / "exp1").glob("exp1_cases_*.csv"):
        links[f.name] = f
    for f in (ROOT / "mappings").glob("mapping_*.csv"):
        links[f.name] = f
    for dst, src in links.items():
        p = work / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, p)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        stage(work)
        run([EXP / "exp1_confirm.py"], work)
        run([EXP / "exp2_analyze.py"], work)
        run([EXP / "exp2b_analyze.py"], work)
        run([EXP / "exp2c_analyze.py"], work)
        for rep in ("exp1_confirm_report.txt", "exp2_report.txt", "exp2b_report.txt", "exp2c_report.txt"):
            shutil.copy2(work / rep, TAB / rep)
            print(f"  wrote results/tables/{rep}")
    for script in ("model_tables.py", "component_tables.py", "exp1_secondary.py", "exp2_snr.py",
                   "weight_sensitivity.py", "sync_drift_summary.py"):
        run([ROOT / "analysis" / script], ROOT)
    print("\nDone. All tables are in results/tables/.")


if __name__ == "__main__":
    main()
