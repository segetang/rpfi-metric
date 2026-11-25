# experiments/

Scripts that produced the files in `results/`. Unlike `analysis/`, most of them need the contact-PPG labels
and the model outputs, which are not redistributed (see the main README, *Data*).

## Working folder

All scripts are run from a working folder laid out as follows (paths are relative to it; most scripts
take `--label-dir`):

```
<work>/
  runs/{ds}_cv5/                     within-dataset model outputs ({model}_f{k}_test_pred.npy, _info.json)
  runs/cross_{src}_to_{tgt}/         cross-dataset model outputs (5 source-fold checkpoints applied to the target)
  results_eval_noSRE/rpfi_participant_{ds}_geometric.csv     v1 evaluation (conventional metrics), within-dataset
  {src}_to_{tgt}/rpfi_participant_{tgt}_geometric.csv        v1 evaluation, cross-dataset
  mapping_{PURE,UBFC,COHFACE}.csv    copies of mappings/
  exp1_cases_{ubfc,pure,cohface}.csv copies of results/exp1/ (frozen exploration cases)
../real data/{PURE,UBFC,COHFACE}_label.npy
../predictions/{ds}_{METHOD}_prediction.npy   unsupervised extractor outputs (rPPG-Toolbox)
```

Run a script as `python <repo>/experiments/<script>.py`; each script adds `<repo>/rpfi` to the import path.
`ds` is one of `pure`, `ubfc`, `cohface`; the 9 conditions are the 3 within-dataset conditions and the 6
cross-dataset directions.

## Run order and outputs

| Step | Script | Output (in the working folder) | Copied to |
|---|---|---|---|
| 0 | `src/` pipeline (main README of tag `v1`), then `python rpfi/rpfi_eval.py` per condition | `runs/`, `rpfi_participant_*_geometric.csv` | — |
| 1 | `run_newsys_all.py` (calls `recompute_newsys.py`) | `newsys_recompute/newsys_participant_ALL.csv` | `results/scores/` |
| 2 | `run_hr_all.py` (calls `recompute_hr.py`) | `hr_recompute/hr_participant_ALL.csv` | `results/scores/` |
| 3 | `sensitivity_v2.py` | `sensitivity_v2/sensitivity_ALL.csv` | `results/robustness/` |
| 4 | `wcr_capped.py` | `sensitivity_v2/wcr_cap_ALL.csv` | `results/robustness/` |
| 5 | `score_all_v2.py` (release scorer, all 9 conditions) | `rpfi_v2_scores/rpfi_v2_ALL.csv` | `results/scores/` |
| 6a | `exp1_distortion_id.py --label "../real data/{DS}_label.npy" --mapping mapping_{DS}.csv --fs {fs} --n-chunks 4 --n-base 40 --tag {ds}` | `exp1_cases_{ds}.csv` (exploration; frozen) | `results/exp1/` |
| 6b | `run_exp1_confirm.py` (calls `exp1_distortion_id.py --split holdout` and `exp1_confirm.py`) | `exp1_cases_{pure,cohface}_holdout.csv`, `exp1_confirm_report.txt` | `results/exp1/`, `results/tables/` |
| 6c | `exp1_refeature.py` | `exp1_refeature/exp1_v2feat_*.csv` | `results/exp1/` |
| 7a | `exp2_extract.py` (pyPPG environment) | `exp2_features/exp2_features_ALL.csv` | `results/exp2/` |
| 7b | `exp2b_extract.py`, `exp2c_extract.py` (pyPPG environment) | `exp2b_prv/exp2b_prv_ALL.csv`, `exp2c_ihr/exp2c_ihr_ALL.csv` | `results/exp2/` |
| 7c | `exp2_analyze.py`, `exp2b_analyze.py`, `exp2c_analyze.py` | reports | `results/tables/` |
| 8 | `final_snr_reliability.py` | `final2/snr_real.csv`, `final2/split_half.csv`, `final2/snr_exp1_*.csv` | `results/scores/`, `results/exp1/` |
| 9 | `pc_final_checks.py` (sampling rate, recording length, lag cap, harmonic-limited SHAPE via `shape_template_k.py`, example figures) | `final_checks/D1-D4_*.csv`, `H_pair_*.png` | `results/robustness/` |
| 10 | `sync_drift_check.py` | `sync_drift/sync_drift.csv` | `results/robustness/` |
| 11 | `diag_ubfc_polarity.py` | `diag_ubfc/diag_ubfc_polarity.csv`, figures | `results/robustness/` |
| 12 | `score_unsup_v2.py --pred-dir ../predictions` | `unsup_v2/unsup_v2_ALL.csv` | `results/scores/` |
| — | data-free: `deg17_v2.py`, `deg17_dense.py` (synthetic degradation batteries, use `synth.py`), `test_failure_modes.py` | `deg17_*`, `failure_modes.csv` | `results/robustness/` |

Steps 6b and 7c are also re-run from `results/` by `analysis/reproduce_all.py`.

## Pre-registered analyses

Experiment 1 (`exp1_confirm.py`) and Experiments 2 / 2b / 2c (`exp2*_analyze.py`) state their criteria in
their docstrings; the criteria, families, features, classifier and splits were fixed before the confirmation
data were generated. `run_exp1_confirm.py` checks the SHA-256 of the frozen exploration cases before running.
Code comments were translated to English after the analyses were run; the program logic is unchanged
(verified by comparing the syntax trees with all string literals removed).

## Notes

- `rpfi/rpfi_eval.py` is also the v1 evaluation (BWMD / WCR / EDD); its per-participant output provides the
  conventional metrics (RMSE, CCC) and the v1 RPFI that are reported next to RPFI v2.
- Cross-dataset outputs in `runs/cross_*` apply each of the 5 source-fold checkpoints, without fine-tuning,
  to the whole target dataset; when frame rates differ the input is resampled to the source rate and the
  prediction back to the target rate.
- The pyPPG-based scripts need a separate environment (`requirements-pyppg.txt`).
