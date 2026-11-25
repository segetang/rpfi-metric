# RPFI — rPPG Physiological Fidelity Index (v2)

Reference implementation and analysis code for **RPFI v2**, a reference-based evaluation metric for
remote photoplethysmography (rPPG) that reports *what* is wrong with a reconstructed pulse waveform,
not only *how much*.

RPFI v2 scores a prediction against the contact-PPG reference on three components and reports them
as a profile together with a composite score:

| Component | What it measures | Score |
|---|---|---|
| **SHAPE** | beat morphology: correlation of the first-derivative (VPG) beat templates after lag correction and fundamental-phase alignment; undefined (score 0) when the prediction is not phase-locked to the reference beats | `s_SHAPE = clip(r_VPG, 0, 1)` |
| **WCR** | waveform agreement: 0.6 CCC + 0.4 Spearman, after removing a constant lag capped at a quarter cardiac cycle | `s_WCR = clip(WCR, 0, 1)` |
| **Rhythm (RHY)** | beat timing: 1:1 beat matching within 0.15 x RR (F1) and normalized inter-beat-interval error (nIBIE) | `s_RHY = F1 * (1 - min(nIBIE, 1))` |

```
RPFI = (s_SHAPE * s_WCR * s_RHY)^(1/3)        # equal weights, exact zero if any component is zero
```

Reported alongside: the constant lag Δt (not scored), a polarity-inversion flag (diagnostic), HR MAE,
and, at the model level, the participant-mean RPFI, its coverage (fraction of participants with a defined
SHAPE) and RPFI given coverage.

> The earlier version of this metric (v1: BWMD / WCR / EDD with fixed anchors) is kept in a separate repository,
> [segetang/rpfi-metric_v1](https://github.com/segetang/rpfi-metric_v1).

---

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # 15 data-free regression tests
```

```python
import sys; sys.path.insert(0, "rpfi")
import rpfi_v2 as RV

r = RV.score(ref, pred, fs=30)                               # one recording (1D arrays, same frame rate)
r = RV.score_participant([(ref1, pred1), (ref2, pred2)], fs=30)  # one participant, several recordings
summary = RV.aggregate([r_participant1, r_participant2, ...])    # model level
print(r["RPFI"], r["s_SHAPE"], r["s_WCR"], r["s_RHY"], r["lag_s"], r["polarity_suspect"])
```

Command line:

```bash
python rpfi/rpfi_v2.py --ref label.npy --pred pred.npy --fs 30
```

Signals are cut into non-overlapping 10 s chunks (a trailing remainder is dropped); each chunk is
detrended (smoothness priors, λ = 100), band-passed (0.75–3.0 Hz, zero phase) and z-normalized, identically
for reference and prediction. We recommend at least 60 s per participant. Scoring a 60 s recording takes
about 0.1 s on one CPU core.

---

## Repository layout

```
rpfi/              the metric (released scorer)
  rpfi_v2.py         RPFI v2: score / score_participant / score_chunks / aggregate, CLI
  shape_template.py  SHAPE component
  rhythm.py          rhythm component (beat detection, matching, F1, nIBIE)
  rpfi_eval.py       shared preprocessing, WCR/CCC, spectral HR, statistics; also the v1 evaluation (BWMD/EDD)
tests/             regression tests (synthetic cases, no data needed)
src/               model pipeline used to produce the evaluated predictions (unchanged from v1)
  preprocessing/     ROI + RGB traces, POS extraction, label alignment
  splits/            participant-wise K-fold splits
  models/, training/ 8 learned models refining the POS waveform, POS exported as a baseline
experiments/       scripts that produced the result files (need the labels and model outputs)
analysis/          scripts that build every paper table from results/ (no raw data needed)
results/           participant-level result files and the generated tables
splits/, mappings/ split definitions and recording <-> subject <-> frame-range mappings
```

---

## Reproducing the paper's numbers

Every table and statistic in the paper can be regenerated from the CSV files in `results/`:

```bash
python analysis/reproduce_all.py          # ~5-7 min on a laptop; writes results/tables/
```

| Output in `results/tables/` | Content |
|---|---|
| `table1.csv` | per condition x model: RPFI [95% CI], coverage, profile, lag, HR MAE, RMSE, CCC, SNR, v1 RPFI, polarity-flag rate |
| `pairwise_summary.csv`, `pairwise_detail.csv` | Friedman + pairwise Wilcoxon (Holm) for RPFI, each component, HR MAE, RMSE, CCC, SNR |
| `exp1_confirm_report.txt` | Experiment 1 (distortion identification at matched CCC), pre-registered confirmation on hold-out recordings |
| `exp1_snr_baselines.csv`, `exp1_ablation.csv` | Experiment 1 with SNR-augmented baselines; component ablation |
| `exp2_report.txt`, `exp2_snr.csv` | Experiment 2 (SHAPE vs. pyPPG fiducial-based morphology error), with SNR control |
| `exp2b_report.txt`, `exp2c_report.txt` | Experiment 2b (PRV; criterion not met) and 2c (instantaneous-HR tracking) |
| `overlap_decomposition.csv` | rhythm vs. WCR overlap (fixed-effects partial correlation) |
| `reliability_icc.csv` | split-half (odd/even chunk) ICC(3,1) and Spearman–Brown values |
| `weight_sensitivity.csv` | ranking stability under weight, tolerance and gate-threshold changes |
| `table_unsupervised.csv` | rPPG-Toolbox unsupervised extractors (raw and whole-method sign flip) |
| `example_pairs.csv` | candidate model pairs with similar HR MAE but different RPFI |

The result files themselves were produced by the scripts in `experiments/` from the model outputs and
labels; see [`experiments/README.md`](experiments/README.md) for the run order and the mapping from each
script to its output.

---

## Data

This repository does not redistribute videos, labels or sample-level predictions. The datasets must be
obtained from their providers under their own licenses:

| Dataset | Provider | Used here |
|---|---|---|
| PURE | Stricker et al., TU Ilmenau | 59 recordings / 10 subjects, 30 fps |
| UBFC-rPPG | Bobbia et al., Université de Bourgogne | 40 recordings / 40 subjects, 30 fps (our archive) |
| COHFACE | Heusch et al., Idiap | 132 of 143 recordings / 40 subjects, 20 fps; 11 recordings without subject metadata are excluded |

`mappings/mapping_*.csv` record which recording belongs to which subject and frame range. The files in
`results/` contain participant-level derived values only.

---

## Environment

- Metric, tests and analyses: `requirements.txt` (tested with Python 3.13, NumPy 2.5, SciPy 1.18,
  pandas 3.0, scikit-learn 1.9; also run on Windows with older versions).
- Experiment 2 / 2b / 2c feature extraction uses [pyPPG](https://github.com/godamartonaron/GODA_pyPPG),
  which needs older pandas/NumPy; use a separate environment: `requirements-pyppg.txt`.
- Model training (`src/`): `requirements-train.txt` / `environment.yml` (PyTorch; `mamba-ssm` only for the
  Mamba models).

## License

MIT — see [LICENSE](LICENSE).
