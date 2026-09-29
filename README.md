# RPFI — rPPG Physiological Fidelity Index

Reference implementation for **RPFI (rPPG Physiological Fidelity Index)**, a waveform-level
evaluation metric for remote photoplethysmography (rPPG) that combines three components:

- **BWMD** (Beat-Warping Morphology Distance): beat-level morphology from DTW alignment,
  systolic-peak timing error, and a waveform overlap ratio
- **WCR** (Weighted Concordance Reliability): global agreement from the concordance
  correlation coefficient (CCC) and Spearman rank correlation (`wcr_beta` = α = 0.6)
- **EDD** (Error Distribution Divergence): fraction of residual spectral power inside the
  cardiac band (0.75–3.0 Hz), which separates unstructured noise from a coherent but
  incorrect pulse ("ghost pulse")

Each component is normalized against a **fixed, pool-independent anchor** and the three
scores are combined by a **weighted geometric mean** (non-compensatory; default weights
BWMD 0.375 / WCR 0.375 / EDD 0.25) into a single score in [0, 100].

> Paper: *RPPG Physiological Fidelity Index (RPFI): A Composite Evaluation Metric for
> Remote Photoplethysmography* (IEEE Access, under review / Manuscript ID Access-2026-26900).
> A citation entry will be added here once the DOI is assigned.

---

## 1. Repository structure

```
rpfi-metric/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── preprocessing/
│   │   ├── roi_extract.py          # facial skin ROI extraction and RGB traces
│   │   ├── pos_extract.py          # POS rPPG extraction (Wang et al. 2017, TBME)
│   │   └── label_preprocess.py     # contact-PPG label alignment / preprocessing
│   │
│   ├── splits/
│   │   └── subject_split_v2.py     # participant-wise K-fold split generator (leakage-safe)
│   │
│   ├── models/
│   │   └── models.py               # RNN / LSTM / BiLSTM / PhysDiff / TemporalTransformer /
│   │                                # Mamba / BiMamba model definitions + MODEL_REGISTRY
│   │
│   ├── training/
│   │   ├── train_models.py         # common train/eval harness (7 models, subject-wise CV)
│   │   ├── train_bimamba_blend.py  # BiMamba ensemble-blend variant (outer-group reuse, K_inner=3)
│   │   └── export_pos_baseline.py  # exports POS as a "model" in the same runs/ schema
│   │
│   └── eval/
│       ├── rpfi_eval.py            # RPFI computation: components, fixed anchors,
│       │                           # participant-level bootstrap/posthoc stats
│       └── derive_anchors.py       # derives the fixed BWMD anchor + degradation-benchmark
│                                    # validation (17 synthetic degradation types)
│
├── analysis/                       # robustness / diagnostic studies (not required to
│   │                                # reproduce the headline RPFI numbers, but referenced
│   │                                # in the paper's Discussion / Supplementary Material)
│   ├── compare_existing_sqi.py         # RPFI vs. reference-free SQI comparison
│   ├── hyperparam_sensitivity.py       # one-at-a-time + joint sensitivity (dtw_band_frac, λp, λa, wcr_beta)
│   ├── lag_corrected_beat_template.py  # lag/polarity-corrected beat-template comparison
│   ├── prv_window_agreement.py         # 10 s-chunk vs. full-recording PRV agreement
│   ├── cross_dataset_eval.py           # zero-shot cross-dataset transfer scoring
│   ├── investigate_physdiff_asymmetry.py
│   ├── investigate_polarity_sensitivity.py
│   ├── diagnose_cohface.py
│   ├── bwmd_failure_rate.py
│   ├── rpfi_ranking_validation.py
│   ├── rpfi_gallery.py                 # qualitative quantile/subject reconstruction galleries
│   ├── score_npy_rpfi.py               # RPFI for full-length prediction arrays (unsupervised extractors)
│   └── make_polarity_variants.py       # sign variants (fixed flip / per-recording oracle) for extractors
│
├── splits/                         # generated CV split JSONs (small, versioned)
│   ├── pure_cv5.json
│   ├── ubfc_cv5.json
│   └── cohface_cv5.json
│
├── mappings/                       # dataset mapping CSVs (recording <-> subject <-> frame range)
│   ├── mapping_PURE.csv
│   ├── mapping_UBFC.csv
│   └── mapping_COHFACE.csv
│
├── configs/
│   └── rpfi_anchors.json           # fixed anchors: BWMD [0.00062, 0.90227], WCR [0, 1], EDD [0, 1]
│
├── results/                        # tracked result CSVs (see §5)
│   ├── rpfi_participant_{pure,ubfc,cohface}_geometric.csv   # within-dataset, per participant
│   ├── {source}_to_{target}/        # zero-shot cross-dataset transfer, six directions:
│   │                                # pure_to_ubfc, pure_to_cohface, ubfc_to_pure,
│   │                                # ubfc_to_cohface, cohface_to_pure, cohface_to_ubfc
│   │                                # (config, participant, posthoc, summary CSV/JSON per direction)
│   └── unsupervised/                # GREEN / LGI / PCA / SSR (+ POS check), Supplementary Table S9
│
└── docs/
    ├── figures/                    # final paper figures only
    └── supplementary/              # supplementary-only figures (PURE/COHFACE galleries)
```

Large/derived artifacts (`runs/`, `results_*_final/`, `.pth` weights, `.npy` predictions and
labels, `__pycache__/`) are **not** tracked in git; see [`.gitignore`](./.gitignore) and
§4 below for how to regenerate them. The exception is `results/`, which holds the final
result CSVs themselves (see §5): small files, always tracked.

---

## 2. Environment setup

```bash
conda env create -f environment.yml
conda activate mamba
```

If you prefer pip only (e.g. inside an existing environment):

```bash
pip install -r requirements.txt
```

GPU (CUDA) is used for training; CPU-only works for `rpfi_eval.py` / `derive_anchors.py` /
the `analysis/` scripts, just slower.

`mamba_ssm` (required only for the Mamba / BiMamba model classes) is imported lazily in
`models.py`; if it isn't installed, those two models are simply skipped and all other models
still run.

> **Note on ROI extraction (`src/preprocessing/roi_extract.py`)**: this step depends on
> `face_recognition` / `dlib`, which are *not* included in `environment.yml`/
> `requirements.txt` above. The original environment used for ROI extraction no longer
> exists on the authors' machine, so its exact package versions could not be recovered.
> If you need to re-run ROI extraction, install `face_recognition` and `dlib` separately
> (see the [`face_recognition` install guide](https://github.com/ageitgey/face_recognition#installation);
> `dlib` typically needs `cmake` and a C++ compiler to build). Everything downstream of
> ROI extraction (POS extraction, label preprocessing, splitting, training, evaluation)
> only needs the pinned environment above.

---

## 3. Data

This repository does **not** redistribute the raw video/label data or per-sample predictions.
You need to obtain the following datasets under their own licenses and place them under a
local `data/` directory (not tracked in git):

| Dataset | Source | Notes |
|---|---|---|
| PURE | Stricker et al., TU Ilmenau | 59 recordings / 10 subjects, fs=30 |
| UBFC-rPPG | Bobbia et al., Université de Bourgogne | our archive (downloaded 2021) = 40 recordings / 40 subjects, fs=30 (see `mappings/mapping_UBFC.csv`; commonly cited as 42 subjects elsewhere) |
| COHFACE | Heusch et al., Idiap | 143 recordings / 40 subjects, fs=20; 11 recordings with unresolved subject/session metadata are excluded from evaluation (see `mappings/mapping_COHFACE.csv`, `status` column) |

The `mappings/mapping_*.csv` files are the authoritative record of which recording belongs
to which subject and frame range. Always cross-check against these rather than
re-deriving subject counts from the literature.

---

## 4. Pipeline: full reproduction order

All commands assume you are in the repo root with `data/` populated as above.
Use forward slashes even on Windows/WSL if any path contains spaces.

```bash
# 1. ROI + POS extraction (per dataset)
python src/preprocessing/roi_extract.py   --dataset {PURE,UBFC,COHFACE} ...
python src/preprocessing/pos_extract.py   --dataset {PURE,UBFC,COHFACE} ...

# 2. Label preprocessing / alignment
python src/preprocessing/label_preprocess.py --dataset {PURE,UBFC,COHFACE} ...

# 3. Generate participant-wise K=5 CV splits (writes splits/{ds}_cv5.json)
python src/splits/subject_split_v2.py --dataset pure    --cv 5 --mapping mappings/mapping_PURE.csv    --out splits
python src/splits/subject_split_v2.py --dataset ubfc    --cv 5 --mapping mappings/mapping_UBFC.csv    --out splits
python src/splits/subject_split_v2.py --dataset cohface --cv 5 --mapping mappings/mapping_COHFACE.csv --out splits

# 4. Export POS as a baseline "model" in the same runs/ schema (no training)
python src/training/export_pos_baseline.py --dataset {ds} --pred data/{DS}_POS_prediction.npy \
    --cv-split splits/{ds}_cv5.json --out runs

# 5. Train the 7 learned models (subject-wise K=5 CV, SmoothL1, epoch cap 200,
#    ReduceLROnPlateau + early stopping; see the paper's Experiments section for the full protocol)
python src/training/train_models.py --model all --dataset {ds} \
    --pred data/{DS}_POS_prediction.npy --label data/{DS}_label.npy \
    --cv-split splits/{ds}_cv5.json --out runs

# 6. (Optional) BiMamba ensemble-blend variant
python src/training/train_bimamba_blend.py --dataset {ds} \
    --pred data/{DS}_POS_prediction.npy --label data/{DS}_label.npy \
    --cv-split splits/{ds}_cv5.json --out runs

# 7. Derive the fixed BWMD anchor + validate against 17 synthetic degradations
#    (run once; anchors are dataset/model independent)
python src/eval/derive_anchors.py --out configs

# 8. Compute RPFI (per-participant scores, posthoc tests, HR-MAE, PRV)
python src/eval/rpfi_eval.py --dataset {ds} --runs runs/{ds}_cv5 \
    --label data/{DS}_label.npy --anchors configs/rpfi_anchors.json \
    --agg geometric --out results_{ds}_final
```

Repeat steps 4–6 and 8 for `ds ∈ {pure, ubfc, cohface}`.

Always pass `--anchors configs/rpfi_anchors.json` in step 8; all results in the paper were
computed with these anchors.

Step 8 writes a full local output folder `results_{ds}_final/` (config, per-participant
scores, posthoc tests, summary; gitignored, regenerable). The single file
`rpfi_participant_{ds}_geometric.csv` from each of those runs is additionally copied into
the tracked `results/` directory at the repo root (see §5).

### Cross-dataset transfer (`analysis/cross_dataset_eval.py`)

Applies each of the five source-fold checkpoints, without fine-tuning, to the complete
target dataset for all six source→target directions. When source and target sampling
rates differ, the target signal is resampled to the source rate before inference and the
prediction is resampled back before scoring. Outputs are in `results/{source}_to_{target}/`.
See the script's `--help` for its arguments.

### Unsupervised extractors (`analysis/score_npy_rpfi.py`, `analysis/make_polarity_variants.py`)

Scores full-length 1D prediction arrays (e.g. `data/PURE_GREEN_prediction.npy`) with the
same RPFI pipeline, using `splits/{ds}_cv5.json` for the recording→participant mapping.
Applied to POS, it reproduces the within-dataset POS scores (29.35 / 26.78 / 5.99).

```bash
python analysis/score_npy_rpfi.py --dataset pure --label data/PURE_label.npy \
    --split splits/pure_cv5.json \
    --pred POS=data/PURE_POS_prediction.npy GREEN=data/PURE_GREEN_prediction.npy \
    --out results/unsupervised

# fixed sign flip (no reference) and per-recording sign alignment (oracle, uses the reference)
python analysis/make_polarity_variants.py data/PURE_label.npy splits/pure_cv5.json \
    data/PURE_GREEN_prediction.npy GREEN variants
```

See `results/unsupervised/README.md` for details. CHROM is not included, because about 83%
of the samples in the available CHROM predictions are exact zeros.

### Robustness / diagnostic analyses (optional, `analysis/`)

These reproduce the auxiliary validation studies referenced in the paper's Discussion and
Supplementary Material (existing-SQI comparison, hyperparameter sensitivity, lag-corrected
beat-template comparison, PRV window agreement, polarity sensitivity, ranking-validation
galleries). Each script documents its own CLI arguments; most consume the same `runs/`
outputs from step 5 above and directly import functions from `src/eval/rpfi_eval.py` to
avoid duplicating the metric implementation.

---

## 5. Per-sample results

`results/` contains the **final result CSVs**, committed directly to this repository (not
gitignored, not regenerated-on-demand):

| File / folder | Contents |
|---|---|
| `rpfi_participant_pure_geometric.csv` | Per-participant RPFI (geometric aggregation) + component breakdown (BWMD/WCR/EDD), HR MAE, RMSE; PURE, all nine models (POS + eight learned models), subject-wise 5-fold CV |
| `rpfi_participant_ubfc_geometric.csv` | Same, UBFC-rPPG |
| `rpfi_participant_cohface_geometric.csv` | Same, COHFACE |
| `{source}_to_{target}/` | Zero-shot cross-dataset transfer, six directions (eight learned models; POS has no source dataset): run config, per-participant scores, posthoc tests, summary |
| `unsupervised/` | GREEN, LGI, PCA, SSR (and POS as a reproduction check): summaries and per-participant scores for the raw outputs, a fixed sign flip, and per-recording sign alignment (Supplementary Table S9) |

These are published alongside the code (rather than only the aggregate numbers reported in
the paper's tables) so that every score in the paper can be traced back to an individual
participant/model/fold record. This directly addresses the reviewers' request that
per-sample results, not just code, be made available during the review period.

---

## 6. Reproducibility notes

- **Subject-wise (participant-level) K=5 cross-validation** is used for all three datasets,
  not a time-based split. This is enforced by `src/splits/subject_split_v2.py`, which
  writes `dataset` and `n_frames_total` into each split JSON and is checked at load time
  (`verify_split_matches()`) to prevent silently applying the wrong dataset's split.
- **Deterministic per-fold/per-model seeding**: `train_models.py` derives its run seed via
  a stable hash (`hashlib.md5`), not Python's built-in `hash()`, which is randomized per
  process (`PYTHONHASHSEED`) and would otherwise make runs non-reproducible.
- **Non-overlapping evaluation windows**: training uses a sliding window with stride < window
  length as augmentation (safe, since it never crosses a train/test subject boundary);
  evaluation always uses non-overlapping 10 s windows.
- **RPFI anchors are fixed, not percentile-normalized against the evaluated model pool.**
  See `derive_anchors.py` for the ideal/null synthetic ensembles used to derive the BWMD
  anchor, and the degradation-sweep validation (17 degradation types × 3 components)
  confirming that no component moves in the wrong direction as severity increases.

---

## 7. License

Code: [MIT License](./LICENSE).
Datasets referenced above (PURE / UBFC-rPPG / COHFACE) retain their own original licenses;
this repository does not redistribute them.

---

## 8. Citation

```bibtex
@article{wi2026rpfi,
  title   = {RPPG Physiological Fidelity Index (RPFI): A Composite Evaluation Metric for Remote Photoplethysmography},
  author  = {Wi, Taehyun and Kim, Dae-Yeol and Lee, Yaesop},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript ID Access-2026-26900, under review}
}
```
