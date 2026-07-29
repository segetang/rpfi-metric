# RPFI — rPPG Physiological Fidelity Index

Reference implementation for **RPFI (rPPG Physiological Fidelity Index)**, a composite
evaluation metric for remote photoplethysmography (rPPG) that combines four components —
**BWMD** (waveform morphology, DTW-based), **SRE** (SNR-weighted residual error),
**WCR** (waveform concordance, CCC-based agreement), and **EDD** (spectral error
distribution diagnostic) — into a single percentile-anchored, non-compensatory score
in [0, 100].

> Paper: *RPPG Physiological Fidelity Index (RPFI): A Composite Evaluation Metric for
> Remote Photoplethysmography* (IEEE Access, under review / Manuscript ID Access-2026-26900).
> A citation entry will be added here once the DOI is assigned.

---

## 1. Repository structure

```
rpfi/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── preprocessing/
│   │   ├── roi_extract.py          # face ROI extraction (CONT pipeline, face_recognition CNN)
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
│       └── derive_anchors.py       # derives fixed BWMD/SRE anchors + degradation-benchmark
│                                    # validation (17 synthetic degradation types)
│
├── analysis/                       # robustness / diagnostic studies (not required to
│   │                                # reproduce the headline RPFI numbers, but referenced
│   │                                # in the paper's Response to Reviewers / Discussion)
│   ├── compare_existing_sqi.py         # RPFI vs. reference-free SQI comparison
│   ├── hyperparam_sensitivity.py       # one-at-a-time sensitivity (dtw_band_frac, λp, λa, Sc, Ss, wcr_beta)
│   ├── lag_corrected_beat_template.py  # lag/polarity-corrected beat-template comparison
│   ├── prv_window_agreement.py         # 10s-chunk vs. full-recording PRV agreement
│   ├── investigate_physdiff_asymmetry.py
│   ├── cross_dataset_eval.py
│   ├── bwmd_failure_rate.py
│   ├── rpfi_ranking_validation.py
│   └── rpfi_gallery.py                 # qualitative quantile/subject reconstruction galleries
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
│   └── rpfi_anchors.json           # fixed BWMD/SRE anchor values (see derive_anchors.py)
│
├── results/                        # per-participant RPFI scores (see §5 "Per-sample results")
│   ├── rpfi_participant_pure_geometric.csv
│   ├── rpfi_participant_ubfc_geometric.csv
│   └── rpfi_participant_cohface_geometric.csv
│
└── docs/
    ├── figures/                    # final paper figures only (galleries etc. kept out of git history)
    └── supplementary/               # supplementary-only figures (PURE/COHFACE galleries, meta CSVs)
```

Large/derived artifacts (`runs/`, `results_*_final/`, `.pth` weights, `.npy` predictions,
`__pycache__/`) are **not** tracked in git — see [`.gitignore`](./.gitignore) and
§4 below for how to obtain or regenerate them. The exception is `results/`, which holds
the final per-participant RPFI scores themselves (see §5) — small CSVs, always tracked.

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
`models.py` — if it isn't installed, those two models are simply skipped, all other models
still run.

> **Note on ROI extraction (`src/preprocessing/roi_extract.py`)**: this step depends on
> `face_recognition` / `dlib`, which are *not* included in `environment.yml`/
> `requirements.txt` above. The original environment used for ROI extraction no longer
> exists on the authors' machine, so its exact package versions could not be recovered.
> If you need to re-run ROI extraction, install `face_recognition` and `dlib` separately
> (see the [`face_recognition` install guide](https://github.com/ageitgey/face_recognition#installation) —
> `dlib` typically needs `cmake` and a C++ compiler to build). Everything downstream of
> ROI extraction (POS extraction, label preprocessing, splitting, training, evaluation)
> only needs the pinned environment above.

---

## 3. Data

This repository does **not** redistribute the raw video/label data. You need to obtain the
following datasets under their own licenses and place them under a local `data/` directory
(not tracked in git):

| Dataset | Source | Notes |
|---|---|---|
| PURE | Stricker et al., Uni Bonn | 59 recordings / 10 subjects, fs=30 |
| UBFC-rPPG | Bobbia et al. | our archive (downloaded 2021) = 40 recordings / 40 subjects (see `mappings/mapping_UBFC.csv`; commonly cited as 42 subjects elsewhere) |
| COHFACE | Heusch et al., Idiap | 143 recordings / 40 subjects, fs=20; 11 recordings with unresolved subject/session metadata are excluded from evaluation (see `mappings/mapping_COHFACE.csv`, `status` column) |

The `mappings/mapping_*.csv` files are the authoritative record of which recording belongs
to which subject and frame range — always cross-check against these rather than
re-deriving subject counts from the literature.

---

## 4. Pipeline — full reproduction order

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
#    ReduceLROnPlateau + early stopping — see paper §Methods for full protocol)
python src/training/train_models.py --model all --dataset {ds} \
    --pred data/{DS}_POS_prediction.npy --label data/{DS}_label.npy \
    --cv-split splits/{ds}_cv5.json --out runs

# 6. (Optional) BiMamba ensemble-blend variant
python src/training/train_bimamba_blend.py --dataset {ds} \
    --pred data/{DS}_POS_prediction.npy --label data/{DS}_label.npy \
    --cv-split splits/{ds}_cv5.json --out runs

# 7. Derive fixed RPFI anchors (BWMD/SRE) + validate against 17 synthetic degradations
#    — run once, anchors are dataset/model independent
python src/eval/derive_anchors.py --out configs

# 8. Compute RPFI (per-participant scores, posthoc tests, HR-MAE, PRV, sensitivity)
python src/eval/rpfi_eval.py --dataset {ds} --runs runs/{ds}_cv5 \
    --label data/{DS}_label.npy --anchors configs/rpfi_anchors.json \
    --agg geometric --out results_{ds}_final
```

Repeat steps 4–6, 8 for `ds ∈ {pure, ubfc, cohface}`.

Step 8 writes a full local output folder `results_{ds}_final/` (config, per-participant
scores, posthoc tests, summary — gitignored, regenerable). The single file
`rpfi_participant_{ds}_geometric.csv` from each of those runs is additionally copied into
the tracked `results/` directory at the repo root — see §5.

### Robustness / diagnostic analyses (optional, `analysis/`)

These reproduce the auxiliary validation studies referenced in the paper's Discussion /
Response to Reviewers (existing-SQI comparison, hyperparameter sensitivity, lag-corrected
beat-template comparison, PRV window agreement, cross-dataset transfer, ranking-validation
galleries). Each script documents its own CLI arguments; most consume the same `runs/`
outputs from step 5 above and directly import functions from `src/eval/rpfi_eval.py` to
avoid duplicating the metric implementation.

---

## 5. Per-sample results

`results/` contains the **final per-participant RPFI scores** for all three datasets,
committed directly to this repository (not gitignored, not regenerated-on-demand):

| File | Contents |
|---|---|
| `rpfi_participant_pure_geometric.csv` | Per-participant RPFI (geometric aggregation) + component breakdown (BWMD/SRE/WCR/EDD), PURE, all 8 models × 5-fold CV |
| `rpfi_participant_ubfc_geometric.csv` | Same, UBFC-rPPG |
| `rpfi_participant_cohface_geometric.csv` | Same, COHFACE |

These are published alongside the code (rather than only the aggregate numbers reported in
the paper's tables) so that every score in the paper can be traced back to an individual
participant/model/fold record. This directly addresses the reviewers' request that
per-sample results, not just code, be made available during the review period.

---

## 6. Reproducibility notes

- **Subject-wise (participant-level) K=5 cross-validation** is used for all three datasets
  — not a time-based split. This is enforced by `src/splits/subject_split_v2.py`, which
  writes `dataset` and `n_frames_total` into each split JSON and is checked at load time
  (`verify_split_matches()`) to prevent silently applying the wrong dataset's split.
- **Deterministic per-fold/per-model seeding**: `train_models.py` derives its run seed via
  a stable hash (`hashlib.md5`), not Python's built-in `hash()`, which is randomized per
  process (`PYTHONHASHSEED`) and would otherwise make runs non-reproducible.
- **Non-overlapping evaluation windows**: training uses a sliding window with stride < window
  length as augmentation (safe, since it never crosses a train/test subject boundary);
  evaluation always uses non-overlapping windows.
- **RPFI anchors are fixed, not percentile-normalized against the evaluated model pool** —
  see `derive_anchors.py` for the ideal/null synthetic-degradation ensembles used to derive
  them, and the degradation-sweep validation (17 degradation types × 4 components) confirming
  monotonic, correctly-signed response.

---

## 7. License

Code: [MIT License](./LICENSE) (adjust if you prefer a different license).
Datasets referenced above (PURE / UBFC-rPPG / COHFACE) retain their own original licenses —
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
