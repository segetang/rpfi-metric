# RPFI for additional unsupervised extractors (Supplementary Sec. S5, Table S9)

Scripts are in `analysis/` and import `src/eval/rpfi_eval.py` automatically. Run them from the repository root.

## Scripts (`analysis/`)
- `score_npy_rpfi.py`: scores full-length 1D prediction arrays (`*_prediction.npy`) with the paper's RPFI pipeline
  (10 s non-overlapping chunks; detrend lambda=100, 0.75-3.0 Hz zero-phase bandpass, z-normalization; EDD without bandpass;
  BWMD averaged over chunks, WCR/EDD on each participant's concatenated chunks; fixed anchors BWMD [0.00062, 0.90227],
  WCR/EDD [0, 1]; weights 0.375/0.375/0.25; weighted geometric mean; participant cluster bootstrap, 2,000 resamples).
  Recording-to-participant mapping comes from `splits/{dataset}_cv5.json`.
  Applied to POS it reproduces Table 1 of the main text (29.35 / 26.78 / 5.99).
- `make_polarity_variants.py`: builds the two sign variants reported in Table S9
  - `flip`: one fixed sign inversion for the whole method (no reference used)
  - `recsign`: per-recording sign alignment to the reference (oracle; uses the reference, not deployable)

Example:
    python analysis/score_npy_rpfi.py --dataset pure --label PURE_label.npy --split splits/pure_cv5.json \
        --pred POS=PURE_POS_prediction.npy GREEN=PURE_GREEN_prediction.npy --out results/unsupervised
    python analysis/make_polarity_variants.py PURE_label.npy splits/pure_cv5.json PURE_GREEN_prediction.npy GREEN variants

## Results
- `unsupervised_rpfi_all.csv`: all method x dataset x variant summaries
- `raw_*.csv`, `polarity_variants_*.csv`: per-dataset summaries
- `participant/`: participant-level BWMD, WCR, EDD, RPFI, HR MAE, RMSE for every run

CHROM is not included, because about 83% of the samples in the provided CHROM predictions are exact zeros.
