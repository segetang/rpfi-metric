# figures/

Scripts for the paper figures. Outputs (PDF for LaTeX, PNG for preview) go to `docs/figures/`.
Style is shared through `figstyle.py` (DejaVu Sans, bold "(A)" panel labels at the top left, reference in black).
Each figure is drawn close to its printed width (full page 7.16 in, single column 3.5 in); `figstyle.save` enlarges any
text that would print below 8 pt.

| Figure | Script | Data |
|---|---|---|
| 1 Pipeline (schematic) | drawn by hand (not scripted) | none |
| 2 SHAPE | `fig2_shape.py` | synthetic pulse, release code |
| 3 WCR with capped lag alignment | `fig3_wcr_lag.py` | synthetic pulse, release code |
| 4 Rhythm | `fig4_rhythm.py` | synthetic pulse, release code |
| 5 Targeted degradations | `fig5_degradation.py` | `results/robustness/deg17_dense_raw.csv` |
| 6 Experiment 1 | `fig6_exp1.py` | `results/exp1/` |
| 7 Experiments 2 and 2c | `fig7_exp2.py` | `results/exp2/`, `results/scores/` |
| 8 Model comparison | `fig8_models.py` | `results/tables/table1.csv` |
| 9 Real examples (UBFC S08, S25) | `fig9_examples.py` | labels and model outputs (run from the working folder) |

All figures are saved at 300 dpi (PNG; the PDF is vector).

```bash
python figures/make_all.py                                   # figures 2-8, from the repository root
python figures/make_all.py --label-dir <labels> --runs-dir <runs>   # also figure 9
```
