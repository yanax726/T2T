# Blood-only EASI learning: bounded model upgrade, 7 November 2026

Two additional approaches were implemented and tested: learning from each patient's W0-to-W12 molecular change, and using recorded ordinal IGA to assist EASI learning during training. Neither approach produced a blood-only EASI reader with adequate individual agreement. The compact three-marker procedure improved somewhat; its remaining errors are about 7–8 EASI points, not about 1 point. The frozen BSA–three-marker estimator was not changed.

## Matched results

All entries below use the same original patient splits and visits. The reported value is the mean of metrics calculated separately for each repeat; predictions are not averaged before error calculation. MAE is in EASI points.

| Blood inputs and procedure | W0 MAE | W12 MAE | W24 MAE | W12 Spearman | W24 Spearman | W12 CCC | W24 CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Three markers, previous ridge | 7.63 | 7.57 | 8.69 | 0.071 | 0.114 | 0.103 | 0.051 |
| Three markers, nested candidate selection | 7.50 | 7.21 | 7.80 | 0.104 | 0.061 | 0.131 | 0.060 |
| QC19, ridge | 7.53 | 7.28 | 8.39 | 0.066 | 0.239 | 0.093 | 0.097 |
| QC19, paired-change RBF family | 7.51 | 7.08 | 8.31 | 0.097 | 0.176 | 0.122 | 0.091 |
| QC19, ordinal-IGA-assisted reduced rank | 7.53 | 7.24 | 8.39 | 0.078 | 0.243 | 0.099 | 0.099 |
| QC19, nested candidate selection | 7.41 | 7.26 | 8.49 | 0.088 | 0.177 | 0.116 | 0.084 |

The three-marker selection procedure reduced MAE relative to the previous three-marker ridge by 0.13 at W0 (95% patient-bootstrap interval for the difference, −0.50 to 0.18), 0.36 at W12 (−0.67 to −0.04), and 0.89 at W24 (−1.20 to −0.57). These are exploratory, conditional comparisons after extensive previous use of this cohort. They do not establish prospective superiority. Absolute agreement remains poor: only 5.6% of W12 and 4.7% of W24 predictions from the new three-marker procedure fell within one EASI point. An improvement in MAE without a corresponding improvement in rank correlation is possible because the estimates move closer to the mean without distinguishing patients better.

The QC19 procedure did not provide a stable advantage over the three-marker reference or over QC19 ridge. Its W0/W12/W24 MAE differences versus QC19 ridge were −0.13, −0.02 and +0.10; all three bootstrap intervals included zero. The table preserves the individual family results rather than retrospectively choosing whichever family happens to look best at each visit.

## What was actually learned

The compact panel consists of CCL17, CCL26 and IL-5. The expanded panel uses the 19 markers already designated discovery-eligible in `config/marker_registry.csv`. CCL18, IL-13, IL-22 and IL-31 were not silently promoted into a deployment panel: their earlier assay-coverage limitations remain. Their descriptive biological results can still be reported elsewhere.

The new models take only the current blood sample as prediction input. They do not take BSA, current EASI, current IGA, symptoms, treatment, visit number, centre, age, sex, or previous clinical records. Patient identity and visit are used to create training folds and, for the paired method, to align the two observed training visits. IGA is an auxiliary training label only. Consequently, these models address the user's blood-only question rather than relabelling a clinical-input model as objective.

Each candidate family uses the already stored log2 concentrations. Missing molecular measurements are replaced by the median calculated in the training fold, then features are standardized within that fold. No clinical target is imputed. No missingness indicators are added. The reference three-marker data are complete; QC19 missingness is explicitly diagnosed below.

Three fixed candidate families were compared:

1. **Weighted ridge.** A linear model with an unpenalized intercept and four fixed ridge penalties (0.1, 1, 10, 100). Each patient has equal total training weight, irrespective of having one or two development visits.
2. **Paired-change RBF.** A 32-component Nyström approximation learns nonlinear molecular patterns. Alongside the EASI-level error it may penalize disagreement between each training patient's observed EASI change and the difference between that patient's two model estimates. The objective is
   `sum_i w_i (y_i − f(x_i))² + λ(n_train/n_pairs) sum_p [(y_p12 − y_p0) − (f(x_p12) − f(x_p0))]² + α||β||²`.
   The fixed grid is two kernel widths, three ridge penalties and four pair weights (0, 0.3, 1, 3). A zero pair weight allows ordinary nonlinear ridge when the paired constraint does not help. Prediction remains `clip(intercept + beta × blood_basis(current_sample), 0, 72)`; it does not require the patient's baseline sample.
3. **Ordinal-IGA-assisted reduced rank.** Weighted ridge jointly fits standardized EASI and the five ordered indicators `IGA≥1`, …, `IGA≥5`. The IGA block is scaled to comparable total weight to EASI. Training fitted responses define a rank-one or rank-two projection; only the EASI output is retained at prediction. This is an explicitly defined ridge-plus-projection estimator, not a claim that the recorded 0–5 IGA variable has become a validated continuous scale.

There are 34 fixed parameter candidates per panel. An inner three-fold patient split selects candidates by mean per-patient absolute EASI error. Family-specific results and the overall selection procedure are both retained. No outer test result or W24 result is used to choose candidates. The code selected a nonzero paired penalty in 4 of 10 compact-panel folds and 1 of 10 QC19 folds. IGA-assisted candidates were selected in 1 of 10 compact-panel folds and 3 of 10 QC19 folds. Neither added training mechanism consistently displaced ordinary nonlinear or linear learning.

## Patients and validation boundary

The fair core comparison contains 160 patients and 316 visits: W0 152, W12 89, W24 75. Development has 241 W0/W12 observations from 156 patients. The original two repeats of five patient folds are preserved; all visits from a patient remain in the same outer fold. W24 is never included in training or tuning. It is a previously examined same-cohort time-transfer assessment, not untouched external validation.

The model cohort differs from the 163-patient treatment-comparison cohort. It excludes all six patients aged under 12, but retains one baseline observation from a patient whose initial abrocitinib dose cannot be assigned ([restricted source case]). Treatment is not an input, so the measured blood and EASI values remain usable for this model comparison. The inherited A100 label is replaced by `unclassified_initial_dose` in the new diagnostic output; that patient is not assigned to either abrocitinib-dose subgroup. This metadata correction leaves all predictions and overall metrics unchanged.

A separate expanded-data sensitivity removes the original BSA and three-marker-completeness restrictions while requiring observed EASI, age at least 12 years, a linked blood sample, and at least eight measured QC19 markers. This yields 161 patients and 327 visits (155/94/78 at W0/W12/W24), including 249 W0/W12 development visits from 157 patients. Existing patients retain their original fold assignments. Its QC19 selection MAE is 7.51/7.44/8.23. Evaluating that expanded training on only the original core visits gives 7.46/7.40/8.40. Relaxing these eligibility restrictions therefore does not explain away the weak results.

Confidence intervals resample patients in the saved out-of-fold predictions, with 2,000 draws. Paired contrasts use identical sampled patient IDs. They are conditional on the saved fits and fixed folds; they do not account for the full history of analyst/model adaptation on this cohort. The wider project has already tried many model families. This amendment is an exploratory procedure comparison, not a fresh confirmatory validation.

## Error diagnosis

The problem is not confined to severe baseline disease. Among the 44 baseline patients with EASI >21, the new three-marker model has MAE 12.88 and mean error −12.77: severe disease is underestimated. Among 68 W24 patients with EASI ≤7, MAE is 8.40 and mean error +8.37: low disease activity is overestimated. The model compresses both groups towards an intermediate score. At W12, mean observed EASI is 4.30 while mean predicted EASI is 10.28; at W24, these are 2.60 and 10.22. This is a failure of individual calibration, not just a small-sample fluctuation in a correlation coefficient.

The broader model also fails among visits with all 19 assays actually observed. In these complete-assay subsets, its W0/W12/W24 MAE is 7.70/7.47/7.94 (n=99/59/44). Imputing absent blood values is therefore not the sole source of the problem. This is a diagnostic subset assessment of the same models, not a separately optimized complete-case model.

Individual change was calculated within patient and within each repeat, then evaluated; no mean prediction was subtracted from another mean. For the compact selection procedure, W0→W12 change MAE is 10.89 (85 paired patients), W0→W24 11.50 (67), and W12→W24 3.88 (55). Its corresponding correlations are 0.223, 0.223 and 0.240. These results do not justify repackaging the current blood-only estimator as an accurate personal monitoring score.

The baseline severe group is sizeable, but W12 contains only one observed EASI>21 patient and W24 none. Performance in severe treated patients cannot be established from those visits. Severity strata are used only after prediction for diagnosis; actual EASI never routes a patient to a model.

## Implication for the paper

The current evidence supports retaining the frozen BSA–three-marker estimator as a **clinical–molecular composite** if it remains the paper's chosen estimator. The accuracy of that composite must not be attributed to blood alone. The present blood-only experiments belong in a compact supplementary comparison with the absolute errors and agreement measures visible. They should not be renamed a clinically validated inflammation score or replace the existing formula.

The negative result is informative: expanding the panel, adding a nonlinear response surface, learning paired molecular changes, and assisting training with ordinal IGA did not recover sufficiently precise individual EASI values. It does not prove that every conceivable model or assay is exhausted. It does remove the unsupported explanation that only a simple linear formula had been tried, and it provides no basis for continuing unconstrained tuning until a one-point error appears on this repeatedly used cohort.

## Reproduction and audit

Run from the project root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python -m cytot2t.curation.molecular_upgrade_20261107
```

The configuration was written before the new model fits. The main formula and seven protected input/configuration files are SHA-256 checked before and after execution. Original three-marker predictions were reproduced across all 632 repeat-by-visit predictions to a maximum absolute difference of 3.6×10⁻¹⁴. Independent NumPy/SciPy calculations verified MAE, Spearman and CCC to 8.9×10⁻¹⁶. Synthetic checks verify ridge equivalence, the full-rank multi-target limit, training-pair penalty direction, and prediction independence from all clinical/time columns.

The local BLAS-backed reference kernel raised floating-point-status warnings despite finite results. The executable uses direct squared distances and explicit tensor contractions, and its Nyström transform is checked against the reference implementation. Repeating the full comparison using this numerical implementation changed predictions by at most 4.8×10⁻¹⁰; no scientific result changed. All fitted coefficients and raw predictions must be finite before clipping. Warnings from the isolated reference test are recorded in `verification.json`; the execution log remains clean.

Primary outputs:

- `results/tables/update_20261107/model/manuscript_comparison.csv`: compact blood-only table for supplementary use.
- `overall_metrics.csv`, `metrics.csv`: overall, severity, treatment, centre and complete-assay diagnostics, including calibration and uncertainty.
- `paired_differences.csv`: same-patient model contrasts.
- `paired_change_metrics.csv`: correctly paired longitudinal changes.
- `coverage.csv`, `selection_counts.csv`, `historical_method_inventory.csv`: coverage, model-selection frequencies and audited prior methods.
- `predictions_PRIVATE.csv`, `fit_audit_PRIVATE.csv`, `fold_map_PRIVATE.csv`: patient-linked reproducibility records; retain locally rather than treating them as public aggregate source data.
- `reports/update_20261107/model/verification.json`, `execution.log`: actual numerical tests, software versions, input/output hashes and execution results.

No new deployable molecular formula is released by this comparison.
