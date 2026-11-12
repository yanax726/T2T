# Frozen E-EASI calculator

E-EASI means estimated EASI, the manuscript name for the existing hybrid BSA–serum estimator. The coefficients, input conventions and versioned calculator are unchanged. The authoritative parameter file is [unified_bsa_three_v1_20261017.json](../config/unified_bsa_three_v1_20261017.json), version `BSA-three-common-v1-20261017`. The calculator is [unified_bsa_three_calculator.py](../src/cytot2t/curation/unified_bsa_three_calculator.py).

## Equation

Let `clip(x, 0, 72) = min(72, max(0, x))`. The full-precision calculation is:

```text
EASI estimate = clip(BSA × [
    0.5127215790257503
  + 0.009847079991894058 × log2(CCL17)
  + 0.009646929477400354 × log2(CCL26)
  - 0.008270558237379214 × log2(IL-5)
], 0, 72)
```

| Input | Required convention |
|---|---|
| `bsa` | Percentage from 0 to 100; enter 10 for 10% |
| `CCL17` | Positive finite recorded concentration in pg/mL |
| `CCL26` | Positive finite recorded concentration in pg/mL |
| `IL-5` | Positive finite recorded concentration in pg/mL |

The concentration convention is the numeric `Summary.xlsx` export recorded in the frozen parameter file. Positive exports were used unchanged before log2 transformation. No additional dilution multiplier, pseudocount or half-minimum replacement is applied by this calculator. The laboratory protocol's dilution must not be applied a second time by assumption. Equivalence to a different assay or export convention has not been established.

Values reported as `<...` or `>...`, nonpositive concentrations, missing values, nonfinite values and BSA outside 0-100 are rejected. Positive numeric values are not proof of validated quantification. Assay bridging and validated quantification limits remain unestablished; the reported IL-5 software-bound sensitivity is a separate analysis. All four inputs are required even when BSA is zero. The zero result at BSA zero is a model constraint, not evidence that serum inflammation is absent.

## CSV use

Raw-concentration input:

```bash
PYTHONPATH=src python3 -m cytot2t.curation.unified_bsa_three_calculator \
  examples/unified_easi_input.csv /tmp/easi_output.csv --input-scale raw
```

For already transformed inputs, columns must be `bsa,log2_CCL17,log2_CCL26,log2_IL-5`:

```bash
PYTHONPATH=src python3 -m cytot2t.curation.unified_bsa_three_calculator \
  examples/unified_easi_log2_input.csv /tmp/easi_log2_output.csv --input-scale log2
```

Raw and log2 examples describe the same synthetic values. The calculator retains the input columns and appends `EASI_estimate`. Create the output's parent directory first. No drug, stage, age, BMI, week or previous score is used by the EASI calculation.

## Expected example and rounding

For BSA 10, CCL17 1024, CCL26 128 and IL-5 2, the log2 values are 10, 7 and 1. The output is `6.704503270491142`. [The expected CSV](../examples/unified_easi_expected.csv) includes this row and a zero-BSA row. `./run.sh demo` checks all fields with absolute tolerance `1e-12`.

The shorter manuscript display is:

```text
clip(BSA/100 × [51.27 + 0.98 log2(CCL17)
                      + 0.96 log2(CCL26) - 0.83 log2(IL-5)], 0, 72)
```

It returns 6.696 for the first synthetic example, a difference of about 0.00850 points. This is a display approximation, not the calculator implementation. Use full precision for threshold decisions. The complete [rounding audit](../results/tables/update_20261017/model/rounding/) distinguishes final-formula arithmetic differences from cross-validation prediction error.

## Scope of the results

The final coefficients were fitted to 241 W0/W12 visits from 156 patients after model selection. Primary performance assesses a nested development procedure, rather than independent validation of this one final fit. Week 24 was previously inspected within the same cohort. Baseline severe disease had larger errors, and few severe observations were present at follow-up. An accuracy advantage from adding the serum markers was not established.

PP-NRS, adult DLQI and recorded IGA have separately fitted formulas and parameters in `results/tables/update_20261017/model/auxiliary_targets/`. They do not inherit the EASI performance and did not show comparable accuracy with this restricted panel.

## Matched input comparisons

The [input comparison analysis](../results/tables/update_20261025/ablation/) evaluates EASI estimation from three markers alone, PP-NRS plus markers, BSA plus markers and BSA plus markers plus PP-NRS. BSA-only, CCL17-only and clinical-only controls are retained. These are separately refitted exploratory models, not revisions to the calculator above. BSA is a clinical assessment of affected area, and PP-NRS is the patient's current itch rating; models containing either are clinical-serum estimates rather than wholly molecular scores.

Each comparison uses identical eligible visits and patient splits, with model structure and shrinkage chosen inside the training folds. W0/W12 provide development data; W24 is same-cohort temporal application. The current-itch comparison contains 315 visits from 160 patients (W0: 151; W12: 89; W24: 75). The adult DLQI extension is refitted separately on its matched 207-visit subset, so its errors should not be compared directly with the larger cohort's errors.

| EASI estimator in the current-itch comparison | W0 MAE | W12 MAE | W24 MAE |
|---|---:|---:|---:|
| Three serum markers | 7.67 | 7.48 | 8.56 |
| PP-NRS plus three markers | 7.14 | 4.54 | 4.05 |
| BSA plus three markers | 3.20 | 1.29 | 1.19 |
| BSA plus three markers plus PP-NRS | 3.08 | 1.17 | 1.04 |

MAE is the mean absolute difference from observed EASI, in EASI points, calculated within each repeat before averaging. [Paired comparisons](../results/tables/update_20261025/ablation/paired_comparisons.csv) include patient-bootstrap intervals; [all metrics](../results/tables/update_20261025/ablation/overall_metrics.csv) also report RMSE, bias, Spearman correlation, concordance and calibration. Adding itch improved follow-up accuracy relative to BSA plus three markers. The controls do not establish a consistent accuracy advantage from the three markers, or that all three are necessary. Their biological associations are distinct from evidence of predictive benefit. These comparisons were conducted after earlier results were known and have not undergone independent validation.

A training-only reference-consistency sensitivity is reported in Supplementary Table S4 and [the aggregate directory](../results/tables/update_20261017/reference_sensitivity/). It repeats fitting within the original patient folds after excluding ten discrepant development visits. It does not change the frozen coefficients, adjudicate either clinical reference, or replace the primary performance estimates.

## Molecular-only extension and formula transfer

The [7 November comparison](../results/tables/update_20261107/model/manuscript_comparison.csv) tested ridge, nonlinear radial-basis learning with paired-change constraints, and ordinal-IGA-assisted reduced-rank learning. Patient folds and W0/W12 development boundaries were retained. IGA was a training outcome only, and the prediction inputs were serum measurements alone. The selected three-marker procedure gave W0/W12/W24 MAE of 7.50/7.21/7.80; the 19-marker procedure gave 7.41/7.26/8.49. These exploratory results did not yield an accurate blood-only clinical reader and did not replace E-EASI.

The [OMIX benchmark](../results/tables/external_validation_20261101/) applied separate, previously fitted molecular-only formulas without refitting. At W0/W16 their three-marker MAE was 11.51/6.92, versus 11.27/3.89 for unchanged p-EASI on the same participants. Higher W16 correlation did not mean smaller absolute error. OMIX lacks BSA, was previously examined, has possible participant overlap and has no established assay bridge. It is a retrospective formula-transfer comparison, not independent validation of the frozen hybrid formula.

Patients with unclassified initial dose can contribute to treatment-free model development. Treatment-group holdouts exclude them from both fitting and evaluation; this does not change the frozen formula or primary overall validation.

## How the compact panel was retained

The initial discovery registry used at least 70% baseline assay availability to define 19 candidates from 23 measured markers. A separate early supervised panel-size search favoured CCL17 alone. CCL17, CCL26 and IL-5 were subsequently retained as a biologically motivated, well-covered compact candidate, initially tested as an equally weighted standardized molecular index. They were not the three highest-ranked markers from that automated search. In the common E-EASI analysis the marker identities stayed fixed; inner validation selected the formula structure and ridge penalty. Earlier panel formation was exploratory and was not repeated inside the later outer validation. Matched controls and removal of individual markers have not established that this exact panel is optimal or that every marker gives a stable accuracy gain. The initial relative molecular index, the older six-marker inflammation score and the current BSA–serum EASI equation are distinct calculations.
