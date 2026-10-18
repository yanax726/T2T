# Frozen EASI calculator

The authoritative parameter file is [unified_bsa_three_v1_20261017.json](../config/unified_bsa_three_v1_20261017.json), version `BSA-three-common-v1-20261017`. The calculator is [unified_bsa_three_calculator.py](../src/cytot2t/curation/unified_bsa_three_calculator.py).

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

