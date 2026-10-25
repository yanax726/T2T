# Independent review of the IgE missingness analysis

Reviewed on 2026-10-25. Read-only review of `src/cytot2t/curation/ige_missingness_20261025.py`, its configuration, aggregate outputs, numerical verification and source-coverage tables. Five short-chain draws and an independently calculated scalar pooling example were executed without writing clinical data or production results. The full 400-imputation chains were not rerun by this reviewer.

## Conclusion

No blocking arithmetic, repeated-measure-unit, observed-value-preservation or pooling error was found. The analysis is suitable as a clearly labelled post hoc sensitivity analysis. It does not establish a corrected true IgE treatment effect, and the abro 200 mg W24 direction remains dependent on missing-data assumptions.

## Computation and design checks

- The independent unit is the patient. W0/W12/W24 are columns of one patient record, not three independent observations.
- Normal-regression coefficient draws include residual-variance uncertainty through an inverse-chi-square draw. PMM compares donor fitted means with recipient means from a coefficient draw, corresponding to type-1 matching. Donors are measured values from the same nominal visit and initial treatment. This matches the core method described in the [mice PMM documentation](https://amices.org/mice/reference/mice.impute.pmm.html).
- The short-chain check confirmed that all measured entries remained bitwise identical, every PMM-filled entry came from the declared treatment/visit donor pool, and the input arrays were unchanged.
- The normal-log alternative draws residuals as well as coefficients. Conversion back to log(1+IgE) is numerically correct, and measured values are explicitly reset to the exact observed log1p values.
- Rubin total variance and the Barnard–Rubin degrees-of-freedom calculation were independently reconstructed; maximum numerical difference was below 3×10^-17. Group means and paired changes use patient-based complete-data degrees of freedom. See the [mice scalar pooling documentation](https://amices.org/mice/reference/pool.scalar.html).
- The supplied independent statsmodels check confirms the weighted coefficients and patient-cluster sandwich covariance to numerical precision. Two follow-up rows are grouped by patient before forming cluster scores. The adjusted contrasts retain initial treatment and hold the previously estimated overlap weights fixed.
- The likelihood model uses all measured IgE rows with a patient random intercept. Its treatment-by-visit contrasts do not use fitted individual random effects as new-patient predictions. Both fits converged without reported optimizer warnings.
- The complete primary-visit clinical population is 163, with 16 patients having no measured W0/W12/W24 IgE. The available-data models use 125 CAMS patients/229 measurements and 147 whole-cohort patients/258 measurements. The imputation populations contain 130 and 163 patients, respectively. These counts must not be interchanged.
- There are no measured W4 IgE entries in the audited source, and the code does not create a W4 trajectory.
- Values recorded as 5000 are retained without declaring that number a validated assay ceiling. No hidden upper truncation or chosen negative trend was found.

Targeted check output: `ige_independent_targeted_checks.json`.

## Interpretation requirements

1. **Keep centre support visible.** Huashan has no measured dupi IgE at W12 or W24. Whole-cohort imputation borrows across centres under model assumptions; it cannot create direct evidence in those cells. CAMS is the better-supported sensitivity population. Restricting the primary sensitivity to CAMS does not make its follow-up missingness ignorable.

2. **Show the method disagreement.** In CAMS abro 200 mg, the W0–W24 ratio is 1.92 (95% CI 0.88–4.18) with PMM and 0.688 (0.311–1.522) with normal-log imputation. PMM has limited low-baseline donor support; the normal model extrapolates beyond the observed range. Neither direction should be chosen because it looks more plausible or agrees with expectations. The random-intercept model is another assumption-based sensitivity, not a gold-standard adjudicator.

3. **Label the transformation correctly.** `exp(change)` here is a ratio for **1 + recorded IgE**, not exactly a raw IgE fold change. `expm1(mean(log1p(IgE)))` is a shifted geometric mean, not an arithmetic mean or median. Use those terms in axes and captions. Generic p values in the level table test a log-scale mean against zero; they are not efficacy tests and should not appear as treatment-effect stars.

4. **Keep distributional extrapolation explicit.** Normal-log imputation produces some draws far above the recorded range, with isolated maxima around 0.7–0.9 million. These are posterior predictive tails, not measured values. The units and any top-coding remain unconfirmed; treating all draws as demonstrated biology or imposing an unverified 5000 ceiling would both be inappropriate. PMM and Gaussian results should remain visible together.

5. **Distinguish reconstruction from prediction.** Other waves, including later EASI/IgE, are auxiliary predictors in retrospective missing-data reconstruction. That is not temporal leakage for this descriptive sensitivity, but it would be inappropriate to describe it as a prospective prediction model. Auxiliary EASI draws are internal to imputation and are not clinical validation outcomes.

6. **Limit the treatment claim.** Initial strategy is retained after reductions or switching. The weighted analysis conditions on fixed estimated overlap weights and does not propagate their estimation uncertainty. The models do not identify a randomized dose effect, and undocumented exposure-related missingness is not repaired by including current EASI.

7. **Report exploratory uncertainty.** New intervals and p values are not a multiplicity-controlled confirmatory family. The scalar within-group changes and the adjusted between-group contrasts answer different questions. An apparently rising available-case median is not itself evidence of individual worsening.

## Diagnostics and remaining limitations

Across exported summaries, maximum split R-hat was approximately 1.028, and the largest Monte Carlo SE was about 5.6% of total inferential SE. These support numerical stability of the chosen algorithms; they do not verify MAR, correct the observed-support gaps or prove that either distributional model is true. Fraction of missing information is high for some change estimates (up to about 0.85). The analyst reports a complete deterministic replay of the final aggregate results; the full replay was not independently rerun in this review.

No production code change is requested by this review. The interpretation and labelling requirements above were communicated to the integration owner and the IgE analyst before manuscript integration.
