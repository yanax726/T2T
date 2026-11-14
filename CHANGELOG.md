# Changelog

## 2026.11.14-rc1

- Add training-only selection of compact baseline panels for week-12 itch response, including matched clinical controls, panel stability and source-file holdout results.
- Add current-cohort week-12 three-marker prediction of week-24 EASI75, retaining its negative molecular result and the predictor-imputation sensitivity.
- Present these exploratory analyses in Supplementary Tables S18–S20, with conditional uncertainty and a short main-text summary. The E-EASI equation and figure set are unchanged.

## 2026.11.12-rc3

- Add an exploratory baseline three-marker comparison for measured week-12 itch improvement, with matched clinical controls and a predictor-imputation sensitivity.
- Report repeated patient-held-out AUC, probability errors and uncertainty in Supplementary Table S18; distinguish these models from the frozen E-EASI equation.

## 2026.11.12-rc2

- Identify complete-three-visit and all-available-pair molecular populations directly on figures and in their captions.
- Align the clinical narrative, molecular correction families and unmeasured laboratory-point descriptions across publication outputs.
- Recheck the current figure set against distributed aggregate results; preserve model parameters and numerical estimates.

## 2026.11.12-rc1

- Use all-available standardized IgE/EOS trajectories with explicit centre and temporal-shape sensitivities; retain earlier imputation analyses.
- Clarify the fixed three-marker candidate history and its distinct supervised-search comparison.
- Unify title-free figure typography, simplify the main input display and move molecular significance marks beside their rows.
- Supply editable supplementary tables and a complete labelled review deck.


The optional CI workflow is distributed at `docs/ci/public-checks.yml`. It is not active until a maintainer with workflow permission installs it under `.github/workflows/`; local verification is unaffected.

## 2026.11.07-rc1

- Name the unchanged frozen hybrid estimator E-EASI (estimated EASI); retain its coefficients and primary performance.
- Rebuild the local framework and fixed-patient W0/W12/W24 molecular display, using separate visit colours and a declared 138-hypothesis family.
- Add baseline molecular summary tables, an explicitly estimated dupi W4 eosinophil sensitivity, and single-visit IEC-compatible states.
- Report nonlinear/paired-change/ordinal-IGA-assisted molecular-only learning and the retrospective OMIX/p-EASI benchmark, including negative findings.
- Add the authorized-input `update-analysis` command, extend `analyse`, and retain aggregate-only `rebuild`.
- Publish reviewed code, configurations and screened aggregate tables. Exclude participant outputs, local correspondence and original planning materials.

## 2026.10.31-rc5

- Apply one semantic colour system to all five main and 21 supplementary figures.
- Simplify duplicate numeric annotations and enlarge the clinical panels; preserve exact counts in the existing tables and source data.
- Check colour semantics, final rendering and numerical preservation. The formula, estimates and uncertainty intervals are unchanged.

## 2026.10.31-rc4

- Correct near-boundary response-classification intervals without changing predictions or point estimates; provide counts and Wilson intervals for each validation repetition.
- Restore designated early clinical visits, show key available denominators, and distinguish sparse molecular estimates from inferential results.
- Clarify IgE observation patterns and eosinophil coverage; add paired laboratory summaries and IGA-free target sensitivity.
- Report early itch associations and persistent W24 differences separately, retaining the observational interpretation.
- Preserve the frozen formula, all continuous model metrics, original records and earlier presentation.

## 2026.10.25-rc3

- Add matched comparisons of molecular, clinical and combined EASI inputs, including CCL17 and clinical-only controls, paired uncertainty and an adult DLQI extension.
- Add observed-pair, longitudinal-model and multiple-imputation IgE analyses, retaining method-dependent results and measured values unchanged.
- Include the new analysis code, protocols, reviewed aggregates and verification records in the public package; add a restricted-data `sensitivity` command.
- Keep the frozen EASI formula and calculator unchanged.

## 2026.10.25-rc2

- Align figure placement to the original 44-slide source deck: restore observed clinical courses to the main article and combine absolute-error and EASI75 classification panels.
- Remove repeated or retired visual panels while preserving their numerical source results and limitations. Organize 21 supplementary figures by clinical question.
- Rebuild the clinical framework and cohort flow as connected stages with distinct clinical and serum-model eligibility branches.
- Keep the frozen formula, patient records and original presentation unchanged.

## 2026.10.25-rc1

- Consolidate the main displays into clinical targets, IGA and treatment pathways, early serum changes and individual EASI estimation.
- Reduce repeated supplementary displays while retaining all aggregate findings and controls; group 22 supplementary figures by clinical question.
- Make baseline quartile interpolation and adult DLQI eligibility explicit. Independent reconciliation left the baseline values and frozen model unchanged.
- Retain observed gaps for unmeasured visits and separate fixed-patient IgE trajectories from available-case summaries.

## 2026.10.18-rc3

- Organize the manuscript around the clinical framework, three-group clinical course, complete serum panel, EASI estimation and individual response monitoring.
- Use equal-size count cells instead of variable-area bubbles; display equation coefficients in Methods.
- Add the training reference-exclusion sensitivity, weighted missing-outcome bounds, safety-documentation windows and input/reference specifications.
- Retain the primary formula, original results and all numerical controls. The earlier local presentation is not updated in this manuscript revision.

## 2026.10.18-rc2

- Provide a documented synthetic calculator demonstration and expected numerical output.
- Document the frozen equation, recorded pg/mL inputs, rounding convention and limits of interpretation.
- Separate aggregate figure/manuscript rebuilding from refitting on restricted participant snapshots.
- Include the current figure catalogue and all current public aggregate results, including controls and negative results.
- Restrict the publication package to research code, source tables, manuscript materials and necessary reproducibility records.
- Add software citation metadata, dependency specifications and continuous integration for public tests.

The frozen equation remains `BSA-three-common-v1-20261017`. This repository version does not indicate independent clinical validation or a manuscript publication.
