# Changelog

The optional CI workflow is distributed at `docs/ci/public-checks.yml`. It is not active until a maintainer with workflow permission installs it under `.github/workflows/`; local verification is unaffected.

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
