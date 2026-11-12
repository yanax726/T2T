# T2T clinical and serum analysis

Code and aggregate results for a Chinese observational atopic dermatitis cohort, including clinical treatment comparisons, 23 serum markers and a common BSA-serum equation for EASI estimation.

E-EASI means estimated EASI. The equation uses BSA, CCL17, CCL26 and IL-5; this name does not change the frozen coefficients or calculator. Its primary evaluation uses patient-level nested validation; week 24 is previously inspected within-cohort temporal application. An accuracy advantage from adding the serum markers was not established. These are research estimates, not a validated replacement for clinical assessment.

## Read the study

| Material | Location |
|---|---|
| Manuscript and supplementary information | [Main PDF](manuscript/main.pdf), [appendix PDF](manuscript/appendix.pdf) |
| All current figures and captions | [Figure book](manuscript/figure_book.pdf), [review deck](manuscript/figure_deck.pdf), [PPTX](manuscript/figure_deck.pptx), [figure catalogue](manuscript/figure_catalog.json) |
| Editable supplementary tables | [Word appendix](manuscript/appendix.docx) |
| Numerical results, including controls and negative findings | [Aggregate tables](results/tables/update_20261017/) |
| Matched input comparisons and IgE missing-data sensitivity | [Model comparisons](results/tables/update_20261025/ablation/), [IgE analyses](results/tables/update_20261025/ige/) |
| Laboratory coverage and clinical definition sensitivity | [Aggregate checks](results/tables/update_20261031/), [repetition-specific response intervals](results/tables/update_20261017/redesign_20261018/monitoring_response_per_repeat_uncertainty.csv) |
| Current molecular, laboratory and framework analyses | [7 November tables](results/tables/update_20261107/), [formula-transfer benchmark](results/tables/external_validation_20261101/) |
| Figure source data | [Source tables](manuscript/source_data/), [source-to-result map](docs/source_map.md) |
| Frozen formula and input conventions | [Model documentation](docs/model.md), [parameter file](config/unified_bsa_three_v1_20261017.json) |

## Quick demonstration

The frozen EASI calculator requires Python 3.9 or newer and the standard library only. From the repository root:

```bash
./run.sh demo
```

The synthetic example with BSA 10%, CCL17 1024 pg/mL, CCL26 128 pg/mL and IL-5 2 pg/mL returns **6.704503270491142** EASI points. The command checks the output against [the expected CSV](examples/unified_easi_expected.csv) and writes `results/demo/`. The second example has BSA zero and returns zero.

## Reproduce figures and tables

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
./run.sh test
./run.sh rebuild
```

PDF rebuilding also requires pdfLaTeX and BibTeX. The [reproduction guide](docs/reproduction.md) documents dependencies, tested environment, runtimes and each command. Figure identities and counts follow the live catalogue rather than a fixed count in this README.

The current [tests](reports/update_20261112/tests.log), [quality checks](reports/update_20261112/final_quality_assurance.json) and [isolated public rebuild](reports/update_20261112/public_reproducibility.json) document the checks actually run. The core rebuild reproduces publication figures, tables and TeX PDFs from distributed aggregates; Office exports have a separate [portable workflow](docs/editable_exports.md). Earlier records retain their original dates and do not certify later changes. The [figure style guide](docs/figure_style.md) defines the shared colour roles and display rules. An optional [GitHub Actions template](docs/ci/public-checks.yml) is supplied but is not active. After granting a publishing credential permission to modify workflows, copy it to `.github/workflows/public-checks.yml` to enable Actions.

`rebuild` uses the distributed aggregates. `analyse` refits the analyses only after authorized restricted snapshots pass the declared hash checks. `sensitivity` reruns the earlier sensitivity analyses. `update-analysis` runs the new laboratory, molecular-learning and formula-transfer comparisons in an authorized workspace with existing curated inputs and predictions. It skips only the external benchmark when its OMIX workbook is absent. Participant records, individual predictions, fold membership, clinical dates and free-text notes are excluded from this repository. See [data access](DATA_ACCESS.md).

The input comparisons refit each model on the same eligible visits and patient splits. Adding current PP-NRS improved follow-up accuracy relative to BSA plus three markers; this does not establish a molecular contribution, and the frozen calculator is unchanged. IgE analyses compare observed pairs, longitudinal modelling and two multiple-imputation methods. Imputed values are estimates, not new measurements. The abrocitinib 200 mg IgE direction depends on the imputation method and remains uncertain.

## Current evidence

The frozen E-EASI procedure has W0/W12/W24 MAE of 3.19/1.30/1.19 points. It is a hybrid clinical–molecular estimate: BSA contributes clinical information, and a consistent molecular accuracy gain was not established. New blood-only learning used nonlinear patterns, paired changes and ordinal IGA during training. Its three-marker MAE remained 7.50/7.21/7.80; a 19-marker panel did not improve performance consistently.

Supplementary Tables S12–S17 cover baseline molecular distributions, framework definitions, the new molecular-only comparisons, the existing formula-transfer benchmark, post hoc single-visit IEC-compatible EASI-and-itch states and all-available adjusted IgE changes. Figure 4 uses patients with all three molecular visits; Figures S11–S13 use every available pair for their stated interval, without requiring the third visit. Both use paired sign-flip tests, with different patient sets and declared multiplicity families. In the dupi W0/W12 available-pair analysis six markers meet the FDR threshold, compared with two in the complete-three-visit subset at W12. These findings describe different populations rather than conflicting tests. All W4 IgE and dupi W4 eosinophil values are unmeasured; hollow display points are explicit conditional temporal interpolations, not recovered measurements. The revised IgE display uses all available responses and shows CAMS-only sensitivity for missing Huashan dupi follow-up. OMIX002850 has no BSA and cannot validate E-EASI; prior inspection, possible overlap and assay differences also prevent an independent-validation claim.

## Version, citation and license

This research snapshot is version [2026.11.12-rc2](VERSION). Cite the version and Git commit using [CITATION.cff](CITATION.cff); no manuscript DOI or manuscript author list is asserted. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

Original code, data and manuscript materials have no assigned reuse license. The existing Apache-2.0 terms for the third-party panel-alignment helper are retained. See [license status](LICENSE_STATUS.md) and [third-party notice](NOTICE.md).

Software questions can be filed in the [repository issue tracker](https://github.com/yanax726/T2T/issues). Do not include participant data in a public issue.
