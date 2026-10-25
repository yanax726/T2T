# T2T clinical and serum analysis

Code and aggregate results for a Chinese observational atopic dermatitis cohort, including clinical treatment comparisons, 23 serum markers and a common BSA-serum equation for EASI estimation.

The equation uses BSA, CCL17, CCL26 and IL-5. Its primary evaluation uses patient-level nested validation; week 24 is previously inspected within-cohort temporal application. An accuracy advantage from adding the serum markers was not established. These are research estimates, not a validated replacement for clinical assessment.

## Read the study

| Material | Location |
|---|---|
| Manuscript and supplementary information | [Main PDF](manuscript/main.pdf), [appendix PDF](manuscript/appendix.pdf) |
| All current figures and captions | [Figure book](manuscript/figure_book.pdf), [figure catalogue](manuscript/figure_catalog.json) |
| Numerical results, including controls and negative findings | [Aggregate tables](results/tables/update_20261017/) |
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

The current revision passed 31 local tests and an isolated aggregate rebuild; 156 distributed manuscript and figure files were byte-identical. See the [verification record](reports/revision_20261025/public_verification.json). An optional [GitHub Actions template](docs/ci/public-checks.yml) is supplied but is not active. After granting a publishing credential permission to modify workflows, copy it to `.github/workflows/public-checks.yml` to enable Actions.

`rebuild` uses the distributed aggregates. `analyse` refits the analyses only after authorized restricted snapshots pass the declared hash checks. Participant records, individual predictions, fold membership, clinical dates and free-text notes are excluded from this repository. See [data access](DATA_ACCESS.md).

## Version, citation and license

This research snapshot is version [2026.10.25-rc1](VERSION). Cite the version and Git commit using [CITATION.cff](CITATION.cff); no manuscript DOI or manuscript author list is asserted. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

Original code, data and manuscript materials have no assigned reuse license. The existing Apache-2.0 terms for the third-party panel-alignment helper are retained. See [license status](LICENSE_STATUS.md) and [third-party notice](NOTICE.md).

Software questions can be filed in the [repository issue tracker](https://github.com/yanax726/T2T/issues). Do not include participant data in a public issue.
