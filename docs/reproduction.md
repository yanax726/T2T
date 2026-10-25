# Reproduction guide

## Environment and installation

The recorded analysis environment is Python 3.9.6 on macOS 26.6.2, Apple Silicon arm64. It uses NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1, scikit-learn 1.6.1, matplotlib 3.9.4 and statsmodels 0.14.6. No GPU or non-standard hardware is required.

The frozen EASI calculator and `demo` use the Python standard library only. The analysis, figures and full tests require third-party libraries:

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

[requirements.lock.txt](../requirements.lock.txt) records the complete tested environment. [requirements.txt](../requirements.txt) lists the direct numerical dependencies for readers using their own environment; transitive versions will then be resolved by pip. `T2T_PYTHON=/path/to/python ./run.sh test` selects another interpreter. Python 3.9 or newer is required by the calculator; the pinned full environment was tested on 3.9.6.

The optional [GitHub Actions template](ci/public-checks.yml) defines Python 3.9/3.12 calculator checks, public tests and an aggregate build. It is supplied as documentation and is not active: the publishing OAuth credential lacks permission to modify workflows. Local checks have passed; no remote CI run is claimed. To enable Actions, first grant the publishing credential the required workflow permission, then copy the unchanged template to `.github/workflows/public-checks.yml` and commit it. Check the resulting run status before claiming cross-platform verification.

PDF builds additionally require `pdflatex` and `bibtex`, with Latin Modern, natbib, booktabs, longtable, amsmath, geometry, float and hyperref. `caption` and `microtype` are optional. Use a TeX Live or MacTeX installation that provides these commands. Plots use the matplotlib-distributed DejaVu Sans font. No browser or external API is required for the scientific rebuild.

## Commands

Run commands from the repository root. Except for the demo, output paths match the supplied manuscript and result folders, so use a fresh checkout to compare rebuilt files with the distributed snapshot.

| Command | Inputs | Outputs and boundary |
|---|---|---|
| `./run.sh demo` | Synthetic example and frozen EASI JSON | `results/demo/unified_easi_output.csv` and measured `runtime.json`; verifies expected values |
| `./run.sh test` | Synthetic tests, formula parameters and public aggregates | Calculator, rounding, pairing, denominator and package-boundary checks; no patient data |
| `./run.sh figures` | Current aggregate tables, style and parameters | PDF/SVG/PNG figures, `manuscript/figure_catalog.json`, figure source data and layout checks |
| `./run.sh manuscript` | Catalogue, aggregate tables, manuscript sections and bibliography | Main, appendix and figure-book TeX/PDF; requires TeX |
| `./run.sh rebuild` | Same public inputs as the preceding two rows | Runs figures, then manuscript compilation; does not refit models |
| `./run.sh sensitivity` | Authorized snapshots plus existing primary model predictions | Rebuilds clinical weights, runs IgE missing-data and matched-input analyses, then rebuilds figures and manuscript |
| `./run.sh analyse` | Exact authorized snapshots in `config/reproduction_inputs.json` | Recomputes clinical, regional, EASI, rounding, published-formula sensitivity, auxiliary-target, reference-exclusion, IgE missing-data, matched-input and display analyses, then rebuilds figures and manuscript |

The optional local `guide` command concerns presentation notes and requires local authoring materials; those materials are not distributed with the research repository.

## Runtime and expected output

Creating a fresh standard-library virtual environment took 1.21 seconds; running the complete two-row demo in it took 0.072 seconds, of which approximately 0.027 seconds ran the frozen calculator. The 31 public tests took 27.25 seconds on first use of a fresh pinned numerical environment and 2.32 seconds in the existing environment. `./run.sh demo` records a fresh elapsed time and checks the expected EASI values `6.704503270491142` and `0.0`. This demonstrates calculation and input conventions, not clinical validity.

Runtime records for the current public demonstration and tests are included in `docs/runtime.json`. All pinned dependencies were installed successfully in a fresh Python 3.9.6 environment, and `pip check` found no broken requirements. Download/install time was not instrumented; it depends on network access and wheel availability. The standard-library calculator needs no package installation.

The timings above are earlier environment benchmarks, not a runtime guarantee for this expanded analysis. The current [test log](../reports/update_20261025/integration/final_tests.log), [quality checks](../reports/update_20261025/integration/final_quality_assurance.json) and [rebuild comparison](../reports/update_20261025/integration/final_reproducibility_check.json) record the release checks. Public rebuilding uses the published aggregates; it does not refit the clinical models or regenerate imputations.

## Additional analyses

The [matched-input protocol](../config/model_ablation_20261025.json) retains the original patient folds and compares refitted estimators on identical complete-case visits. It includes serum-only, clinical-only and combined inputs, a CCL17 reference and removal of individual markers. The adult DLQI comparison has its own matched adult subset. Patient-bootstrap comparisons preserve pairing across models and repeats. Its [verification](../reports/update_20261025/ablation/verification.json) and [replay record](../reports/update_20261025/ablation/replay_verification.json) are distributed with the numerical aggregates. The primary formula and its parameters are unchanged.

The [IgE protocol](../config/ige_missingness_20261025.json) analyses W0/W12/W24 with all available observed pairs, an available-data longitudinal model and two patient-wide multiple-imputation methods. Each method uses four chains and 400 retained completed datasets per population. Observed values remain fixed. Imputed clinical values are auxiliary variables in the retrospective IgE analysis and never validation outcomes for the EASI calculator. CAMS is analysed separately because follow-up IgE is entirely unobserved for Huashan dupilumab; whole-cohort imputation requires additional cross-centre assumptions. The direction for abrocitinib 200 mg differs between imputation methods. The analysis reports this sensitivity rather than choosing a preferred direction. See [methods](../reports/update_20261025/ige/methods_en.md), [numerical checks](../reports/update_20261025/ige/numerical_verification.json) and [replay verification](../reports/update_20261025/ige/replay_verification.json).

`sensitivity` requires `results/tables/update_20261017/model/predictions.csv` from an earlier authorized `analyse` run. It first regenerates the clinical overlap weights needed by the IgE analysis. On a fresh authorized workspace, use `analyse`, which creates all these dependencies in order. Neither command can run from public aggregates alone.

## Reproduction scope

1. Public calculation verifies a frozen formula on synthetic inputs.
2. Public aggregate rebuilding reproduces figures, tables and manuscript materials from distributed numerical outputs.
3. Authorized full refitting reproduces the analyses from versioned curated participant snapshots. It cannot be performed from aggregate means and does not recreate undocumented spreadsheet adjudication.

`analyse` and `sensitivity` begin with a SHA-256 preflight and stop when private files are missing or changed. They create participant-level outputs locally. Keep these restricted. Public sensitivity outputs are selected by explicit filename allowlists and checked for identifiers. Fold-level fitting diagnostics and coefficients are included; individual predictions and patient membership are excluded. See [DATA_ACCESS.md](../DATA_ACCESS.md).

The live figure catalogue is the authority for current figure identifiers and source dependencies. `SHA256SUMS` in the publication package records distributed file contents. PDF bytes may differ with TeX, font and platform versions even when numerical results agree. Review numerical equality separately from rendering equality.

