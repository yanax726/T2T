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
| `./run.sh sensitivity` | Authorized snapshots plus existing primary model predictions | Rebuilds clinical weights, runs IgE missing-data, matched-input and IGA-definition analyses, reconciles laboratory/clinical results, updates classification bounds, then rebuilds figures and manuscript |
| `./run.sh update-analysis` | Authorized snapshots and existing frozen/ablation predictions | Runs laboratory revision, molecular-only extension and local paired benchmark; runs OMIX benchmark only if its workbook is present; does not regenerate figures |
| `./run.sh itch-update` | Authorized clinical and serum snapshots | Refits the fixed-three-marker W0-to-W12 itch comparisons and rebuilds Table S18 |
| `./run.sh itch-panel-update` | Authorized snapshots and saved fixed-three itch OOF controls | Repeats training-only panel selection and whole-source-file holdout, then rebuilds Table S19 |
| `./run.sh landmark-update` | Authorized clinical and serum snapshots | Refits W12 three-marker prediction of observed W24 EASI75 and rebuilds Table S20 |
| `./run.sh analyse` | Exact authorized snapshots in `config/reproduction_inputs.json` | Recomputes clinical, regional, EASI, rounding, published-formula sensitivity, auxiliary-target, reference-exclusion, IgE missing-data, matched-input and display analyses, followed by the 7 November laboratory/molecular and paired-benchmark analyses, then rebuilds figures and manuscript |

The retired local `guide` command exits with a notice; its earlier figure numbering is no longer current. Figure explanations are in the figure book and review-deck notes.

## Runtime and expected output

Creating a fresh standard-library virtual environment took 1.21 seconds; running the complete two-row demo in it took 0.072 seconds, of which approximately 0.027 seconds ran the frozen calculator. The 31 public tests took 27.25 seconds on first use of a fresh pinned numerical environment and 2.32 seconds in the existing environment. `./run.sh demo` records a fresh elapsed time and checks the expected EASI values `6.704503270491142` and `0.0`. This demonstrates calculation and input conventions, not clinical validity.

The earlier environment benchmark is recorded in `docs/runtime.json`. At that check, all pinned dependencies were installed successfully in a fresh Python 3.9.6 environment, and `pip check` found no broken requirements. Download/install time was not instrumented; it depends on network access and wheel availability. The standard-library calculator needs no package installation.

The timings above are earlier environment benchmarks, not a runtime guarantee for this expanded analysis. The 31 October release passed 53 tests, including semantic colour and image-transform checks. These historical counts are not a claim about the current revision. The [test log](../reports/visual_revision_20261031/tests.log), [quality checks](../reports/visual_revision_20261031/final_quality_assurance.json) and [rebuild comparison](../reports/visual_revision_20261031/public_reproducibility.json) record the release checks. Public rebuilding uses the published aggregates; it does not refit the clinical models or regenerate imputations. The latest rebuild uses a fresh copy of the public package with the existing numerical and TeX environment; it is not a new dependency-installation test.

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

## 7 November analysis update

`./run.sh update-analysis` is for an authorized workspace that already contains the curated participant snapshots and saved primary/ablation predictions. It runs the laboratory revision (including single-visit IEC-compatible states), the molecular-only extension and the local same-patient p-EASI benchmark. `analyse` produces the preceding dependencies and then runs the same update sequence. The external formula benchmark additionally requires `data/clinical_alignment/external/OMIX002850-01.xlsx`. If absent, the runner prints an explicit skip message and still completes the other update analyses; published external aggregates remain available for `rebuild`.

The new configurations are `laboratory_molecular_revision_20261107.json`, `molecular_upgrade_20261107.json`, `local_paired_benchmark_20261101.json` and `external_molecular_benchmark_20261101.json`. No raw participant data, individual predictions, fold membership, local email records or original planning documents are included in the public package. New CSV outputs pass the same identifier and column screening as the earlier aggregates. Table S12–S16 source data are distributed with the manuscript package.

`rebuild` reads aggregate results and redraws the manuscript; it never trains models or fills missing patient values. `run.sh` sets both `MPLCONFIGDIR` and `XDG_CACHE_HOME` below `results/.cache/`, giving Matplotlib and Fontconfig a project-local writable cache. BLAS thread counts default to one and can be overridden by the environment. These caches are generated locally and are not publication inputs.

## 12 November laboratory and editable-document update

`./run.sh lab-update` requires the authorized curated clinical snapshot. It recomputes the all-available IgE/eosinophil response-profile models and explicit W4 interpolation sensitivities, then rebuilds figures and manuscript PDFs. These steps also run inside `update-analysis` and `analyse`. Interpolated points never enter observed data, treatment tests or prediction truth. `rebuild` uses the distributed aggregates and does not refit these models.

The package also contains editable supplementary tables in `manuscript/appendix.docx` and a review deck in `manuscript/figure_deck.pptx`/`.pdf`. Those review exports are separate from the core numerical/TeX rebuild. Their optional export module documents additional document-rendering dependencies. Publication figures remain vector PDF/SVG; the review deck uses exact PNG previews to avoid SVG font-substitution differences.

## Baseline prediction of week-12 itch improvement

`./run.sh itch-update` requires the declared clinical and molecular participant snapshots and refits the fixed three-marker, clinical-only and combined logistic comparisons. The target uses two observed PP-NRS values; missing outcomes are never imputed. Training-fold-only predictor imputation is a separate 105-patient sensitivity. Aggregate `rebuild` reproduces Supplementary Table S18 from distributed metrics without refitting. Individual probabilities and fold membership remain under the restricted subdirectory. The two probability aggregations and their conditional intervals are reported separately; no final prognostic calculator or clinical risk threshold is validated.

## Compact panels and the week-12 landmark analysis

Supplementary Table S19 reports fixed-size training-selection procedures for three, four, five, six and 19 baseline markers, plus an inner-validation simplicity rule. Preprocessing and marker selection are repeated within training; the held-out outcome does not select the panel size. The roughly 0.80 AUROC includes baseline clinical information. It does not apply to markers alone or to a fixed list selected after seeing the full cohort. The paired fourth-versus-third marker gain is uncertain. Whole-source-file holdout is reported alongside patient-level cross-validation because the compact models were sensitive to source and case composition; report-file groups are not confirmed physical assay batches.

`./run.sh itch-panel-update` requires both private OOF control files produced by `itch-update`, in addition to authorized clinical and molecular snapshots. It refits the panel-size procedures, verifies their saved metrics and training selections, runs the limited source-file stress test and rebuilds the manuscript. The control filenames and hashes are recorded in the protocol and result receipts. This command cannot run from the aggregate-only public checkout. To replay metrics from existing restricted predictions without fitting, an authorized workspace can run `PYTHONPATH=src .venv/bin/python -m cytot2t.curation.itch_panel_size_20261112 --replay-only`.

`./run.sh landmark-update` uses measured W0 EASI, W12 information and measured W24 EASI. The target is W24 EASI75 relative to W0, not an additional 75% improvement after W12. The primary population has 68 patients, including only nine nonresponders; the 72-patient sensitivity permits training-only imputation of missing W12 predictors. Clinical and combined controls use information available by W12, and no W24 predictor is admitted. All three model comparisons remain in Table S20 despite the negative molecular result.

Tables S18–S20 score each repeated held-out prediction set separately before averaging. Their patient-bootstrap intervals condition on saved predictions; neither these intervals nor high AUPRC in a predominantly responsive sample prove clinical utility or external validity. An alternative aggregation that scores each patient's averaged probabilities is explicitly separate in the source data. The public `rebuild` reproduces all three tables from distributed aggregates without patient records. No future-response calculator is provided.
