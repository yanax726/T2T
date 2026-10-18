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
| `./run.sh analyse` | Exact authorized snapshots in `config/reproduction_inputs.json` | Recomputes clinical, regional, EASI, rounding, published-formula sensitivity, auxiliary-target and display aggregates, then rebuilds figures and manuscript |

The optional local `guide` command concerns presentation notes and requires local authoring materials; those materials are not distributed with the research repository.

## Runtime and expected output

Creating a fresh standard-library virtual environment took 1.21 seconds; running the complete two-row demo in it took 0.072 seconds, of which approximately 0.027 seconds ran the frozen calculator. The 31 public tests took 27.25 seconds on first use of a fresh pinned numerical environment and 2.32 seconds in the existing environment. `./run.sh demo` records a fresh elapsed time and checks the expected EASI values `6.704503270491142` and `0.0`. This demonstrates calculation and input conventions, not clinical validity.

Runtime records for the current public demonstration and tests are included in `docs/runtime.json`. All pinned dependencies were installed successfully in a fresh Python 3.9.6 environment, and `pip check` found no broken requirements. Download/install time was not instrumented; it depends on network access and wheel availability. The standard-library calculator needs no package installation.

In an isolated copy of the public package, `rebuild` took 34.74 seconds. All figure PDF/SVG/PNG files and all three manuscript PDFs matched the supplied files byte for byte; all 205 checked numerical and source files were unchanged. Demo and tests also passed, and restricted-input preflight stopped as expected. The [public rebuild report](../reports/update_20261018/revision2/qa/public_rebuild.json) records commands, versions and source hashes. The historical scientific replay logs report analysis timings and hash comparisons for their own recorded source snapshots; those logs do not assert a new model refit during this aggregate rebuild.

The subsequent CI-template relocation changed documentation and package selection only. Scientific code, parameters, numerical results and renderings remained byte-identical to the verified package; the recorded rebuild hashes identify that earlier verification snapshot.

## Reproduction scope

1. Public calculation verifies a frozen formula on synthetic inputs.
2. Public aggregate rebuilding reproduces figures, tables and manuscript materials from distributed numerical outputs.
3. Authorized full refitting reproduces the analyses from versioned curated participant snapshots. It cannot be performed from aggregate means and does not recreate undocumented spreadsheet adjudication.

`analyse` begins with a SHA-256 preflight and stops when private files are missing or changed. It creates participant-level outputs locally. Keep these restricted. See [DATA_ACCESS.md](../DATA_ACCESS.md).

The live figure catalogue is the authority for current figure identifiers and source dependencies. `SHA256SUMS` in the publication package records distributed file contents. PDF bytes may differ with TeX, font and platform versions even when numerical results agree. Review numerical equality separately from rendering equality.

