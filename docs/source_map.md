# Source-to-result map

The current figure inventory is [manuscript/figure_catalog.json](../manuscript/figure_catalog.json). Each entry contains a figure identifier, title, caption and source paths. It is generated with the figures, so supplementary renumbering does not require a separate hand-maintained count.

| Analysis | Implementation | Numerical outputs |
|---|---|---|
| Clinical outcomes, operational targets, adjusted initial-treatment comparisons and molecular changes | `src/cytot2t/curation/clinical_evidence_20261017.py` | `results/tables/update_20261017/clinical/` |
| Regional EASI and whole-body itch | `src/cytot2t/curation/regional_evidence_20261017.py` | `results/tables/update_20261017/regional/` |
| Common EASI model, nested validation and controls | `src/cytot2t/curation/unified_model_20261017.py` | `results/tables/update_20261017/model/` |
| Display precision and threshold sensitivity | `src/cytot2t/curation/rounding_20261017.py` | `results/tables/update_20261017/model/rounding/` |
| Published p-EASI clipping sensitivity | `src/cytot2t/curation/published_clipping_sensitivity_20261017.py` | `model/published_clipping_*.csv` under the current result directory |
| Independently fitted PP-NRS, adult DLQI and recorded IGA | `src/cytot2t/curation/auxiliary_targets_20261017.py` | `results/tables/update_20261017/model/auxiliary_targets/` |
| Cohort flow, analysis-specific balance, paired monitoring and uncertainty | `src/t2t_paper/redesign_evidence_20261018.py` and its imported modules | `results/tables/update_20261017/redesign_20261018/` |
| Composite definitions, endpoint-specific balance, missing-outcome bounds, baseline-itch sensitivity and conditional prediction diagnostics | `src/t2t_paper/revision_checks_20261018.py` | `results/tables/update_20261017/revision2_20261018/` |
| Training-only exclusion of reference-inconsistent development visits | `src/cytot2t/curation/reference_sensitivity_20261018.py` | `results/tables/update_20261017/reference_sensitivity/` |
| Figure rendering | `src/t2t_paper/figures.py` and current rendering modules | `manuscript/figures/`, `manuscript/source_data/`, figure catalogue |
| Tables, narrative assembly and TeX compilation | `src/t2t_paper/build.py` | `manuscript/tables/`, main, appendix and figure-book files |

All paths abbreviated as `model/...` above are under `results/tables/update_20261017/`. The date in a directory identifies the analysis series; it does not imply that subsequent presentation revisions were analysed independently.

## Reading numerical tables

CSV headers give the analysis grouping and measure. The methods and figure captions define the estimand, uncertainty and multiplicity family. Do not substitute a display bin or group mean for a participant-level record. Blank or non-estimable cells are not zero effects. Within-treatment molecular changes and between-treatment contrasts answer different questions; time windows can share patients.

The numerical archive retains BSA-only and molecular-only controls, alternative structures, auxiliary-target limitations, published-formula sensitivities, sparse cells and negative results even when a redundant display is removed. The appendix links to the complete machine-readable source-table catalogue in `manuscript/source_data/index.csv`. Figure-specific source copies complement the full result tables; they do not replace them.

## Provenance and verification

The parameter/configuration files define the frozen formulas, modelling choices and input snapshot hashes. Necessary method notes and hash-based replay records are retained under `reports/update_20261017/`, with final aggregate verification summaries under `reports/update_20261018/revision2/`; public paths are sanitized where they contain a local home or temporary directory. Each replay record applies to its own recorded source hashes.

The generated publication package uses an explicit file allowlist, screens numerical tables for identifier columns and study IDs, and includes only figures listed in the current catalogue. Its packaging audit records selected and excluded aggregate files and verifies that copied numerical tables are byte-identical to their source. Internal discussion, author checklists, independent review reports, presentation source decks and raw participant inputs are not publication materials.

