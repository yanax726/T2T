# Current independent-target results

These are three separate models sharing four measurements, not three outputs copied from the EASI formula. All results below use new patient-held-out fits on current clinical records.

|Target|W0 n|W0 result|W12 n|W12 result|W24 n|W24 result|
|---|---:|---:|---:|---:|---:|---:|
|PP-NRS, MAE in points|151|2.350|89|2.776|75|2.533|
|DLQI, MAE in points; age >=18|105|6.255|58|5.904|44|6.281|
|Recorded IGA, normalized RPS|152|0.0711|89|0.0988|75|0.1245|
|Recorded IGA, exact category agreement|152|54.3%|89|35.4%|75|28.0%|

EASI's relatively low follow-up error does not extend to these three outcomes. Follow-up PP-NRS and DLQI were systematically overestimated under a common formula without treatment-phase input. For example, mean signed errors at W12 were +2.40 PP-NRS points and +5.13 DLQI points. Their molecular-only references also performed poorly. This is consistent with the measured inputs containing less information about subjective itch and quality of life than about lesion extent, but it does not establish a biological mechanism.

Adding three molecules to BSA did not show a stable improvement for PP-NRS or DLQI: every week-specific conditional interval for the paired MAE difference included zero. For recorded IGA, the three-molecule model had worse ranked probability scores than BSA alone at W12 and W24, by 0.00539 (95% conditional interval 0.00097–0.01003) and 0.01004 (0.00479–0.01534), respectively. These exploratory findings should be retained in the supplement and should not be presented as successful substitutes for all four clinical targets.

The differences from historical output are intentional and scientifically important. The historical models used different panels and explicitly separated baseline from follow-up treatment groups. The current models use the same small panel as the main EASI work and have no phase or drug input. DLQI now restricts all fitting and validation to age >=18; the older adult scope included age 17. These new numbers cannot be replaced by the smaller errors from the historical analysis.

Files: overall_metrics.csv contains the four model strategies at every visit; metrics.csv includes treatment and center strata; paired_comparisons.csv contains matched reference comparisons; formula_pp_nrs.json, formula_dlqi.json and formula_iga.json hold the independent full-development coefficients. W24 is same-cohort temporal transfer, not external validation.

Run: `PYTHONPATH=src python -m cytot2t.curation.auxiliary_targets_20261017 --root /path/to/project`.
Calculate a synthetic example: `PYTHONPATH=src python -m cytot2t.curation.auxiliary_targets_calculator_20261017 examples/auxiliary_targets_input.csv output.csv --target pp_nrs --input-scale raw --root /path/to/project`.

Keep predictions_private.csv, input_private.csv and fit_audit_private.csv out of the public GitHub package. The reproduction directory also contains private inputs and must remain local.
