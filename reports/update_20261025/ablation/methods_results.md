# Input ablation: matched comparisons and interpretation

This is a post-result exploratory extension. The frozen primary equation and its published-in-project results were not replaced. The analysis was run with `PYTHONPATH=src .venv/bin/python -m cytot2t.curation.model_ablation_20261025`; `run.log` and `verification.json` record the execution.

## What was compared

Three serum inputs always mean log2 CCL17, log2 CCL26 and log2 IL-5. BSA, current PP-NRS and current adult DLQI are supplied as recorded clinical values. Neither treatment, time, baseline history nor EASI components other than the explicitly declared BSA input enter these models.

| Comparison population | W0 | W12 | W24 | Development |
|---|---:|---:|---:|---|
| Original complete core | 152 | 89 | 75 | 241 visits, 156 patients |
| Current itch available | 151 | 89 | 75 | 240 visits, 156 patients |
| Adults with current itch and DLQI | 105 | 58 | 44 | 163 visits, 107 patients |

The populations contain 160, 160 and 111 unique patients, respectively. The missing baseline itch observation was excluded only from the itch comparison, without removing that patient's other eligible visits. Adult DLQI requires age ≥18. No clinical outcome or deployment input was imputed.

All models within a population were trained and evaluated on the same visits. Outer folds retain the original assignment of all 160 core patients, with five folds and two repetitions; all visits of a patient remain in the same fold. Models fit W0/W12 only. Three-fold inner patient validation selects the ridge penalty and, where BSA is available, linear, BSA-quadratic or BSA-times-other-input structures. Serum-only and itch-plus-serum use linear ridge. Each development patient's fitting weights sum to the same total. Predictions are restricted to 0–72. W24 is previously inspected same-cohort temporal application, not untouched external validation.

The input ladder compares complete modelling procedures, not identical algebra with one coefficient erased. Two additional controls therefore hold the area structure fixed: BSA alone versus BSA times the three-marker component, and the latter versus removal of each single marker. The CCL17-only and BSA-plus-CCL17 comparators test whether the complete panel beats a single familiar marker on the same core population.

Metrics are scored within each repetition and then averaged. For differences, the same patients are resampled jointly across models and repetitions in 2,000 bootstrap draws. Intervals are conditional on saved held-out predictions and do not include refitting uncertainty. They are exploratory, unadjusted intervals across several comparisons. Paired score changes are calculated within patient and repetition before subtraction, with no averaging of repeat predictions before absolute errors.

## Main matched input comparison

| Inputs | W0 MAE | W12 MAE | W24 MAE |
|---|---:|---:|---:|
| Three serum markers | 7.67 | 7.48 | 8.56 |
| PP-NRS + three markers | 7.14 | 4.54 | 4.05 |
| BSA alone | 3.16 | 1.21 | 1.05 |
| BSA + three markers | 3.20 | 1.29 | 1.19 |
| BSA + PP-NRS | 3.06 | 1.10 | 0.96 |
| BSA + three markers + PP-NRS | 3.08 | 1.17 | 1.04 |

Here every row uses 151/89/75 observations, respectively. MAE is mean absolute EASI error. Full uncertainty intervals, Spearman correlation, concordance correlation, RMSE, signed error, calibration and within-one/two-point proportions are in `main_input_comparison.csv` and `overall_metrics.csv`.

Adding BSA to three serum markers reduced MAE by 4.46/6.19/7.38 points at W0/W12/W24. Adding PP-NRS to the BSA–three-marker procedure reduced follow-up MAE by 0.127 points at W12 (95% CI −0.236 to −0.011) and 0.145 at W24 (−0.246 to −0.047); baseline reduction was 0.125 (−0.277 to 0.026). These are modest symptom-input gains. BSA plus itch without serum still had lower point-estimate error. The results therefore do not establish added EASI accuracy from the molecular panel.

## Single-marker and fixed-structure checks

On the original 152/89/75 core observations, three serum markers did not lower MAE relative to CCL17 alone. Adding CCL26 and IL-5 to BSA plus CCL17 also did not establish an advantage: the full-minus-single-marker MAE differences were +0.032, +0.050 and +0.042 points, with all three intervals spanning zero.

When the area interaction structure was held fixed, BSA plus three markers gave W12/W24 MAEs of 1.146/1.007 versus 1.178/1.022 with BSA alone. The paired changes were −0.031 (95% CI −0.086 to 0.032) and −0.015 (−0.044 to 0.014). This small uncertain difference cannot be called an established molecular benefit. Baseline error was greater with the molecular terms (3.255 versus 3.123).

Removing CCL17 from the fixed area structure increased W12 MAE by about 0.062 points, but this was not consistent across all visits. CCL26 and IL-5 were not consistently beneficial when tested separately. Selection and coefficient stability are reported by structure; coefficients of different structural forms are not averaged together. The negative IL-5 coefficient is a conditional prediction coefficient and must not be described as a protective biological effect.

## Adult DLQI extension

On the separate 105/58/44 adult observations, BSA plus three markers had MAE 2.823/1.339/1.218. Adding itch and DLQI gave 2.834/1.132/0.924. The corresponding no-serum BSA-plus-itch-plus-DLQI model gave 2.728/1.126/0.996. Thus, the serum panel reduced W24 error by only 0.072 points (95% CI −0.124 to −0.021) within this 44-person adult subset, while increasing baseline error by 0.107 (0.035 to 0.186); the W12 difference was +0.006 (−0.035 to 0.047).

This isolated small W24 gain is an exploratory subset finding, not a robust all-visit molecular advantage. The 0.924-point value must not be compared with a full-cohort value from 75 W24 patients. More inputs reduce practicality and retain clinical subjectivity. These models are not nominated as a replacement frozen calculator.

## Suggested manuscript wording

Methods: We compared the three serum markers alone, BSA alone, BSA plus serum markers, PP-NRS plus serum markers, and BSA with serum markers and PP-NRS using matched complete-input visits and identical patient-level outer folds. All structures and penalties were selected within W0/W12 training folds. BSA-plus-PP-NRS controls isolated molecular inclusion. A separate adult complete-input analysis evaluated DLQI; its comparators were refitted within the same adult subset. Fixed-area and single-marker analyses assessed the dependence of apparent component effects on structural selection. Differences and confidence intervals used paired patient resampling of the held-out predictions.

Results: Lesion extent supplied most of the accuracy gained over the three-marker-only estimate. On the common 151/89/75-visit comparison, adding BSA reduced MAE from 7.67/7.48/8.56 to 3.20/1.29/1.19 points. Current itch further reduced follow-up MAE to 1.17 and 1.04 points. Molecular inclusion did not show a consistent accuracy advantage over matched clinical-input controls; fixed-area differences were small and uncertain. Adult DLQI extensions were evaluated separately because complete-input coverage was lower.

## What the evidence supports

The score can be described as a clinical–serum EASI estimate. The blood measurements are objective assay inputs, but the combined score is not wholly observer-independent because BSA supplies substantial clinical information. Molecular treatment responsiveness and the incremental accuracy of molecular inputs are separate questions. The current data can support the former for some markers and comparisons without establishing the latter for all three markers.

The study should not claim that three markers are necessary, optimal, or better than CCL17 on the evidence above. The panel arose from earlier exploration. There is also no evidence here that disagreement with EASI means the model is more correct than the clinician. A new benefit claim would need a prospectively chosen endpoint independent of the EASI training reference, such as assay reproducibility or independently adjudicated residual inflammation; those data are not supplied by this ablation.
