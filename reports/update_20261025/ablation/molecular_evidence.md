# Molecular interpretation separate from EASI accuracy

The current calculation is BSA multiplied by a fitted molecular factor. It is a hybrid readout; the factor is not itself a validated inflammation scale. The three markers were fixed after earlier exploratory work, not chosen prospectively from the following papers.

## Evidence already present in this cohort

The full 23-marker analysis retains its original multiple-testing family. Over W0–W12, CCL17 declined in dupi (mean log2 change −1.664; q=0.00107) and abro 200 mg (−1.224; q=0.00176), while the abro 100 mg estimate was −0.372 (q=0.242). CCL26 declined in dupi (−0.195; q=0.00790), but its other two treatment estimates did not meet the original FDR threshold. IL-5 changes were negative in all three groups but none met that panel-wide FDR threshold (q=0.394, 0.0871 and 0.0926).

Among the complete molecular–clinical change family, IL-5 change correlated with itch change in abro 100 mg over W0–W12 (31 pairs, Spearman 0.742, q=0.00156). This is an exploratory treatment-specific association, not a validated universal itch calculator or proof that adding IL-5 improves EASI prediction.

Source tables: `results/tables/update_20261017/clinical/molecular_paired_changes.csv` and `molecular_clinical_change_correlations.csv`. All markers and nonsignificant results remain available; this note extracts the already fixed three-marker panel, not a newly selected positive panel.

## Primary literature and the limits of its support

1. [Lebrikizumab Rapidly Lowers Inflammatory Biomarkers with Clinical Correlations in Moderate-to-Severe Atopic Dermatitis](https://pubmed.ncbi.nlm.nih.gov/40663228/). Trial biomarker analyses reported baseline EASI/IGA associations with IL-5 and CCL26 and treatment-related changes in several type-2 markers, including CCL17. This supports biological plausibility and pharmacodynamic measurement. It does not validate our coefficients, our three-marker choice or transfer across treatments.

2. [Effect of abrocitinib on skin biomarkers in patients with moderate-to-severe atopic dermatitis](https://onlinelibrary.wiley.com/doi/10.1111/all.15969). Skin CCL17 changes correlated with EASI change, whereas CCL26 did not correlate with the studied clinical efficacy measures. Tissue gene expression and serum protein concentrations are different measurements. The study illustrates that a molecular response need not reproduce every clinical outcome.

3. [Interleukin-22 and Interleukin-18 as Potential Blood Biomarkers in Dupilumab-Treated Atopic Dermatitis](https://pmc.ncbi.nlm.nih.gov/articles/PMC13342774/). The longitudinal analysis found that biomarker–severity relationships varied during dupilumab treatment; CCL17's association was mainly early, while IL-22/IL-18 were more consistent. This argues against assuming that one type-2 panel is universally calibrated to clinical severity over time. It does not establish that expanding our panel would improve held-out prediction.

4. [Dupilumab reduces inflammatory biomarkers in pediatric patients with moderate-to-severe atopic dermatitis](https://pubmed.ncbi.nlm.nih.gov/39178993/). Randomized-trial biomarker analyses support treatment-associated reductions in CCL17 and IgE. The reported analysis used last-observation-carried-forward and rescue-treatment censoring; that missing-data procedure should not be copied as a justification for the present observational data.

These sources support measuring treatment-responsive molecular information alongside clinical extent. They do not repair a negative or uncertain ablation result. The defensible manuscript distinction is **molecular treatment response** versus **molecular added predictive accuracy**.
