"""Laboratory and exploratory baseline itch-prediction tables."""
from pathlib import Path

import pandas as pd


def itch_prediction(root):
    folder = root/'results/tables/update_20261112/itch_prediction'
    data = pd.read_csv(folder/'metrics.csv')
    differences = pd.read_csv(folder/'paired_differences.csv')
    labels = {'three_markers':'Three baseline markers',
              'clinical':'Baseline clinical information',
              'clinical_three_markers':'Clinical + three markers'}
    text = r'''\begin{table}[htbp]
\centering\small
\caption{Exploratory baseline prediction of week-12 itch improvement.}\label{tab:itch_prediction}
\begin{tabular}{p{51mm}p{43mm}rr}\toprule
Inputs & AUROC (95\% CI) & AUPRC & Brier score \\ \midrule
'''
    for evaluation,title in [('mean_repetition','Mean repetition-specific performance')]:
        for population,pop_label in [('complete_three_markers','Complete markers: 102 patients, 88 responders'),
                                     ('available_serum_imputation_sensitivity','Predictor-imputation sensitivity: 105 patients, 89 responders')]:
            text += r'\multicolumn{4}{l}{\textbf{' + pop_label + '}}' + r' \\' + '\n'
            z = data[data.population.eq(population) & data.evaluation.eq(evaluation)]
            for model,label in labels.items():
                m = z[z.model.eq(model)].set_index('metric')
                a = m.loc['AUROC']
                text += (label + f' & {a.estimate:.3f} ({a.ci_low:.3f}--{a.ci_high:.3f})'
                         + f' & {m.loc["AUPRC","estimate"]:.3f} & {m.loc["Brier","estimate"]:.4f}'
                         + r' \\' + '\n')
            text += r'\addlinespace' + '\n'
    d = differences[differences.population.eq('complete_three_markers')
                    & differences.evaluation.eq('mean_repetition')
                    & differences.model.eq('clinical_three_markers')].set_index('metric')
    auc,brier = d.loc['AUROC'],d.loc['Brier']
    text += r'''\bottomrule\end{tabular}
\par\small\raggedright Response required measured baseline PP-NRS at least 4 and a measured decrease of at least 4 points by W12. The three markers are CCL17, CCL26 and IL-5. These newly fitted prognostic logistic models are separate from E-EASI. All models use identical patients and splits within each population. This table scores each outer repetition before averaging. Metrics of averaged patient-held-out probabilities, matching the historical aggregation, are reported separately in the source tables. Conditional 95\% intervals use 2,000 paired patient resamples without refitting. The sensitivity includes three CCL26 results below the reporting limit, using training-fold median predictor imputation; response labels are never imputed. Response prevalence was 86.3\% and 84.8\%, respectively, providing the no-information precision reference for AUPRC. Training-fold event-rate prediction gave mean Brier scores of 0.1186 and 0.1294. Clinical inputs, preprocessing and nested model selection are specified in the Supplementary Methods.

'''
    text += (f'In the complete-marker analysis, the combined-minus-clinical mean-repetition AUROC difference was '
             f'{auc.difference:.3f} (95\\% CI, {auc.ci_low:.3f} to {auc.ci_high:.3f}); the Brier difference was '
             f'{brier.difference:.5f} ({brier.ci_low:.5f} to {brier.ci_high:.5f}). '
             'The data did not establish molecular added predictive value. Only 14 complete-marker patients were nonresponders. '
             'This is a post hoc internal exploratory analysis. Complete metrics, calibration and repetition-specific results '
             'are in the distributed itch-prediction source tables.\n')
    text += r'\end{table}'+'\n'
    (root/'manuscript/tables/itch_prediction.tex').write_text(text)


def run(root):
    root = Path(root)
    folder = root/'results/tables/update_20261112/labs'
    changes = pd.read_csv(folder/'longitudinal_changes.csv')
    coverage = pd.read_csv(folder/'coverage_by_centre.csv')
    labels = {'dupilumab': 'dupi', 'abrocitinib_100mg': 'abro 100 mg',
              'abrocitinib_200mg': 'abro 200 mg'}
    text = r'''\begin{table}[htbp]
\centering\small
\caption{All-available IgE coverage and baseline-relative adjusted changes.}\label{tab:ige_all_available}
\begin{tabular}{p{19mm}p{25mm}rrrp{31mm}p{31mm}}\toprule
Scope & Initial treatment & W0 n & W12 n & W24 n & W12/W0 ratio (95\% CI) & W24/W0 ratio (95\% CI) \\ \midrule
'''
    for scope, model, scope_label in [('all_163', 'shared_centre', 'Both centres'),
                                       ('CAMS', 'single_centre', 'CAMS')]:
        for arm, label in labels.items():
            z = changes[changes.outcome.eq('IgE') & changes.scope.eq(scope)
                        & changes.model.eq(model) & changes.treatment.eq(arm)].set_index('week')
            c = coverage[coverage.outcome.eq('IgE') & coverage.treatment.eq(arm)]
            if scope == 'CAMS':
                c = c[c.centre.eq('CAMS')]
            n = c.groupby('week').n_observed.sum()
            values = [f'{z.loc[w,"ratio"]:.2f} ({z.loc[w,"ratio_low"]:.2f}--{z.loc[w,"ratio_high"]:.2f})'
                      for w in [12, 24]]
            text += f'{scope_label} & {label} & {int(n.loc[0])} & {int(n.loc[12])} & {int(n.loc[24])} & ' + ' & '.join(values) + r' \\' + '\n'
        if scope == 'all_163':
            text += r'\addlinespace' + '\n'
    text += r'''\bottomrule\end{tabular}
\par\small\raggedright Counts are measured IgE values, not interpolations. The full-cohort model uses 258 measurements from 147 patients; CAMS uses 229 measurements from 125 patients. A patient need not have all three visits. Adjusted ratios compare geometric mean (IgE+1) at follow-up with the same treatment group's modelled W0 value. Maximum-likelihood random-intercept models include initial-treatment-by-visit effects, log(1+baseline EASI), age, sex, BMI and centre where variable. Treatment-specific baseline IgE means are unconstrained. Intervals are model-based 95\% confidence intervals. The pooled model assumes centre differences can be shared across treatments because Huashan has no dupi follow-up IgE; CAMS-only and centre-by-visit models assess that dependence. There are no observed W4 IgE values; illustrative W4 interpolation does not enter this table or any treatment test. The abro 100-versus-200 mg ratio of baseline-relative change at W24 was 0.81 (0.39--1.69; Holm-adjusted P=1.00), without a clear dose difference. Complete pairwise estimates and correction families are in \texttt{between\_treatment\_changes.csv}; coverage and fitted trajectories are in \texttt{coverage\_by\_centre.csv}, \texttt{longitudinal\_levels.csv} and \texttt{longitudinal\_changes.csv}. Earlier multiple-imputation sensitivity remains in Table S8.
\end{table}
'''
    (root/'manuscript/tables/ige_all_available.tex').write_text(text)
    itch_prediction(root)
