"""Tables accompanying the all-available laboratory display."""
from pathlib import Path

import pandas as pd


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
