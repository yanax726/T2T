"""Supplementary tables for exploratory future-response prediction."""
from pathlib import Path

import pandas as pd


POPULATIONS = [
    ('complete_three_markers', 'Complete original panel: 102 patients, 88 responders'),
    ('available_serum_imputation_sensitivity', 'Predictor-imputation sensitivity: 105 patients, 89 responders'),
]


def select(data, **keys):
    for key, value in keys.items():
        data = data[data[key].eq(value)]
    if len(data) != 1:
        raise ValueError('Expected one aggregate row: ' + str(keys))
    return data.iloc[0]


def auc(data, **keys):
    row = select(data, metric='AUROC', **keys)
    return f'{row.estimate:.3f} ({row.ci_low:.3f}--{row.ci_high:.3f})'


def brier(data, **keys):
    return f'{select(data, metric="Brier", **keys).estimate:.4f}'


def panel_size(root):
    folder = root/'results/tables/update_20261112/itch_panel_size'
    data = pd.read_csv(folder/'metrics.csv')
    data = data[data.evaluation.eq('mean_repetition')]
    stress = pd.read_csv(folder/'batch_stress/aggregate_metrics.csv')
    text = r'''{\small\setlength{\tabcolsep}{4pt}
\begin{longtable}{p{34mm}p{42mm}p{42mm}rr}
\caption{Exploratory panel-size comparisons for baseline prediction of W12 itch improvement.}\label{tab:itch_panel_size}\\
\toprule Panel procedure & Serum AUROC (95\% CI) & Clinical + serum AUROC (95\% CI) & Serum Brier & Combined Brier \\ \midrule\endfirsthead
\toprule Panel procedure & Serum AUROC (95\% CI) & Clinical + serum AUROC (95\% CI) & Serum Brier & Combined Brier \\ \midrule\endhead
'''
    panels = [('Clinical reference', None, 'clinical'),
              ('Original fixed three', 'fixed_three_serum', 'clinical_fixed_three')]
    panels += [(f'Training-selected {k}', f'serum_ranked{k}', f'clinical_serum_ranked{k}')
               for k in [3, 4, 5, 6, 19]]
    panels += [('Inner-selected size 3--6', 'serum_compact', 'clinical_serum_compact')]

    def line(frame, population, label, serum, combined):
        def value(model, metric):
            if model is None:
                return '---'
            return (auc if metric == 'AUROC' else brier)(frame, population=population, model=model)
        return (label + ' & ' + ' & '.join([value(serum, 'AUROC'), value(combined, 'AUROC'),
                                           value(serum, 'Brier'), value(combined, 'Brier')])
                + r' \\' + '\n')

    for population, label in POPULATIONS:
        text += r'\multicolumn{5}{l}{\textbf{Patient-held-out validation: ' + label + '}}' + r' \\' + '\n'
        for name, serum, combined in panels:
            text += line(data, population, name, serum, combined)
        text += r'\addlinespace' + '\n'
    for population, label in POPULATIONS:
        text += r'\multicolumn{5}{l}{\textbf{Entire report-file holdout: ' + label + '}}' + r' \\' + '\n'
        for name, serum, combined in [
                ('Clinical reference', None, 'clinical'),
                ('Original fixed three', 'fixed_three_serum', 'clinical_fixed_three'),
                ('Inner-selected size 3--6', 'serum_compact', 'clinical_serum_compact')]:
            text += line(stress, population, name, serum, combined)
        text += r'\addlinespace' + '\n'
    text += r'''\bottomrule\end{longtable}}
\noindent\footnotesize Response is a measured PP-NRS decrease of at least four points from baseline to W12, among patients with baseline PP-NRS at least four. The clinical reference and original fixed CCL17/CCL26/IL-5 panel reproduce Table S18. Candidate panels use the 19 discovery-eligible markers; ranking, median imputation, scaling and penalty selection are fitted within training folds. Fixed-size rows evaluate selection procedures, not one fixed marker list. All rows within each population use the same patients, outcomes and outer partitions. AUROC measures discrimination; lower Brier score indicates lower squared probability error. Complete AUPRC, calibration, log-loss and paired comparisons are in the source tables.

The first two blocks average performance over five outer repetitions. Conditional 95\% intervals use 2,000 paired patient resamples without refitting. For the size-selected procedure, each inner validation chooses the smallest size from 3--6 within one standard error of the lowest inner log-loss; this is a simplicity heuristic, not a noninferiority test. The clinical-plus-serum procedure chose three markers in all 25 overlapping training folds in each population. Serum-only selection chose three in 24 folds and four in one. The primary three-marker procedure produced eight different marker sets; no fixed new panel was validated. Clinical inputs are specified in Table S18 and the Supplementary Methods.

The final two blocks hold out each of four original assay-report files in turn, repeat all selection using the remaining patients, and pool the held-out predictions. Intervals resample patients within held-out files, conditional on these four sources and fitted models. A file containing responders only has no within-file AUROC; its patients remain in the pooled metrics and Brier score. Report file is not a verified physical assay batch. Files differ in centre, treatment and response prevalence, so this is a joint source-and-case-mix stress test, not external validation or proof of a technical batch effect. These intervals are not interchangeable with the patient-held-out intervals. No simultaneous inferential claim is made across panel sizes.\normalsize
'''
    (root/'manuscript/tables/itch_panel_size.tex').write_text(text)


def landmark(root):
    folder = root/'results/tables/update_20261114/landmark_three'
    data = pd.read_csv(folder/'metrics.csv')
    differences = pd.read_csv(folder/'paired_differences.csv')
    null = pd.read_csv(folder/'null_reference.csv')
    labels = [('three_markers', 'W12 three markers'), ('clinical', 'Clinical information through W12'),
              ('clinical_three_markers', 'Clinical information + W12 markers')]
    text = r'''\begin{table}[htbp]
\centering\small
\caption{Exploratory prediction of W24 EASI75 from W12 measurements.}\label{tab:landmark_three}
\begin{tabular}{p{61mm}p{43mm}rr}\toprule
Inputs & AUROC (95\% CI) & AUPRC & Brier score \\ \midrule
'''
    notes = []
    for population, _ in POPULATIONS:
        z = data[data.population.eq(population) & data.evaluation.eq('mean_repetition')]
        n, events, non = [int(z.iloc[0][name]) for name in ['n', 'events', 'nonevents']]
        label = 'Complete W12 markers' if population == 'complete_three_markers' else 'Predictor-imputation sensitivity'
        text += (r'\multicolumn{4}{l}{\textbf{' + f'{label}: {n} patients, {events} responders, {non} nonresponders'
                 + '}}' + r' \\' + '\n')
        for model, label in labels:
            ap = select(z, model=model, metric='AUPRC').estimate
            text += (label + ' & ' + auc(z, model=model) + f' & {ap:.3f} & '
                     + brier(z, model=model) + r' \\' + '\n')
        text += r'\addlinespace' + '\n'
        d = select(differences, population=population, evaluation='mean_repetition',
                   model='clinical_three_markers', metric='AUROC')
        p = null[null.population.eq(population)]
        notes.append(f'{n}-patient analysis: response prevalence {events/n:.3f}, mean training-fold-event-rate Brier '
                     f'{p.Brier.mean():.4f}, combined-minus-clinical AUROC difference {d.difference:.3f} '
                     f'(95\\% CI, {d.ci_low:.3f} to {d.ci_high:.3f}).')
    text += r'''\bottomrule\end{tabular}
\par\small\raggedright W24 EASI75 means a measured EASI reduction of at least 75\% from the patient's pretreatment W0 score. Eligibility requires observed positive W0 EASI, observed W24 EASI and a W12 visit; it does not require EASI75 at W12. The response therefore includes both maintained and newly achieved target states. The three markers are the measured W12 concentrations of CCL17, CCL26 and IL-5, rather than W0--W12 differences. The complete-marker and sensitivity populations use the same current age-at-least-12 eligibility rule as the main study. No W24 outcome is imputed.

Three prespecified logistic models use W12 markers alone, clinical information available through W12, or their combination. Clinical inputs comprise pretreatment EASI, age, sex, BMI, centre, initial treatment and W12 EASI, PP-NRS, IGA, BSA and adult DLQI; adolescent DLQI is masked. Initial treatment does not encode subsequent regimen changes. Five outer patient folds are repeated five times, with three inner folds selecting the ridge penalty by log-loss. All preprocessing is fitted within training folds. Metrics are scored within each repetition before averaging. Conditional 95\% intervals use 2,000 paired patient resamples of saved held-out predictions without refitting; they do not account for the prior exploration of this cohort. AUROC concerns W24 target discrimination, not continuous EASI error or validation of the contemporaneous E-EASI formula. AUPRC must be read against the high response prevalence.

'''
    text += ' '.join(notes) + '\n'
    text += r'\end{table}' + '\n'
    (root/'manuscript/tables/landmark_three.tex').write_text(text)


def run(root):
    root = Path(root)
    panel_size(root)
    landmark(root)
