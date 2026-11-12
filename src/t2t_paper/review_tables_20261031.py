"""Laboratory coverage, endpoint definitions and classification uncertainty."""
import pandas as pd
from .sensitivity_tables_20261025 import start, end

ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
LABELS = dict(zip(ARMS, ['dupi', 'abro 100 mg', 'abro 200 mg']))


def tables(root, tex):
    out = root/'manuscript/tables'
    folder = root/'results/tables/update_20261031'
    coverage = pd.read_csv(folder/'laboratory_coverage.csv')
    paired = pd.read_csv(folder/'laboratory_paired_changes.csv')
    s = start('Laboratory coverage and observed paired eosinophil changes',
              'tab:laboratory_support', 'Measure & Initial group & W0 & W4 & W12 & W24',
              r'llrrrr')
    for target, name in [('ige_numeric', 'IgE'), ('eo_count_numeric', 'Eosinophils')]:
        for arm in ARMS:
            d = coverage[coverage.target.eq(target) & coverage.scope.eq('all') & coverage.treatment.eq(arm)].set_index('week')
            cells = [f'{int(d.loc[w,"n_observed"])}/{int(d.loc[w,"n_initial"])}' for w in [0,4,12,24]]
            s += ' & '.join([name, LABELS[arm], *cells]) + r' \\' + '\n'
    s += end()
    s += (r'\noindent Observed/initial patients. Zero denotes no recorded measurement, not a laboratory value of zero. '
          r'Patients may differ between visits. All values remain in their recorded units. Counts describe observed measurements only. Hollow W4 points in Figures S4--S5 are conditional temporal interpolations; they are not included in these counts or tests.'+'\n\n')
    s += (r'{\small\begin{tabular}{llrr}\toprule Interval & Initial group & Pairs & '
          r'Mean eosinophil change (95\% CI) \\ \midrule'+'\n')
    for a,b in [(0,12),(0,24),(12,24)]:
        for arm in ARMS:
            r = paired[paired.target.eq('eo_count_numeric') & paired.scope.eq('all') & paired.treatment.eq(arm)
                       & paired.start.eq(a) & paired.end.eq(b)].iloc[0]
            s += (f'W{a}--W{b} & {LABELS[arm]} & {int(r.n)} & '
                  f'${r.mean_change:.3f}$ (${r.ci_low:.3f}$, ${r.ci_high:.3f}$)' + r' \\' + '\n')
    s += (r'\bottomrule\end{tabular}}'+'\n\n'+
          r'Later minus earlier measurements in all available pairs, without requiring the third visit. '
          r'Descriptive t-based intervals are not multiplicity-adjusted treatment-comparison tests. '
          r'Baseline-adjusted group contrasts and centre-specific coverage remain in the aggregate sources. '
          r'No follow-up eosinophil measurements were recorded for Huashan dupi patients; an overall observed change cannot represent that unobserved cell.'+'\n')
    (out/'laboratory_support.tex').write_text(s)

    d = pd.read_csv(folder/'iga_definition_paired_comparison.csv')
    d = d[d.cohort.eq('all_eligible') & d.domain.eq('combined') & d.treatment.isin(ARMS)]
    s = start('Combined-target sensitivity without an IGA component', 'tab:iga_definition',
              r'Week & Initial group & Paired N & Recorded & No IGA & Changed', r'llrrrr')
    for week in [4,12,24]:
        for arm in ARMS:
            r = d[d.week.eq(week) & d.treatment.eq(arm)].iloc[0]
            n = int(r.n_both_evaluable)
            s += (f'W{week} & {LABELS[arm]} & {n} & {int(r.recorded_success_paired)}/{n} & '
                  f'{int(r.iga_free_success_paired)}/{n} & {int(r.n_classification_changed_both_evaluable)}'+r' \\'+'\n')
    s += end()
    s += (r'Both definitions are evaluated in the same patients within each row. Recorded denotes the reconstructed framework, '
          r'not a verified consensus endpoint. No IGA removes only the IGA alternative from the physician domain, retaining EASI50 '
          r'at W4, EASI75 at W12 and EASI75 or EASI$\leq5$ at W24; the patient domain is unchanged. '
          r'This is a different, exploratory target definition, not a recoding of IGA.'+'\n\n')
    rates = pd.read_csv(folder/'iga_definition_rates.csv')
    rates = rates[rates.cohort.eq('all_eligible') & rates.domain.eq('combined') & rates.treatment.isin(ARMS)]
    s += (r'{\small\begin{tabular}{llrr}\toprule Week & Initial group & Recorded, all evaluable & No IGA, all evaluable \\ \midrule'+'\n')
    for week in [4,12,24]:
        for arm in ARMS:
            q = rates[rates.week.eq(week) & rates.treatment.eq(arm)].set_index('definition')
            cells = [f'{int(q.loc[k,"n_success"])}/{int(q.loc[k,"n_evaluable"])}' for k in ['recorded_framework','iga_free_sensitivity']]
            s += ' & '.join([f'W{week}',LABELS[arm],*cells])+r' \\'+'\n'
    s += (r'\bottomrule\end{tabular}}'+'\n\n'+
          r'The second block uses each definition\textquotesingle s own evaluable population; a change in denominator is not a change in the same patient\textquotesingle s response. '
          r'Unknown components follow three-valued AND/OR rules and are never filled with clinical scores. '
          r'Full transition counts, physician-domain results and exclusion of the 22 patients with recorded baseline IGA5 are supplied in the source tables. '
          r'The protocol inclusion text specifies 0--4 but its IGA appendix defines grades 0--5, including grade 5 as very severe. '
          r'All original grades are therefore retained; actual implementation requires confirmation.'+'\n')
    (out/'iga_definition_support.tex').write_text(s)

    d = pd.read_csv(root/'results/tables/update_20261017/redesign_20261018/monitoring_response_per_repeat_uncertainty.csv')
    s = start('Response classification by validation repetition', 'tab:classification_intervals',
              r'Visit / target & Repeat & Sensitivity & Specificity & Agreement', r'llrrr')
    for (window,endpoint,rep), group in d.groupby(['window','endpoint','repeat'],sort=True):
        cells = []
        for metric in ['sensitivity','specificity','agreement']:
            r = group.set_index('metric').loc[metric]
            cells.append(r'\shortstack{'+f'{int(r.successes)}/{int(r.denominator)}'+r'\\'+
                         f'{100*r.estimate:.1f} ({100*r.wilson_ci_low:.1f}, {100*r.wilson_ci_high:.1f})'+r'}')
        s += ' & '.join([window.replace('W0_','')+' / '+endpoint,str(rep+1),*cells])+r' \\[4pt]'+'\n'
    s += end()
    s += (r'Each cell gives correctly classified patients/eligible patients and percentage (Wilson 95\% interval). '
          r'The repetitions contain the same patients and are not pooled as independent observations. '
          r'The main figure averages repetition-specific percentages. Its conservative bounds average limits of '
          r'two-sided 97.5\% Clopper--Pearson intervals, using Bonferroni across the two repetitions within each target, visit and metric. '
          r'These bounds assume independent Bernoulli patient outcomes within each repetition, permit dependence between repetitions, '
          r'and condition on saved predictions. They do not propagate shared training fits or model-development uncertainty and do not '
          r'provide simultaneous coverage across all targets and metrics. W24 remains same-cohort temporal application.'+'\n')
    (out/'classification_intervals.tex').write_text(s)
