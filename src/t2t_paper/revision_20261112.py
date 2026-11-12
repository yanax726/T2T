"""Available-record laboratory trajectories with explicit unmeasured-visit estimates."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import palette


def _legend(fig, base):
    handles = [Line2D([], [], color=palette.register(fig, 'treatment.' + arm),
                      marker='o', lw=1.35, ms=4, label=label)
               for arm, label in zip(base.ARMS, base.LABELS)]
    handles.append(Line2D([], [], color='#55616A', marker='o', markerfacecolor='white',
                          lw=0, ms=4, label='Interpolated W4'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .995),
               ncol=4, frameon=False, columnspacing=1.0, handlelength=1.5)


def _model_curve(ax, data, base, *, ratio=False):
    columns = ['ratio', 'ratio_low', 'ratio_high'] if ratio else ['estimate', 'ci_low', 'ci_high']
    for j, arm in enumerate(base.ARMS):
        z = data[data.treatment.eq(arm)].sort_values('week')
        color = palette.color('treatment.' + arm)
        x = z.week.to_numpy(float) + (j - 1) * .18
        value, low, high = (z[c].to_numpy(float) for c in columns)
        ax.plot(x, value, color=color, lw=1.35)
        for k, row in enumerate(z.itertuples()):
            hollow = row.kind == 'assumption_based_interpolation'
            ax.errorbar(x[k], value[k], yerr=[[max(0, value[k]-low[k])],
                                             [max(0, high[k]-value[k])]],
                        fmt='o', color=color, ms=4, lw=1.05, capsize=1.8,
                        markerfacecolor='white' if hollow else color,
                        markeredgewidth=1.05, zorder=4)
    ax.set(xticks=[0, 4, 12, 24], xlim=(-1, 25), xlabel='Week')
    ax.grid(axis='y', color='#E8EEF0', lw=.45, zorder=0)
    if ratio:
        ax.set_yscale('log')
        ax.set(ylim=(.2, 2), yticks=[.25, .5, 1, 2],
               yticklabels=['0.25', '0.5', '1', '2'])
        ax.axhline(1, color='#B2BCC2', lw=.65, zorder=0)
    ax.minorticks_off()


def _ige_scale(ax):
    ax.set_yscale('log')
    ax.set(ylim=(25, 6500), yticks=[50, 100, 500, 1000, 5000],
           yticklabels=['50', '100', '500', '1000', '5000'])
    ax.minorticks_off()


def laboratories(root, base, finish):
    root = Path(root)
    source = root/'results/tables/update_20261112/labs/longitudinal_levels.csv'
    levels = pd.read_csv(source)
    observed_path = root/'results/tables/update_20261017/clinical/clinical_observed.csv'
    observed = pd.read_csv(observed_path)
    fig, matrix = plt.subplots(2, 2, figsize=(183/25.4, 153/25.4))
    fig.subplots_adjust(left=.13, right=.97, top=.90, bottom=.10, wspace=.50, hspace=.44)
    axes = list(matrix.flat)
    _legend(fig, base)
    for row, (outcome, variable, ylabel, weeks) in enumerate([
        ('IgE', 'ige_numeric', 'Observed IgE (recorded units)', [0, 12, 24]),
        ('EOS', 'eo_count_numeric', 'Observed eosinophils\n(recorded units)', [0, 4, 12, 24]),
    ]):
        ax, model_ax = axes[row*2:row*2+2]
        for j, arm in enumerate(base.ARMS):
            z = observed[observed.target.eq(variable) & observed.treatment.eq(arm)
                         & observed.week.isin(weeks) & observed.n.gt(0)].sort_values('week')
            color = palette.color('treatment.' + arm)
            x = z.week.to_numpy(float) + (j-1)*.18
            ax.errorbar(x, z['median'], yerr=[z['median']-z.q25, z.q75-z['median']],
                        fmt='o-', color=color, lw=1.35, ms=4, capsize=1.8)
        ax.set(xticks=[0, 4, 12, 24], xlim=(-1, 25), xlabel='Week', ylabel=ylabel)
        ax.grid(axis='y', color='#E8EEF0', lw=.45)
        data = levels[levels.outcome.eq(outcome) & levels.scope.eq('all_163')
                      & levels.model.eq('shared_centre')]
        _model_curve(model_ax, data, base)
        model_ax.set_ylabel('Adjusted IgE (recorded units)' if row == 0
                            else 'Adjusted eosinophils\n(recorded units)')
        if row == 0:
            _ige_scale(ax)
            _ige_scale(model_ax)
        else:
            limit = max(float(observed[observed.target.eq(variable) & observed.week.isin(weeks)].q75.max()),
                        float(data.ci_high.max()))
            ax.set_ylim(0, np.ceil(limit*2)/2)
            model_ax.set_ylim(0, np.ceil(limit*2)/2)
    for i, ax in enumerate(axes):
        ax.annotate('abcd'[i], (0, 1), xycoords='axes fraction', xytext=(-16, 10),
                    textcoords='offset points', fontsize=9.5, weight='bold')
    finish(fig, 4, 'Observed laboratories and available-data adjusted trajectories',
           'a,c, Available-case medians and interquartile ranges at measured visits, using every eligible observed value; '
           'patients can differ between visits. Lines connect summaries and do not create measurements at unobserved visits. '
           'b,d, Maximum-likelihood random-intercept models use all available IgE or eosinophil measurements, including patients '
           'with only one recorded value. Initial treatment-by-categorical-visit terms allow treatment-specific W0 levels and '
           'changes; adjustment includes log(1+baseline EASI), age, sex, BMI and centre. Fixed-effect log(1+x) predictions are '
           'standardized over the initial cohort, then back-transformed; these are shifted geometric means, not observed medians. '
           'Bars in b,d are model-based 95% confidence intervals. Filled points are estimates at visits with measurements in the '
           'source population. Hollow points denote entirely unmeasured cells: every W4 IgE cell and dupi W4 eosinophils. '
           'They are separate temporal illustrations, obtained one-third of the way from modelled W0 to W12 on the log(1+x) scale. '
           'Their intervals propagate endpoint covariance conditional on that shape assumption; no W4 test or measurement is created. '
           'Such interpolation can miss an early transient eosinophil rise. Huashan has no follow-up dupi IgE, so pooled predictions '
           'assume transferable centre differences; CAMS-only and centre-by-visit sensitivities are provided separately. '
           'All denominators and contrasts remain in source tables. Initial treatment reflects observational management, '
           'including later regimen changes, and does not identify causal dose effects.',
           [observed_path, source, root/'results/tables/update_20261112/labs/coverage_by_centre.csv'],
           root, base)


def ige_sensitivity(root, base, finish):
    root = Path(root)
    folder = root/'results/tables/update_20261112/labs'
    levels = pd.read_csv(folder/'longitudinal_levels.csv')
    changes = pd.read_csv(folder/'longitudinal_changes.csv')
    fig, matrix = plt.subplots(2, 2, figsize=(183/25.4, 153/25.4))
    fig.subplots_adjust(left=.13, right=.97, top=.90, bottom=.10, wspace=.50, hspace=.44)
    axes = list(matrix.flat)
    _legend(fig, base)
    for row, (scope, model) in enumerate([('all_163', 'shared_centre'), ('CAMS', 'single_centre')]):
        for col, (data, ratio) in enumerate([(levels, False), (changes, True)]):
            ax = axes[row*2+col]
            d = data[data.outcome.eq('IgE') & data.scope.eq(scope) & data.model.eq(model)]
            _model_curve(ax, d, base, ratio=ratio)
            ax.set_ylabel('IgE: ratio to own W0\n(geometric mean of IgE + 1)' if ratio
                          else 'Adjusted IgE (recorded units)')
            if not ratio:
                _ige_scale(ax)
                ax.set(ylim=(35, 750), yticks=[50, 100, 250, 500],
                       yticklabels=['50', '100', '250', '500'])
            ax.annotate('abcd'[row*2+col], (0, 1), xycoords='axes fraction', xytext=(-16, 10),
                        textcoords='offset points', fontsize=9.5, weight='bold')
    finish(fig, 21, 'All-available IgE trajectories and centre sensitivity',
           'a,b, All-centre available-data model; c,d, CAMS-only sensitivity. a,c, Adjusted shifted geometric mean IgE; '
           'b,d, The modelled ratio of geometric mean (IgE+1) to the same treatment group’s own W0 value. A ratio below one '
           'indicates a lower follow-up level. The all-centre analysis uses 258 observed values from 147 patients; CAMS uses '
           '229 values from 125 patients. No complete-three-visit requirement is imposed; patients with only one measured visit '
           'can contribute. Models include initial treatment-by-categorical-visit effects, log(1+baseline EASI), age, sex and BMI, '
           'plus centre in a,b, with a patient random intercept. The baseline IgE response is modelled jointly, with free '
           'treatment-specific baseline means; this is not a follow-up ANCOVA that explicitly conditions on each observed baseline IgE. '
           'Predictions are standardized to the baseline population of the stated scope. Lines and 95% confidence intervals '
           'describe adjusted group trajectories, not individual prediction or causal drug effects. There are no W4 IgE measurements. '
           'Hollow W4 points are illustrative log-scale interpolations between modelled W0 and W12, with conditional intervals; '
           'no hypothesis test uses these points. Huashan has baseline and abro follow-up IgE but no dupi follow-up IgE. '
           'The all-centre model therefore assumes shared centre differences, examined using CAMS-only and centre-by-visit sensitivity. '
           'Two previous multiple-imputation methods, their differing assumptions and results remain in Table S8 and the complete source tables. '
           'Drug-curve ordering alone does not establish a dose effect. Exact coverage, model diagnostics, interpolation alternatives '
           'and baseline-relative between-treatment tests remain available in source data.',
           [folder/'longitudinal_levels.csv', folder/'longitudinal_changes.csv',
            folder/'between_treatment_changes.csv', folder/'coverage_by_centre.csv'], root, base)
