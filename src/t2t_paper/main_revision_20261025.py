"""Main displays organized around clinical targets, molecular response and measurement."""
from pathlib import Path
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import figures as base
from . import palette
from .main_redesign_20261018 import label, clean, legend

TARGETS = [
    ('EASI50', 'EASI50', [4, 12, 24]),
    ('EASI75', 'EASI75', [4, 12, 24]),
    ('EASI90', 'EASI90', [4, 12, 24]),
    ('EASI_le5', 'EASI ≤5', [0, 4, 12, 24]),
    ('EASI_le3', 'EASI ≤3', [0, 4, 12, 24]),
    ('IGA_drop2', 'IGA decrease ≥2', [4, 12, 24]),
    ('IGA_01', 'IGA 0/1', [0, 4, 12, 24]),
    ('PP_NRS_le4', 'PP-NRS ≤4', [0, 2, 4, 12, 24]),
    ('PP_NRS_01', 'PP-NRS 0/1', [0, 4, 12, 24]),
    ('DLQI_le10', 'Adult DLQI ≤10', [0, 4, 12, 24]),
    ('DLQI_le4', 'Adult DLQI ≤4', [0, 4, 12, 24]),
]
PATHS = [
    ('W12_missing', 'W12 outcome unknown'),
    ('W12_not_EASI75', 'W12 EASI75 not reached'),
    ('responder_no_recorded_reduction', 'Reached; no reduction'),
    ('responder_reduced_after_W12', 'Reached; later reduction'),
    ('responder_stopped_or_switched', 'Reached; stopped /\nswitched'),
    ('responder_exposure_unknown', 'Reached; exposure\nunknown'),
    ('responder_exposure_conflict', 'Reached; exposure\nconflict'),
]


def clinical_targets(root):
    folder = root / 'results/tables/update_20261017/clinical'
    rates = pd.read_csv(folder / 'endpoint_rates.csv')
    observed = pd.read_csv(folder / 'clinical_observed.csv')
    fig, matrix = plt.subplots(5, 3, figsize=(183 / 25.4, 235 / 25.4))
    fig.subplots_adjust(left=.09, right=.98, top=.93, bottom=.055, wspace=.40, hspace=.70)
    axes = list(matrix.flat)
    legend(fig, at=(.52, .989))
    weeks = [0, 12, 24]
    for ax, variable, title in zip(axes[:3], ['easi_total', 'pp_nrs', 'dlqi'],
                                    ['EASI', 'PP-NRS', 'Adult DLQI']):
        for j, (arm, color) in enumerate(zip(base.ARMS, base.COLORS)):
            d = observed[observed.target.eq(variable) & observed.treatment.eq(arm)].set_index('week').reindex(weeks)
            assert d[['median', 'q25', 'q75']].notna().all().all()
            x = np.array(weeks, float) + (j - 1) * .22
            ax.errorbar(x, d['median'],
                        yerr=np.vstack([d['median']-d.q25, d.q75-d['median']]),
                        fmt='o-', color=color, lw=1.35, elinewidth=.85, capsize=1.4, ms=3.4,
                        markeredgecolor=palette.role_style('treatment.' + arm)['outline'], markeredgewidth=.4)
        upper = {'easi_total': 35, 'pp_nrs': 11, 'dlqi': 30}[variable]
        ax.set(title=title, xticks=weeks, xlim=(-1, 25), ylim=(0, upper),
               xlabel='Week', ylabel='Score (points)')
    for ax, (target, title, weeks) in zip(axes[3:], TARGETS):
        for j, (arm, color) in enumerate(zip(base.ARMS, base.COLORS)):
            d = rates[rates.target.eq(target) & rates.treatment.eq(arm)].set_index('week').reindex(weeks)
            assert d.rate.notna().all()
            x = np.array(weeks, float) + (j - 1) * .22
            ax.errorbar(x, 100*d.rate,
                        yerr=np.maximum(0, np.vstack([100*(d.rate-d.ci_low), 100*(d.ci_high-d.rate)])),
                        fmt='o-', color=color, lw=1.35, elinewidth=.85, capsize=1.4, ms=3.4,
                        markeredgecolor=palette.role_style('treatment.' + arm)['outline'], markeredgewidth=.4)
        ax.set(title=title, xticks=weeks, xlim=(-1, 25), ylim=(0, 103),
               yticks=[0, 50, 100], xlabel='Week')
    ax = axes[-1]
    for j, (arm, color) in enumerate(zip(base.ARMS, base.COLORS)):
        d = rates[rates.target.eq('overall') & rates.treatment.eq(arm)].set_index('week').reindex([4, 12, 24])
        ax.errorbar(np.array([4, 12, 24]) + (j-1)*.45, 100*d.rate,
                    yerr=np.maximum(0, np.vstack([100*(d.rate-d.ci_low), 100*(d.ci_high-d.rate)])),
                    fmt='o', color=color, lw=1.1, ms=3.5, capsize=1.5)
    ax.set(title='Combined stage-specific target', xticks=[4, 12, 24],
           xlim=(-1, 25), ylim=(0, 103), yticks=[0, 50, 100], xlabel='Week')
    for i, ax in enumerate(axes):
        label(ax, string.ascii_lowercase[i]); clean(ax)
        ax.tick_params(labelsize=6.5)
        if i >= 3 and i % 3 == 0:
            ax.set_ylabel('Target attainment (%)')
    base.save(fig, 'Figure2', 'Observed clinical activity and target attainment by initial treatment',
              'a–c, Observed medians and interquartile ranges for EASI, PP-NRS and adult DLQI. '
              'd–n, Clinical threshold attainment, with pointwise Wilson 95% confidence intervals. '
              'Each visit uses its available observations, not a fixed patient set. Group positions are slightly offset for legibility. '
              'Improvement thresholds require observed baseline and follow-up; adult DLQI requires age ≥18. '
              'o, Combined physician-and-patient targets from Fig. 1, with Wilson intervals. These stage-specific criteria differ, '
              'so points are not connected. Colour denotes initial treatment throughout; subsequent treatment changes do not reassign groups. '
              'No outcomes are imputed. Counts and missing-outcome bounds are in Supplementary Table S1 and source data; '
              'baseline-adjusted treatment comparisons and additional designated visits are in the supplement. '
              'These available-case, observational trajectories do not establish causal treatment differences.',
              [str(folder/'clinical_observed.csv'), str(folder/'endpoint_rates.csv')], root)


def _forest(ax, data, color, *, stars=True):
    y = np.arange(len(data))
    valid = np.isfinite(data.effect.to_numpy())
    interval = valid & np.isfinite(data.ci_low.to_numpy()) & np.isfinite(data.ci_high.to_numpy())
    ax.errorbar(data.effect.to_numpy()[interval], y[interval],
                xerr=[data.effect.to_numpy()[interval]-data.ci_low.to_numpy()[interval],
                      data.ci_high.to_numpy()[interval]-data.effect.to_numpy()[interval]],
                fmt='o', color=color, ms=3.2, lw=1.1, capsize=1.5)
    ax.scatter(data.effect.to_numpy()[valid & ~interval], y[valid & ~interval],
               s=11, facecolor=color, edgecolor=color, zorder=3)
    if stars:
        for i, q in enumerate(data.q):
            if np.isfinite(q) and q < .05:
                ax.text(1.075, i, '*', transform=ax.get_yaxis_transform(), va='center',
                        ha='center', fontsize=9, weight='bold')
    ax.axvline(0, color='#B8C1C6', lw=.65, zorder=0)
    ax.grid(axis='y', color='#E9EEF0', lw=.35)
    ax.set(ylim=(len(data)-.45, -.55), yticks=y)
    clean(ax)


def molecular_response(root):
    folder = root / 'results/tables/update_20261017/clinical'
    paired = pd.read_csv(folder/'molecular_paired_changes.csv')
    paired = paired[paired.start.eq(0) & paired.end.eq(12)]
    contrasts = pd.read_csv(folder/'molecular_adjusted_contrasts.csv')
    contrasts = contrasts[contrasts.start.eq(0) & contrasts.end.eq(12)
                          & contrasts.scope.eq('all_centers') & contrasts.analysis.eq('primary')]
    fig, matrix = plt.subplots(1, 5, figsize=(183/25.4, 178/25.4))
    fig.subplots_adjust(left=.145, right=.965, bottom=.13, top=.87, wspace=.28)
    axes = list(matrix)
    lower = np.floor(paired[['effect', 'ci_low']].min().min())
    upper = np.ceil(paired[['effect', 'ci_high']].max().max())
    for i, (ax, arm, title) in enumerate(zip(axes[:3], base.ARMS,
                                            ['Dupilumab', 'Abrocitinib\n100 mg', 'Abrocitinib\n200 mg'])):
        d = paired[paired.treatment.eq(arm)].set_index('marker').reindex(base.MARKERS)
        color = palette.register(fig, 'treatment.'+arm)
        _forest(ax, d, color)
        ax.set(yticklabels=base.MARKERS if i == 0 else ['']*23, xlim=(lower, upper),
               xticks=np.arange(lower, upper+1, 2), title=title,
               xlabel='Log₂ change')
    comparisons = [('A200 - A100', 'A200 minus\nA100'), ('D - pooled A', 'D minus\npooled A')]
    current = contrasts[contrasts.comparison.isin([r[0] for r in comparisons])]
    bound = np.ceil(np.abs(current[['ci_low','ci_high','effect']]).max().max())
    for ax, (key, title) in zip(axes[3:], comparisons):
        d = contrasts[contrasts.comparison.eq(key)].set_index('marker').reindex(base.MARKERS)
        _forest(ax, d, palette.register(fig, 'method.current'))
        ax.set(yticklabels=['']*23, xlim=(-bound, bound), xticks=[-bound, 0, bound],
               title=title, xlabel='Adjusted difference')
    for i, ax in enumerate(axes): label(ax, string.ascii_lowercase[i])
    left = axes[0].get_position().x0; middle = axes[2].get_position().x1
    a = axes[3].get_position().x0; b = axes[4].get_position().x1
    fig.text((left+middle)/2, .97, 'Within-person change: W0 → W12', ha='center', fontsize=8, weight='bold')
    fig.text((a+b)/2, .97, 'Treatment contrast: W0 → W12', ha='center', fontsize=8, weight='bold')
    fig.text(.53, .04, 'Point and 95% CI   ·   * FDR q<0.05   ·   All 23 measured markers',
             ha='center', fontsize=6.5)
    base.save(fig, 'Figure4', 'Serum molecular response during the first 12 weeks',
              'a–c, Mean within-patient change in log₂ concentration from W0 to W12 and 95% t confidence intervals for '
              'all 23 markers, by initial treatment. A change of −1 corresponds to half the baseline concentration. '
              'd–e, Direct adjusted differences in those changes for A200 minus A100 and dupilumab minus pooled abrocitinib, '
              'with HC3-robust 95% confidence intervals. Models adjust for starting log₂ concentration, W0 EASI, age, sex, BMI and centre. '
              'Pooled abrocitinib preserves the observed dose mixture. Negative contrasts mean a larger decline in the first group. '
              'Stars show q<0.05 under the original declared BH families; no tests or multiplicity families were changed for this display. '
              'Each marker uses its own available pairs. Estimates with fewer than five pairs have no inferential interval or star; '
              'missing estimates are left blank. Exact sample sizes, estimates, intervals and q values remain in source tables. '
              'Within-group and between-group panels have separately labelled x axes and cannot be interpreted as the same effect. '
              'The remaining pairwise treatment contrasts, later windows, fixed three-visit trajectories and assay-batch sensitivities '
              'are reported in the supplement. Treatment assignment was observational.',
              [str(folder/'molecular_paired_changes.csv'), str(folder/'molecular_adjusted_contrasts.csv')], root)


def model_performance(root):
    from .sensitivity_figures_20261025 import model_performance as render
    render(root)


def clinical_composition(root):
    folder=root/'results/tables/update_20261017/clinical'
    iga=pd.read_csv(folder/'IGA_composition.csv')
    paths=pd.read_csv(folder/'maintenance_rates.csv')
    paths=paths[paths.classification.eq('group')]
    grades=palette.sequence('recorded_iga_0_to_5')
    fig, matrix=plt.subplots(2,3,figsize=(183/25.4,190/25.4),gridspec_kw={'height_ratios':[1,1.22]})
    fig.subplots_adjust(left=.19,right=.98,top=.87,bottom=.12,wspace=.38,hspace=.73)
    axes=list(matrix.flat)
    for ax,tx,title in zip(axes[:3],base.ARMS,base.LABELS):
        z=iga[iga.treatment.eq(tx)&iga.week.isin([0,4,12,24])]
        weeks=[0,4,12,24];bottom=np.zeros(4)
        for grade,color in enumerate(grades):
            rows=[z[z.week.eq(w)&z.category.eq(grade)].iloc[0] for w in weeks]
            values=np.array([r.proportion*100 for r in rows])
            ax.bar(np.arange(4),values,bottom=bottom,color=color,width=.72,edgecolor='white',linewidth=.3)
            bottom+=values
        ns=[int(z[z.week.eq(w)].n.iloc[0]) for w in weeks]
        ax.set(title=title,xticks=range(4),xticklabels=[f'W{w}\nn={n}' for w,n in zip(weeks,ns)],
               ylim=(0,100),yticks=[0,50,100])
    axes[0].set_ylabel('Observed IGA composition (%)')
    fig.legend(handles=[Patch(facecolor=c,edgecolor='#B9C5CC',lw=.4,label=f'IGA {i}') for i,c in enumerate(grades)],
               loc='upper center',bbox_to_anchor=(.57,.965),ncol=6,frameon=False,handlelength=1,columnspacing=.9,fontsize=6.5)
    for j,(ax,tx,title) in enumerate(zip(axes[3:],base.ARMS,base.LABELS)):
        z=paths[paths.treatment.eq(tx)].set_index('group').reindex([r[0] for r in PATHS])
        values=z.n_group.fillna(0).astype(int)
        n=int(values.sum());assert n==[65,56,42][j]
        proportions=values.to_numpy()/n*100
        ax.barh(range(len(PATHS)),proportions,color='#6B747C',height=.64,
                edgecolor='white',linewidth=.3)
        for i,(total,proportion) in enumerate(zip(values,proportions)):
            ax.text(proportion+1.2,i,str(total),ha='left',va='center',fontsize=6.3)
        ax.set(yticks=range(len(PATHS)),yticklabels=[r[1] for r in PATHS] if j==0 else ['']*len(PATHS),
               ylim=(len(PATHS)-.5,-.5),xlim=(0,52),xticks=[0,25,50],
               xlabel='Initial group (%)',title=f'{title}\nN={n}')
        ax.tick_params(axis='y',length=0,labelsize=6.3)
    for i,ax in enumerate(axes):label(ax,string.ascii_lowercase[i]);clean(ax)
    base.save(fig,'Figure3','Recorded skin severity and treatment-reduction pathways',
              'a–c, Recorded IGA compositions at W0, W4, W12 and W24. Each bar totals 100% of the evaluable patients; '
              'n below the bar is its patient denominator. The ordered blue scale identifies recorded IGA grades 0–5 and is '
              'not a treatment colour. No score or count is superimposed on the segments. All grade-specific counts and '
              'proportions remain in source tables. d–f, Future-defined pathways formed by W12 EASI75 status and subsequent '
              'recorded treatment changes. Horizontal bar length is the proportion of the complete initial group (N=65/56/42), '
              'and the number at its end is the pathway size. The seven mutually exclusive rows exhaust each initial group, '
              'including zero-size and uncertain-exposure rows. Reached refers to W12 EASI75; later reduction means a '
              'documented reduction after W12. These are descriptive pathways, not randomized reduction strategies; '
              'the bars do not encode subsequent efficacy or relapse. Instrument identity remains unverified for the recorded IGA scale. '
              'Complete pathway counts are available in source data.',
              [str(folder/'IGA_composition.csv'),str(folder/'maintenance_rates.csv')],root)


def run(root):
    from .clinical_flow_20261025 import framework
    root=Path(root).resolve()
    framework(root,base)
    clinical_targets(root)
    clinical_composition(root)
    molecular_response(root)
    model_performance(root)
