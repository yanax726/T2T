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

MODEL = 'unified_selected'
SHORT = ['D', 'A100', 'A200']
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
    fig, matrix = plt.subplots(4, 3, figsize=(183 / 25.4, 225 / 25.4))
    fig.subplots_adjust(left=.09, right=.98, top=.925, bottom=.065, wspace=.40, hspace=.65)
    axes = list(matrix.flat)
    legend(fig, at=(.52, .987))
    for ax, (target, title, weeks) in zip(axes, TARGETS):
        for arm, color in zip(base.ARMS, base.COLORS):
            d = rates[rates.target.eq(target) & rates.treatment.eq(arm)].set_index('week').reindex(weeks)
            assert d.rate.notna().all()
            x = np.array(weeks, float)
            ax.plot(x, 100*d.rate, 'o-', color=color, lw=1.35, ms=3.4,
                    markeredgecolor=palette.role_style('treatment.' + arm)['outline'], markeredgewidth=.4)
            ax.fill_between(x, 100*d.ci_low.to_numpy(), 100*d.ci_high.to_numpy(),
                            color=color, alpha=.13, lw=0)
        ax.set(title=title, xticks=weeks, xlim=(-1, 25), ylim=(0, 103),
               yticks=[0, 50, 100], xlabel='Week')
        ax.tick_params(axis='x', labelsize=6)
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
        if i % 3 == 0: ax.set_ylabel('Patients reaching target (%)')
    base.save(fig, 'Figure2', 'Clinical target attainment across initial treatment groups',
              'a–k, Observed attainment of every specified clinical threshold. Lines join available-case proportions; '
              'shaded bands are pointwise Wilson 95% confidence intervals, not longitudinal model intervals. '
              'EASI50/75/90 and IGA decrease ≥2 require a recorded baseline and follow-up; absolute thresholds also show W0. '
              'PP-NRS ≤4 additionally includes W2. The adult DLQI denominator is restricted to age ≥18. '
              'l, Combined physician-and-patient targets from Fig. 1, with Wilson intervals. These criteria change by stage; '
              'points are deliberately not connected. All other lines use the same definition over time. '
              'Colours consistently identify initial treatment. Later dose changes do not reassign groups. '
              'These are unadjusted observed rates; changing observed denominators and baseline treatment differences preclude '
              'a causal comparison. All achieved/evaluable counts, missing-outcome bounds and adjusted comparisons accompany '
              'the source tables; numerical rates and denominators are tabulated in the supplement. No outcomes are imputed.',
              [str(folder/'endpoint_rates.csv')], root)


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
    folder = root/'results/tables/update_20261017'
    metrics = pd.read_csv(folder/'model/model_metrics.csv')
    ecdf = pd.read_csv(root/'manuscript/source_data/error_ecdf.csv')
    response = pd.read_csv(folder/'redesign_20261018/monitoring_response_uncertainty.csv')
    change = pd.read_csv(folder/'redesign_20261018/monitoring_change_uncertainty.csv')
    fig, matrix = plt.subplots(2, 2, figsize=(183/25.4, 158/25.4))
    fig.subplots_adjust(left=.12, right=.96, top=.90, bottom=.13, wspace=.62, hspace=.80)
    axes = list(matrix.flat)
    for week in [0,12,24]:
        color = palette.register(fig, f'visit.W{week}')
        d = ecdf[ecdf.week.eq(week) & ecdf.absolute_error.le(10)].sort_values('absolute_error')
        axes[0].plot(d.absolute_error, 100*d.proportion, color=color, lw=1.5, label=f'Week {week}')
        axes[0].fill_between(d.absolute_error.to_numpy(), 100*d.ci_low.to_numpy(),
                             100*d.ci_high.to_numpy(), color=color, lw=0, alpha=.14)
    axes[0].axvline(2, color='#A9B4BA', lw=.7)
    axes[0].set(xlim=(0,10), ylim=(0,102), xlabel='Absolute EASI error (points)',
                ylabel='Predictions within threshold (%)', title='Individual error distribution')
    axes[0].legend(loc='lower right', frameon=False, fontsize=6.3)
    d = metrics[metrics.model.eq(MODEL) & metrics.week.eq(0) & metrics.group_type.eq('severity')].set_index('group')
    keys = ['EASI_le7','EASI_7_to21','EASI_gt21']
    labels = ['≤7', '>7 to 21', '>21']
    for y, key in enumerate(keys):
        r = d.loc[key]
        axes[1].errorbar(r.MAE, y, xerr=[[r.MAE-r.MAE_lo],[r.MAE_hi-r.MAE]],
                         fmt='o', color=palette.color('visit.W0'), ms=4, lw=1.35, capsize=2)
    axes[1].set(yticks=range(3), yticklabels=[f'{name} (n={int(d.loc[k,"n_patients"])})' for k,name in zip(keys,labels)],
                xlabel='Mean absolute EASI error (points)', xlim=(0,7), ylim=(2.6,-.6), title='Baseline error by severity')
    windows = ['W0_W12','W0_W24','W12_W24']
    d = change[change.metric.eq('MAE')].set_index('window').loc[windows]
    for y, (window, row) in enumerate(d.iterrows()):
        axes[2].errorbar(row.estimate, y, xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],
                         fmt='o', color='#59666F', ms=4, lw=1.35, capsize=2)
    axes[2].set(yticks=range(3),yticklabels=[f'{w.replace("_", " → ")}\n(n={int(d.loc[w,"n_patients"])})' for w in windows],
                xlim=(0,5.3), ylim=(2.6,-.6), xlabel='Mean absolute change error (points)', title='Within-person change accuracy')
    ax=axes[3]
    names = ['Sensitivity','Specificity','Agreement']
    for j,week in enumerate([12,24]):
        color=palette.color(f'visit.W{week}')
        for k,metric in enumerate(['sensitivity','specificity','agreement']):
            r=response[response.window.eq(f'W0_W{week}') & response.endpoint.eq('EASI75') & response.metric.eq(metric)].iloc[0]
            ax.errorbar(100*r.estimate,k+(j-.5)*.25,
                        xerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],
                        fmt='o',color=color,ms=4,lw=1.35,capsize=2)
    ax.set(yticks=range(3),yticklabels=names,xlim=(45,102),xticks=[50,75,100],ylim=(2.6,-.6),
           xlabel='Classification performance (%)',title='EASI75 response classification')
    ax.legend(handles=[Line2D([],[],color=palette.color(f'visit.W{w}'),marker='o',lw=0,label=f'Week {w}') for w in [12,24]],
              frameon=False,loc='upper center',bbox_to_anchor=(.5,-.36),ncol=2,fontsize=6.3)
    for i,ax in enumerate(axes):label(ax,string.ascii_lowercase[i]);clean(ax)
    base.save(fig,'Figure5','Accuracy of the common EASI estimate and paired response monitoring',
              'a, Cumulative absolute-error proportions for the common BSA–CCL17–CCL26–IL-5 procedure. Each repetition is scored '
              'before averaging; bands are patient-bootstrap 95% confidence intervals conditional on saved predictions. All predictions '
              'enter the proportions, although thresholds are displayed only through ten points. The two-point line is a reading reference, '
              'not an accepted replacement margin. W0/W12/W24 include 152/89/75 patients. b, Baseline MAE by observed severity, '
              'with patient-bootstrap intervals and unique-patient counts. Follow-up severe observations were too sparse to demonstrate '
              'accuracy across the full severity range. c, Mean absolute error in paired score changes, with patient-bootstrap 95% intervals. '
              'Intervals are distinct comparisons, not sequential categories, and can share patients. d, EASI75 sensitivity, specificity and '
              'agreement with observed response, with patient-bootstrap 95% intervals. Each response uses its corresponding observed or '
              'estimated W0 and follow-up score. W12/W24 sensitivity denominators are 49/54 observed responders; specificity denominators '
              'are 36/13 nonresponders; agreement includes 85/67 patients. W0/W12 use nested patient-level validation; W24 is previously '
              'inspected same-cohort temporal application. Metrics assess the development procedure, not independent validation of the final '
              'formula. MAE is mean absolute error in EASI points. Numerical metrics and matched controls are in Table 2; full agreement '
              'and response analyses are in the supplement.',
              [str(root/'manuscript/source_data/error_ecdf.csv'),str(folder/'model/model_metrics.csv'),
               str(folder/'redesign_20261018/monitoring_response_uncertainty.csv'),
               str(folder/'redesign_20261018/monitoring_change_uncertainty.csv')],root)


def clinical_composition(root):
    folder=root/'results/tables/update_20261017/clinical'
    iga=pd.read_csv(folder/'IGA_composition.csv')
    paths=pd.read_csv(folder/'maintenance_rates.csv')
    paths=paths[paths.classification.eq('group')]
    grades=palette.sequence('recorded_iga_0_to_5')
    statuses=palette.sequence('pathway_w24_status')
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
        values=z[['n_group','n_evaluable','k','n_missing']].fillna(0).astype(int)
        n=int(values.n_group.sum());assert n==[65,56,42][j]
        counts=np.column_stack([values.k, values.n_evaluable-values.k, values.n_missing])
        assert np.array_equal(counts.sum(axis=1),values.n_group)
        start=np.zeros(len(PATHS))
        for column,color in enumerate(statuses):
            width=counts[:,column]/n*100
            ax.barh(range(len(PATHS)),width,left=start,color=color,height=.64,edgecolor='white',linewidth=.3)
            for i,k in enumerate(counts[:,column]):
                if k>=3:
                    ax.text(start[i]+width[i]/2,i,str(k),ha='center',va='center',fontsize=6.3,
                            color='white' if column==1 else '#24333C')
            start+=width
        for i,total in enumerate(values.n_group):
            ax.text(start[i]+1.2,i,str(total),ha='left',va='center',fontsize=6.3)
        ax.set(yticks=range(len(PATHS)),yticklabels=[r[1] for r in PATHS] if j==0 else ['']*len(PATHS),
               ylim=(len(PATHS)-.5,-.5),xlim=(0,52),xticks=[0,25,50],
               xlabel='Initial group (%)',title=f'{title}\nN={n}')
        ax.tick_params(axis='y',length=0,labelsize=6.3)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(statuses,
                ['W24 EASI75 reached','W24 EASI75 not reached','W24 outcome missing'])],
               loc='upper center',bbox_to_anchor=(.57,.055),ncol=3,frameon=False,fontsize=6.3,
               handlelength=1,columnspacing=1)
    for i,ax in enumerate(axes):label(ax,string.ascii_lowercase[i]);clean(ax)
    base.save(fig,'Figure3','Recorded skin severity and treatment-reduction pathways',
              'a–c, Recorded IGA compositions at W0, W4, W12 and W24. Each bar totals 100% of the evaluable patients; '
              'n below the bar is its patient denominator. The ordered blue scale identifies recorded IGA grades 0–5 and is '
              'not a treatment colour. No score or count is superimposed on the segments. All grade-specific counts and '
              'proportions remain in source tables. d–f, Future-defined pathways formed by W12 EASI75 status and subsequent '
              'recorded treatment changes. Horizontal bar length is the proportion of the complete initial group (N=65/56/42), '
              'and the number at its end is the pathway size. Warm segments show observed W24 EASI75 status and grey shows '
              'missing W24 outcomes; segment counts of at least three are printed. The seven mutually exclusive rows exhaust '
              'each initial group, including zero-size and uncertain-exposure rows. Reached refers to W12 EASI75; later reduction '
              'means a documented reduction after W12. A W24 EASI75 failure does not by itself establish the timing or cause of relapse. '
              'These are descriptive paths selected partly by later response, not randomized reduction strategies. Instrument identity '
              'remains unverified for the recorded IGA scale. Complete counts and conditional W24 response intervals are available in '
              'source data.',
              [str(folder/'IGA_composition.csv'),str(folder/'maintenance_rates.csv')],root)


def run(root):
    from .clinical_framework_20261018 import framework
    root=Path(root).resolve()
    framework(root,base)
    clinical_targets(root)
    clinical_composition(root)
    molecular_response(root)
    model_performance(root)
