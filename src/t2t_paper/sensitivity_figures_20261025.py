"""Matched input comparisons and observed-versus-imputed IgE sensitivity displays."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import palette
from . import figures as base
from .main_redesign_20261018 import clean, label

MODEL_ROWS = [
    ('serum3', 'Three serum markers'),
    ('bsa_serum3', 'BSA + three markers'),
    ('bsa_serum3_itch', 'BSA + three markers\n+ PP-NRS'),
    ('itch_serum3', 'PP-NRS + three markers'),
    ('bsa', 'BSA'),
    ('bsa_itch', 'BSA + PP-NRS'),
]


def model_performance(root):
    root = Path(root)
    source = root/'results/tables/update_20261025/ablation/main_input_comparison.csv'
    metrics = pd.read_csv(source)
    old = root/'results/tables/update_20261017/redesign_20261018'
    ecdf_path = root/'manuscript/source_data/error_ecdf.csv'
    ecdf = pd.read_csv(ecdf_path)
    response = pd.read_csv(old/'monitoring_response_uncertainty.csv')
    fig = plt.figure(figsize=(183/25.4, 185/25.4))
    top = fig.add_gridspec(1, 3, left=.265, right=.93, bottom=.59, top=.925, wspace=.42)
    positions = [0, 1, 2, 3, 4.5, 5.5]
    for i, week in enumerate([0, 12, 24]):
        ax = fig.add_subplot(top[0, i])
        color = palette.register(fig, f'visit.W{week}')
        d = metrics[metrics.week.eq(week)].set_index('model')
        assert d.n_patients.nunique() == 1
        for y, (key, name) in zip(positions, MODEL_ROWS):
            row = d.loc[key]
            ax.errorbar(row.MAE, y, xerr=[[row.MAE-row.MAE_lo],[row.MAE_hi-row.MAE]],
                        fmt='o', color=color, lw=1.3, ms=4, capsize=2)
            ax.text(1.04, y, f'{row.MAE:.2f}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=6.5)
        ax.axhline(3.75, color='#D7DEE2', lw=.6)
        ax.set(yticks=positions, yticklabels=[x[1] for x in MODEL_ROWS] if i==0 else ['']*6,
               ylim=(6.1,-.6), xlim=(0,10), xticks=[0,4,8], xlabel='MAE (EASI points)',
               title=f'Week {week} · n={int(d.n_patients.iloc[0])}')
        ax.tick_params(axis='y', length=0, labelsize=6.5)
        clean(ax); label(ax,'abc'[i])
    fig.text(.265, .972, 'Matched input comparison', fontsize=8, weight='bold')
    lower = fig.add_gridspec(1, 2, left=.10, right=.97, bottom=.12, top=.405, wspace=.65)
    ax = fig.add_subplot(lower[0, 0])
    for week in [0,12,24]:
        color = palette.register(fig, f'visit.W{week}')
        d = ecdf[ecdf.week.eq(week) & ecdf.absolute_error.le(10)].sort_values('absolute_error')
        ax.plot(d.absolute_error,100*d.proportion,color=color,lw=1.5,label=f'Week {week}')
        ax.fill_between(d.absolute_error.to_numpy(),100*d.ci_low.to_numpy(),100*d.ci_high.to_numpy(),
                        color=color,alpha=.13,lw=0)
    ax.axvline(2,color='#A9B4BA',lw=.7)
    ax.set(xlim=(0,10),ylim=(0,102),xlabel='Absolute EASI error (points)',
           ylabel='Predictions within threshold (%)',title='Primary model: individual error')
    ax.legend(loc='lower right',frameon=False,fontsize=6.5)
    clean(ax);label(ax,'d')
    ax = fig.add_subplot(lower[0,1])
    for j,week in enumerate([12,24]):
        color=palette.color(f'visit.W{week}')
        for k,metric in enumerate(['sensitivity','specificity','agreement']):
            row=response[response.window.eq(f'W0_W{week}') & response.endpoint.eq('EASI75') & response.metric.eq(metric)].iloc[0]
            ax.errorbar(100*row.estimate,k+(j-.5)*.25,
                        xerr=[[100*(row.estimate-row.ci_low)],[100*(row.ci_high-row.estimate)]],
                        fmt='o',color=color,ms=4,lw=1.3,capsize=2)
    ax.set(yticks=range(3),yticklabels=['Sensitivity','Specificity','Agreement'],
           xlim=(45,102),xticks=[50,75,100],ylim=(2.6,-.6),xlabel='Classification performance (%)',
           title='Primary model: EASI75')
    ax.legend(handles=[Line2D([],[],color=palette.color(f'visit.W{w}'),marker='o',lw=0,label=f'Week {w}') for w in [12,24]],
              loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2,frameon=False,fontsize=6.5)
    clean(ax);label(ax,'e')
    base.save(fig,'Figure5','Input contributions and performance of the common EASI estimator',
              'a–c, Post hoc input-set comparisons on identical patients with all three serum markers, BSA, PP-NRS and observed EASI. '
              'W0/W12/W24 contain 151/89/75 visits. Each model was refitted using the same patient outer folds; W0/W12 alone supplied training '
              'data. Preprocessing, structure and penalty selection used training folds. Serum markers are CCL17, CCL26 and IL-5; clinical inputs '
              'are from the current visit. Points and labels show mean absolute error (MAE), with 95% paired-patient bootstrap intervals. '
              'Clinical-only controls appear below the divider. Lower MAE after adding BSA to markers does not demonstrate molecular added accuracy; '
              'the relevant molecular comparisons retain the same clinical inputs. d–e, The unchanged primary BSA–serum procedure, evaluated on '
              'its original population. d, Cumulative absolute-error proportions among 152/89/75 patients, with bootstrap 95% intervals. All errors '
              'enter the calculation; only thresholds up to ten points are displayed. Two points is a visual reference, not an accepted interchangeability '
              'margin. e, EASI75 sensitivity, specificity and agreement, with bootstrap 95% intervals; W12/W24 responder denominators are 49/54, '
              'nonresponder denominators 36/13, and total paired denominators 85/67. Observed and estimated responses each use their own baseline '
              'and follow-up scores. All metrics are scored within each repetition before averaging. Bootstrap intervals condition on saved predictions. '
              'W24 was previously inspected and is same-cohort temporal application. Additional molecular-panel and adult-DLQI comparisons are in '
              'Supplementary Table S6 and its complete source tables. The new input comparisons do not replace the frozen primary equation.',
              [str(source),str(ecdf_path),str(old/'monitoring_response_uncertainty.csv')],root)


def ige_sensitivity(root, base, finish):
    root = Path(root)
    folder = root/'results/tables/update_20261025/ige'
    fixed_path = root/'results/tables/update_20261017/clinical/clinical_fixed_triplets.csv'
    fixed = pd.read_csv(fixed_path).query("target == 'ige_numeric'")
    paired = pd.read_csv(folder/'paired_log_changes.csv')
    longitudinal = pd.read_csv(folder/'available_longitudinal_changes.csv')
    imputed = pd.read_csv(folder/'mi_log_changes.csv')
    fig = plt.figure(figsize=(183/25.4, 165/25.4))
    top = fig.add_gridspec(1,3,left=.15,right=.97,bottom=.60,top=.90,wspace=.35)
    bottom = fig.add_gridspec(1,3,left=.30,right=.97,bottom=.13,top=.40,wspace=.30)
    fig.text(.15,.965,'Identical three-visit patients · all centres',fontsize=8,weight='bold')
    fig.text(.15,.47,'W0 to W24 · missing-data sensitivity in CAMS',fontsize=8,weight='bold')
    method_names=['Paired observations','Available-data model','Log-normal imputation','Donor imputation']
    for j, arm in enumerate(base.ARMS):
        color=palette.register(fig,'treatment.'+arm)
        ax=fig.add_subplot(top[0,j])
        d=fixed[fixed.treatment.eq(arm)].sort_values('week')
        assert d.n.nunique()==1 and d.ci_low.gt(0).all()
        ax.errorbar(d.week,d['median'],yerr=[d['median']-d.ci_low,d.ci_high-d['median']],
                    fmt='o-',color=color,lw=1.35,ms=4,capsize=2)
        ax.set_yscale('log');ax.minorticks_off()
        ax.set(xticks=[0,12,24],xlim=(-1,25),ylim=(25,6500),
               yticks=[50,100,500,1000,5000],yticklabels=['50','100','500','1000','5000'],
               xlabel='Week',title=f'{palette.treatment_label(arm)} · n={int(d.n.iloc[0])}')
        if j==0:ax.set_ylabel('Observed IgE (recorded units)')
        clean(ax);label(ax,'abc'[j])
        ax=fig.add_subplot(bottom[0,j])
        q=paired[paired.scope.eq('CAMS') & paired.treatment.eq(arm) & paired.window.eq('W0_W24') & paired.method.eq('all_available_pairs')].iloc[0]
        l=longitudinal[longitudinal.scope.eq('CAMS') & longitudinal.treatment.eq(arm) & longitudinal.window.eq('W0_W24')].iloc[0]
        rows=[q,l]
        for method in ['normal_log','pmm']:
            rows.append(imputed[imputed.scope.eq('CAMS') & imputed.treatment.eq(arm) & imputed.window.eq('W0_W24') & imputed.scenario.eq('MAR') & imputed.method.eq(method)].iloc[0])
        for i,row in enumerate(rows):
            ax.errorbar(row.ratio,i,xerr=[[row.ratio-row.ratio_low],[row.ratio_high-row.ratio]],
                        fmt='o',color=color,lw=1.35,ms=4,capsize=2)
        ax.axvline(1,color='#A9B4BA',lw=.7)
        ax.set_xscale('log');ax.minorticks_off()
        ax.set(yticks=range(4),yticklabels=method_names if j==0 else ['']*4,
               ylim=(3.5,-.5),xlim=(.1,6),xticks=[.1,.5,1,5],xticklabels=['0.1','0.5','1','5'],
               xlabel='W24/W0 ratio of (IgE + 1)',title=palette.treatment_label(arm))
        ax.tick_params(axis='y',length=0,labelsize=6.5)
        clean(ax);label(ax,'def'[j])
    finish(fig,21,'IgE: fixed-patient observations and missing-data sensitivity',
           'a–c, Observed medians and patient-bootstrap 95% confidence intervals in the same complete three-visit patients at every visit: '
           '11 dupi, 9 abro 100 mg and 10 abro 200 mg patients across both centres. The vertical scale is logarithmic, labelled in recorded '
           'units. d–f, W0–W24 geometric mean ratios of IgE+1 from four analyses within CAMS. The observed paired analysis uses all available '
           'W0/W24 pairs (14/10/12 by treatment); it does not require W12. The available-data random-intercept model uses 229 measured visits '
           'from 125 patients across the three treatments. Multiple-imputation analyses target all 130 CAMS patients (51/38/41), using patient-wide '
           'chained equations with either posterior log-normal draws or within-treatment predictive mean matching (donor imputation); 400 retained '
           'datasets are pooled with Rubin variance and finite-sample degrees of freedom. Both assume missingness can be explained by the included '
           'observed information. The panels target the same direction of change but rely on different populations or missingness assumptions. '
           'Ratios below one indicate lower W24 values; the line at one denotes no change. Bars are 95% confidence intervals. No observed value '
           'was replaced. Sparse donor support makes the abro 200 mg estimate sensitive to the imputation method; neither imputation confirms its '
           'direction of change. CAMS is used because Huashan has no follow-up IgE for dupi. Full-cohort, other-window and missing-not-at-random '
           'analyses are retained in Supplementary Table S8 and source data. No W4 IgE was observed, so no W4 point is imputed or plotted. '
           'The recorded value 5000 was not assumed to be a verified censoring threshold. These observational analyses do not identify causal drug effects.',
           [fixed_path,folder/'paired_log_changes.csv',folder/'available_longitudinal_changes.csv',folder/'mi_log_changes.csv'],root,base)
