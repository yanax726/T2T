"""Claim-led main figures using the frozen analysis and a shared visual system."""
from pathlib import Path
import json
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

from . import figures as base

MODEL = 'unified_selected'
SHORT = {'dupilumab': 'D', 'abrocitinib_100mg': 'A100', 'abrocitinib_200mg': 'A200'}
PANEL = ['CCL17', 'CCL26', 'IL-5']


def label(ax, letter):
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-18, 10),
                textcoords='offset points', weight='bold', fontsize=8,
                annotation_clip=False)


def clean(ax):
    ax.tick_params(length=2.5, width=.7, pad=3)
    ax.spines[['top', 'right']].set_visible(False)


def legend(fig, at=(.51, .988), kind='treatment'):
    if kind == 'treatment':
        names, colors = base.LABELS, base.COLORS
        markers, styles = base.TREATMENT_MARKERS, ['-'] * 3
    else:
        names, colors = [f'Week {w}' for w in base.TIMES], base.TIME_COLORS
        markers, styles = base.TIME_MARKERS, base.TIME_STYLES
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls=s, ms=3, lw=1, label=n)
                        for c, n, m, s in zip(colors, names, markers, styles)],
               loc='upper center', bbox_to_anchor=at, ncol=3, frameon=False)


def count_strip(ax, rows, columns):
    """Place available counts below a trajectory without encoding them as outcomes."""
    table = ax.table(cellText=rows, rowLabels=['D', 'A100', 'A200'],
                     colLabels=[str(w) for w in columns], cellLoc='center',
                     bbox=(.02, -.48, .96, .29))
    table.auto_set_font_size(False); table.set_fontsize(6.3)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('none'); cell.PAD = .02
        if r == 0: cell.set_text_props(color='#69747C')
    ax.set_xlabel('Week', labelpad=0)


def count_key(fig, position):
    handles=[Line2D([],[],marker='o',ms=np.sqrt(9*n),ls='',color=base.STYLE['neutral_dark'],
                    alpha=.65,label=str(n)) for n in [1,5,10]]
    fig.legend(handles=handles,title='Patients per rounded coordinate',loc='upper center',
               bbox_to_anchor=position,ncol=3,frameon=False,fontsize=6.3,title_fontsize=6.3)


def clinical_context(root):
    folder = root / 'results/tables/update_20261017/clinical'
    observed = pd.read_csv(folder / 'clinical_observed.csv')
    rates = pd.read_csv(folder / 'endpoint_rates.csv')
    contrasts = pd.read_csv(folder / 'clinical_trajectory_contrasts.csv')
    fig = plt.figure(figsize=(183 / 25.4, 168 / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.58], left=.10, right=.84,
                           bottom=.09, top=.90, wspace=.52, hspace=.90)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    legend(fig)
    fig.text(.47,.945,'Observed trajectories: median; shading indicates the interquartile range',ha='center',fontsize=6.3)
    for ax, variable, title in zip(axes[:2], ['easi_total', 'pp_nrs'], ['Disease activity', 'Itch']):
        counts = []
        for arm, color in zip(base.ARMS, base.COLORS):
            g = observed[observed.target.eq(variable) & observed.treatment.eq(arm)].sort_values('week')
            wanted = [0, 2, 4, 12, 24] if variable == 'pp_nrs' else [0, 4, 12, 24]
            g = g[g.week.isin(wanted)]
            x = g.week.to_numpy(float)
            ax.fill_between(x, g.q25.to_numpy(float), g.q75.to_numpy(float), color=color, alpha=.10, lw=0)
            ax.plot(x, g['median'], marker=base.STYLE['treatment_markers'][arm], color=color, lw=1.15, ms=3)
            counts.append([int(g.set_index('week').loc[w, 'n']) for w in wanted])
        ax.set(title=title, xlabel='Week', ylabel='PP-NRS (points)' if variable == 'pp_nrs' else 'EASI (points)',
               xticks=wanted, xlim=(-1, 25), ylim=(0, 11 if variable == 'pp_nrs' else 35))
        count_strip(ax, counts, wanted)
        ax.text(-.20, -.25, 'N', transform=ax.transAxes, fontsize=6.3)
    ax = axes[2]
    pairs = [('dupilumab', 'abrocitinib_100mg'), ('dupilumab', 'abrocitinib_200mg'),
             ('abrocitinib_100mg', 'abrocitinib_200mg')]
    for j, week in enumerate([12, 24]):
        subset = contrasts[contrasts.target.eq('easi_total') & contrasts.week.eq(week)
                           & contrasts.analysis.eq('overlap_weighted')]
        for i, (first, second) in enumerate(pairs):
            r = subset[subset.group1.eq(first) & subset.group2.eq(second)].iloc[0]
            ax.errorbar(r.effect, i + (j - .5) * .24,
                        xerr=[[r.effect - r.ci_low], [r.ci_high - r.effect]],
                        fmt=['o', 's'][j], color=base.TIME_COLORS[j + 1], ms=3.5, capsize=2, lw=.8)
    ax.axvline(0, color='#B9C2C7', lw=.7, zorder=0)
    ax.set(yticks=range(3), yticklabels=[SHORT[a] + ' − ' + SHORT[b] for a, b in pairs],
           ylim=(2.65, -.65), title='Baseline-adjusted comparison',
           xlabel='EASI difference (first − second)')
    ax.legend(handles=[Line2D([], [], color=base.TIME_COLORS[j + 1], marker=m, lw=0, ms=3.5,
                              label=f'Week {w}') for j, (w, m) in enumerate([(12, 'o'), (24, 's')])],
              loc='lower left', bbox_to_anchor=(-.04, -.02), ncol=1, fontsize=6.5, frameon=False)
    ax = axes[3]
    positions, names = [], []
    for block, (week, target, heading) in enumerate([(2, 'PP_NRS_le4', 'Week 2: itch target'),
                                                    (4, 'overall', 'Week 4: early target'),
                                                    (12, 'overall', 'Week 12: stringent target'),
                                                    (24, 'overall', 'Week 24: low-activity target')]):
        start = block * 4.1
        ax.text(0, start - .10, heading, fontsize=6.5, weight='bold', va='bottom')
        for i, (arm, color) in enumerate(zip(base.ARMS, base.COLORS)):
            r = rates[rates.week.eq(week) & rates.target.eq(target) & rates.treatment.eq(arm)].iloc[0]
            y = start + .75 + i * .82
            ax.errorbar(r.rate * 100, y,
                        xerr=[[(r.rate - r.ci_low) * 100], [(r.ci_high - r.rate) * 100]],
                        fmt=base.STYLE['treatment_markers'][arm], color=color, ms=3.2, capsize=1.6, lw=.8)
            ax.text(1.04, y, f'{100*r.rate:.1f}%  {int(r.k)}/{int(r.n_evaluable)}',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=6.3)
            positions.append(y); names.append(SHORT[arm])
    ax.text(1.04, 1.015, 'Rate   Achieved/N', transform=ax.transAxes, fontsize=6.3, va='bottom')
    ax.set(yticks=positions, yticklabels=names, xlim=(0, 103), xticks=[0, 50, 100],
           ylim=(15.5, -.8), title='Stage-specific clinical targets', xlabel='Target attainment (%)')
    ax.tick_params(axis='y', length=0)
    for i, axis in enumerate(axes):
        label(axis, string.ascii_lowercase[i]); clean(axis)
    base.save(fig, 'Figure1', 'Clinical course and target attainment during treatment',
              'a,b, Observed medians and interquartile bands by initial treatment; count strips give patients available at each visit. '
              'c, EASI contrasts from the common-reference overlap-weighted model, adjusted for baseline EASI, age, sex, BMI and centre, '
              'with patient-cluster robust 95% confidence intervals. Negative differences favour the first named group for lower EASI. '
              'd, Stage-specific observed target rates with Wilson 95% confidence intervals. Every row explicitly pairs its treatment with '
              'the achieved/evaluable count. D, dupilumab; A100/A200, initial abrocitinib 100/200 mg. W2 uses PP-NRS ≤4; later both-domain '
              'targets require at least one physician and one patient criterion. W4: EASI50 or IGA drop ≥2, plus PP-NRS ≤4 or adult DLQI ≤10. '
              'W12: EASI75 or IGA 0/1, plus PP-NRS 0/1 or adult DLQI ≤4. W24 adds absolute EASI ≤5 as a physician criterion. Criteria differ by stage; points are not '
              'joined as a single common endpoint. Missing outcomes are not counted as observed failures. Initial groups are observational; '
              'later actual regimens and complete adjusted target contrasts are presented in the supplement.',
              [str(folder / f) for f in ['clinical_observed.csv', 'clinical_trajectory_contrasts.csv', 'endpoint_rates.csv']], root)


def molecular_landscape(root):
    folder = root / 'results/tables/update_20261017/clinical'
    paired = pd.read_csv(folder / 'molecular_three_visit_paired.csv')
    differences = pd.read_csv(folder / 'molecular_adjusted_contrasts.csv')
    fig = plt.figure(figsize=(183 / 25.4, 168 / 25.4))
    grid = fig.add_gridspec(2, 6, height_ratios=[2.45, 1], left=.145, right=.975,
                           bottom=.115, top=.90, wspace=.90, hspace=.82)
    left = fig.add_subplot(grid[0, :3]); right = fig.add_subplot(grid[0, 3:])
    bottom = [fig.add_subplot(grid[1, k:k + 2]) for k in [0, 2, 4]]
    cmap = LinearSegmentedColormap.from_list('t2t_signed', base.STYLE['diverging_colors'])
    cmap.set_bad(base.STYLE['unknown_color'])
    values = np.full((23, 6), np.nan)
    nvalues = np.zeros_like(values)
    for j, arm in enumerate(base.ARMS):
        for k, week in enumerate([12, 24]):
            g = paired[paired.treatment.eq(arm) & paired.week.eq(week)].set_index('marker').reindex(base.MARKERS)
            values[:, 2*j+k] = g.effect; nvalues[:, 2*j+k] = g.n
    im = left.imshow(values, cmap=cmap, vmin=-2, vmax=2, aspect='auto')
    left.set(xticks=range(6), xticklabels=['12', '24'] * 3, yticks=range(23),
             yticklabels=base.MARKERS, xlabel='Follow-up week; each change is relative to own W0',
             title='Within-person molecular change')
    for j, name in enumerate(['D', 'A100', 'A200']):
        left.text((2*j+1)/6, 1.03, name, transform=left.transAxes, ha='center', fontsize=6.5)
    for j in [1.5, 3.5]:left.axvline(j, color='white', lw=2)
    for i in range(23):
        for j in range(6):
            missing = not np.isfinite(values[i,j])
            rgb=cmap((np.clip(values[i,j],-2,2)+2)/4)[:3] if not missing else (1,1,1)
            dark_fill=np.dot(rgb,[.2126,.7152,.0722]) < .45
            left.text(j, i, '×' if missing else str(int(nvalues[i,j])), ha='center', va='center',
                      fontsize=6.3, color='white' if dark_fill else '#25343F')
            if 0 < nvalues[i,j] < 5:
                left.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,ls=':',lw=.65,ec='#25343F'))
    left.set_title('Within-person change\nSame three-visit patients; cells show n', pad=23, fontsize=6.8)
    contrasts = ['D - A100', 'D - A200', 'A200 - A100', 'D - pooled_A']
    current = differences[differences.start.eq(0) & differences.end.eq(12)
                          & differences.scope.eq('all_centers') & differences.analysis.eq('primary')]
    # Preserve all requested contrasts; only the initial treatment interval is the main panel.
    keys = current.comparison.unique().tolist()
    if 'D - pooled_A' not in keys:
        contrasts[-1] = next(k for k in keys if 'pooled' in k)
    matrix = current.pivot(index='marker', columns='comparison', values='effect').reindex(index=base.MARKERS, columns=contrasts)
    qvalues = current.pivot(index='marker', columns='comparison', values='q').reindex(index=base.MARKERS, columns=contrasts)
    im2 = right.imshow(matrix, cmap=cmap, vmin=-2, vmax=2, aspect='auto')
    right.set(xticks=range(4), xticklabels=['D−A100', 'D−A200', 'A200−A100', 'D−pooled A'],
              yticks=range(23), yticklabels=['']*23,
              title='Between-treatment change difference\nAll eligible W0/W12 pairs', xlabel='W0 → W12; first − second')
    right.tick_params(axis='x', labelrotation=25)
    for i, j in zip(*np.where(qvalues.to_numpy() < .05)):
        right.text(j, i, '•', ha='center', va='center', fontsize=6.5)
    for i, j in zip(*np.where(~np.isfinite(matrix.to_numpy()))):
        right.text(j, i, '×', ha='center', va='center', fontsize=6.5)
    for ax in [left, right]:
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    for axis, image, text in [(left, im, 'Mean within-person log₂ change'),
                              (right, im2, 'Adjusted difference in log₂ change')]:
        position = axis.get_position()
        cax = fig.add_axes([position.x0+.025, .385, position.width-.05, .011])
        cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[-2, 0, 2])
        cb.set_label(text, fontsize=6.3, labelpad=1)
        cax.tick_params(length=2, labelsize=6.3)
    for ax, marker in zip(bottom, PANEL):
        position=ax.get_position()
        ax.set_position([position.x0,position.y0-.012,position.width,position.height])
        counts=[]
        for arm, color in zip(base.ARMS, base.COLORS):
            g = paired[paired.marker.eq(marker) & paired.treatment.eq(arm)].sort_values('week')
            base.errorline(ax, g.week, g.effect, g.ci_low, g.ci_high, color)
            counts.append(f'{SHORT[arm]} {int(g.n.iloc[0])}')
        ax.axhline(0, color='#B9C2C7', lw=.6)
        ax.set(title=marker, xlabel='Week', ylabel='Log₂ change from W0', xticks=[0,12,24], xlim=(-1,25))
        ax.text(.5,-.37,'Fixed n: '+', '.join(counts),ha='center',transform=ax.transAxes,fontsize=6.3)
        clean(ax)
    legend(fig, at=(.56, .337))
    for i, ax in enumerate([left, right] + bottom):label(ax, string.ascii_lowercase[i])
    base.save(fig, 'Figure2', 'Molecular response across treatments and the three-marker input panel',
              'a, Mean paired log₂ concentration changes among each marker/treatment-specific fixed set with measurements at W0, W12 and W24. '
              'W0 is the zero reference; the two displayed columns per treatment show change to W12 and W24. '
              'b, All prespecified between-treatment contrasts for W0–W12 from HC3-robust linear models, adjusted for starting molecular '
              'concentration, W0 EASI, age, BMI, sex and centre; dots denote the declared FDR-adjusted q<0.05. These use all eligible two-visit pairs, a different population '
              'from panel a. D, dupilumab; A100/A200, initial abrocitinib dose; pooled A, the observed mixture of the two doses. '
              'Both heatmaps use the same signed log₂ scale, but panel b is a difference between treatment changes, not a within-group change. '
              'Purple denotes a negative value; zero is no change/difference. Cells marked × lack an estimate. Panel a prints paired n; '
              'dotted cell borders indicate fewer than five pairs. Each estimand has its own labelled colour bar. '
              'c–e, W0/W12/W24 paired trajectories and 95% t confidence intervals for the three frozen model inputs. Their inclusion is based on '
              'the existing panel, not selection of significant trajectories. Exact marker-specific counts, intervals, all three windows and '
              'the complete comparison families are retained in source tables and supplementary figures.',
              [str(folder / f) for f in ['molecular_three_visit_paired.csv', 'molecular_adjusted_contrasts.csv']], root)


def model_readout(root):
    folder = root / 'manuscript/source_data'
    metrics = pd.read_csv(root / 'results/tables/update_20261017/model/overall_metrics.csv')
    metrics = metrics[metrics.model.eq(MODEL)].set_index('week')
    bins = pd.read_csv(folder / 'model_plot_bins.csv')
    ecdf_path = folder / 'error_ecdf.csv'
    if not ecdf_path.exists():
        raise FileNotFoundError('The audited repeated-validation ECDF is required: ' + str(ecdf_path))
    ecdf = pd.read_csv(ecdf_path)
    fig = plt.figure(figsize=(183/25.4, 165/25.4))
    grid = fig.add_gridspec(3,3,height_ratios=[.70,1.25,1.08],left=.09,right=.975,
                           bottom=.085,top=.95,wspace=.45,hspace=.68)
    schematic = fig.add_subplot(grid[0,:]);schematic.set_axis_off()
    schematic.text(0,.97,'Final equation · four inputs matched by nominal visit',weight='bold',fontsize=7)
    schematic.text(0,.66,'BSA (%)   +   CCL17   +   CCL26   +   IL-5 (each pg/mL in the study export)',fontsize=6.8)
    schematic.text(0,.35,'EASI ≈ clip{ BSA/100 × [51.27 + 0.98 log₂(CCL17) + 0.96 log₂(CCL26) − 0.83 log₂(IL-5)], 0, 72 }',fontsize=6.5)
    schematic.text(0,.04,'BSA 10% = 10. No treatment or prior score input. Full-precision coefficients are retained in the calculator.',fontsize=6.3)
    scatter = [fig.add_subplot(grid[1,j]) for j in range(3)]
    for ax, week, color in zip(scatter,base.TIMES,base.TIME_COLORS):
        d=bins[bins.visit_week.eq(week)]
        high=max(10,np.ceil(max(d.observed_bin.max(),d.predicted_bin.max())/10)*10)
        ax.plot([0,high],[0,high],ls='--',color='#9BA6AD',lw=.8)
        ax.scatter(d.observed_bin,d.predicted_bin,s=9*d['count'],color=base.STYLE['neutral_dark'],alpha=.65,lw=0)
        ax.set(xlim=(-.8,high+.8),ylim=(-.8,high+.8),xlabel='Observed EASI (points)',ylabel='Estimated EASI (points)',
               title=f'Week {week} · {int(metrics.loc[week,"n_patients"])} patients\n'+('Temporal application · repetition 1' if week==24 else 'Held-out repetition 1'))
        clean(ax)
    fig.text(.09,.758,'Scatter coordinates are rounded to one EASI point.',fontsize=6.3)
    count_key(fig,(.78,.795))
    curve=fig.add_subplot(grid[2,:2]);numbers=fig.add_subplot(grid[2,2]);numbers.set_axis_off()
    for week,color,style in zip(base.TIMES,base.TIME_COLORS,base.TIME_STYLES):
        d=ecdf[ecdf.week.eq(week)&ecdf.absolute_error.le(10)].sort_values('absolute_error')
        curve.fill_between(d.absolute_error.to_numpy(),100*d.ci_low.to_numpy(),100*d.ci_high.to_numpy(),color=color,alpha=.10,lw=0)
        curve.plot(d.absolute_error,100*d.proportion,color=color,lw=1.25,ls=style,label=f'Week {week}')
    curve.axvline(2,color='#B9C2C7',lw=.6,ls=':')
    curve.set(xlim=(0,10),ylim=(0,101),yticks=[0,25,50,75,100],xlabel='Absolute EASI error threshold (points)',
              ylabel='Predictions within threshold (%)',title='Individual error · both validation repetitions')
    curve.legend(loc='lower right',fontsize=6.5,frameon=False);clean(curve)
    numbers.text(0,1.07,'Primary results · both repetitions',fontsize=6.5,weight='bold',va='top')
    numbers.text(0,.96,'MAE [95% CI], points',fontsize=6.3,va='top')
    for i,week in enumerate(base.TIMES):
        r=metrics.loc[week];y=.78-i*.30
        numbers.text(0,y,f'Week {week}',fontsize=7,weight='bold')
        numbers.text(0,y-.09,f'{r.MAE:.2f} [{r.MAE_lo:.2f}, {r.MAE_hi:.2f}]',fontsize=7)
        numbers.text(0,y-.18,f'EASI >21: n = {dict(zip(base.TIMES,[44,1,0]))[week]}',fontsize=6.3)
    for i,ax in enumerate([schematic]+scatter+[curve,numbers]):label(ax,string.ascii_lowercase[i])
    base.save(fig,'Figure3','A common four-input equation estimates individual EASI readings',
              'a, The final common equation, shown with scaled two-decimal coefficients for readability; the calculator retains full precision. '
              'BSA is entered as percent (10% as 10), molecular concentrations as positive pg/mL values in the study export, and output is clipped to 0–72. '
              'No additional dilution or assay-conversion factor is applied. Exact blood/clinical timing within a nominal visit is not fully verified. '
              'No treatment, visit or previous score enters the equation. b–d, First patient-held-out repetition, with coordinates rounded to '
              'one EASI point and symbol area proportional to count. Dashed lines show exact agreement; axes cover the observed range at each visit. '
              'e, Cumulative absolute-error distribution, computed within each validation repetition and then averaged; bands are patient-bootstrap '
              '95% confidence intervals; thresholds from 0 to 10 points are displayed, with all errors contributing to each proportion. '
              'The two-point line is a reading reference, not an accepted replacement margin. f, Primary nested-validation '
              'MAE and patient-bootstrap intervals, conditional on saved held-out predictions. These describe the nested fitting procedure, '
              'not an independent test of the final all-development coefficient vector. W0/W12 used patient-grouped model selection; W24 is '
              'previously inspected same-cohort temporal application. EASI >21 counts show the marked change in severity mix. '
              'Full numerical metrics, severity strata, calibration, area-only controls and assay sensitivity analyses are supplied in source tables. '
              'Added predictive accuracy from the molecular terms has not been established.',
              [str(folder/'model_plot_bins.csv'),str(ecdf_path),str(root/'results/tables/update_20261017/model/overall_metrics.csv')],root)


def monitoring(root):
    folder = root/'results/tables/update_20261017/redesign_20261018'
    changes = pd.read_csv(folder/'monitoring_change_bins_repeat0.csv')
    response = pd.read_csv(folder/'monitoring_response_uncertainty.csv')
    counts = pd.read_csv(folder/'monitoring_confusion_counts.csv')
    metrics = pd.read_csv(folder/'monitoring_change_uncertainty.csv')
    fig = plt.figure(figsize=(183/25.4,160/25.4))
    fig.text(.53,.985,'Paired-change plots: held-out repetition 1. Summary performance: both repetitions.',ha='center',fontsize=6.3)
    grid = fig.add_gridspec(2,6,left=.09,right=.975,bottom=.10,top=.92,
                           wspace=1.2,hspace=.62,height_ratios=[1.1,1])
    axes=[fig.add_subplot(grid[0,k:k+2]) for k in [0,2,4]]
    for ax,window,color in zip(axes,['W0_W12','W0_W24','W12_W24'],[base.TIME_COLORS[1],base.TIME_COLORS[2],base.STYLE['neutral_dark']]):
        d=changes[changes.window.eq(window)]
        low=np.floor(min(d.observed_change_bin.min(),d.predicted_change_bin.min())/5)*5
        high=np.ceil(max(d.observed_change_bin.max(),d.predicted_change_bin.max())/5)*5
        low,high=min(low,-5),max(high,5)
        ax.plot([low,high],[low,high],'--',color='#9BA6AD',lw=.8)
        ax.axhline(0,color='#D9DEE2',lw=.6);ax.axvline(0,color='#D9DEE2',lw=.6)
        ax.scatter(d.observed_change_bin,d.predicted_change_bin,s=9*d['count'],color=color,alpha=.75,lw=0)
        m=metrics[metrics.window.eq(window)&metrics.metric.eq('MAE')].iloc[0]
        ax.set(xlim=(low-1,high+1),ylim=(low-1,high+1),xlabel='Observed EASI change',ylabel='Estimated EASI change',
               title=window.replace('_',' → ')+f' · {int(m.n_patients)} patients')
        ax.text(.04,.95,f'Change MAE {m.estimate:.2f}',transform=ax.transAxes,va='top',fontsize=6.5)
        clean(ax)
    # The lower forest includes its row labels and numeric column within the full panel width.
    ax=fig.add_axes([.27,.10,.46,.30]); axes.append(ax)
    row_labels=[]; positions=[]
    for k,window in enumerate(['W0_W12','W0_W24']):
        week=[12,24][k]; color=base.TIME_COLORS[k+1]; marker=base.TIME_MARKERS[k+1]
        for j,(metric,name) in enumerate([('sensitivity','Sensitivity'),('specificity','Specificity'),('agreement','Agreement')]):
            r=response[response.window.eq(window)&response.endpoint.eq('EASI75')&response.metric.eq(metric)].iloc[0]
            y=k*4+j
            c=counts[counts.window.eq(window)&counts.endpoint.eq('EASI75')&counts['repeat'].eq(0)]
            n=int(c[c.observed_response.eq(1 if metric=='sensitivity' else 0)].n_observed_class.iloc[0]) if metric!='agreement' else int(r.n_patients)
            ax.errorbar(r.estimate*100,y,xerr=[[(r.estimate-r.ci_low)*100],[(r.ci_high-r.estimate)*100]],
                        fmt=marker,color=color,ms=4,lw=1,capsize=2)
            ax.text(1.05,y,f'{100*r.estimate:.1f} [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]',
                    transform=ax.get_yaxis_transform(),va='center',fontsize=6.5)
            row_labels.append(f'W{week}  {name} (n={n})'); positions.append(y)
    ax.text(1.05,1.025,'Estimate [95% CI]',transform=ax.transAxes,fontsize=6.5)
    ax.set(yticks=positions,yticklabels=row_labels,ylim=(6.65,-.65),xlim=(0,100),
           xticks=[0,25,50,75,100],xlabel='EASI75 classification performance (%)',
           title='Primary repeated-validation estimates')
    ax.tick_params(axis='y',length=0); clean(ax)
    fig.text(.09,.475,'Scatter coordinates are rounded to one EASI point.',fontsize=6.3)
    count_key(fig,(.79,.515))
    for i,ax in enumerate(axes):label(ax,string.ascii_lowercase[i])
    base.save(fig,'Figure4','Within-patient change and clinical response from the common equation',
              'a–c, Each point represents paired predictions and observed EASI from the same patient in the first held-out repetition. '
              'Change is end minus start; negative values indicate improvement. Coordinates are rounded to one point; symbol area represents count. '
              'Every window is separate, with its own paired population. Dashed lines show exact agreement. MAE annotations use both validation '
              'repetitions, scoring each before averaging; patient-bootstrap intervals are provided in source tables. '
              'd, EASI75 classification from estimated versus observed percentage improvement from W0, computed separately within each '
              'held-out repetition and then averaged. Intervals are patient-bootstrap 95% confidence intervals conditional on the saved predictions. '
              'Sensitivity denominators are actual responders (W12 n=49; W24 n=54); specificity denominators are nonresponders (36 and 13). '
              'Agreement uses all 85 and 67 evaluable patients. '
              'Only positive observed and predicted W0 values enter percentage-response classification. '
              'All thresholds (EASI50/75/90), excluded counts and both-repetition summary metrics are retained in Supplementary Fig. S38 and its source tables. '
              'W24 is same-cohort temporal application, not external validation. These results assess clinical monitoring, not treatment causality.',
              [str(folder/f) for f in ['monitoring_change_bins_repeat0.csv','monitoring_response_uncertainty.csv','monitoring_confusion_counts.csv','monitoring_change_uncertainty.csv']],root)


def cohort_flow(root):
    folder=root/'results/tables/update_20261017/redesign_20261018'
    nodes=pd.read_csv(folder/'flow_nodes.csv').set_index('node')
    exclusions=pd.read_csv(folder/'model_ordered_exclusions.csv')
    fig,ax=plt.subplots(figsize=(183/25.4,210/25.4));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    ax.set(xlim=(0,1),ylim=(0,1));ax.set_axis_off()
    def box(x,y,w,h,text,accent=False):
        ax.add_patch(Rectangle((x,y),w,h,fc='#F1F3F4' if accent else 'white',
                               ec='#65737C',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7,linespacing=1.4)
    def arrow(start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,color='#65737C',lw=.85))
    def counts(key,unit='visits'):
        r=nodes.loc[key];return f'{int(r.n_patients)} patients; {int(r.n_records)} {unit}'
    ax.text(.015,.98,'a  Clinical treatment comparison',weight='bold',fontsize=8,va='top')
    box(.015,.84,.28,.10,'Linked clinical records\n170 patients\n736 observed visit records')
    box(.365,.84,.255,.10,'Age ≥12 years\n164 patients')
    box(.69,.84,.295,.10,'Initial treatment known\n163 patients\nD 65 · A100 56 · A200 42',True)
    arrow((.295,.89),(.365,.89));arrow((.62,.89),(.69,.89))
    box(.17,.71,.31,.065,'Excluded: age <12\n6 patients')
    box(.57,.71,.31,.065,'Excluded: initial dose uncertain\n1 patient')
    arrow((.33,.89),(.33,.775));arrow((.655,.89),(.655,.775))
    ax.text(.015,.665,'b  Four-input EASI model',weight='bold',fontsize=8,va='top')
    box(.035,.54,.44,.085,'Linked serum records\n'+counts('molecular_source','serum records'))
    box(.035,.425,.44,.085,'W0, W12 or W24 serum records\n'+counts('molecular_target_weeks','serum records'))
    box(.035,.31,.44,.085,'Age ≥12 years\n'+counts('molecular_age_eligible','serum records'))
    box(.035,.185,.44,.085,'Complete EASI, BSA and three markers\n'+counts('model_complete'),True)
    for high,low in [(.54,.51),(.425,.395),(.31,.27)]:arrow((.255,high),(.255,low))
    box(.58,.492,.40,.065,'Excluded visit records\nW48: 24')
    box(.58,.377,.40,.065,'Excluded visit records\nAge <12: 12')
    box(.58,.235,.40,.10,'Excluded sequentially: 24 visits\nMissing EASI 13; BSA 1\nCCL26 8; IL-5 2')
    arrow((.255,.525),(.58,.525));arrow((.255,.410),(.58,.410));arrow((.255,.285),(.58,.285))
    box(.015,.025,.47,.095,'W0/W12 model development\n'+counts('development')+'\nW0 n=152; W12 n=89',True)
    box(.565,.025,.42,.095,'W24 temporal application\n'+counts('temporal')+'\nSame cohort; previously inspected',True)
    ax.plot([.255,.255,.775],[.185,.15,.15],color='#65737C',lw=.85)
    arrow((.255,.15),(.255,.12));arrow((.775,.15),(.775,.12))
    fig.text(.50,.013,'Shared patients: clinical and complete-input model, 159; development and W24, 71.',ha='center',fontsize=6.3)
    base.save(fig,'FigureS39','Patient and data flow through the parallel analyses',
              'Counts are recomputed from the locked clinical, serum and model inputs. The clinical table contains 1,700 scheduled slots '
              '(170 patients × 10 visits), including empty rows; only 736 have recorded visit data. Clinical treatment comparisons include '
              '163 age-eligible patients with an unambiguous initial dose. Serum availability creates a separate branch, not a subset of those '
              '163 patients. Sequential model exclusions count visits once in the displayed order; loss of a visit does not necessarily exclude '
              'the patient. The resulting 160 patients contribute 316 complete observed EASI/BSA/three-marker visits. Development and W24 '
              'have disjoint visit rows but overlapping people; no patient crosses training and test within a validation split. '
              'The two analysis populations share 159 patients: four are clinical-only and one is model-only. Development and W24 share 71 '
              'patients; 85 appear only in development and four only at W24. Fifty-one patients have all three complete model visits. '
              'The source count is not a screened or consented recruitment denominator. All intersections are supplied in linked flow tables. '
              'Missing clinical outcomes are never filled as validation truth.',
              [str(folder/f) for f in ['flow_nodes.csv','flow_edges.csv','flow_exclusions.csv','model_ordered_exclusions.csv','cohort_overlap.csv','model_visit_patterns.csv']],root)


def balance_diagnostics(root):
    folder=root/'results/tables/update_20261017/redesign_20261018'
    d=pd.read_csv(folder/'baseline_balance_full.csv');d=d[d['sample'].eq('clinical_baseline')]
    ps=pd.read_csv(folder/'propensity_overlap_bins.csv')
    ess=pd.read_csv(folder/'analysis_effective_sample_sizes.csv')
    fig=plt.figure(figsize=(183/25.4,155/25.4))
    grid=fig.add_gridspec(2,3,left=.13,right=.975,bottom=.11,top=.89,wspace=.50,hspace=.64)
    axes=[fig.add_subplot(grid[r,c]) for r in range(2) for c in range(3)]
    variables=d.variable.drop_duplicates().tolist()
    names={'easi_total':'Baseline EASI','bmi':'BMI','age':'Age','sex_male':'Male sex','center_HUASHAN':'Huashan centre'}
    pairs=d[['group1','group2']].drop_duplicates().itertuples(index=False,name=None)
    for ax,(one,two) in zip(axes[:3],pairs):
        for analysis,color,marker in [('unweighted','#9AA5AE','o'),('overlap_weighted',base.STYLE['neutral_dark'],'s')]:
            z=d[d.group1.eq(one)&d.group2.eq(two)&d.analysis.eq(analysis)].set_index('variable').reindex(variables)
            ax.plot(z.absolute_SMD,range(len(variables)),marker=marker,color=color,lw=0,ms=3,label='Observed' if analysis=='unweighted' else 'Overlap weighted')
        ax.axvline(.1,color='#AEB9BE',lw=.7,ls=':')
        ax.set(yticks=range(len(variables)),yticklabels=[names.get(v,v.replace('_',' ')) for v in variables],
               ylim=(len(variables)-.5,-.5),xlim=(-.02,.9),xticks=[0,.3,.6,.9],xlabel='Absolute standardized difference',title=SHORT[one]+' versus '+SHORT[two]);clean(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.98),ncol=2,frameon=False)
    for ax,probability_arm in zip(axes[3:],base.ARMS):
        for arm,color in zip(base.ARMS,base.COLORS):
            z=ps[ps.probability_arm.eq(probability_arm)&ps.treatment.eq(arm)].sort_values('bin_low')
            ax.stairs(z.fraction.to_numpy()*100,np.r_[z.bin_low.to_numpy(),z.bin_high.iloc[-1]],color=color,lw=1,ls=['-','--',':'][base.ARMS.index(arm)],label=base.LABELS[base.ARMS.index(arm)])
        ax.set(xlim=(0,1),ylim=(0,30),xlabel='Estimated treatment probability',ylabel='Patients within initial group (%)',title='Probability of '+SHORT[probability_arm]);clean(ax)
    for i,ax in enumerate(axes):label(ax,string.ascii_lowercase[i])
    z=ess[ess['sample'].eq('clinical_baseline')&ess.center.eq('all')]
    summary=' · '.join(SHORT[r.treatment]+f': N {r.n_patients}, ESS {r.ESS:.1f}' for r in z.itertuples())
    fig.text(.55,.015,summary,ha='center',fontsize=6.5)
    fig.legend(*axes[3].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.505),ncol=3,fontsize=6.5,frameon=False)
    base.save(fig,'FigureS40','Baseline balance, treatment overlap and effective sample size',
              'a–c, Absolute standardized mean differences for every covariate in the initial-treatment propensity model, before and after '
              'generalized overlap weighting. Differences use the unweighted pairwise pooled standard deviation in both displays. The dotted '
              '0.10 line is a descriptive balance guide, not a statistical significance threshold. Residual imbalance remains. '
              'd–f, Baseline multinomial treatment probabilities, stratified by actual initial treatment; each distribution uses all patients '
              'in that initial group and sums to 100%. These show common support rather than longitudinal drug exposure. '
              'The footer reports original N and effective sample size, (sum w)²/sum(w²). All centre, sex, age, BMI and EASI variables are shown. '
              'Overlap weighting targets comparable baseline patients; it cannot remove unmeasured confounding or correct outcome-driven dose changes. '
              'Endpoint- and marker-specific availability/ESS diagnostics are retained in the source tables.',
              [str(folder/f) for f in ['baseline_balance_full.csv','propensity_overlap_bins.csv','analysis_effective_sample_sizes.csv']],root)


def run(root):
    root=Path(root).resolve()
    clinical_context(root)
    molecular_landscape(root)
    model_readout(root)
    monitoring(root)
    cohort_flow(root)
    balance_diagnostics(root)

