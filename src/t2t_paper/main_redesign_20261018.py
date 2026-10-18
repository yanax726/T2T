"""Claim-led main figures using the frozen analysis and a shared visual system."""
from pathlib import Path
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import figures as base
from . import palette
from . import countmaps

MODEL = 'unified_selected'
SHORT = {'dupilumab': 'D', 'abrocitinib_100mg': 'A100', 'abrocitinib_200mg': 'A200'}
PANEL = ['CCL17', 'CCL26', 'IL-5']
LINE_WIDTH = 1.35
MARKER_SIZE = 4
BAND_ALPHA = .16


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
        markers, styles = ['o'] * 3, ['-'] * 3
        roles = ['treatment.' + arm for arm in base.ARMS]
    else:
        names, colors = [f'Week {w}' for w in base.TIMES], base.TIME_COLORS
        markers, styles = ['o'] * 3, ['-'] * 3
        roles = [f'visit.W{week}' for week in base.TIMES]
    fig.legend(handles=[Line2D([], [], color=palette.register(fig, role, c), marker=m, ls=s,
                              ms=MARKER_SIZE, lw=LINE_WIDTH, mec=palette.role_style(role)['outline'],
                              mew=.55, label=n)
                        for c, n, m, s, role in zip(colors, names, markers, styles, roles)],
               loc='upper center', bbox_to_anchor=at, ncol=3, frameon=False)


def count_strip(ax, rows, columns, separate_table=False):
    """Align denominator rows to the actual visit positions on the time axis."""
    if separate_table:
        ax.text(.5, -.30, 'Available n by visit', transform=ax.transAxes,
                ha='center', va='center', fontsize=6)
        table = ax.table(cellText=rows, rowLabels=['D', 'A100', 'A200'],
                         colLabels=[f'W{w}' for w in columns], cellLoc='center',
                         bbox=(.02, -.77, .96, .38))
        table.auto_set_font_size(False)
        table.set_fontsize(6.3)
        for cell in table.get_celld().values():
            cell.set_edgecolor('none')
            cell.PAD = .02
        ax.set_xlabel('Week', labelpad=0)
        return
    ax.text(.5, -.30, 'Available n', transform=ax.transAxes, ha='center',
            va='center', fontsize=6)
    for row, name in enumerate(['D', 'A100', 'A200']):
        ypos = -.44 - .12 * row
        ax.text(-.12, ypos, name, transform=ax.transAxes, ha='right',
                va='center', fontsize=6.3, clip_on=False)
        for week, count in zip(columns, rows[row]):
            ax.text(week, ypos, str(count), transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=6.3, clip_on=False)
    ax.set_xlabel('Week', labelpad=0)


def clinical_context(root):
    folder = root / 'results/tables/update_20261017/clinical'
    observed = pd.read_csv(folder / 'clinical_observed.csv')
    rates = pd.read_csv(folder / 'endpoint_rates.csv')
    contrasts = pd.read_csv(folder / 'clinical_trajectory_contrasts.csv')
    # One clinical question, with symptom burden, response depth and direct
    # treatment comparison assigned complementary panel roles.
    fig = plt.figure(figsize=(183 / 25.4, 212 / 25.4))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.32], left=.10, right=.975,
                           bottom=.075, top=.91, wspace=.55, hspace=1.08)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
    axes += [fig.add_subplot(grid[2, :2]), fig.add_subplot(grid[2, 2])]
    legend(fig)
    for ax, variable, title in zip(axes[:3], ['easi_total', 'pp_nrs', 'dlqi'],
                                    ['EASI', 'PP-NRS', 'Adult DLQI']):
        counts = []
        for arm, color in zip(base.ARMS, base.COLORS):
            g = observed[observed.target.eq(variable) & observed.treatment.eq(arm)].sort_values('week')
            wanted = [0, 2, 4, 12, 24] if variable == 'pp_nrs' else [0, 4, 12, 24]
            g = g[g.week.isin(wanted)]
            x = g.week.to_numpy(float)
            ax.fill_between(x, g.q25.to_numpy(float), g.q75.to_numpy(float), color=color, alpha=BAND_ALPHA, lw=0)
            ax.plot(x, g['median'], marker='o', color=color, lw=LINE_WIDTH, ms=MARKER_SIZE,
                    markeredgecolor=palette.role_style('treatment.' + arm)['outline'], markeredgewidth=.55)
            counts.append([int(g.set_index('week').loc[w, 'n']) for w in wanted])
        upper = {'easi_total': 35, 'pp_nrs': 11, 'dlqi': 30}[variable]
        ax.set(title=title, xlabel='Week', ylabel='Score (points)',
               xticks=wanted, xlim=(-1, 25), ylim=(0, upper))
        count_strip(ax, counts, wanted, separate_table=(variable == 'pp_nrs'))
    for ax, target in zip(axes[3:6], ['EASI50', 'EASI75', 'EASI90']):
        counts = []
        wanted = [4, 12, 24]
        for arm, color in zip(base.ARMS, base.COLORS):
            d = rates[rates.target.eq(target) & rates.treatment.eq(arm)
                      & rates.week.isin(wanted)].set_index('week').reindex(wanted)
            if d.rate.isna().any():
                raise ValueError('A specified EASI response visit is unavailable.')
            ax.errorbar(wanted, 100*d.rate,
                        yerr=np.maximum(0, np.vstack([100*(d.rate-d.ci_low), 100*(d.ci_high-d.rate)])),
                        fmt='o-', color=color, lw=LINE_WIDTH, ms=MARKER_SIZE, capsize=1.5,
                        markeredgecolor=palette.role_style('treatment.' + arm)['outline'],
                        markeredgewidth=.55)
            counts.append(d.n_evaluable.astype(int).tolist())
        ax.set(title=target, xlabel='Week', ylabel='Target attainment (%)',
               xticks=wanted, xlim=(-1, 25), ylim=(-2, 103), yticks=[0, 50, 100])
        count_strip(ax, counts, wanted)
    ax = axes[7]
    pairs = [('dupilumab', 'abrocitinib_100mg'), ('dupilumab', 'abrocitinib_200mg'),
             ('abrocitinib_100mg', 'abrocitinib_200mg')]
    for j, week in enumerate([12, 24]):
        palette.register(fig, f'visit.W{week}', base.TIME_COLORS[j + 1])
        subset = contrasts[contrasts.target.eq('easi_total') & contrasts.week.eq(week)
                           & contrasts.analysis.eq('overlap_weighted')]
        for i, (first, second) in enumerate(pairs):
            r = subset[subset.group1.eq(first) & subset.group2.eq(second)].iloc[0]
            ax.errorbar(r.effect, i + (j - .5) * .24,
                        xerr=[[r.effect - r.ci_low], [r.ci_high - r.effect]],
                        fmt='o', color=base.TIME_COLORS[j + 1],
                        ms=MARKER_SIZE, capsize=2, lw=LINE_WIDTH)
    ax.axvline(0, color='#B9C2C7', lw=.7, zorder=0)
    ax.set(yticks=range(3), yticklabels=[SHORT[a] + ' − ' + SHORT[b] for a, b in pairs],
           ylim=(3.3, -.65), title='Adjusted EASI contrast',
           xlabel='EASI difference (points)')
    ax.legend(handles=[Line2D([], [], color=base.TIME_COLORS[j + 1], marker=m, lw=0, ms=MARKER_SIZE,
                              label=f'Week {w}') for j, (w, m) in enumerate([(12, 'o'), (24, 'o')])],
              loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2,
              fontsize=6.3, frameon=False, handlelength=1, columnspacing=.8)
    ax = axes[6]
    positions, names = [], []
    for block, (week, target, heading) in enumerate([(2, 'PP_NRS_le4', 'Week 2: itch target'),
                                                    (4, 'overall', 'Week 4: early target'),
                                                    (12, 'overall', 'Week 12: stringent target'),
                                                    (24, 'overall', 'Week 24: combined target')]):
        start = block * 4.1
        ax.text(0, start - .65, heading, fontsize=6.5, weight='bold', va='top')
        for i, (arm, color) in enumerate(zip(base.ARMS, base.COLORS)):
            r = rates[rates.week.eq(week) & rates.target.eq(target) & rates.treatment.eq(arm)].iloc[0]
            y = start + .75 + i * .94
            ax.errorbar(r.rate * 100, y,
                        xerr=[[(r.rate - r.ci_low) * 100], [(r.ci_high - r.rate) * 100]],
                        fmt='o', color=color, ms=MARKER_SIZE, capsize=1.6, lw=LINE_WIDTH,
                        markeredgecolor=palette.role_style('treatment.' + arm)['outline'], markeredgewidth=.55)
            ax.text(110, y, f'{int(r.k)}/{int(r.n_evaluable)}',
                    va='center', fontsize=6.3)
            positions.append(y); names.append(SHORT[arm])
    ax.text(110, -1.1, 'Achieved/N', fontsize=6.3, va='bottom')
    ax.set(yticks=positions, yticklabels=names, xlim=(0, 145), xticks=[0, 50, 100],
           ylim=(15.5, -.8), title='Stage-specific targets', xlabel='Target attainment (%)')
    ax.spines['bottom'].set_bounds(0, 100)
    ax.tick_params(axis='y', length=0, labelsize=6.3)
    for i, axis in enumerate(axes):
        label(axis, string.ascii_lowercase[i]); clean(axis)
    base.save(fig, 'Figure2', 'Clinical course, response depth and target attainment',
              'a–c, Observed EASI, peak-pruritus numerical rating scale (PP-NRS) and adult Dermatology Life Quality Index (DLQI) '
              'medians with interquartile bands by initial treatment. Adult DLQI requires age ≥18. d–f, Observed EASI50/75/90 '
              'response rates at W4/W12/W24, with Wilson 95% confidence intervals; response requires positive observed baseline EASI. '
              'Counts are in D/A100/A200 order, aligned to actual visits except the separately headed PP-NRS visit table. Lines describe available-case summaries, '
              'which need not retain the same patients over time. g, Observed stage-specific target rates with Wilson intervals and '
              'achieved/evaluable counts. W2 uses PP-NRS ≤4; later combined targets require a physician and a patient criterion, as '
              'defined in Fig. 1. Stage criteria differ, so these points are not joined. Missing outcomes remain unknown rather than '
              'observed failures. h, EASI differences from the common-reference generalized-overlap-weighted model, adjusted for baseline '
              'EASI, age, sex, BMI and centre, with patient-cluster-robust 95% confidence intervals. Negative values favour the first '
              'named group for lower EASI; colour distinguishes W12/W24. D, dupilumab; A100/A200, initial abrocitinib 100/200 mg. '
              'Treatment was clinician directed, and later regimens could change. These are initial-strategy associations, not randomized '
              'fixed-dose effects. Remaining outcomes, IGA composition, actual regimens, missing-outcome bounds and complete adjusted '
              'target contrasts are retained in the supplement.',
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
    cmap = palette.cmap('ocean_coral_diverging')
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
            count_label = '×' if missing else str(int(nvalues[i,j])) + ('†' if 0 < nvalues[i,j] < 5 else '')
            left.text(j, i, count_label, ha='center', va='center',
                      fontsize=6.3, color='white' if dark_fill else '#25343F')
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
    base.save(fig, 'Figure3', 'Molecular response across treatments and the three-marker input panel',
              'a, Mean paired log₂ concentration changes among each marker/treatment-specific fixed set with measurements at W0, W12 and W24. '
              'W0 is the zero reference; the two displayed columns per treatment show change to W12 and W24. '
              'b, All specified between-treatment contrasts for W0–W12 from HC3-robust linear models, adjusted for starting molecular '
              'concentration, W0 EASI, age, BMI, sex and centre; dots denote the declared FDR-adjusted q<0.05. These use all eligible two-visit pairs, a different population '
              'from panel a. D, dupilumab; A100/A200, initial abrocitinib dose; pooled A, the observed mixture of the two doses. '
              'Both heatmaps use the same signed log₂ scale, but panel b is a difference between treatment changes, not a within-group change. '
              'Cool blue denotes a negative value, warm coral a positive value; the near-white centre is zero change/difference. Cells marked × lack an estimate. Panel a prints paired n; '
              '† marks fewer than five pairs; these estimates are descriptive. Each estimand has its own labelled colour bar. '
              'c–e, W0/W12/W24 paired trajectories and 95% t confidence intervals for the three frozen model inputs. Their inclusion is based on '
              'the existing panel, not selection of significant trajectories. Exact marker-specific counts, intervals, all three windows and '
              'the complete comparison families are retained in source tables and supplementary figures.',
              [str(folder / f) for f in ['molecular_three_visit_paired.csv', 'molecular_adjusted_contrasts.csv']], root)


def model_readout(root):
    folder = root / 'manuscript/source_data'
    cells = pd.read_csv(folder / 'model_count_cells.csv')
    ecdf_path = folder / 'error_ecdf.csv'
    ecdf = pd.read_csv(ecdf_path)
    metric_path = root / 'results/tables/update_20261017/model/model_metrics.csv'
    metrics = pd.read_csv(metric_path)
    severity = metrics[metrics.model.eq(MODEL) & metrics.week.eq(0)
                       & metrics.group_type.eq('severity')].set_index('group')
    expected = {0: 152, 12: 89, 24: 75}
    actual = cells.groupby('visit_week')['count'].sum().astype(int).to_dict()
    if actual != expected:
        raise ValueError('The first-repetition EASI count maps must retain every model visit.')
    maximum = next(n for n in [2, 5, 10, 20, 50, 100, 200] if n >= cells['count'].max())
    high = np.ceil((cells[['cell_x', 'cell_y']].to_numpy().max() + 2) / 10) * 10
    fig = plt.figure(figsize=(183/25.4, 160/25.4))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, .90], left=.10, right=.975,
                           bottom=.12, top=.91, wspace=1.15, hspace=.92)
    axes = [fig.add_subplot(grid[0, k:k+2]) for k in [0, 2, 4]]
    for ax, week in zip(axes, base.TIMES):
        d = cells[cells.visit_week.eq(week)]
        image = countmaps.draw(ax, d, 'cell_x', 'cell_y', maximum=maximum)
        ax.plot([0, high], [0, high], color='#AAB5BC', lw=.65)
        ax.set(xlim=(0, high), ylim=(0, high), aspect='equal',
               xlabel='Observed EASI (points)', ylabel='Estimated EASI (points)',
               title=f'Week {week} · n={expected[week]}',
               xticks=np.arange(0, high+1, 20), yticks=np.arange(0, high+1, 20))
        clean(ax)
    cax = fig.add_axes([.36, .505, .34, .011])
    countmaps.colorbar(fig, image, cax, maximum, label='Patients per cell')
    curve = fig.add_subplot(grid[1, :3])
    boundary = fig.add_subplot(grid[1, 3:])
    for week, color in zip(base.TIMES, base.TIME_COLORS):
        palette.register(fig, f'visit.W{week}', color)
        d = ecdf[ecdf.week.eq(week) & ecdf.absolute_error.le(10)].sort_values('absolute_error')
        curve.fill_between(d.absolute_error.to_numpy(), 100*d.ci_low.to_numpy(),
                           100*d.ci_high.to_numpy(), color=color, alpha=BAND_ALPHA, lw=0)
        curve.plot(d.absolute_error, 100*d.proportion, color=color, lw=LINE_WIDTH,
                   ls='-', label=f'Week {week}')
    curve.axvline(2, color='#B9C2C7', lw=.65)
    curve.set(xlim=(0, 10), ylim=(0, 101), yticks=[0, 25, 50, 75, 100],
              xlabel='Absolute EASI error (points)', ylabel='Predictions within threshold (%)',
              title='Individual error distribution')
    curve.legend(loc='lower right', fontsize=6.3, frameon=False)
    clean(curve)
    strata = [('EASI_le7', '≤7'), ('EASI_7_to21', '>7 to 21'), ('EASI_gt21', '>21')]
    for ypos, (key, name) in enumerate(strata):
        r = severity.loc[key]
        boundary.errorbar(r.MAE, ypos, xerr=[[r.MAE-r.MAE_lo], [r.MAE_hi-r.MAE]],
                          fmt='o', color=base.TIME_COLORS[0], lw=LINE_WIDTH,
                          ms=MARKER_SIZE, capsize=2)
    boundary.set(yticks=range(3),
                 yticklabels=[f'{name} (n={int(severity.loc[key, "n_patients"])})'
                              for key, name in strata],
                 ylim=(2.6, -.6), xlim=(0, 7), xticks=[0, 2, 4, 6],
                 xlabel='MAE (EASI points)',
                 title='Error by baseline EASI')
    clean(boundary)
    for ax, left in [(curve, .10), (boundary, .64)]:
        position = ax.get_position()
        ax.set_position([left, position.y0, .335, position.height])
    for i, ax in enumerate(axes + [curve, boundary]):
        label(ax, string.ascii_lowercase[i])
    base.save(fig, 'Figure4', 'Individual EASI estimation and its severity boundary',
              'a–c, Estimated versus observed EASI in the first patient-held-out repetition at W0/W12 and same-cohort temporal '
              'application at W24. Each fixed 2×2-EASI-point cell contains all patients whose exact score pair falls within its '
              'boundaries. Colour gives patient count on the shared logarithmic count scale; white cells have no observations. '
              'All 152/89/75 eligible patients are retained, and identical axes preserve the severity-range comparison. Grey lines '
              'denote exact agreement, not a fitted association. d, Cumulative absolute-error proportions, calculated separately '
              'in each validation repetition and then averaged. Bands are patient-bootstrap 95% confidence intervals conditional '
              'on saved predictions. All errors enter each proportion; only thresholds from zero to ten points are displayed. '
              'The two-point line is a reading reference, not an accepted replacement margin. e, Baseline MAE by observed EASI '
              'stratum, with patient-bootstrap 95% confidence intervals and unique-patient counts. These estimates average the '
              'two repetition-specific metrics. W12/W24 have only one/no EASI>21 observation, respectively. W0/W12 use nested '
              'patient-level structure and penalty selection; earlier marker selection is not repeated. W24 was previously '
              'inspected within the same cohort. These results assess the development procedure, not independent validation of '
              'the final formula. Primary overall metrics and numerical controls appear in Table 2; molecular added accuracy '
              'has not been established.',
              [str(folder/'model_count_cells.csv'), str(ecdf_path), str(metric_path)], root)


def monitoring(root):
    folder = root / 'results/tables/update_20261017/redesign_20261018'
    cell_path = root / 'manuscript/source_data/monitoring_count_cells.csv'
    changes = pd.read_csv(cell_path)
    response = pd.read_csv(folder/'monitoring_response_uncertainty.csv')
    counts = pd.read_csv(folder/'monitoring_confusion_counts.csv')
    metrics = pd.read_csv(folder/'monitoring_change_uncertainty.csv')
    windows = ['W0_W12', 'W0_W24', 'W12_W24']
    expected = {'W0_W12': 85, 'W0_W24': 67, 'W12_W24': 55}
    if changes.groupby('window')['count'].sum().astype(int).to_dict() != expected:
        raise ValueError('The first-repetition change maps must retain every paired patient.')
    maximum = next(n for n in [2, 5, 10, 20, 50, 100, 200] if n >= changes['count'].max())
    coordinates = changes[['cell_x', 'cell_y']].to_numpy()
    low = min(-5, np.floor(coordinates.min()/10)*10)
    high = max(5, np.ceil((coordinates.max()+2)/10)*10)
    fig = plt.figure(figsize=(183/25.4, 162/25.4))
    grid = fig.add_gridspec(2, 6, left=.10, right=.975, bottom=.105, top=.91,
                           wspace=1.15, hspace=.95, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(grid[0, k:k+2]) for k in [0, 2, 4]]
    for ax, window in zip(axes, windows):
        d = changes[changes.window.eq(window)]
        image = countmaps.draw(ax, d, 'cell_x', 'cell_y', maximum=maximum)
        ax.plot([low, high], [low, high], '-', color='#AAB5BC', lw=.65)
        ax.axhline(0, color='#D9DEE2', lw=.5)
        ax.axvline(0, color='#D9DEE2', lw=.5)
        ax.set(xlim=(low, high), ylim=(low, high), aspect='equal',
               xlabel='Observed EASI change', ylabel='Estimated EASI change',
               title=window.replace('_', ' → ')+f' · n={expected[window]}',
               xticks=np.arange(np.ceil(low/20)*20, high+1, 20),
               yticks=np.arange(np.ceil(low/20)*20, high+1, 20))
        clean(ax)
    cax = fig.add_axes([.36, .505, .34, .011])
    countmaps.colorbar(fig, image, cax, maximum, label='Patients per cell')
    # Leave room on the left for response-class denominators and on the right
    # for the numerical estimates, without squeezing the three count maps.
    ax = fig.add_subplot(grid[1, 1:5])
    axes.append(ax)
    row_labels, positions = [], []
    for k, window in enumerate(['W0_W12', 'W0_W24']):
        week = [12, 24][k]
        color = palette.register(fig, f'visit.W{week}', base.TIME_COLORS[k+1])
        for j, (metric, name) in enumerate([('sensitivity', 'Sensitivity'),
                                           ('specificity', 'Specificity'),
                                           ('agreement', 'Agreement')]):
            r = response[response.window.eq(window) & response.endpoint.eq('EASI75')
                         & response.metric.eq(metric)].iloc[0]
            y = k*4+j
            c = counts[counts.window.eq(window) & counts.endpoint.eq('EASI75') & counts['repeat'].eq(0)]
            n = (int(c[c.observed_response.eq(1 if metric == 'sensitivity' else 0)]
                     .n_observed_class.iloc[0]) if metric != 'agreement' else int(r.n_patients))
            ax.errorbar(100*r.estimate, y,
                        xerr=[[100*(r.estimate-r.ci_low)], [100*(r.ci_high-r.estimate)]],
                        fmt='o', color=color, ms=MARKER_SIZE, lw=LINE_WIDTH, capsize=2)
            ax.text(1.04, y, f'{100*r.estimate:.1f} [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=6.3)
            row_labels.append(f'W{week}  {name} (n={n})')
            positions.append(y)
    ax.text(1.04, 1.025, 'Estimate [95% CI]', transform=ax.transAxes, fontsize=6.3)
    ax.set(yticks=positions, yticklabels=row_labels, ylim=(6.65, -.65), xlim=(0, 100),
           xticks=[0, 25, 50, 75, 100], xlabel='EASI75 classification performance (%)',
           title='Estimated versus observed response')
    ax.tick_params(axis='y', length=0)
    clean(ax)
    for i, ax in enumerate(axes):
        label(ax, string.ascii_lowercase[i])
    base.save(fig, 'Figure5', 'Within-patient change and EASI75 response classification',
              'a–c, Estimated versus observed EASI changes in the first patient-held-out repetition. Change is end minus start, '
              'so negative values indicate improvement. Fixed 2×2-EASI-point cells count exact paired score changes, using one '
              'logarithmic count scale and identical axes across intervals; white cells contain no observations. All 85, 67 and '
              '55 paired patients are retained in their respective windows, which can share patients. Grey diagonal lines denote '
              'exact agreement; horizontal and vertical lines mark no change. Primary change MAEs over W0–W12, W0–W24 and '
              'W12–W24 are 3.87, 3.47 and 1.49 points, respectively, with intervals in the source tables. These metrics score '
              'each of two repetitions before averaging, whereas the count maps illustrate one repetition. d, EASI75 sensitivity, '
              'specificity and overall agreement, averaged after classification within each repetition, with patient-bootstrap '
              '95% confidence intervals conditional on saved predictions. Sensitivity denominators are observed responders '
              '(W12 n=49; W24 n=54); specificity denominators are nonresponders (36 and 13). Agreement includes 85 and 67 patients. '
              'Observed and estimated response each use their corresponding W0 and follow-up scores; percentage response '
              'requires positive observed and estimated W0. W24 is previously inspected same-cohort temporal application, not '
              'external validation. Complete EASI50/75/90 classifications and exclusions remain in the supplement. These results '
              'assess monitoring agreement, not treatment causality or clinical interchangeability.',
              [str(cell_path)] + [str(folder/f) for f in ['monitoring_response_uncertainty.csv',
               'monitoring_confusion_counts.csv', 'monitoring_change_uncertainty.csv']], root)


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
        for analysis,role in [('unweighted','method.reference'),('overlap_weighted','method.current')]:
            color=palette.register(fig,role); marker='o'
            z=d[d.group1.eq(one)&d.group2.eq(two)&d.analysis.eq(analysis)].set_index('variable').reindex(variables)
            ax.plot(z.absolute_SMD,range(len(variables)),marker=marker,color=color,lw=0,ms=MARKER_SIZE,label='Observed' if analysis=='unweighted' else 'Overlap weighted')
        ax.axvline(.1,color='#B9C2C7',lw=.7,ls='-')
        ax.set(yticks=range(len(variables)),yticklabels=[names.get(v,v.replace('_',' ')) for v in variables],
               ylim=(len(variables)-.5,-.5),xlim=(-.02,.9),xticks=[0,.3,.6,.9],xlabel='Absolute standardized difference',title=SHORT[one]+' versus '+SHORT[two]);clean(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.98),ncol=2,frameon=False)
    for ax,probability_arm in zip(axes[3:],base.ARMS):
        for arm,color in zip(base.ARMS,base.COLORS):
            z=ps[ps.probability_arm.eq(probability_arm)&ps.treatment.eq(arm)].sort_values('bin_low')
            ax.stairs(z.fraction.to_numpy()*100,np.r_[z.bin_low.to_numpy(),z.bin_high.iloc[-1]],color=color,lw=LINE_WIDTH,ls='-',label=base.LABELS[base.ARMS.index(arm)])
        ax.set(xlim=(0,1),ylim=(0,30),xlabel='Estimated treatment probability',ylabel='Patients within initial group (%)',title='Probability of '+SHORT[probability_arm]);clean(ax)
    for i,ax in enumerate(axes):label(ax,string.ascii_lowercase[i])
    z=ess[ess['sample'].eq('clinical_baseline')&ess.center.eq('all')]
    summary=' · '.join(SHORT[r.treatment]+f': N {r.n_patients}, ESS {r.ESS:.1f}' for r in z.itertuples())
    fig.text(.55,.015,summary,ha='center',fontsize=6.5)
    fig.legend(*axes[3].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.505),ncol=3,fontsize=6.5,frameon=False)
    base.save(fig,'FigureS40','Baseline balance, treatment overlap and effective sample size',
              'a–c, Absolute standardized mean differences for every covariate in the initial-treatment propensity model, before and after '
              'generalized overlap weighting. Differences use the unweighted pairwise pooled standard deviation in both displays. The light '
              '0.10 reference line is a descriptive balance guide, not a statistical significance threshold. Residual imbalance remains. '
              'd–f, Baseline multinomial treatment probabilities, stratified by actual initial treatment; each distribution uses all patients '
              'in that initial group and sums to 100%. These show common support rather than longitudinal drug exposure. '
              'The footer reports original N and effective sample size, (sum w)²/sum(w²). All centre, sex, age, BMI and EASI variables are shown. '
              'Overlap weighting targets comparable baseline patients; it cannot remove unmeasured confounding or correct outcome-driven dose changes. '
              'Endpoint- and marker-specific availability/ESS diagnostics are retained in the source tables.',
              [str(folder/f) for f in ['baseline_balance_full.csv','propensity_overlap_bins.csv','analysis_effective_sample_sizes.csv']],root)


def run(root):
    from .clinical_framework_20261018 import framework, cohort_flow
    root=Path(root).resolve()
    framework(root, base)
    clinical_context(root)
    molecular_landscape(root)
    model_readout(root)
    monitoring(root)
    cohort_flow(root, base)
    balance_diagnostics(root)

