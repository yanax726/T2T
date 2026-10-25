"""Detailed model diagnostics retained outside the main response displays."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import figures as base
from . import countmaps
from . import main_redesign_20261018 as previous


def individual_scores(root):
    path = root / 'manuscript/source_data/model_count_cells.csv'
    cells = pd.read_csv(path)
    expected = {0: 152, 12: 89, 24: 75}
    assert cells.groupby('visit_week')['count'].sum().astype(int).to_dict() == expected
    maximum = next(n for n in [2, 5, 10, 20, 50, 100, 200] if n >= cells['count'].max())
    high = np.ceil((cells[['cell_x', 'cell_y']].to_numpy().max() + 2) / 10) * 10
    fig = plt.figure(figsize=(183/25.4, 107/25.4))
    grid = fig.add_gridspec(1, 3, left=.10, right=.975, bottom=.34, top=.88, wspace=.55)
    for i, week in enumerate(base.TIMES):
        ax = fig.add_subplot(grid[0, i])
        d = cells[cells.visit_week.eq(week)]
        image = countmaps.draw(ax, d, 'cell_x', 'cell_y', maximum=maximum)
        ax.plot([0, high], [0, high], color='#AAB5BC', lw=.65)
        ax.set(xlim=(0, high), ylim=(0, high), aspect='equal',
               xlabel='Observed EASI (points)', ylabel='Estimated EASI (points)',
               title=f'Week {week} · n={expected[week]}',
               xticks=np.arange(0, high+1, 20), yticks=np.arange(0, high+1, 20))
        previous.label(ax, 'abc'[i]); previous.clean(ax)
    countmaps.colorbar(fig, image, fig.add_axes([.36, .14, .34, .018]), maximum)
    base.save(fig, 'FigureS44', 'Observed and estimated EASI at each visit',
              'a–c, Estimated versus observed EASI in the first patient-held-out repetition at W0/W12 and same-cohort temporal '
              'application at W24. Each fixed 2×2-EASI-point cell counts all exact score pairs within its boundaries. '
              'Colour gives patient count on one shared logarithmic scale; white cells contain no observations. '
              'All 152/89/75 eligible patients are retained. Identical axes preserve the severity-range comparison; '
              'grey diagonals denote exact agreement. W0/W12 use nested patient-level structure and penalty selection; '
              'earlier marker selection is not repeated. W24 had previously been inspected. These results assess the development '
              'procedure rather than independent validation of the final coefficient vector. Overall performance is in Table 2; '
              'the main figure displays the complete absolute-error distribution.', [str(path)], root)


def paired_changes(root):
    path = root / 'manuscript/source_data/monitoring_count_cells.csv'
    metric_path = root / 'results/tables/update_20261017/redesign_20261018/monitoring_change_uncertainty.csv'
    changes, metrics = pd.read_csv(path), pd.read_csv(metric_path)
    expected = {'W0_W12': 85, 'W0_W24': 67, 'W12_W24': 55}
    assert changes.groupby('window')['count'].sum().astype(int).to_dict() == expected
    maximum = next(n for n in [2, 5, 10, 20, 50, 100, 200] if n >= changes['count'].max())
    coordinates = changes[['cell_x', 'cell_y']].to_numpy()
    low = min(-5, np.floor(coordinates.min()/10)*10)
    high = max(5, np.ceil((coordinates.max()+2)/10)*10)
    fig = plt.figure(figsize=(183/25.4, 152/25.4))
    grid = fig.add_gridspec(2, 3, left=.10, right=.975, bottom=.12, top=.91,
                           wspace=.55, hspace=.95, height_ratios=[1, .8])
    for i, (window, n) in enumerate(expected.items()):
        ax = fig.add_subplot(grid[0, i])
        d = changes[changes.window.eq(window)]
        image = countmaps.draw(ax, d, 'cell_x', 'cell_y', maximum=maximum)
        ax.plot([low, high], [low, high], color='#AAB5BC', lw=.65)
        ax.axhline(0, color='#D9DEE2', lw=.5); ax.axvline(0, color='#D9DEE2', lw=.5)
        ax.set(xlim=(low, high), ylim=(low, high), aspect='equal',
               xlabel='Observed EASI change', ylabel='Estimated EASI change',
               title=window.replace('_', ' → ')+f' · n={n}',
               xticks=np.arange(np.ceil(low/20)*20, high+1, 20),
               yticks=np.arange(np.ceil(low/20)*20, high+1, 20))
        previous.clean(ax); previous.label(ax, 'abc'[i])
    countmaps.colorbar(fig, image, fig.add_axes([.36, .49, .34, .012]), maximum)
    lower = fig.add_gridspec(1, 1, left=.21, right=.77, bottom=.12, top=.365)
    ax = fig.add_subplot(lower[0, 0])
    for i, window in enumerate(expected):
        row = metrics[metrics.window.eq(window) & metrics.metric.eq('MAE')].iloc[0]
        ax.errorbar(row.estimate, i, xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                    fmt='o', color='#536875', lw=1.35, ms=4, capsize=2)
        ax.text(1.04, i, f'{row.estimate:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]',
                transform=ax.get_yaxis_transform(), va='center', fontsize=6.5)
    ax.set(yticks=range(3), yticklabels=[w.replace('_', ' → ') for w in expected],
           ylim=(2.6, -.6), xlim=(0, 5), xticks=range(6),
           xlabel='Mean absolute error of EASI change (points)', title='Paired-change error')
    ax.text(1.04, 1.03, 'MAE [95% CI]', transform=ax.transAxes, fontsize=6.5)
    previous.clean(ax); previous.label(ax, 'd')
    base.save(fig, 'FigureS45', 'Within-patient EASI change and absolute error',
              'a–c, Estimated versus observed EASI changes in the first patient-held-out repetition. Change is end minus start: '
              'negative values indicate improvement. Fixed 2×2-EASI-point cells count exact paired changes on one logarithmic '
              'count scale. Identical axes retain all 85, 67 and 55 pairs; the intervals can share patients and are not consecutive '
              'categories on a time axis. White cells contain no observations; diagonals mark exact agreement. d, Change MAE '
              'calculated within each of two repetitions before averaging, with 2,000 patient-bootstrap 95% confidence intervals '
              'conditional on saved predictions. Both scores are paired within each repetition before subtraction. W24 is '
              'previously inspected same-cohort temporal application. EASI75 classification is shown in the main model figure; '
              'complete EASI50/75/90 classifications are retained separately.', [str(path), str(metric_path)], root)


def supporting_figures(root):
    from .clinical_flow_20261025 import cohort_flow
    cohort_flow(root, base)
    previous.balance_diagnostics(root)
    individual_scores(root)
    paired_changes(root)
