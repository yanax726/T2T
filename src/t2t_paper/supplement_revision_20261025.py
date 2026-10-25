"""Focused supplementary figures for the clinical and molecular manuscript."""
from pathlib import Path
import json
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from . import palette
from . import supplement_redesign_20261018 as previous

SOURCES = [3, 4, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 23, 24, 25, 26, 38]
ARMS = previous.ARMS
LABELS = previous.LABELS
MARKERS = previous.MARKERS
TIMES = [0, 12, 24]
REGIONS = ['head_neck', 'upper_limb', 'trunk', 'lower_limb']
REGION_LABELS = ['Head/neck', 'Upper limbs', 'Trunk', 'Lower limbs']
SHORT = ['D', 'A100', 'A200']


def initialize():
    previous.COLORS = [palette.color('treatment.' + x) for x in ARMS]
    previous.TIME_COLORS = [palette.color('visit.W' + str(w)) for w in TIMES]
    previous.TIME_MARKERS = ['o'] * 3
    previous.TREATMENT_MARKERS = ['o'] * 3
    previous.CURRENT = palette.color('method.current')
    previous.CONTROL = palette.color('method.reference')
    previous.POOLED = palette.color('treatment.pooled_abrocitinib')
    previous.CMAP = palette.cmap('ocean_coral_diverging')
    previous.PROPORTION_CMAP = palette.cmap('ocean_sequential')
    previous.IGA_COLORS = palette.sequence('recorded_iga_0_to_5')


def finish(fig, number, title, caption, sources, root, base):
    previous.finish(fig, number, title, caption, sources, root, base,
                    title, 'See labelled axes', 'See labelled axes',
                    'Patient-level available observations; source-specific denominators')


def light_grid(ax):
    ax.grid(axis='y', color='#E7EBED', lw=.35, zorder=0)
    ax.tick_params(length=2.5, width=.6)


def courses(root, base):
    folder = root / 'results/tables/update_20261017/clinical'
    observed = pd.read_csv(folder / 'clinical_observed.csv')
    adjusted = pd.read_csv(folder / 'clinical_trajectory_adjusted.csv')
    groups = [(3, ['bsa', 'scorad']), (4, ['ige_numeric', 'eo_count_numeric'])]
    for number, variables in groups:
        fig, axes = previous.grid(len(variables), 2, 195 if number == 2 else 151,
                                  left=.13, top=.90, bottom=.13, hspace=.72, wspace=.40)
        previous.legend(fig)
        for row, var in enumerate(variables):
            weeks = [0, 12, 24] if var == 'ige_numeric' else [0, 4, 12, 24]
            if var == 'pp_nrs':
                weeks = [0, 2, 4, 12, 24]
            obsax, adjax = axes[row * 2:row * 2 + 2]
            for j, tx in enumerate(ARMS):
                z = observed[observed.target.eq(var) & observed.treatment.eq(tx)].set_index('week').reindex(weeks)
                # Explicit missing rows break the line: absent W4 eosinophils are not interpolated.
                previous.line(obsax, weeks, z['median'], z.q25, z.q75, previous.COLORS[j])
                z = adjusted[adjusted.target.eq(var) & adjusted.treatment.eq(tx)].set_index('week').reindex(weeks[1:])
                v = z[['adjusted_mean_scale', 'ci_low', 'ci_high']].to_numpy()
                if var in ['ige_numeric', 'eo_count_numeric']:
                    v = np.expm1(v)
                previous.line(adjax, np.array(weeks[1:]) + (j - 1) * .20,
                              v[:, 0], v[:, 1], v[:, 2], previous.COLORS[j])
            for ax, x, title in [(obsax, weeks, 'Observed median and IQR'),
                                  (adjax, weeks[1:], 'Baseline-adjusted estimate and 95% CI')]:
                ax.set(xticks=x, xlabel='Week', ylabel=previous.DISPLAY[var],
                       title=title if row == 0 else '')
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                ax.set_xlim(min(x) - 1, 25)
                light_grid(ax)
        caption = ('Left, observed medians with interquartile ranges by initial treatment; all available values are used at each specified visit. '
                   'Right, generalized-overlap-weighted marginal estimates with patient-cluster robust 95% confidence intervals, '
                   'adjusted for baseline outcome, age, sex, BMI and centre. Patient sets can differ between visits. '
                   'Exact outcome-specific counts and numerical comparisons are in the source tables; they are not replaced by a common denominator. ')
        if number == 2:
            caption += 'Adult DLQI is restricted to age ≥18. '
        if number == 4:
            caption += ('IgE and eosinophils use log(1+x) models and back-transformation; adjusted estimates are not arithmetic means. '
                        'IgE is shown only at W0/W12/W24. No dupilumab W4 eosinophil measurement is available; both the observed and adjusted '
                        'line break at that visit. No multiple-imputation value is displayed as an observation. '
                        'The separate same-patient IgE analysis tests the effect of changing sample composition. ')
        caption += 'Initial treatment groups describe observational management strategies, including subsequent regimen changes, rather than randomized fixed-dose effects.'
        finish(fig, number, 'Observed and adjusted ' + ', '.join(previous.DISPLAY[v].replace('\n', ' ') for v in variables),
               caption, [folder / 'clinical_observed.csv', folder / 'clinical_trajectory_adjusted.csv'], root, base)


def molecule_windows(root, base):
    folder = root / 'results/tables/update_20261017/clinical'
    source = folder / 'molecular_paired_changes.csv'
    data = pd.read_csv(source)
    for number, start, end in [(6, 12, 24), (7, 0, 24)]:
        d = data[data.start.eq(start) & data.end.eq(end)]
        fig, axes = previous.grid(1, 3, 176, left=.16, top=.91, bottom=.12, wspace=.40)
        bounds = (min(-.4, d[['effect', 'ci_low']].min().min() - .15),
                  max(.4, d[['effect', 'ci_high']].max().max() + .15))
        for j, (ax, tx) in enumerate(zip(axes, ARMS)):
            z = d[d.treatment.eq(tx)].set_index('marker').reindex(MARKERS)
            previous.forest(ax, z, np.arange(23), previous.COLORS[j])
            ax.axvline(0, color='#AAB6BC', lw=.6)
            ax.set(yticks=range(23), yticklabels=MARKERS if j == 0 else [''] * 23,
                   ylim=(22.6, -.6), xlim=bounds, xlabel='Mean log₂ change', title=LABELS[j])
            ax.grid(axis='y', color='#EDF0F2', lw=.35)
            for y, r in enumerate(z.itertuples()):
                if np.isfinite(r.q) and r.q < .05:
                    ax.text(1.025, y, '*', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=10)
        previous.window_header(fig, f'W{start} → W{end}: within-patient molecular change')
        finish(fig, number, f'Molecular changes from W{start} to W{end}',
               'Mean paired log₂ concentration changes with 95% t confidence intervals. Negative values indicate lower later concentration. '
               'Each marker/treatment uses all available pairs for this interval; exact counts are in the source table. All 23 measured markers are retained. '
               'Estimates based on fewer than five pairs are descriptive and have no inferential interval; unavailable estimates are blank. '
               'Stars denote BH q<0.05 across the full 207 marker-by-treatment-by-window family, not a separate family for the displayed interval.',
               [source], root, base)
    source = folder / 'molecular_adjusted_contrasts.csv'
    data = pd.read_csv(source)
    for number, start, end in [(8, 0, 12), (9, 12, 24), (10, 0, 24)]:
        comparisons = ['D - A100', 'D - A200'] if number == 8 else ['D - A100', 'D - A200', 'A200 - A100', 'D - pooled A']
        d = data[data.start.eq(start) & data.end.eq(end) & data.scope.eq('all_centers') & data.analysis.eq('primary')]
        fig, axes = previous.grid(1, len(comparisons), 176, left=.17, top=.90, bottom=.12, wspace=.30)
        bound = np.ceil(np.nanmax(np.abs(d[['ci_low', 'ci_high']].to_numpy())) * 2) / 2
        for j, (ax, comp) in enumerate(zip(axes, comparisons)):
            z = d[d.comparison.eq(comp)].set_index('marker').reindex(MARKERS)
            previous.forest(ax, z, np.arange(23), previous.CURRENT)
            ax.axvline(0, color='#AAB6BC', lw=.6)
            ax.set(yticks=range(23), yticklabels=MARKERS if j == 0 else [''] * 23,
                   ylim=(22.6, -.6), xlim=(-bound, bound), xlabel='First − second\n(log₂ change)',
                   title=comp.replace(' - ', ' minus\n'))
            ax.grid(axis='y', color='#EDF0F2', lw=.35)
            for y, r in enumerate(z.itertuples()):
                if np.isfinite(r.q) and r.q < .05:
                    ax.text(1.025, y, '*', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=10)
        previous.window_header(fig, f'W{start} → W{end}: baseline-adjusted molecular contrasts')
        finish(fig, number, f'Adjusted molecular comparisons from W{start} to W{end}',
               'Differences in within-patient log₂ changes with HC3 95% confidence intervals. Negative values mean a smaller, more negative change '
               'in the first named treatment. Models adjust starting log₂ concentration, baseline total EASI, age, sex, BMI and centre. '
               'D, dupilumab; A100/A200, initial abrocitinib 100/200 mg. Pooled A retains the observed dose mixture. '
               'Stars denote the original BH q<0.05 family; no multiplicity family is recalculated after display selection. '
               'Blank positions have no estimable contrast. Complete counts, diagnostics, centre/batch sensitivities and tests remain in the source table. '
               + ('The W0–W12 D-minus-pooled-A and A200-minus-A100 comparisons are in the main molecular figure. ' if number == 8 else '')
               + 'These are adjusted observational associations, not randomized current-dose effects.', [source], root, base)


def selected_associations(root, base):
    source = root / 'results/tables/update_20261017/clinical/molecular_clinical_change_correlations.csv'
    full = pd.read_csv(source)
    d = full[full.q.lt(.05) & full.rho.notna()].sort_values(['start', 'end', 'treatment', 'marker', 'outcome']).copy()
    assert len(d) == 6
    fig, axes = previous.grid(1, 1, 122, left=.40, top=.88, bottom=.15)
    ax = axes[0]
    fig.subplots_adjust(right=.89)
    labels = []
    outcome_names = {'easi_total': 'EASI', 'pp_nrs': 'PP-NRS', 'dlqi': 'Adult DLQI', 'bsa': 'BSA'}
    for y, r in enumerate(d.itertuples()):
        color = previous.COLORS[ARMS.index(r.treatment)]
        ax.errorbar(r.rho, y, xerr=[[r.rho - r.ci_low], [r.ci_high - r.rho]],
                    color=color, fmt='o', ms=4, capsize=2, lw=1.35)
        labels.append(f'{r.marker} · {outcome_names[r.outcome]}\n{SHORT[ARMS.index(r.treatment)]} · W{r.start}–W{r.end} · n={r.n}')
        ax.text(1.02, y, f'{r.q:.4f}', transform=ax.get_yaxis_transform(), va='center', fontsize=6.5)
    ax.text(1.02, 1.03, 'FDR q', transform=ax.transAxes, va='bottom', fontsize=6.5)
    ax.set(yticks=range(len(d)), yticklabels=labels, ylim=(len(d) - .5, -.5),
           xlim=(-.05, 1), xlabel='Spearman correlation (95% CI)')
    ax.axvline(0, color='#AAB6BC', lw=.6)
    ax.grid(axis='x', color='#EDF0F2', lw=.35)
    previous.legend(fig)
    finish(fig, 11, 'Molecular and clinical changes associated after multiplicity correction',
           'Every association meeting BH q<0.05 in the full declared change-association family is displayed; selection spans 23 markers, '
           'five clinical outcomes, three initial treatments and three intervals. No W12–W24 association met this threshold. '
           'Points are Spearman correlations between within-patient clinical change and log₂ molecular change, both later minus earlier, '
           'with patient-bootstrap 95% confidence intervals. Positive correlations indicate changes in the same direction. '
           'Labels give paired-patient counts and the interval; colours indicate initial treatment only. Adult DLQI requires age ≥18. '
           'All tested coefficients, non-significant results and unestimable pairs remain in the complete source table. '
           'This significance-selected display is exploratory and is not independent validation, numerical agreement or a causal molecular effect.',
           [source], root, base)


def baseline_heatmap(root, base):
    source = root / 'results/tables/update_20261017/clinical/molecular_distributions.csv'
    d = pd.read_csv(source)
    d = d[d.week.eq(0)]
    fig, axes = previous.grid(1, 3, 182, left=.17, top=.92, bottom=.15, wspace=.22)
    low = np.floor(d.q25.min()); high = np.ceil(d.q75.max())
    cmap = palette.cmap('ocean_sequential')
    for j, (ax, column, title) in enumerate(zip(axes, ['q25', 'median', 'q75'], ['Lower quartile', 'Median', 'Upper quartile'])):
        matrix = d.pivot(index='marker', columns='treatment', values=column).reindex(index=MARKERS, columns=ARMS)
        im = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect='auto', interpolation='none')
        for y in range(23):
            for x in range(3):
                value = matrix.iloc[y, x]
                if pd.notna(value):
                    col = previous.cell_text_color(cmap((value - low) / (high - low)))
                    ax.text(x, y, ('0.0' if abs(value)<.05 else f'{value:.1f}'), ha='center', va='center', fontsize=6.3, color=col)
        ax.set(yticks=range(23), yticklabels=MARKERS if j == 0 else [''] * 23,
               xticks=range(3), xticklabels=SHORT, title=title)
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([.36, .065, .39, .018])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Log₂ recorded concentration', fontsize=6.5)
    cb.ax.tick_params(labelsize=6.3, length=2)
    finish(fig, 18, 'Baseline distributions of all 23 serum markers',
           'Cells display the lower quartile, median and upper quartile of baseline log₂ recorded concentration for each initial treatment. '
           'All three panels use the same numeric colour scale without row normalization. Compare treatments within a marker: concentration '
           'differences between distinct analytes do not indicate differences in biological activity. D denotes dupilumab; A100/A200 initial '
           'abrocitinib 100/200 mg. Every measured marker is retained. The source table includes exact counts, censoring-related availability, '
           'ranges and distribution summaries; quartiles summarize available positive numeric observations and do not replace censored values.',
           [source], root, base)


def triplet_forests(root, base):
    source = root / 'results/tables/update_20261017/clinical/molecular_three_visit_paired.csv'
    d = pd.read_csv(source)
    fig, axes = previous.grid(1, 3, 222, left=.16, top=.91, bottom=.11, wspace=.28)
    previous.legend(fig, [f'Week {w}' for w in TIMES], previous.TIME_COLORS, markers=['o'] * 3)
    bound = np.ceil(np.nanmax(np.abs(d[['ci_low', 'ci_high']].to_numpy())))
    for j, (ax, tx) in enumerate(zip(axes, ARMS)):
        for k, week in enumerate(TIMES):
            z = d[d.treatment.eq(tx) & d.week.eq(week)].set_index('marker').reindex(MARKERS)
            previous.forest(ax, z, np.arange(23) * 1.5 + (k - 1) * .36,
                            previous.TIME_COLORS[k], marker='o')
        ax.set(yticks=np.arange(23) * 1.5, yticklabels=MARKERS if j == 0 else [''] * 23,
               ylim=(34, -.9), xlim=(-bound, bound), xlabel='Log₂ change from W0', title=LABELS[j])
        ax.axvline(0, color='#B7C0C4', lw=.65)
        ax.grid(axis='y', color='#E9EEF0', lw=.4)
    finish(fig, 19, 'Three-visit molecular trajectories in fixed patient sets',
           'Within each marker and treatment, the same patients contribute at W0, W12 and W24. W0 is zero by definition; W12/W24 are mean '
           'within-patient log₂ changes from W0. Light blue, teal and deep blue encode W0/W12/W24 consistently across all panels. '
           'Rows are spaced to separate the three visits; horizontal intervals are 95% t confidence intervals for n≥5. Smaller sets are '
           'descriptive without inferential intervals, and absent estimates remain blank. All 23 markers are retained. Fixed triplet counts '
           'are available in the source table and may differ across markers or treatments, but never across the three displayed visits within a marker/treatment.',
           [source], root, base)


def ige_fixed(root, base):
    source = root / 'results/tables/update_20261017/clinical/clinical_fixed_triplets.csv'
    d = pd.read_csv(source)
    d = d[d.target.eq('ige_numeric')]
    if not d[['median', 'ci_low', 'ci_high']].gt(0).all().all():
        raise ValueError('The fixed-patient IgE log-axis display requires positive estimates and bounds.')
    fig, axes = previous.grid(1, 1, 110, left=.14, top=.81, bottom=.18)
    ax = axes[0]
    labels = []
    for j, tx in enumerate(ARMS):
        z = d[d.treatment.eq(tx)].sort_values('week')
        assert z.n.nunique() == 1
        previous.line(ax, z.week + (j - 1)*.22, z['median'], z.ci_low, z.ci_high, previous.COLORS[j])
        labels.append(f'{LABELS[j]} (n={int(z.n.iloc[0])})')
    ax.set_yscale('log')
    ax.set(xticks=TIMES, xlim=(-1, 25), ylim=(25, 6500), xlabel='Week',
           ylabel='IgE (recorded units; logarithmic scale)', title='The same patients at each visit',
           yticks=[50, 100, 500, 1000, 5000], yticklabels=['50', '100', '500', '1000', '5000'])
    ax.minorticks_off()
    light_grid(ax)
    previous.legend(fig, labels, previous.COLORS, y=.98)
    finish(fig, 21, 'IgE with identical patients at W0, W12 and W24',
           'Medians and patient-bootstrap 95% confidence intervals in fixed complete three-visit sets: 11 dupilumab, 9 A100 and 10 A200 patients. '
           'The same patients contribute at every visit within each treatment. The vertical axis is logarithmic with ticks labelled in the original '
           'recorded units; estimates and interval endpoints are unchanged. Small horizontal offsets separate treatment intervals at the same visit. '
           'No missing value is filled. These fixed-set summaries differ from available-case visit summaries and baseline-adjusted estimates because '
           'their patients and estimands differ. This check describes observation-pattern sensitivity: the small selected subsets neither estimate '
           'a causal treatment effect nor establish arithmetic mean changes. Laboratory units require source confirmation.',
           [source], root, base)


def regional_composite(root, base):
    clinical = root / 'results/tables/update_20261017/clinical'
    regional = root / 'results/tables/update_20261017/regional'
    corr = pd.read_csv(clinical / 'molecular_regional_baseline_correlations.csv')
    burden = pd.read_csv(regional / 'regional_normalized_burden.csv')
    change = pd.read_csv(regional / 'regional_paired_improvement.csv')
    itch = pd.read_csv(regional / 'whole_body_PP_NRS_improvement.csv')
    fig = plt.figure(figsize=(183 / 25.4, 215 / 25.4))
    gs = fig.add_gridspec(12, 12, left=.145, right=.975, top=.91, bottom=.12, wspace=.80, hspace=1.50)
    a = fig.add_subplot(gs[:, :5])
    b = fig.add_subplot(gs[:5, 7:])
    c = fig.add_subplot(gs[6:9, 7:])
    e = fig.add_subplot(gs[10:, 7:])
    axes = [a, b, c, e]
    for ax, label in zip(axes, 'abcd'):
        ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-17, 11), textcoords='offset points',
                    fontsize=8, weight='bold', annotation_clip=False)
    matrix = corr.pivot(index='marker', columns='region', values='rho').reindex(index=MARKERS, columns=REGIONS)
    q = corr.pivot(index='marker', columns='region', values='q').reindex(index=MARKERS, columns=REGIONS)
    texts = [[('*' if q.iloc[y, x] < .05 else '') for x in range(4)] for y in range(23)]
    im = previous.heat(a, matrix, MARKERS, ['Head/\nneck', 'Upper\nlimbs', 'Trunk', 'Lower\nlimbs'], 'Baseline serum–region association', 1, texts)
    cax = a.inset_axes([.15, -.105, .70, .018])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, 0, 1]); cb.set_label('Spearman correlation', fontsize=6.3)
    cax.tick_params(labelsize=6.3, length=2)
    rows = [(tx, region) for tx in ARMS for region in REGIONS]
    values = np.array([[float(burden[burden.scope.eq(tx) & burden.region.eq(region) & burden.week.eq(w)]['mean'].iloc[0]) * 100 for w in TIMES] for tx, region in rows])
    im = b.imshow(values, cmap=previous.PROPORTION_CMAP, vmin=0, vmax=100, aspect='auto', interpolation='none')
    for y in range(12):
        for x in range(3):
            b.text(x, y, f'{values[y,x]:.0f}', ha='center', va='center', fontsize=6.3,
                   color=previous.cell_text_color(previous.PROPORTION_CMAP(values[y,x]/100)))
    b.set(yticks=range(12), yticklabels=[f'{SHORT[ARMS.index(tx)]} · {REGION_LABELS[REGIONS.index(region)]}' for tx, region in rows],
          xticks=range(3), xticklabels=['W0', 'W12', 'W24'], title='Regional burden (% of maximum)')
    for y in [3.5, 7.5]: b.axhline(y, color='white', lw=2)
    cbax = b.inset_axes([.14, -.095, .72, .025]); cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=[0, 50, 100]); cbax.tick_params(labelsize=6.3, length=2)
    columns = [(start, end, scope) for start, end in [(0, 12), (0, 24), (12, 24)] for scope in ['dupilumab', 'pooled_abrocitinib']]
    values = np.array([[float(change[change.region.eq(region) & change.start.eq(start) & change.end.eq(end) & change.scope.eq(scope) & change.outcome.eq('normalized_improvement')]['mean'].iloc[0])*100 for start,end,scope in columns] for region in REGIONS])
    limit = max(5, np.ceil(np.nanmax(np.abs(values)) / 5) * 5)
    im = previous.heat(c, values, REGION_LABELS, ['D', 'A']*3, 'Paired regional decrease', limit,
                       [[('0' if abs(v)<.5 else f'{v:.0f}') for v in row] for row in values])
    for x, label in zip([.1667, .5, .8333], ['W0–12', 'W0–24', 'W12–24']):
        c.text(x, 1.06, label, transform=c.transAxes, ha='center', fontsize=6.3)
    c.set_title('Paired decrease (% of maximum)', pad=25)
    for x in [1.5, 3.5]: c.axvline(x, color='white', lw=2)
    cbax = c.inset_axes([.14, -.23, .72, .045]); cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=[-limit, 0, limit]); cbax.tick_params(labelsize=6.3, length=2)
    for j, (scope, color) in enumerate([('dupilumab', previous.COLORS[0]), ('pooled_abrocitinib', previous.POOLED)]):
        z = itch[itch.scope.eq(scope)].set_index(['start', 'end']).reindex([(0,12),(0,24),(12,24)])
        previous.forest(e, z, np.arange(3)+(j-.5)*.20, color, key='mean')
    e.set(yticks=range(3), yticklabels=['W0–12', 'W0–24', 'W12–24'], ylim=(2.5, -.5), xlabel='PP-NRS decrease (points)', title='Whole-body itch')
    e.axvline(0, color='#AAB6BC', lw=.6)
    handles = [Line2D([],[], color=c, marker='o', lw=1.35, ms=3.5, label=l) for c,l in [(previous.COLORS[0], 'D: dupilumab'),(previous.POOLED,'A: pooled abrocitinib')]]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98,.995), frameon=False, ncol=2, fontsize=6.3)
    for ax in [a,b,c]:
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    finish(fig, 14, 'Regional clinical burden and molecular associations',
           'a, Baseline Spearman associations between all 23 markers and the four regional EASI contributions; stars mark BH q<0.05, with complete coefficients, '
           'patient counts and bootstrap intervals retained in the source. b, Available-case mean regional EASI divided by its theoretical maximum and multiplied '
           'by 100; cells show percentages, not patient counts. c, Mean paired regional decrease as a percentage of the same regional maximum for three separately '
           'labelled intervals; positive values indicate improvement. D is dupilumab and A is the observed pooled abrocitinib dose mixture. These heatmaps summarize '
           'magnitude; exact 95% t intervals and counts are in the source tables. d, Whole-body PP-NRS mean decreases with 95% t intervals, not region-specific itch. '
           'Regional maxima are 7.2, 14.4, 21.6 and 28.8 for head/neck, upper limbs, trunk and lower limbs. All regional analyses require complete, bounded regional '
           'records consistent with total EASI. Intervals can share patients and are not successive cumulative effects. Colour bars encode distinct quantities and '
           'are not interchangeable. No baseline-adjusted regional treatment contrast passed the FDR threshold; full separate-dose comparisons remain available.',
           [clinical/'molecular_regional_baseline_correlations.csv', regional/'regional_normalized_burden.csv',
            regional/'regional_paired_improvement.csv', regional/'whole_body_PP_NRS_improvement.csv',regional/'regional_treatment_contrasts.csv'], root, base)


def run(root, base=None):
    if base is None:
        from . import figures as base
    root = Path(root).resolve()
    initialize()
    previous.DESIGNS.clear()
    courses(root, base)
    molecule_windows(root, base)
    selected_associations(root, base)
    regional_composite(root, base)
    baseline_heatmap(root, base)
    triplet_forests(root, base)
    ige_fixed(root, base)
    previous.treatment_contrasts(root, base)
    previous.monitoring_classification(root, base)
    actual = [int(r['id'].removeprefix('FigureS')) for r in previous.DESIGNS]
    if set(actual) != set(SOURCES) or len(actual) != len(SOURCES):
        raise ValueError('Supplementary revision coverage differs from the agreed source list.')
    folder = root / 'reports/revision_20261025/supplement'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(previous.DESIGNS).to_csv(folder / 'design_and_source_map.csv', index=False)
    (folder/'display_contract.json').write_text(json.dumps({'retained_source_ids':SOURCES,
        'full_numeric_tables_retained':True, 'clinical_missing_observations_imputed':False,
        'selected_associations':'All six q<0.05 coefficients from the original complete family; exploratory display only'}, indent=2)+'\n')
    return previous.DESIGNS
