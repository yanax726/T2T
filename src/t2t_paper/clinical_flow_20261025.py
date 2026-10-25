"""Source-based clinical framework and auditable analysis-population diagrams."""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from . import palette

INK = '#26343C'
MUTED = '#66747C'
RULE = '#B8C7CD'
ACCENT = '#587F8E'
CLINICAL_FILL = '#EDF3F5'
PATIENT_FILL = '#FAEEE8'
RESEARCH_FILL = '#F3F0ED'
NEUTRAL_FILL = '#F6F7F7'


def _text(ax, x, y, text, *, size=7.5, weight='normal', ha='center', va='center', color=INK):
    return ax.text(x, y, text, fontsize=size, weight=weight, ha=ha,
                   va=va, color=color, linespacing=1.3, zorder=4)


def _panel(ax, letter, title):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-12, 7),
                textcoords='offset points', weight='bold', fontsize=8.5,
                color=INK, annotation_clip=False)
    ax.annotate(title, (0, 1), xycoords='axes fraction', xytext=(0, 7),
                textcoords='offset points', weight='bold', fontsize=8,
                color=INK, annotation_clip=False)


def _box(ax, x, y, w, h, *, face='white', edge=RULE, accent=None):
    patch = Rectangle((x, y), w, h, linewidth=.65, edgecolor=edge,
                      facecolor=face, zorder=1)
    ax.add_patch(patch)
    if accent:
        ax.add_patch(Rectangle((x, y + h - .012), w, .012,
                               facecolor=accent, edgecolor='none', zorder=2))
    return patch


def _arrow(ax, start, end, *, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=7.0, lw=.8, color=color,
                                shrinkA=0, shrinkB=0, zorder=2))


def _line(ax, xs, ys, *, color=MUTED):
    ax.plot(xs, ys, color=color, lw=.8, solid_capstyle='butt', zorder=2)


def _card(ax, x, y, w, h, title, detail, *, face='white', accent=None,
          title_size=7.5, detail_size=7.5):
    _box(ax, x, y, w, h, face=face, accent=accent)
    _text(ax, x + w / 2, y + .72 * h, title,
          size=title_size, weight='bold')
    _text(ax, x + w / 2, y + .31 * h, detail, size=detail_size)


def framework(root, base):
    """Render the framework's development, milestone logic and research branch."""
    root = Path(root)
    source = root / 'config/clinical_framework.json'
    spec = json.loads(source.read_text())
    development = spec['development_summary']
    assert spec['version'] == 'framework_20260903'
    assert spec['adult_DLQI_analysis_minimum_age'] == 18
    assert development['planned_agreement_criterion'] == .75
    assert development['reported_returned_questionnaires'] == 49
    assert development['original_ballots_available'] is False
    assert development['round_specific_agreement_available'] is False
    expected = {
        '2': {'physician': None, 'patient': ['pp_nrs <= 4']},
        '4': {'physician': ['easi_pct_improvement >= 50', 'iga_baseline - iga >= 2'],
              'patient': ['pp_nrs <= 4', 'dlqi <= 10']},
        '12': {'physician': ['easi_pct_improvement >= 75', 'iga <= 1'],
               'patient': ['pp_nrs <= 1', 'dlqi <= 4']},
        '24': {'physician': ['easi_pct_improvement >= 75', 'easi_total <= 5', 'iga <= 1'],
               'patient': ['pp_nrs <= 1', 'dlqi <= 4']},
    }
    assert all(spec['visits'][week] == values for week, values in expected.items())

    fig = plt.figure(figsize=(183 / 25.4, 180 / 25.4), facecolor='white')
    grid = fig.add_gridspec(3, 1, height_ratios=[.95, 1.8, 1.05],
                           left=.055, right=.985, top=.945, bottom=.045,
                           hspace=.28)
    a, b, c = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    _panel(a, 'a', 'Development of the clinical target framework')
    _panel(b, 'b', 'Scheduled clinical targets')
    _panel(c, 'c', 'Clinical care and parallel research measurement')

    for x, title, detail in [
        (.01, 'Candidate clinical measures', 'EASI · IGA\nPP-NRS · DLQI'),
        (.365, 'Two-round review', 'Questionnaires\nand discussion'),
        (.72, 'Target framework', 'Planned agreement\ncriterion ≥75%'),
    ]:
        _card(a, x, .27, .27, .65, title, detail, face=CLINICAL_FILL)
    _arrow(a, (.28, .595), (.365, .595))
    _arrow(a, (.635, .595), (.72, .595))
    _text(a, .5, .085, '49 returned questionnaires reported', size=7.2, color=MUTED)

    # Aligned visit columns expose the OR/AND logic without implying a new rule.
    columns = [.16, .375, .59, .805]
    widths = .185
    physicians = [None, 'EASI50\nor IGA decrease ≥2', 'EASI75\nor IGA 0/1',
                  'EASI75\nor EASI ≤5\nor IGA 0/1']
    patients = ['PP-NRS ≤4', 'PP-NRS ≤4\nor adult DLQI ≤10',
                'PP-NRS 0/1\nor adult DLQI ≤4', 'PP-NRS 0/1\nor adult DLQI ≤4']
    _text(b, .01, .625, 'Physician\ndomain', size=7.2, weight='bold', ha='left')
    _text(b, .01, .225, 'Patient\ndomain', size=7.2, weight='bold', ha='left')
    for x, week, physician, patient in zip(columns, ['W2', 'W4', 'W12', 'W24'], physicians, patients):
        _box(b, x, .015, widths, .95, face=NEUTRAL_FILL, edge='#D5DEE2')
        _text(b, x + widths / 2, .9, week, size=8, weight='bold')
        _box(b, x + .009, .475, widths - .018, .30,
             face='white' if physician is None else CLINICAL_FILL, edge='#DCE3E6')
        _text(b, x + widths / 2, .625, 'Not assessed' if physician is None else physician,
              size=7.0, color=MUTED if physician is None else INK)
        if physician is not None:
            _line(b, [x + widths / 2] * 2, [.475, .427], color=RULE)
            _text(b, x + widths / 2, .395, 'AND', size=7, weight='bold')
            _line(b, [x + widths / 2] * 2, [.363, .315], color=RULE)
        _box(b, x + .009, .115, widths - .018, .20,
             face=PATIENT_FILL, edge='#E5D7CE')
        _text(b, x + widths / 2, .215, patient, size=7.0)
    for i in range(3):
        _arrow(b, (columns[i] + widths + .003, .9), (columns[i + 1] - .003, .9))

    # Care and measurement have separate paths; the score is not a treatment rule.
    _box(c, .01, .54, .98, .42, face=CLINICAL_FILL, edge='#D7E1E5')
    _text(c, .035, .86, 'Clinical care', size=7.4, weight='bold', ha='left')
    _text(c, .17, .68, 'Clinical scores\nand safety')
    _text(c, .50, .68, 'Review targets\nand individual context')
    _text(c, .83, .68, 'Clinician-directed care\nand reassessment')
    _arrow(c, (.295, .68), (.365, .68))
    _arrow(c, (.635, .68), (.70, .68))
    _box(c, .01, .015, .98, .42, face=RESEARCH_FILL, edge='#DFD7D1')
    _text(c, .035, .335, 'Research measurement', size=7.4, weight='bold', ha='left')
    _text(c, .22, .16, 'BSA + CCL17, CCL26, IL-5', size=7.2)
    _text(c, .60, .16, 'Common EASI estimator', size=7.2)
    _text(c, .875, .16, 'Estimated EASI', size=7.2)
    _arrow(c, (.405, .16), (.465, .16), color=ACCENT)
    _arrow(c, (.735, .16), (.785, .16), color=ACCENT)

    base.save(
        fig, 'Figure1', 'Clinical target framework and parallel research EASI estimation',
        'a, Framework development described in the investigator materials: candidate clinical '
        'measures, a two-round questionnaire and discussion procedure, 49 returned questionnaires, '
        'and a planned agreement criterion of at least 75%. Original ballots and round-specific '
        'agreement were unavailable; these are source-reported development details, not independently '
        'verified consensus results. b, Scheduled targets transcribed from the project framework. '
        'Within each domain, any listed criterion is sufficient; from W4 onward both domains '
        'must be met. W2 assesses only PP-NRS ≤4. EASI50 and EASI75 denote at least 50% and 75% '
        'improvement from a positive observed baseline EASI. IGA decrease ≥2 is relative to baseline; '
        'IGA 0/1 and PP-NRS 0/1 denote observed values 0 or 1. DLQI targets apply to adults aged ≥18 '
        'years. Missing components are handled with three-valued logic, rather than automatically '
        'classified as target failure. The exact IGA instrument and approved implemented framework '
        'version require investigator confirmation. Visits are schematic and not spaced in proportion '
        'to time. c, Clinician-directed assessment and care are separated from the research estimator '
        'using BSA, CCL17, CCL26 and IL-5. The estimator was not a dose-change rule, and this observational '
        'analysis does not establish the effectiveness of the proposed integrated management strategy.',
        [str(source)], root,
    )


def cohort_flow(root, base):
    """Render independent clinical/serum entry paths with exact exclusion units."""
    root = Path(root)
    folder = root / 'results/tables/update_20261017/redesign_20261018'
    names = ['flow_nodes.csv', 'flow_edges.csv', 'flow_exclusions.csv',
             'model_ordered_exclusions.csv', 'cohort_overlap.csv', 'model_visit_patterns.csv']
    nodes, edges, exclusions, ordered, overlaps, patterns = [pd.read_csv(folder / name) for name in names]
    nodes = nodes.set_index('node')
    order = ordered.set_index('reason')
    groups_source = root / 'results/tables/update_20261017/clinical/baseline_group_counts.csv'
    groups = pd.read_csv(groups_source)
    groups = groups[groups.variable.eq('sex_male')].groupby('treatment').n.sum()
    assert groups.to_dict() == {'dupilumab': 65, 'abrocitinib_100mg': 56, 'abrocitinib_200mg': 42}
    assert int(groups.sum()) == int(nodes.loc['clinical_main', 'n_patients'])
    assert list(ordered.remaining_rows) == [340, 327, 326, 326, 318, 316]
    assert list(ordered.excluded_rows) == [12, 13, 1, 0, 8, 2]
    assert int(patterns.n_patients.sum()) == int(nodes.loc['model_complete', 'n_patients'])
    assert int(nodes.loc['development', 'n_records'] + nodes.loc['temporal', 'n_records']) == int(nodes.loc['model_complete', 'n_records'])
    clinical_exclusions = exclusions[exclusions.branch.eq('clinical')]
    assert int(nodes.loc['clinical_source', 'n_patients'] - clinical_exclusions.n_patients.sum()) == int(nodes.loc['clinical_main', 'n_patients'])
    assert {('clinical_source', 'clinical_age_eligible'), ('molecular_source', 'molecular_target_weeks')} <= set(zip(edges.source, edges.target))
    clinical_model = overlaps[(overlaps.population1 == 'clinical_main') & (overlaps.population2 == 'model_complete')].iloc[0]
    dev_temporal = overlaps[(overlaps.population1 == 'development') & (overlaps.population2 == 'temporal')].iloc[0]
    assert int(clinical_model.intersection) == 159 and int(dev_temporal.intersection) == 71
    w48_records = int(exclusions.loc[exclusions.branch.eq('molecular_time'), 'n_records'].sum())
    assert w48_records == 24

    fig = plt.figure(figsize=(183 / 25.4, 190 / 25.4), facecolor='white')
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.8],
                           left=.055, right=.985, top=.947, bottom=.035,
                           hspace=.19)
    a, b = [fig.add_subplot(grid[i, 0]) for i in range(2)]
    _panel(a, 'a', 'Clinical comparison population')
    _panel(b, 'b', 'Serum-linked model population')

    def n(key, column='n_patients'):
        return int(nodes.loc[key, column])

    _card(a, .01, .69, .27, .25, 'Clinical records',
          f"{n('clinical_source')} patients", face=CLINICAL_FILL, detail_size=8)
    _card(a, .365, .69, .27, .25, 'Age ≥12 years',
          f"{n('clinical_age_eligible')} patients", face=CLINICAL_FILL, detail_size=8)
    _card(a, .72, .69, .27, .25, 'Clinical comparison set',
          f"{n('clinical_main')} patients", face=CLINICAL_FILL, detail_size=8)
    _arrow(a, (.28, .815), (.365, .815))
    _arrow(a, (.635, .815), (.72, .815))
    for x, label in [(.3225, '6 excluded\nAge <12 years'),
                      (.6775, '1 excluded\nInitial dose ambiguous')]:
        _line(a, [x, x], [.815, .62], color=RULE)
        _text(a, x, .535, label, size=7.2, color=MUTED)
    _line(a, [.855, .855], [.69, .365])
    _line(a, [.145, .855], [.365, .365])
    for x, key, title in [(.01, 'dupilumab', 'Dupilumab'),
                          (.365, 'abrocitinib_100mg', 'Abrocitinib 100 mg'),
                          (.72, 'abrocitinib_200mg', 'Abrocitinib 200 mg')]:
        _arrow(a, (x + .135, .365), (x + .135, .285))
        colour = palette.register(fig, 'treatment.' + key)
        _card(a, x, .035, .27, .25, title, f'{int(groups[key])} patients',
              accent=colour, detail_size=8)

    _card(b, .01, .81, .27, .16, 'Linked serum source',
          f"{n('molecular_source')} patients · {n('molecular_source', 'n_records')} records",
          face=CLINICAL_FILL, detail_size=7.2)
    _card(b, .365, .81, .27, .16, 'W0, W12 or W24',
          f"{n('molecular_target_weeks')} patients · {n('molecular_target_weeks', 'n_records')} records",
          face=CLINICAL_FILL, detail_size=7.2)
    _card(b, .72, .81, .27, .16, 'Age ≥12 years',
          f"{n('molecular_age_eligible')} patients · {n('molecular_age_eligible', 'n_records')} records",
          face=CLINICAL_FILL, detail_size=7.2)
    _arrow(b, (.28, .89), (.365, .89))
    _arrow(b, (.635, .89), (.72, .89))
    for x, label in [(.3225, '24 W48 records\nexcluded'),
                      (.6775, '12 records excluded\n6 patients aged <12')]:
        _line(b, [x, x], [.89, .785], color=RULE)
        _text(b, x, .74, label, size=7.0, color=MUTED)
    _line(b, [.855, .855, .50], [.81, .67, .67])
    _arrow(b, (.50, .67), (.50, .64))

    _box(b, .01, .345, .98, .295, face=NEUTRAL_FILL, edge='#D4DDE1')
    _text(b, .035, .602, 'Sequential selection of observed outcomes and valid inputs',
          size=7.5, weight='bold', ha='left')
    steps = [(.11, 'Age-eligible', n('molecular_age_eligible', 'n_records'), None)]
    for x, label, reason in [(.305, 'EASI observed', 'missing_easi_total'),
                              (.50, 'BSA observed', 'missing_bsa'),
                              (.695, 'CCL26 valid', 'missing_log2_CCL26'),
                              (.89, 'IL-5 valid', 'missing_log2_IL-5')]:
        step = order.loc[reason]
        steps.append((x, label, int(step.remaining_rows), int(step.excluded_rows)))
    for x, label, count, excluded in steps:
        _text(b, x, .536, label, size=7.1)
        _text(b, x, .493, f'{count} records', size=7.5, weight='bold')
        if excluded is not None:
            _text(b, x, .449, f'{excluded} excluded', size=7, color=MUTED)
    for i in range(4):
        _arrow(b, (steps[i][0] + .079, .493), (steps[i + 1][0] - .079, .493), color=RULE)
    _text(b, .035, .384, 'CCL17: no additional exclusions', size=7.1, ha='left', color=MUTED)
    _text(b, .97, .384, f"Complete model: {n('model_complete')} patients · {n('model_complete', 'n_records')} visits", size=7.4,
          weight='bold', ha='right')

    _line(b, [.50, .50], [.345, .29])
    _line(b, [.255, .765], [.29, .29])
    for x in [.255, .765]:
        _arrow(b, (x, .29), (x, .235))
    _box(b, .01, .035, .49, .20, face=CLINICAL_FILL, edge='#CEDCE2')
    _text(b, .255, .194, 'Development · W0 and W12', size=7.8, weight='bold')
    _text(b, .255, .132, f"{n('development')} patients · {n('development', 'n_records')} visits", size=8)
    _text(b, .255, .075, f"W0: {n('model_W0', 'n_records')} visits · W12: {n('model_W12', 'n_records')} visits", size=7.3, color=MUTED)
    _box(b, .54, .035, .45, .20, face=RESEARCH_FILL, edge='#DFD7D1')
    _text(b, .765, .194, 'Temporal application · W24', size=7.8, weight='bold')
    _text(b, .765, .132, f"{n('temporal')} patients · {n('temporal', 'n_records')} visits", size=8)
    _text(b, .765, .075, 'Previously inspected within-cohort data', size=7.3, color=MUTED)

    base.save(
        fig, 'FigureS39', 'Patient and record flow into the clinical and model analyses',
        'a, The curated clinical source contains 170 patients, not a screened or consented recruitment '
        'denominator. Six patients aged <12 years and one age-eligible patient with an ambiguous '
        'initial dose are excluded, leaving 163 patients classified by initial treatment: dupilumab, '
        '65; abrocitinib 100 mg, 56; and abrocitinib 200 mg, 42. b, The linked serum source is a '
        'separate entry branch; model eligibility is not conditional on membership of the clinical '
        'comparison population. Removing 24 W48 records leaves 168 patients with 352 W0/W12/W24 '
        'records; age exclusions remove 12 records from six patients. Further ordered exclusions are '
        'mutually exclusive at the record level: 13 EASI, one BSA, zero CCL17, eight CCL26 and two IL-5 '
        'inputs are unavailable, leaving 316 complete model visits from 160 patients. A valid molecular '
        'input requires an available positive concentration for the log2 transformation. Loss of a '
        'record does not necessarily remove a patient. Development uses 241 W0/W12 visits from 156 '
        'patients, with patient-level validation splits; W24 includes 75 visits from 75 patients and '
        'is previously inspected within-cohort temporal application, not external validation. '
        'Development and W24 share 71 patients, with 85 development-only and four W24-only patients. '
        'Fifty-one patients have all three complete visits. The clinical and complete-input model '
        'populations share 159 patients, with four clinical-only and one model-only patient; the '
        'treatment-free model can retain an ambiguous initial dose. Scheduled empty visit slots '
        'are not counted as observed visits, and no filled-in clinical outcome is used as validation truth.',
        [str(folder / name) for name in names] + [str(groups_source)], root,
    )
