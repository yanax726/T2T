"""Clinical framework and parallel cohort flow, rendered from verified sources.

The framework is a source-grounded schematic, not a new consensus experiment
or a validated treatment algorithm. The two population branches are not nested.
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from . import palette


INK = '#26343C'
MUTED = '#66747C'
RULE = '#CBD2D5'
TEAL = '#3C7378'
PALE = '#EFF4F4'


def _panel(ax, letter, title):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-13, 7),
                textcoords='offset points', weight='bold', fontsize=8,
                color=INK, annotation_clip=False)
    ax.text(0, 1.025, title, transform=ax.transAxes, color=INK,
            fontsize=8, weight='bold', va='bottom')


def _arrow(ax, start, end, *, dashed=False, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=7, lw=.75, color=color,
                                linestyle='--' if dashed else '-'))


def _text(ax, x, y, text, *, size=7, weight='normal', color=INK,
          ha='left', va='center'):
    return ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
                   weight=weight, linespacing=1.35)


def framework(root, base):
    """Save main Figure 1, a compact development, target and research schematic."""
    root = Path(root)
    source = root / 'config/clinical_framework.json'
    specification = json.loads(source.read_text())
    development = specification['development_summary']
    assert specification['version'] == 'framework_20260903'
    assert specification['adult_DLQI_analysis_minimum_age'] == 18
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
    assert all(specification['visits'][w] == values for w, values in expected.items())

    # Three complementary evidence roles share one left/right panel boundary.
    fig = plt.figure(figsize=(183 / 25.4, 140 / 25.4), facecolor='white')
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.7, 1.2],
                           left=.065, right=.975, top=.935, bottom=.045,
                           hspace=.35)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    a, b, c = axes
    _panel(a, 'a', 'Source-reported framework development')
    _panel(b, 'b', 'Stage-specific clinical targets')
    _panel(c, 'c', 'Clinical assessment and the proposed research addition')

    # Open process labels retain the development status without a wall of boxes.
    for x, heading, detail in [
        (.0, 'Candidate measures', 'EASI · IGA\nPP-NRS · DLQI'),
        (.36, 'Two-round procedure', 'Questionnaire review\nand discussion'),
        (.73, 'Draft target framework', 'Planned agreement\ncriterion ≥' +
         f"{100 * development['planned_agreement_criterion']:.0f}%"),
    ]:
        _text(a, x, .76, heading, weight='bold')
        _text(a, x, .44, detail)
    _arrow(a, (.28, .64), (.335, .64))
    _arrow(a, (.64, .64), (.705, .64))
    _text(a, 0, .04, f"{development['reported_returned_questionnaires']} returned questionnaires reported",
          size=6.7, color=MUTED, va='bottom')

    # Visit milestones carry the reading direction. Each later visit joins two
    # clinical domains; the early itch milestone has no physician-domain link.
    milestones = [
        (.20, 'W2', 'Not assessed', 'PP-NRS ≤4'),
        (.43, 'W4', 'EASI50\nOR IGA decrease ≥2',
         'PP-NRS ≤4\nOR adult DLQI ≤10'),
        (.66, 'W12', 'EASI75\nOR IGA 0/1',
         'PP-NRS 0/1\nOR adult DLQI ≤4'),
        (.89, 'W24', 'EASI75\nOR IGA 0/1\nOR EASI ≤5',
         'PP-NRS 0/1\nOR adult DLQI ≤4'),
    ]
    b.plot([.20, .89], [.87, .87], color=RULE, lw=1.0, zorder=0)
    _text(b, 0, .60, 'Physician\ndomain', size=6.5, weight='bold')
    _text(b, 0, .135, 'Patient\ndomain', size=6.5, weight='bold')
    for x, week, physician, patient in milestones:
        colour = (palette.register(fig, 'visit.' + week)
                  if week in ['W12', 'W24'] else '#B9C2C7')
        b.scatter([x], [.87], s=30, color=colour, edgecolor='white',
                  linewidth=.6, zorder=2)
        _text(b, x, .985, week, weight='bold', ha='center')
        _text(b, x, .60, physician, size=6.7, ha='center',
              color=MUTED if week == 'W2' else INK)
        _text(b, x, .135, patient, size=6.7, ha='center')
        if week != 'W2':
            b.plot([x, x], [.445, .405], color=RULE, lw=.7)
            _text(b, x, .35, 'AND', size=6.5, weight='bold', ha='center')
            b.plot([x, x], [.295, .255], color=RULE, lw=.7)

    _text(c, 0, .80, 'Measured skin severity\nSymptoms, quality of life\nand safety information')
    _text(c, .39, .80, 'Review clinical targets\nand individual context')
    _text(c, .76, .80, 'Clinician-directed care\nand reassessment')
    _arrow(c, (.31, .80), (.365, .80))
    _arrow(c, (.69, .80), (.735, .80))
    c.plot([0, 1], [.53, .53], color=RULE, lw=.55)
    _text(c, 0, .29, 'Research addition', weight='bold')
    _text(c, .25, .29, 'BSA + CCL17, CCL26, IL-5')
    _arrow(c, (.66, .29), (.735, .29), color=TEAL)
    _text(c, .76, .29, 'EASI research readout', weight='bold')
    _text(c, .25, .025, 'Clinical decision use remains untested', size=6.7,
          color=MUTED, va='bottom')

    base.save(
        fig, 'Figure1', 'Source framework, clinical targets and the research readout',
        'a, The project development summary describes candidate measures, a two-round questionnaire '
        'procedure and a planned ≥75% agreement criterion, and reports 49 returned questionnaires. '
        'Original ballots and round-specific agreement results are unavailable; this is not a newly '
        'verified consensus experiment. b, The operational target schedule follows the source '
        'framework; milestone positions are schematic rather than time-scaled. EASI50/75 denotes '
        'at least 50%/75% improvement from positive observed baseline EASI. IGA decrease ≥2 is change '
        'from baseline; 0/1 denotes an observed grade or score of 0 or 1. OR applies within a domain '
        'and AND across domains. W2 is an itch-only target, not a combined physician-patient target. '
        'Adult DLQI requires age ≥18. Missing components use three-valued logic; adult DLQI is unknown '
        'below age 18. The IGA instrument and approved, implemented framework version require '
        'confirmation; these are exploratory recorded-score targets. c, Clinical assessment '
        'supports clinician-directed management. The BSA-serum readout is a parallel research '
        'addition, with clinical decision use untested. It does not replace examination, symptoms, '
        'quality of life or safety review. The cohort does not test framework effectiveness. '
        'Detailed definitions and provenance are provided in the Supplementary Methods.',
        [str(source)], root,
    )


def cohort_flow(root, base):
    """Save internal FigureS39, published as S37, from exact aggregate counts."""
    root = Path(root)
    folder = root / 'results/tables/update_20261017/redesign_20261018'
    names = ['flow_nodes.csv', 'flow_edges.csv', 'flow_exclusions.csv',
             'model_ordered_exclusions.csv', 'cohort_overlap.csv', 'model_visit_patterns.csv']
    nodes, edges, exclusions, ordered, overlaps, patterns = [pd.read_csv(folder / n) for n in names]
    nodes = nodes.set_index('node')
    clinical_model = overlaps[overlaps.population1.eq('clinical_main')
                              & overlaps.population2.eq('model_complete')].iloc[0]
    development_temporal = overlaps[overlaps.population1.eq('development')
                                    & overlaps.population2.eq('temporal')].iloc[0]
    clinical_exclusions = exclusions[exclusions.branch.eq('clinical')].set_index('reason')
    molecular_time = exclusions[exclusions.branch.eq('molecular_time')].n_records.sum()
    assert int(nodes.loc['clinical_source', 'n_patients']) - int(clinical_exclusions.n_patients.sum()) == int(nodes.loc['clinical_main', 'n_patients'])
    assert int(nodes.loc['molecular_source', 'n_records']) - int(molecular_time) == int(nodes.loc['molecular_target_weeks', 'n_records'])
    assert int(ordered.excluded_rows.sum()) == int(nodes.loc['molecular_target_weeks', 'n_records'] - nodes.loc['model_complete', 'n_records'])
    assert int(patterns.n_patients.sum()) == int(nodes.loc['model_complete', 'n_patients'])
    assert {('clinical_source', 'clinical_age_eligible'),
            ('molecular_source', 'molecular_target_weeks')} <= set(zip(edges.source, edges.target))

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 190 / 25.4))
    fig.subplots_adjust(left=.055, right=.985, top=.935, bottom=.075, wspace=.13)
    left, right = axes
    _panel(left, 'a', 'Clinical treatment comparisons')
    _panel(right, 'b', 'Linked-serum model population')

    def count(key, unit):
        row = nodes.loc[key]
        return f'{int(row.n_patients)} patients\n{int(row.n_records)} {unit}'

    def node(ax, y, title, detail, *, x=.015, width=.66, height=.115, accent=False):
        ax.add_patch(Rectangle((x, y - height / 2), width, height,
                               fc=PALE if accent else 'white', ec=TEAL if accent else RULE, lw=.7))
        _text(ax, x + width / 2, y + .026, title, size=7, weight='bold',
              color=INK, ha='center')
        _text(ax, x + width / 2, y - .024, detail, size=7, ha='center')

    def downward(ax, upper, lower, height=.115):
        _arrow(ax, (.345, upper - height / 2), (.345, lower + height / 2))

    def excluded(ax, midpoint, text):
        _arrow(ax, (.345, midpoint), (.735, midpoint))
        _text(ax, .755, midpoint, text, size=6.7, color=MUTED)

    node(left, .91, 'Curated clinical source',
         f"{int(nodes.loc['clinical_source', 'n_patients'])} patients\n"
         f"{int(nodes.loc['clinical_observed_slots', 'n_records'])} nonempty visit slots")
    node(left, .70, 'Age ≥12 years',
         f"{int(nodes.loc['clinical_age_eligible', 'n_patients'])} patients")
    node(left, .49, 'Initial group recorded',
         f"{int(nodes.loc['clinical_main', 'n_patients'])} patients", accent=True)
    downward(left, .91, .70)
    downward(left, .70, .49)
    excluded(left, .805, 'Age <12\n' + str(int(clinical_exclusions.loc['age_below_12', 'n_patients'])) + ' patients')
    excluded(left, .595, 'Initial dose\nambiguous\n' + str(int(clinical_exclusions.loc['ambiguous_initial_dose_after_age_filter', 'n_patients'])) + ' patient')
    _text(left, .02, .365, 'Initial treatment groups', size=7, weight='bold', color=MUTED)
    # Group sizes are computed from the exact baseline source, not inferred from serum rows.
    group_source = root / 'results/tables/update_20261017/clinical/baseline_group_counts.csv'
    groups = pd.read_csv(group_source)
    sizes = groups[groups.variable.eq('sex_male')].groupby('treatment').n.sum()
    assert int(sizes.sum()) == int(nodes.loc['clinical_main', 'n_patients'])
    for y, treatment, label in [(.315, 'dupilumab', 'Dupilumab'),
                                (.265, 'abrocitinib_100mg', 'Abrocitinib 100 mg'),
                                (.215, 'abrocitinib_200mg', 'Abrocitinib 200 mg')]:
        _text(left, .02, y, label)
        _text(left, .64, y, str(int(sizes.loc[treatment])), ha='right')
    left.plot([.02, .66], [.17, .17], color=RULE, lw=.6)
    _text(left, .02, .115, f"Shared with complete-input model\n{int(clinical_model.intersection)} patients", size=7)
    _text(left, .02, .035,
          f"Clinical only {int(clinical_model.only1)} · Model only {int(clinical_model.only2)}",
          size=6.7, color=MUTED)

    node(right, .91, 'Linked serum source', count('molecular_source', 'serum records'))
    node(right, .70, 'W0, W12 or W24', count('molecular_target_weeks', 'serum records'))
    node(right, .49, 'Age ≥12 years', count('molecular_age_eligible', 'serum records'))
    node(right, .28, 'Complete model visits', count('model_complete', 'visits'), accent=True)
    for upper, lower in [(.91, .70), (.70, .49), (.49, .28)]:
        downward(right, upper, lower)
    excluded(right, .805, f'W48 outside\nmodel scope\n{int(molecular_time)} records')
    age = ordered[ordered.reason.eq('age_below_12_or_unknown')].iloc[0]
    excluded(right, .595, f'Age <12\n{int(age.excluded_rows)} records')
    missing = ordered[~ordered.reason.eq('age_below_12_or_unknown')].set_index('reason')
    excluded(right, .385,
             f'Missing data\n{int(missing.excluded_rows.sum())} records')

    # A true visit partition, with explicit participant overlap rather than a false cohort split.
    right.plot([.345, .345], [.2225, .18], color=MUTED, lw=.75)
    right.plot([.245, .775], [.18, .18], color=MUTED, lw=.75)
    _arrow(right, (.245, .18), (.245, .145))
    _arrow(right, (.775, .18), (.775, .145))
    node(right, .0875, 'W0/W12 development', count('development', 'visits'),
         x=.015, width=.46, accent=True)
    node(right, .0875, 'W24 application', count('temporal', 'visits'),
         x=.56, width=.43, accent=True)
    _text(right, .5, -.018,
          f"W0 {int(nodes.loc['model_W0', 'n_records'])} · W12 {int(nodes.loc['model_W12', 'n_records'])} · "
          f"W24 {int(nodes.loc['model_W24', 'n_records'])} visits", ha='center',
          size=6.7, color=MUTED)
    fig.text(.52, .023,
             f"Development and W24 share {int(development_temporal.intersection)} patients; "
             'visit rows are disjoint, participants overlap.', ha='center',
             fontsize=6.7, color=MUTED)

    base.save(
        fig, 'FigureS39', 'Parallel clinical and complete-input model populations',
        'a, Clinical comparison eligibility starts with 170 curated patients and 1,700 predefined '
        'visit slots; 736 slots contain recorded visit data. Six patients aged <12 and one '
        'age-eligible patient with an ambiguous initial dose are excluded, leaving 163 patients. '
        'This is a record source, not a screened or consented recruitment denominator. '
        'b, The linked serum branch is separate and is not a subset of the clinical comparison. '
        'Serum records count linked sample rows, not independent participants. Excluding 24 W48 '
        'records leaves 168 patients and 352 W0/W12/W24 records; age exclusions remove 12 records '
        'from six patients. Complete-case exclusions are sequential and mutually exclusive at the '
        'record level. Of the remaining 340 records, 13 lack EASI, one BSA, none CCL17, eight CCL26 '
        'and two IL-5, leaving 316 visits from 160 patients. Missing model inputs include '
        'unavailable valid positive molecular values. Losing a visit does not necessarily remove '
        'the patient. Development comprises 156 patients and 241 visits, W0 n=152 and W12 n=89. '
        'W24 comprises 75 patients and 75 visits and is previously inspected within-cohort temporal '
        'application, not external validation. The clinical and model populations share 159 '
        'patients, with four clinical-only and one model-only. Development and W24 share 71 '
        'patients, with 85 development-only and four W24-only. Fifty-one patients have all three '
        'complete model visits. Initial-dose ambiguity may remain in the treatment-free model. '
        'No missing clinical outcome is filled as validation truth. All exact counts and units '
        'remain in the linked flow and overlap source tables.',
        [str(folder / n) for n in names] + [str(group_source)], root,
    )
