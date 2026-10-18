"""Reconcile parallel clinical and serum-model populations without exporting IDs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd

KEYS = ['patient_uid', 'visit_week']
FEATURES = ['easi_total', 'bsa', 'log2_CCL17', 'log2_CCL26', 'log2_IL-5']
INPUTS = ['results/tables/update_20260925/clinical.csv',
          'results/tables/redevelopment/molecular.csv',
          'results/tables/update_20261002/stage_refinement/input.csv']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_masks(frame, conditions):
    """Ordered exclusions count records once; patient losses are separate counts."""
    remain = frame.copy()
    records = []
    for label, predicate in conditions:
        keep = predicate(remain)
        removed = remain.loc[~keep]
        after = remain.loc[keep].copy()
        records.append(dict(reason=label, before_rows=len(remain), excluded_rows=len(removed),
                            remaining_rows=len(after), excluded_patient_records=removed.patient_uid.nunique(),
                            patients_fully_lost=len(set(remain.patient_uid) - set(after.patient_uid)),
                            remaining_patients=after.patient_uid.nunique()))
        remain = after
    return remain, records


def membership_patterns(frame):
    patterns = frame.groupby('patient_uid').visit_week.agg(
        lambda values: '+'.join('W' + str(int(w)) for w in sorted(set(values))))
    return patterns.value_counts().rename_axis('pattern').reset_index(name='n_patients')


def run(root):
    root = Path(root)
    out = root / 'results/tables/update_20261017/redesign_20261018'
    report = root / 'reports/update_20261017/redesign_20261018'
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    c, m, frozen = [pd.read_csv(root / path) for path in INPUTS]
    for frame in [c, m, frozen]:
        assert not frame.duplicated(KEYS).any()
    baseline = c[c.visit_week.eq(0)].copy()
    clinical = c[c.age.ge(12) & ~c.dose_ambiguous].copy()
    linked = m.merge(c, on=KEYS, how='inner', validate='one_to_one')
    assert len(linked) == len(m), 'Unlinked serum rows must be explicitly reconciled.'
    selected = linked[linked.visit_week.isin([0, 12, 24])].copy()
    conditions = [('age_below_12_or_unknown', lambda d: d.age.ge(12))]
    conditions += [('missing_' + col, lambda d, col=col: np.isfinite(d[col])) for col in FEATURES]
    reconstructed, exclusions = exclusive_masks(selected, conditions)
    comparison = reconstructed[KEYS + FEATURES].merge(
        frozen[KEYS + ['observed', 'bsa', 'log2_CCL17', 'log2_CCL26', 'log2_IL-5']],
        on=KEYS, how='outer', indicator=True, validate='one_to_one', suffixes=('', '_frozen'))
    assert comparison['_merge'].eq('both').all()
    assert np.allclose(comparison.easi_total, comparison.observed)
    for col in FEATURES[1:]:
        assert np.allclose(comparison[col], comparison[col + '_frozen'])
    development = frozen[frozen.visit_week.isin([0, 12])]
    temporal = frozen[frozen.visit_week.eq(24)]
    nodes, edges = [], []

    def node(key, label, frame, unit='records', note=''):
        nodes.append(dict(node=key, label=label, n_patients=frame.patient_uid.nunique(),
                          n_records=len(frame), record_unit=unit, note=note))

    node('clinical_source', 'Curated clinical source', c, 'scheduled slots',
         '170 patients each have ten predefined visit slots; empty slots are not observed visits.')
    node('clinical_observed_slots', 'Slots with any recorded visit data', c[c.any_raw_visit_data], 'nonempty slots')
    node('clinical_age_eligible', 'Age at least 12', c[c.age.ge(12)], 'scheduled slots')
    node('clinical_main', 'Clinical treatment comparisons', clinical, 'scheduled slots',
         'Age at least 12 and unambiguous initial dose; not the parent of the model population.')
    node('molecular_source', 'Linked serum source', linked, 'serum rows', 'W0, W12, W24 and W48.')
    node('molecular_target_weeks', 'Serum rows at W0, W12 or W24', selected, 'serum rows')
    node('molecular_age_eligible', 'Serum rows at age at least 12', selected[selected.age.ge(12)], 'serum rows')
    node('model_complete', 'Complete EASI, BSA and three-marker visits', frozen, 'complete visits',
         'Initial dose ambiguity is retained because treatment is not an input; no clinical outcome imputation.')
    node('development', 'W0/W12 development', development, 'complete visits')
    node('temporal', 'W24 within-cohort temporal application', temporal, 'complete visits',
         'Previously inspected W24; overlaps development participants and is not external validation.')
    for w in [0, 12, 24]:
        node(f'model_W{w}', f'Model W{w}', frozen[frozen.visit_week.eq(w)], 'complete visits')
    for a, b, relationship in [
        ('clinical_source', 'clinical_observed_slots', 'descriptive subset'),
        ('clinical_source', 'clinical_age_eligible', 'age eligibility'),
        ('clinical_age_eligible', 'clinical_main', 'initial-dose eligibility'),
        ('clinical_source', 'molecular_source', 'linked independent serum availability branch'),
        ('molecular_source', 'molecular_target_weeks', 'time window'),
        ('molecular_target_weeks', 'molecular_age_eligible', 'age eligibility'),
        ('molecular_age_eligible', 'model_complete', 'complete observed inputs and outcome'),
        ('model_complete', 'development', 'visit partition; overlapping people'),
        ('model_complete', 'temporal', 'visit partition; overlapping people')]:
        target = next(r for r in nodes if r['node'] == b)
        edges.append(dict(source=a, target=b, relationship=relationship,
                          n_patients=target['n_patients'], n_records=target['n_records']))
    clinical_exclusions = [
        dict(branch='clinical', reason='age_below_12', n_patients=int(baseline.age.lt(12).sum()), n_records=int(c.age.lt(12).sum())),
        dict(branch='clinical', reason='age_unknown', n_patients=int(baseline.age.isna().sum()), n_records=int(c.age.isna().sum())),
        dict(branch='clinical', reason='ambiguous_initial_dose_after_age_filter',
             n_patients=int((baseline.age.ge(12) & baseline.dose_ambiguous).sum()),
             n_records=int((c.age.ge(12) & c.dose_ambiguous).sum())),
        dict(branch='molecular_time', reason='W48_outside_current_model_scope',
             n_patients=int(linked[~linked.visit_week.isin([0, 12, 24])].patient_uid.nunique()),
             n_records=int((~linked.visit_week.isin([0, 12, 24])).sum()))]
    sets = {'clinical_main': set(clinical.patient_uid), 'model_complete': set(frozen.patient_uid),
            'development': set(development.patient_uid), 'temporal': set(temporal.patient_uid),
            'molecular_source': set(linked.patient_uid)}
    overlaps = []
    for one, two in [('clinical_main', 'model_complete'), ('development', 'temporal'),
                     ('clinical_main', 'molecular_source')]:
        a, b = sets[one], sets[two]
        overlaps.append(dict(population1=one, population2=two, n1=len(a), n2=len(b),
                             intersection=len(a & b), only1=len(a - b), only2=len(b - a), union=len(a | b)))
    nonoverlap = []
    clinical_only = sets['clinical_main'] - sets['model_complete']
    for uid in clinical_only:
        serum = linked[linked.patient_uid.eq(uid)]
        target = selected[selected.patient_uid.eq(uid)]
        if serum.empty:
            reason = 'no_linked_serum_record'
        elif target.empty:
            reason = 'serum_only_outside_W0_W12_W24'
        else:
            current = target.copy(); reason = 'unresolved'
            for label, predicate in conditions:
                current = current[predicate(current)]
                if current.empty:
                    reason = label; break
        assert reason != 'unresolved'
        nonoverlap.append(dict(population='clinical_only', reason=reason))
    for uid in sets['model_complete'] - sets['clinical_main']:
        row = baseline[baseline.patient_uid.eq(uid)].iloc[0]
        reason = 'ambiguous_initial_dose_retained_in_treatment_free_model' if row.dose_ambiguous else 'other_clinical_eligibility_difference'
        nonoverlap.append(dict(population='model_only', reason=reason))
    nonoverlap = pd.DataFrame(nonoverlap).groupby(['population', 'reason']).size().reset_index(name='n_patients')
    visits = []
    for population, frame in [('clinical_source', c), ('clinical_main', clinical), ('model_complete', frozen)]:
        for week, sub in frame.groupby('visit_week'):
            row = dict(population=population, week=week, n_slots=len(sub), n_patients=sub.patient_uid.nunique())
            if 'any_raw_visit_data' in sub:
                row['n_nonempty_slots'] = int(sub.any_raw_visit_data.sum())
            for col in ['easi_total', 'observed', 'bsa', 'pp_nrs', 'dlqi', 'iga']:
                if col in sub:
                    row['n_observed_' + col] = int(sub[col].notna().sum())
            visits.append(row)
    tables = {'flow_nodes': nodes, 'flow_edges': edges, 'flow_exclusions': clinical_exclusions,
              'model_ordered_exclusions': exclusions, 'cohort_overlap': overlaps,
              'visit_inventory': visits, 'model_visit_patterns': membership_patterns(frozen),
              'cohort_nonoverlap_reasons': nonoverlap}
    public = root / 'manuscript/source_data'; public.mkdir(parents=True, exist_ok=True)
    for name, data in tables.items():
        frame = pd.DataFrame(data)
        frame.to_csv(out / (name + '.csv'), index=False)
        frame.to_csv(public / (name + '.csv'), index=False)
    checks = dict(unique_patient_visit_rows=True, serum_rows_linked_to_clinical_source=True,
                  reconstructed_frozen_model_rows_and_values=True,
                  clinical_and_model_not_assumed_nested=True,
                  overlapping_development_temporal_people_explicit=True,
                  model_rows=int(len(frozen)), model_patients=int(frozen.patient_uid.nunique()))
    manifest = dict(status='PASS', checks=checks,
                    inputs=[dict(path=p, sha256=sha(root / p)) for p in INPUTS],
                    source_sha256=sha(Path(__file__)),
                    tables=[dict(path=str(p.relative_to(root)), sha256=sha(p))
                            for p in sorted(out.glob('*.csv')) if p.stem in tables],
                    exclusions_note='Ordered missing-field exclusions are mutually exclusive at the visit level. Patients may have another retained visit; excluded rows are not excluded patients.',
                    recruitment_boundary='The source begins with 170 curated patients; screened or consented patients before this source are not documented by these inputs and cannot be invented.')
    (report / 'cohort_flow_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (report / 'cohort_flow_methods.md').write_text(
        '# Parallel population flow\n\nClinical comparisons and model development use different eligibility rules. '
        'Start from the curated clinical roster and show separate clinical and linked-serum branches; '
        'connect the resulting populations with an overlap annotation, not a subset arrow. '
        'Clinical source rows are predefined visit slots. Counts of nonempty slots and observed scores are supplied separately. '
        'The model source is reconstructed from current clinical and serum rows, with exact patient/visit and input-value equality to the frozen input. '
        'W0/W12 and W24 partition visits, not people. W24 is previously inspected within-cohort temporal application. '
        'All outputs here are aggregate counts without identifiers.\n')
    print(json.dumps(dict(status='PASS', nodes=nodes, overlaps=overlaps), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])))
    run(parser.parse_args().root)
