"""Aggregate IGA-definition sensitivity without changing observed clinical scores.

Remove only IGA clauses from the current versioned framework. The resulting
definition is a post hoc simplification, not a replacement consensus. Outputs
contain aggregate counts only; patient-level working data remain in memory.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
WEEKS = [4, 12, 24]
STATES = ['failure', 'success', 'unknown']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tri_or(parts):
    if 1 in parts:
        return 1
    return 0 if all(x == 0 for x in parts) else -1


def tri_and(parts):
    if 0 in parts:
        return 0
    return 1 if all(x == 1 for x in parts) else -1


def state(value):
    return {-1: 'unknown', 0: 'failure', 1: 'success'}[value]


def evaluate(expression, current, baseline, adult_minimum):
    match = re.fullmatch(r'(.+?)\s+(<=|>=)\s+([0-9.]+)', expression)
    if not match:
        raise ValueError(f'Unsupported configured rule {expression!r}')
    variable, op, threshold = match.groups()
    threshold = float(threshold)
    if variable == 'easi_pct_improvement':
        start, end = baseline.easi_total, current.easi_total
        if pd.isna(start) or start <= 0 or pd.isna(end):
            return -1
        # Match the frozen endpoint arithmetic at threshold equality.
        assert op == '>='
        return int(end <= start * (1 - threshold / 100))
    if variable == 'iga_baseline - iga':
        value = baseline.iga - current.iga
    elif variable == 'dlqi' and current.age < adult_minimum:
        return -1
    else:
        value = getattr(current, variable)
    if pd.isna(value):
        return -1
    return int(value <= threshold if op == '<=' else value >= threshold)


def wilson(k, n):
    if not n:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return centre-radius, centre+radius


def summary(values):
    n = len(values)
    success = int(values.eq('success').sum())
    failure = int(values.eq('failure').sum())
    unknown = int(values.eq('unknown').sum())
    evaluable = success + failure
    low, high = wilson(success, evaluable)
    assert success + failure + unknown == n
    return dict(n_initial=n, n_evaluable=evaluable, n_success=success,
                n_failure=failure, n_unknown=unknown,
                rate=success/evaluable if evaluable else np.nan,
                ci_low=low, ci_high=high,
                lower_missing_bound=success/n if n else np.nan,
                upper_missing_bound=(success+unknown)/n if n else np.nan)


def run(root):
    out = root / 'results/tables/update_20261031'
    checks = root / 'reports/review_20261031/checks'
    out.mkdir(parents=True, exist_ok=True)
    checks.mkdir(parents=True, exist_ok=True)
    source = root / 'results/tables/update_20260925/clinical.csv'
    config_path = root / 'config/clinical_framework.json'
    existing_path = root / 'results/tables/update_20261017/clinical/endpoint_rates.csv'
    private_endpoint_path = root / 'results/tables/update_20261017/clinical/restricted/endpoint_values.csv'
    input_hashes = {str(p.relative_to(root)): digest(p) for p in [source, config_path, existing_path, private_endpoint_path]}
    raw = pd.read_csv(source)
    cfg = json.loads(config_path.read_text())
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    baseline = c[c.visit_week.eq(0)].set_index('patient_uid')
    assert len(baseline) == 163 and c.patient_uid.nunique() == 163
    assert not c.duplicated(['patient_uid', 'visit_week']).any()
    assert baseline.groupby('treatment').size().to_dict() == dict(zip(ARMS, [65, 56, 42]))
    grade5 = set(c.loc[c.iga.eq(5), 'patient_uid'])
    assert len(grade5) == 22 and c.iga.eq(5).sum() == 22
    assert set(c.loc[c.iga.eq(5), 'visit_week']) == {0}
    # Independent truth-table tests of determining observed components.
    for left, right in itertools.product([-1, 0, 1], repeat=2):
        expected_or = 1 if left == 1 or right == 1 else (0 if left == right == 0 else -1)
        expected_and = 0 if left == 0 or right == 0 else (1 if left == right == 1 else -1)
        assert tri_or([left, right]) == expected_or
        assert tri_and([left, right]) == expected_and
    specs, values = [], []
    for week in WEEKS:
        rule = cfg['visits'][str(week)]
        physician = rule['physician']
        reduced = [x for x in physician if 'iga' not in x]
        assert len(reduced) == len(physician)-1
        for name, clauses in [('recorded_framework', physician), ('iga_free_sensitivity', reduced)]:
            specs.append(dict(week=week, definition=name, physician_rule=' OR '.join(clauses),
                patient_rule=' OR '.join(rule['patient']), combined_rule='physician AND patient',
                unknown_rule='OR true if any observed true; false only if all observed false; otherwise unknown. AND false if any observed false; true only if both true; otherwise unknown.',
                adult_DLQI_minimum_age=cfg['adult_DLQI_analysis_minimum_age'],
                adolescent_DLQI='unknown, not failure and not removed from patient OR rule',
                interpretation='Recorded framework' if name == 'recorded_framework' else 'Post hoc removal of IGA only; not a validated or approved replacement framework',
                config_version=cfg['version']))
        rows = c[c.visit_week.eq(week)]
        assert len(rows) == 163
        for r in rows.itertuples():
            b = baseline.loc[r.patient_uid]
            patient = tri_or([evaluate(x, r, b, cfg['adult_DLQI_analysis_minimum_age']) for x in rule['patient']])
            for name, clauses in [('recorded_framework', physician), ('iga_free_sensitivity', reduced)]:
                ph = tri_or([evaluate(x, r, b, cfg['adult_DLQI_analysis_minimum_age']) for x in clauses])
                for domain, result in [('physician', ph), ('combined', tri_and([ph, patient]))]:
                    values.append(dict(patient_uid=r.patient_uid, treatment=r.treatment, week=week,
                        definition=name, domain=domain, state=state(result), any_recorded_iga5=r.patient_uid in grade5))
    v = pd.DataFrame(values)
    saved = pd.read_csv(private_endpoint_path)
    saved = saved[saved.week.isin(WEEKS) & saved.target.isin(['physician', 'overall'])].copy()
    saved['domain'] = saved.target.map({'physician': 'physician', 'overall': 'combined'})
    saved['saved_state'] = saved.value.map({0: 'failure', 1: 'success'}).fillna('unknown')
    reconciled_patient_values = v[v.definition.eq('recorded_framework')].merge(
        saved[['patient_uid', 'week', 'domain', 'saved_state']],
        on=['patient_uid', 'week', 'domain'], validate='one_to_one', how='outer', indicator=True)
    assert len(reconciled_patient_values) == 163 * 3 * 2
    assert reconciled_patient_values['_merge'].eq('both').all()
    assert reconciled_patient_values.state.eq(reconciled_patient_values.saved_state).all()
    cohort_rows, audits, rates, paired, transitions, reconciliation, exclusion_effect, patient_changes = [], [], [], [], [], [], [], []
    for scope, frame in [('source_170', raw), ('eligible_163', c)]:
        for treatment in ['all'] + ARMS:
            h = frame if treatment == 'all' else frame[frame.treatment.eq(treatment)]
            for week, g in h.groupby('visit_week'):
                audits.append(dict(scope=scope, treatment=treatment, week=int(week), n_records=len(g),
                    n_iga_observed=int(g.iga.notna().sum()), n_iga5_records=int(g.iga.eq(5).sum()),
                    n_iga5_patients=int(g.loc[g.iga.eq(5), 'patient_uid'].nunique()),
                    n_outside_0_to4=int((g.iga.lt(0)|g.iga.gt(4)).sum())))
    for cohort, data in [('all_eligible', v), ('exclude_any_iga5_patient', v[~v.any_recorded_iga5])]:
        for treatment in ['all'] + ARMS:
            g = data if treatment == 'all' else data[data.treatment.eq(treatment)]
            n_full = len(baseline) if treatment == 'all' else int(baseline.treatment.eq(treatment).sum())
            n = g.patient_uid.nunique()
            cohort_rows.append(dict(cohort=cohort, treatment=treatment, n_original=n_full,
                n_retained=n, n_excluded=n_full-n,
                exclusion='none' if cohort == 'all_eligible' else 'Remove every patient with any recorded IGA=5; in these data all are baseline records. Not adjudication of the correct IGA scale.'))
            for (week, domain, definition), group in g.groupby(['week', 'domain', 'definition']):
                rates.append(dict(cohort=cohort, treatment=treatment, week=week, domain=domain,
                                  definition=definition, **summary(group.state)))
            wide = g.pivot(index=['patient_uid', 'week', 'domain'], columns='definition', values='state').reset_index()
            for domain in ['any_domain', 'physician', 'combined']:
                change_rows = wide if domain == 'any_domain' else wide[wide.domain.eq(domain)]
                old, new = change_rows.recorded_framework, change_rows.iga_free_sensitivity
                both = old.ne('unknown') & new.ne('unknown')
                patient_changes.append(dict(cohort=cohort, treatment=treatment, domain=domain,
                    weeks='4;12;24', n_initial=change_rows.patient_uid.nunique(),
                    n_patients_classification_changed_any_visit=change_rows.loc[both & old.ne(new), 'patient_uid'].nunique(),
                    n_patients_evaluability_changed_any_visit=change_rows.loc[old.eq('unknown') != new.eq('unknown'), 'patient_uid'].nunique(),
                    n_patients_any_state_changed_any_visit=change_rows.loc[old.ne(new), 'patient_uid'].nunique(),
                    interpretation='Unique patient union across the specified visits and domain(s); not a sum of visit-level changes'))
            for (week, domain), group in wide.groupby(['week', 'domain']):
                old, new = group.recorded_framework, group.iga_free_sensitivity
                common = old.ne('unknown') & new.ne('unknown')
                n_both = int(common.sum())
                old_k = int((common & old.eq('success')).sum())
                new_k = int((common & new.eq('success')).sum())
                changes = int((common & old.ne(new)).sum())
                paired.append(dict(cohort=cohort, treatment=treatment, week=week, domain=domain,
                    n_initial=len(group), n_both_evaluable=n_both,
                    recorded_success_paired=old_k, iga_free_success_paired=new_k,
                    recorded_rate_paired=old_k/n_both if n_both else np.nan,
                    iga_free_rate_paired=new_k/n_both if n_both else np.nan,
                    paired_rate_difference_iga_free_minus_recorded=(new_k-old_k)/n_both if n_both else np.nan,
                    n_classification_changed_both_evaluable=changes,
                    success_to_failure=int((old.eq('success') & new.eq('failure')).sum()),
                    failure_to_success=int((old.eq('failure') & new.eq('success')).sum()),
                    n_recorded_only_evaluable=int((old.ne('unknown') & new.eq('unknown')).sum()),
                    n_iga_free_only_evaluable=int((old.eq('unknown') & new.ne('unknown')).sum()),
                    n_both_unknown=int((old.eq('unknown') & new.eq('unknown')).sum()),
                    n_any_state_changed=int(old.ne(new).sum())))
                for before, after in itertools.product(STATES, repeat=2):
                    transitions.append(dict(cohort=cohort, treatment=treatment, week=week, domain=domain,
                        recorded_state=before, iga_free_state=after,
                        n=int((old.eq(before) & new.eq(after)).sum())))
                # Removing an OR component cannot turn an observed failure into success.
                assert not (old.eq('failure') & new.eq('success')).any()
                assert n_both + int((~common).sum()) == len(group)
    rates = pd.DataFrame(rates)
    old_rates = pd.read_csv(existing_path)
    for row in rates[rates.cohort.eq('all_eligible') & rates.definition.eq('recorded_framework') & rates.treatment.ne('all')].itertuples():
        target = 'overall' if row.domain == 'combined' else 'physician'
        old = old_rates[old_rates.week.eq(row.week) & old_rates.target.eq(target) & old_rates.treatment.eq(row.treatment)].iloc[0]
        ok = (row.n_initial == old.n_initial and row.n_evaluable == old.n_evaluable and row.n_success == old.k and row.n_unknown == old.not_evaluable)
        reconciliation.append(dict(treatment=row.treatment, week=row.week, domain=row.domain,
            n_initial=row.n_initial, n_evaluable=row.n_evaluable, n_success=row.n_success,
            n_unknown=row.n_unknown, matches_frozen_primary=bool(ok)))
        assert ok
    for (treatment, week, domain, definition), group in rates.groupby(['treatment', 'week', 'domain', 'definition']):
        full = group[group.cohort.eq('all_eligible')].iloc[0]
        kept = group[group.cohort.eq('exclude_any_iga5_patient')].iloc[0]
        exclusion_effect.append(dict(treatment=treatment, week=week, domain=domain, definition=definition,
            n_full=full.n_initial, n_retained=kept.n_initial, n_excluded=full.n_initial-kept.n_initial,
            full_evaluable=full.n_evaluable, retained_evaluable=kept.n_evaluable,
            full_success=full.n_success, retained_success=kept.n_success,
            full_rate=full.rate, retained_rate=kept.rate,
            excluded_success=full.n_success-kept.n_success,
            excluded_failure=full.n_failure-kept.n_failure,
            excluded_unknown=full.n_unknown-kept.n_unknown,
            interpretation='Different populations after exclusion; descriptive sensitivity, not a paired treatment-effect contrast'))
    frames = {'iga_definition_specification': specs, 'iga_grade5_audit': audits,
              'iga_cohort_counts': cohort_rows, 'iga_definition_rates': rates,
              'iga_definition_paired_comparison': paired, 'iga_definition_transitions': transitions,
              'iga_primary_reconciliation': reconciliation, 'iga_grade5_exclusion_effect': exclusion_effect,
              'iga_definition_patient_change_counts': patient_changes}
    for name, frame in frames.items():
        frame = pd.DataFrame(frame)
        assert not any('patient_uid' == col for col in frame.columns)
        frame.to_csv(out / (name+'.csv'), index=False)
    t = pd.DataFrame(transitions)
    for keys, group in t.groupby(['cohort', 'treatment', 'week', 'domain']):
        q = pd.DataFrame(paired)
        for col, key in zip(['cohort', 'treatment', 'week', 'domain'], keys):
            q = q[q[col].eq(key)]
        assert int(group.n.sum()) == int(q.iloc[0].n_initial)
    assert all(digest(root/p) == value for p, value in input_hashes.items())
    check = dict(status='PASS', source_sha256=input_hashes,
        n_eligible_patients=163, n_iga5_records=22, n_iga5_patients=22,
        all_iga5_at_baseline=True, n_retained_after_patient_exclusion=141,
        framework_version=cfg['version'], no_iga_rules={str(w): [x for x in cfg['visits'][str(w)]['physician'] if 'iga' not in x] for w in WEEKS},
        frozen_primary_aggregate_cells_reconciled=len(reconciliation), truth_table_cases_checked=9,
        frozen_primary_patient_endpoint_values_reconciled=len(reconciled_patient_values),
        underlying_clinical_scores_unchanged=True, clinical_outcomes_imputed=False,
        grade5_values_relabelled_or_clipped=False, outputs_aggregate_only=True,
        analysis='Post hoc definition sensitivity; no efficacy, clinical-adoption or consensus-validation claim')
    (checks/'iga_verification_20261031.json').write_text(json.dumps(check, ensure_ascii=False, indent=2)+'\n')
    manifest = dict(inputs=input_hashes, script_sha256=digest(__file__),
        outputs={str((out/(name+'.csv')).relative_to(root)): digest(out/(name+'.csv')) for name in frames})
    (checks/'iga_manifest_20261031.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(check, ensure_ascii=False, indent=2))
    print(pd.DataFrame(paired).query("cohort == 'all_eligible' and treatment == 'all'").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())
