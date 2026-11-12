"""Independently reconcile the current baseline-to-W12 itch-response cohort."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/update_20261112/itch_prediction/cohort_audit'
MARKERS = ['CCL17', 'CCL26', 'IL-5']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'results/tables/update_20260925/clinical.csv',
             ROOT / 'results/tables/redevelopment/molecular.csv',
             ROOT / 'results/tables/redevelopment/assay_long.csv',
             ROOT / 'results/tables/baseline_W12_itch_response_oof.csv',
             ROOT / 'config/endpoint_registry.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    raw, molecular, assay, old = [pd.read_csv(p) for p in paths[:4]]
    registry = json.loads(paths[-1].read_text())
    assert registry['itch_response'] == {
        'baseline_min': 4, 'minimum_reduction': 4, 'unit': 'PP-NRS points'}
    assert not raw.duplicated(['patient_uid', 'visit_week']).any()
    assert not molecular.duplicated(['patient_uid', 'visit_week']).any()
    assert not assay.duplicated(['patient_uid', 'visit_week', 'analyte']).any()
    assert not assay.duplicated(['sample_id', 'analyte']).any()
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    baseline = c[c.visit_week.eq(0)].copy()
    assert len(baseline) == 163
    followup = c[c.visit_week.eq(12)][['patient_uid', 'pp_nrs']].rename(
        columns={'pp_nrs': 'pp_nrs_W12'})
    data = baseline.merge(followup, on='patient_uid', how='left', validate='one_to_one')
    columns = ['log2_' + x for x in MARKERS]
    data = data.merge(molecular[molecular.visit_week.eq(0)][['patient_uid'] + columns],
                      on='patient_uid', how='left', validate='one_to_one')
    data['eligible'] = data.pp_nrs.ge(4) & data.pp_nrs_W12.notna()
    data['response'] = (data.pp_nrs - data.pp_nrs_W12).ge(4)
    data['complete_three'] = np.isfinite(data[columns]).all(axis=1)
    clinical = data[data.eligible]
    primary = data[data.eligible & data.complete_three]
    assert (len(clinical), int(clinical.response.sum())) == (112, 94)
    assert (len(primary), int(primary.response.sum())) == (102, 88)
    assert clinical.pp_nrs.between(4, 10).all() and clinical.pp_nrs_W12.between(0, 10).all()
    counts = []
    for label, frame in [('all_clinical_evaluable', clinical),
                         ('complete_three_baseline_molecules', primary)]:
        for arm, g in [('all', frame)] + list(frame.groupby('treatment')):
            counts.append(dict(population=label, treatment=arm, n=len(g),
                               responders=int(g.response.sum()), nonresponders=int((~g.response).sum()),
                               response_fraction=float(g.response.mean()),
                               age12_to17=int(g.age.between(12, 17).sum())))
    pd.DataFrame(counts).to_csv(OUT / 'cohort_counts.csv', index=False)
    flow = [dict(step='Current clinical comparison population', n=163),
            dict(step='PP-NRS W0 >=4 and W12 observed', n=len(clinical)),
            dict(step='All three baseline molecule readings finite', n=len(primary)),
            dict(step='Primary W12 drop >=4', n=int(primary.response.sum())),
            dict(step='Primary W12 drop <4', n=int((~primary.response).sum()))]
    pd.DataFrame(flow).to_csv(OUT / 'cohort_flow.csv', index=False)
    old = old[old.model.eq('Clinical')].merge(raw[raw.visit_week.eq(0)][
        ['patient_uid', 'age', 'dose_ambiguous']], on='patient_uid', validate='one_to_one')
    old_eligible = old[old.age.ge(12) & ~old.dose_ambiguous]
    missing_old_ids = set(old_eligible.patient_uid) - set(primary.patient_uid)
    assert len(old) == 111 and len(old_eligible) == 105 and len(missing_old_ids) == 3
    assert set(primary.patient_uid) <= set(old_eligible.patient_uid)
    a = assay[assay.visit_week.eq(0) & assay.analyte.isin(MARKERS)]
    assert a[a.patient_uid.isin(missing_old_ids) & a.analyte.eq('CCL26')].state.eq('below_lloq').all()
    missing = []
    for marker in MARKERS:
        j = clinical[['patient_uid']].merge(a[a.analyte.eq(marker)][
            ['patient_uid', 'state']], on='patient_uid', how='left', validate='one_to_one')
        j['state'] = j.state.fillna('no_baseline_assay_row')
        for state, n in j.state.value_counts().items():
            missing.append(dict(marker=marker, state=state, n=int(n)))
    pd.DataFrame(missing).to_csv(OUT / 'baseline_molecular_availability.csv', index=False)
    qc = a.merge(molecular[molecular.visit_week.eq(0)], on=['patient_uid', 'visit_week'],
                 validate='many_to_one')
    errors = []
    for marker, g in qc.groupby('analyte'):
        assert np.allclose(g.log2_concentration, g['log2_' + marker], equal_nan=True)
        h = g[g.state.eq('observed')]
        assert h.concentration.gt(0).all()
        errors.extend(abs(np.log2(h.concentration) - h.log2_concentration))
    report = dict(status='PASS', clinical_initial_n=163,
                  endpoint='Observed W0 PP-NRS >=4 and observed W12 drop >=4 points',
                  all_clinical_n=112, all_clinical_responders=94, all_clinical_nonresponders=18,
                  primary_complete_three_n=102, primary_responders=88, primary_nonresponders=14,
                  historical_n=111, historical_under12=6, historical_age12plus_unambiguous_n=105,
                  additional_excluded_CCL26_below_lloq=3, clinical_evaluable_no_baseline_assay=7,
                  molecular_patient_week_duplicates=0, assay_patient_week_marker_duplicates=0,
                  assay_sample_marker_duplicates=0,
                  maximum_log2_replay_error=float(max(errors)),
                  missing_outcome_imputed=False, input_values_modified=False,
                  linked_by='patient_uid and nominal visit_week; exact dates not verified',
                  preprocessing='Observed positive concentration log2 transformed once; censored/nonpositive/unmeasured remain missing',
                  repeat_handling='Unique integrated assay reading per sample and analyte; no averaging of technical replicates by this audit',
                  source_builder='src/cytot2t/stages/stage_01_data_lock.py:assay_from_raw; src/cytot2t/redevelopment/data_audit.py',
                  clinical_source='results/tables/update_20260925/clinical.csv',
                  molecular_source='results/tables/redevelopment/molecular.csv',
                  clinical_predictor_note='Mask unconfirmed adolescent DLQI or omit it; current manuscript uses adult DLQI from age18 only',
                  source_sha256=hashes)
    assert all(digest(ROOT / name) == expected for name, expected in hashes.items())
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
