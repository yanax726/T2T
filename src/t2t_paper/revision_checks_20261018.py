"""Post-review aggregate audits; never refit or overwrite the frozen EASI model.

Run: PYTHONPATH=src .venv/bin/python -m t2t_paper.revision_checks_20261018
Intervals for prediction diagnostics condition on saved held-out predictions.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from t2t_paper.balance_support_20261018 import effective_n, standardized_differences
from t2t_paper.model_display_evidence_20261018 import arrays
from cytot2t.curation.clinical_evidence_20261017 import or_rule, and_rule
from cytot2t.curation.update_20261002_analysis import parse

ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
BASE = ['easi_total', 'bmi', 'age', 'sex_male']
SEED = 20261018


def propensity(b, include_itch=False):
    X = b[BASE + (['pp_nrs'] if include_itch else [])].astype(float).copy()
    X['HUASHAN'] = b.center.eq('HUASHAN').astype(float)
    fit = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)).fit(X, b.treatment)
    p = fit.predict_proba(X)
    own = p[np.arange(len(b)), [list(fit[-1].classes_).index(t) for t in b.treatment]]
    return (1 / (1 / p).sum(axis=1)) / own


def run(root, model_bootstraps=2000, treatment_bootstraps=1000):
    root = Path(root)
    out = root / 'results/tables/update_20261017/revision2_20261018'
    report = root / 'reports/update_20261018/revision2'
    public = root / 'manuscript/source_data'
    for p in [out, report, public]:
        p.mkdir(parents=True, exist_ok=True)
    inputs = [root / p for p in [
        'results/tables/update_20260925/clinical.csv',
        'results/tables/update_20261017/clinical/restricted/endpoint_values.csv',
        'results/tables/update_20261017/clinical/restricted/overlap_patient_weights.csv',
        'results/tables/update_20261017/clinical/endpoint_rates.csv',
        'results/tables/update_20261017/model/predictions.csv',
        'results/tables/update_20261017/model/overall_metrics.csv',
        'results/tables/update_20261017/model/formula.json',
        'results/tables/update_20261002/exposure_at_visits.csv',
        'results/tables/update_20261002/exposure_source_repaired.csv',
        'results/tables/update_20261017/redesign_20261018/monitoring_response_uncertainty.csv']]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    outputs = []

    def save(rows, name):
        d = pd.DataFrame(rows)
        assert not set(d.columns) & {'patient_uid', 'patient_id', 'source_record_id'}
        d.to_csv(out / (name + '.csv'), index=False)
        d.to_csv(public / (name + '.csv'), index=False)
        outputs.append(name + '.csv')
        return d

    clinical = pd.read_csv(inputs[0])
    clinical = clinical[clinical.age.ge(12) & ~clinical.dose_ambiguous].copy()
    baseline = clinical[clinical.visit_week.eq(0)].reset_index(drop=True)
    assert len(baseline) == baseline.patient_uid.nunique() == 163
    endpoint = pd.read_csv(inputs[1])
    original_w = pd.read_csv(inputs[2]).set_index('patient_uid').reindex(baseline.patient_uid).weight.to_numpy()
    assert np.allclose(propensity(baseline), original_w, rtol=0, atol=1e-12)

    # Three-valued logic is tested exhaustively, then reconciled to saved patients.
    truth = []
    for a, b in itertools.product([0., 1., np.nan], repeat=2):
        expected_or = 1. if a == 1 or b == 1 else (0. if a == 0 and b == 0 else np.nan)
        expected_and = 0. if a == 0 or b == 0 else (1. if a == 1 and b == 1 else np.nan)
        for operation, fn, expected in [('OR', or_rule, expected_or), ('AND', and_rule, expected_and)]:
            actual = fn([a, b])
            assert (np.isnan(actual) and np.isnan(expected)) or actual == expected
            truth.append(dict(operation=operation, component1=a, component2=b, result=actual))
    save(truth, 'composite_truth_table')
    counts = []
    for (week, arm), g in endpoint[endpoint.week.isin([4, 12, 24, 48])].groupby(['week', 'treatment']):
        wide = g.pivot(index='patient_uid', columns='target', values='value')
        ages = g.groupby('patient_uid').age.first().reindex(wide.index)
        pp = wide['PP_NRS_le4' if week == 4 else 'PP_NRS_01']
        dlqi = wide['DLQI_le10' if week == 4 else 'DLQI_le4']
        assert dlqi[ages.lt(18)].isna().all()
        p = np.array([or_rule([x, y]) for x, y in zip(pp, dlqi)])
        assert np.allclose(p, wide.patient, equal_nan=True)
        combined = np.array([and_rule([x, y]) for x, y in zip(wide.physician, p)])
        assert np.allclose(combined, wide.overall, equal_nan=True)
        parts = [wide['EASI50' if week == 4 else ('EASI90' if week == 48 else 'EASI75')], wide['IGA_drop2' if week == 4 else 'IGA_01']]
        if week in [24, 48]:
            parts.append(wide['EASI_le5' if week == 24 else 'EASI_le3'])
        physician = np.array([or_rule(x) for x in zip(*parts)])
        assert np.allclose(physician, wide.physician, equal_nan=True)
        for age_label, use in [('12-17', ages.lt(18)), ('18+', ages.ge(18))]:
            for target in ['physician', 'patient', 'overall']:
                v = wide.loc[use, target]
                # Sensitivity counts only: treat inapplicable adult DLQI as absent
                # from the adolescent rule, so PP-NRS alone defines that domain.
                alt = pp.loc[use] if target == 'patient' else pd.Series(
                    [and_rule([x, y]) for x, y in zip(wide.loc[use, 'physician'], pp.loc[use])], index=v.index)
                changed = int((v.isna() & alt.notna()).sum()) if age_label == '12-17' and target != 'physician' else 0
                counts.append(dict(week=week, treatment=arm, age_group=age_label, target=target,
                                   n_initial=int(use.sum()), n_success=int(v.eq(1).sum()), n_failure=int(v.eq(0).sum()),
                                   n_unknown=int(v.isna().sum()), newly_determinate_if_adolescent_PP_only=changed,
                                   active_paper_composite=week != 48))
    save(counts, 'composite_age_evaluability_audit')

    # Explain source-field discordance without changing original treatment labels.
    exposure = pd.read_csv(inputs[7])
    exposure = exposure[exposure.patient_uid.isin(baseline.patient_uid) & exposure.visit_week.eq(0)]
    source = pd.read_csv(inputs[8]).fillna('')
    assert not source.patient_uid.duplicated().any()
    e = exposure.merge(source, on='patient_uid', validate='one_to_one')
    assert e.initial_treatment.eq(e.treatment).all()
    assert len(e) == 163
    audit = []
    for row in e.itertuples():
        initial_segments = [s for s in parse(row) if s['start_week'] == 0]
        expected = 100 if row.initial_treatment == ARMS[1] else 200
        mismatch = bool(initial_segments and row.initial_treatment != ARMS[0] and initial_segments[0]['dose_mg'] != expected)
        assert mismatch == row.initial_conflict
        audit.append(dict(initial_treatment=row.initial_treatment, previsit_regimen=row.previsit_regimen,
                          initial_conflict=mismatch, source_treatment_agrees_with_group=True,
                          interpretation='source regimen at week zero conflicts with initial group; unresolved, no reclassification' if mismatch else 'source regimen state; unknown states retained'))
    save(pd.DataFrame(audit).groupby(list(audit[0]), dropna=False).size().reset_index(name='n'), 'exposure_W0_source_adjudication')
    assert sum(x['initial_conflict'] for x in audit) == 3

    # Endpoint-specific balance and ESS use the actual available-case samples.
    support, balance = [], []
    for week, target in [(2, 'PP_NRS_le4'), (24, 'overall')]:
        v = endpoint[endpoint.week.eq(week) & endpoint.target.eq(target)].set_index('patient_uid').value
        y = v.reindex(baseline.patient_uid).to_numpy()
        X = baseline[BASE + ['pp_nrs']].copy()
        X['HUASHAN'] = baseline.center.eq('HUASHAN').astype(float)
        use = np.isfinite(y)
        # Missing baseline itch is excluded only for that covariate's balance calculation.
        for covariate in X:
            ok = use & X[covariate].notna().to_numpy()
            for label, w in [('unweighted', np.ones(len(baseline))), ('overlap_weighted', original_w)]:
                balance += standardized_differences(baseline.loc[ok], X.loc[ok, [covariate]], w[ok], label, f'W{week}_{target}_available')
        for arm in ARMS:
            g = baseline.treatment.eq(arm).to_numpy()
            support.append(dict(week=week, target=target, treatment=arm, model='original_baseline_weights',
                                n_initial=int(g.sum()), n_evaluable=int((g & use).sum()), ESS=effective_n(original_w[g & use])))
    save(balance, 'endpoint_available_balance')
    save(support, 'endpoint_available_ESS')

    yall = endpoint[endpoint.week.eq(2) & endpoint.target.eq('PP_NRS_le4')].set_index('patient_uid').value
    complete = baseline.dropna(subset=['pp_nrs']).reset_index(drop=True)
    estimates, contrasts, sensitivity_balance = [], [], []
    for label, b, add_itch in [('original_full_cohort', baseline, False),
                               ('original_covariates_itch_complete', complete, False),
                               ('plus_baseline_itch', complete, True)]:
        y = yall.reindex(b.patient_uid).to_numpy()
        w = propensity(b, add_itch)
        arm = b.treatment.to_numpy()
        groups = [np.flatnonzero(arm == a) for a in ARMS]

        def means(values, ww, labels):
            return np.array([np.average(values[(labels == a) & np.isfinite(values)], weights=ww[(labels == a) & np.isfinite(values)]) for a in ARMS])

        point = means(y, w, arm)
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(treatment_bootstraps):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            bb = b.iloc[idx].reset_index(drop=True)
            draws.append(means(y[idx], propensity(bb, add_itch), arm[idx]))
        draws = np.asarray(draws)
        X = b[BASE + ['pp_nrs']].copy(); X['HUASHAN'] = b.center.eq('HUASHAN').astype(float)
        for covariate in X:
            ok = np.isfinite(y) & X[covariate].notna().to_numpy()
            sensitivity_balance += standardized_differences(b.loc[ok], X.loc[ok, [covariate]], w[ok], label, 'W2_PP_NRS_le4_available')
        for k, a in enumerate(ARMS):
            use = arm == a; observed = use & np.isfinite(y)
            numerator = np.nansum(y[use] * w[use]); total = w[use].sum()
            lo, hi = np.quantile(draws[:, k], [.025, .975])
            estimates.append(dict(model=label, treatment=a, n_initial=int(use.sum()), n_evaluable=int(observed.sum()),
                                  weighted_rate=point[k], ci_low=lo, ci_high=hi, ESS=effective_n(w[observed]),
                                  lower_missing_bound=numerator / total,
                                  upper_missing_bound=(numerator + w[use & ~np.isfinite(y)].sum()) / total,
                                  bootstrap_replicates=treatment_bootstraps))
        temp = []
        for i, j in [(1, 0), (2, 0), (1, 2)]:
            delta = draws[:, i] - draws[:, j]; lo, hi = np.quantile(delta, [.025, .975])
            estimate = point[i] - point[j]; se = delta.std(ddof=1)
            temp.append(dict(model=label, comparison=ARMS[i] + ' minus ' + ARMS[j], risk_difference=estimate,
                             ci_low=lo, ci_high=hi, p=2 * norm.sf(abs(estimate / se)), bootstrap_replicates=treatment_bootstraps))
        for row, adjusted in zip(temp, multipletests([r['p'] for r in temp], method='holm')[1]):
            row['p_holm'] = adjusted
        contrasts.extend(temp)
    save(estimates, 'W2_baseline_itch_sensitivity_rates')
    save(contrasts, 'W2_baseline_itch_sensitivity_contrasts')
    save(sensitivity_balance, 'W2_baseline_itch_sensitivity_balance')
    rates = pd.read_csv(inputs[3])
    save(rates[rates.week.eq(2) | (rates.week.eq(24) & rates.target.eq('overall'))], 'headline_missing_outcome_bounds')

    # Pair comparators within patient, visit, repeat and fold. No fit is changed.
    predictions = pd.read_csv(inputs[4]); primary = predictions[predictions.model.eq('unified_selected')]
    reference_columns = ['easi_head_neck', 'easi_upper_limb', 'easi_trunk', 'easi_lower_limb']
    raw_reference = pd.read_csv(inputs[0])
    ref = primary.merge(raw_reference[['patient_uid', 'visit_week', 'regional_total_diff', 'regional_reliable'] + reference_columns],
                        on=['patient_uid', 'visit_week'], validate='many_to_one')
    reference_audit = []
    for week, g in ref.groupby('visit_week'):
        first = g[g['repeat'].eq(g['repeat'].min())]
        complete = first[reference_columns].notna().all(axis=1)
        assert np.array_equal(first.source_contradiction.to_numpy(), (complete & first.regional_total_diff.abs().gt(.15)).to_numpy())
        strict = g[g.regional_reliable]
        reference_audit.append(dict(week=week, n_model_visits=len(first), n_complete_regional_reference=int(complete.sum()),
                                    n_recorded_contradiction=int(first.source_contradiction.sum()),
                                    n_regional_reference_unknown=int((~complete).sum()),
                                    largest_absolute_total_region_gap=first.loc[complete, 'regional_total_diff'].abs().max(),
                                    n_strict_consistent_visits=strict.patient_uid.nunique(),
                                    primary_MAE=g.assign(error=abs(g.prediction-g.observed)).groupby('repeat').error.mean().mean(),
                                    strict_consistent_MAE=strict.assign(error=abs(strict.prediction-strict.observed)).groupby('repeat').error.mean().mean(),
                                    strict_definition='all four regions observed and absolute total-minus-regional-sum difference <=0.01',
                                    sensitivity_scope='restrict evaluation of unchanged saved predictions; no training exclusion or refit'))
    save(reference_audit, 'model_reference_QC_audit')
    comparison, calibration = [], []
    for week, g in primary.groupby('visit_week'):
        y, p = arrays(g)
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(y), size=(model_bootstraps, len(y)))
        for model in ['bsa_area', 'bsa_linear', 'serum_linear', 'unified_area_interaction']:
            comparator = predictions[predictions.model.eq(model) & predictions.visit_week.eq(week)]
            joint = g.merge(comparator[['patient_uid', 'repeat', 'fold', 'prediction']], on=['patient_uid', 'repeat', 'fold'], validate='one_to_one', suffixes=('', '_control'))
            assert len(joint) == len(g)
            yy, pp = arrays(joint); _, q = arrays(joint, 'prediction_control')
            assert np.allclose(yy, y)
            per_patient = (abs(pp - yy) - abs(q - yy)).mean(axis=0)
            boot = per_patient[indices].mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            comparison.append(dict(week=week, primary='unified_selected', comparator=model,
                                   difference='primary minus comparator MAE', estimate=per_patient.mean(), ci_low=lo, ci_high=hi,
                                   n_patients=len(y), n_repeats=p.shape[0], bootstrap_replicates=model_bootstraps,
                                   interval_scope='patient bootstrap conditional on saved predictions'))

        def calibrate(idx):
            yy, pp = y[idx], p[:, idx]
            slopes = np.array([np.mean((v - v.mean()) * (yy - yy.mean())) / np.var(v) if np.var(v) > 0 else np.nan for v in pp])
            return np.array([np.nanmean(yy.mean() - slopes * pp.mean(axis=1)), np.nanmean(slopes), (pp - yy).mean()])

        point = calibrate(np.arange(len(y))); boot = np.array([calibrate(idx) for idx in indices])
        lo, hi = np.nanquantile(boot, [.025, .975], axis=0)
        old = pd.read_csv(inputs[5]); old = old[old.model.eq('unified_selected') & old.week.eq(week)].iloc[0]
        for j, metric in enumerate(['calibration_intercept', 'calibration_slope', 'bias']):
            assert abs(point[j] - old[metric]) < 1e-12
            calibration.append(dict(week=week, metric=metric, estimate=point[j], ci_low=lo[j], ci_high=hi[j],
                                    n_patients=len(y), n_repeats=p.shape[0], bootstrap_replicates=model_bootstraps,
                                    valid_bootstrap_replicates=int(np.isfinite(boot[:, j]).sum())))
    save(comparison, 'primary_paired_control_MAE_uncertainty')
    save(calibration, 'primary_calibration_uncertainty')
    response = pd.read_csv(inputs[9])
    response = response[response.endpoint.eq('EASI75')].copy()
    class_path = root / 'results/tables/update_20261017/redesign_20261018/monitoring_confusion_counts.csv'
    inputs.append(class_path)
    hashes[str(class_path.relative_to(root))] = hashlib.sha256(class_path.read_bytes()).hexdigest()
    classes = pd.read_csv(class_path)
    classes = classes[classes.endpoint.eq('EASI75')]
    for value, label in [(1, 'n_observed_responders'), (0, 'n_observed_nonresponders')]:
        group = classes[classes.observed_response.eq(value)].groupby('window').n_observed_class
        assert group.nunique().eq(1).all()
        response[label] = response.window.map(group.first())
    response['bootstrap_seed'] = 20261002
    save(response, 'primary_EASI75_uncertainty')
    families = []
    family_specs = {
        'molecular_paired_changes': ['family'],
        'molecular_adjusted_contrasts': ['scope', 'analysis'],
        'molecular_global_interactions': ['family'],
        'molecular_baseline_clinical_correlations': ['family'],
        'molecular_clinical_change_correlations': ['family'],
        'doctor_stratified_response_contrasts': ['family'],
        'endpoint_tests': ['family'],
        'endpoint_weighted_contrasts': ['family']}
    for name, keys in family_specs.items():
        path = root / 'results/tables/update_20261017/clinical' / (name + '.csv')
        if not path.exists():
            continue
        inputs.append(path)
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        d = pd.read_csv(path)
        for key, g in d.groupby(keys):
            label = '|'.join(map(str, key if isinstance(key, tuple) else (key,)))
            families.append(dict(source=name + '.csv', correction='BH exploratory', family=label,
                                 n_source_cells=len(g), n_finite_tests=int(g.p.notna().sum())))
        if name.startswith('endpoint_'):
            for key, g in d[d.family.eq('three_pairwise')].groupby(['week', 'target']):
                families.append(dict(source=name + '.csv', correction='Holm', family='|'.join(map(str, key)),
                                     n_source_cells=len(g), n_finite_tests=int(g.p.notna().sum())))
    save(families, 'multiplicity_family_inventory')
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'Frozen input changed during audit.'
    manifest = dict(status='PASS', seed=SEED, model_bootstraps=model_bootstraps, treatment_bootstraps=treatment_bootstraps,
                    inputs=hashes, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), outputs=outputs,
                    frozen_model_refit=False, frozen_inputs_unchanged=True,
                    prediction_interval_scope='Patient percentile bootstrap, all repeats carried together; metric computed within repeat before averaging; no selection or fitting repeated.',
                    clinical_sensitivity='Stratified patient bootstrap; standardized multinomial logistic C=1; propensity refit each resample. Complete-baseline-itch cohort compared with and without itch covariate. No follow-up imputation or censoring adjustment.',
                    exposure_adjudication='Three W0 source regimen/group-label conflicts reproduced from source fields, not evidence of post-baseline switching; labels retained, clinical record adjudication unresolved.',
                    composite_logic='For ages 12-17 adult DLQI is unknown in the saved composite; PP-NRS failure alone therefore leaves patient-domain unknown. PP-only sensitivity counts are not replacements for primary targets.')
    (report / 'statistical_checks_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', outputs=len(outputs), frozen_inputs_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.root)
