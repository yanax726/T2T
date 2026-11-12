"""All-available laboratory trajectories and explicitly assumed W4 interpolation.

Observed laboratory records remain unchanged. Empty treatment/visit cells are not
estimated by a categorical model: optional W4 points are endpoint interpolations
and are excluded from every treatment comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell(arm, week):
    return f'{arm}__{int(week)}'


def load(root, cfg):
    c = pd.read_csv(root / cfg['source'])
    c = c[c.age.ge(12) & ~c.dose_ambiguous].copy()
    assert c.patient_uid.nunique() == 163
    assert not c.duplicated(['patient_uid', 'visit_week']).any()
    b = c[c.visit_week.eq(0)].copy().set_index('patient_uid').sort_index()
    assert b[['age', 'bmi', 'sex_male', 'easi_total']].notna().all().all()
    return c, b


def coverage(c, b, cfg):
    rows, patterns = [], []
    for outcome, item in cfg['outcomes'].items():
        col = item['column']
        for (centre, arm), base in b.groupby(['center', 'treatment']):
            z = c[c.patient_uid.isin(base.index)]
            for week in [0, 4, 12, 24]:
                x = z[z.visit_week.eq(week)][col].dropna()
                rows.append(dict(outcome=outcome, centre=centre, treatment=arm, week=week,
                    n_initial=len(base), n_observed=len(x), n_missing=len(base)-len(x),
                    median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75))))
            p = z[z.visit_week.isin(item['observed_weeks'])].pivot(index='patient_uid', columns='visit_week', values=col)
            counts = p.notna().sum(1).value_counts()
            for count in range(len(item['observed_weeks'])+1):
                patterns.append(dict(outcome=outcome, centre=centre, treatment=arm,
                    n_observed_visits=count, n_patients=int(counts.get(count, 0)),
                    eligible_visits='/'.join(map(str, item['observed_weeks']))))
    return pd.DataFrame(rows), pd.DataFrame(patterns)


def fit_profile(c, b, outcome, item, scope, model_name):
    col = item['column']
    if scope == 'CAMS':
        b = b[b.center.eq('CAMS')]
    d = c[c.patient_uid.isin(b.index) & c.visit_week.isin(item['observed_weeks']) & c[col].notna()].copy()
    d['y'] = np.log1p(d[col])
    d['baseline_EASI'] = np.log1p(d.patient_uid.map(b.easi_total))
    d['sex'] = d.sex_male
    d['week'] = d.visit_week.astype(int)
    d['cell'] = [cell(a, w) for a, w in zip(d.treatment, d.week)]
    reference = b.assign(baseline_EASI=np.log1p(b.easi_total), sex=b.sex_male)
    formula = 'y ~ 0 + C(cell) + baseline_EASI + age + bmi + sex'
    if b.center.nunique() > 1:
        formula += ' + C(center)'
    if model_name == 'centre_by_visit':
        for week in item['observed_weeks'][1:]:
            name = f'centre_w{week}'
            d[name] = (d.center.eq('HUASHAN') & d.week.eq(week)).astype(float)
            formula += ' + '+name
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = MixedLM.from_formula(formula, groups='patient_uid', re_formula='1', data=d)
        x = np.asarray(model.exog)
        assert np.linalg.matrix_rank(x) == x.shape[1], (outcome, scope, model_name, 'design rank')
        fit = model.fit(reml=False, method=['lbfgs', 'powell'], maxiter=1000, disp=False)
    assert fit.converged
    beta = fit.fe_params.to_numpy()
    cov = fit.cov_params().iloc[:len(beta), :len(beta)].to_numpy()
    assert np.isfinite(beta).all() and np.isfinite(cov).all()
    assert np.linalg.eigvalsh(cov).min() > -1e-8
    xtvx = np.zeros((len(beta), len(beta)))
    xtvy = np.zeros(len(beta))
    for indices in d.reset_index(drop=True).groupby('patient_uid').indices.values():
        indices = np.asarray(indices)
        block = np.eye(len(indices))*fit.scale + float(fit.cov_re.iloc[0, 0])
        vi_x = np.linalg.solve(block, x[indices])
        vi_y = np.linalg.solve(block, d.y.to_numpy()[indices])
        xtvx += x[indices].T @ vi_x
        xtvy += x[indices].T @ vi_y
    gls_difference = float(np.max(abs(np.linalg.solve(xtvx, xtvy)-beta)))
    assert gls_difference < 1e-8
    vectors = {}
    for arm in ARMS:
        for week in item['observed_weeks']:
            if not ((d.treatment == arm) & (d.week == week)).any():
                continue
            ref = reference.assign(cell=cell(arm, week), week=week)
            if model_name == 'centre_by_visit':
                for at in item['observed_weeks'][1:]:
                    ref[f'centre_w{at}'] = (ref.center.eq('HUASHAN') & (week == at)).astype(float)
            vectors[arm, week] = np.asarray(build_design_matrices([model.data.design_info], ref)[0]).mean(0)
    info = dict(outcome=outcome, scope=scope, model=model_name, n_initial=len(b),
        n_patients=d.patient_uid.nunique(), n_observations=len(d), n_fixed_parameters=len(beta),
        n_single_observation_patients=int(d.groupby('patient_uid').size().eq(1).sum()),
        n_complete_primary_visit_patients=int(d.groupby('patient_uid').size().eq(len(item['observed_weeks'])).sum()),
        log_likelihood=float(fit.llf), converged=bool(fit.converged),
        random_intercept_variance=float(fit.cov_re.iloc[0, 0]), residual_variance=float(fit.scale),
        independent_GLS_max_absolute_coefficient_difference=gls_difference,
        formula=formula, warnings=[str(w.message) for w in caught],
        standardization_population='all baseline patients in the declared scope')
    coefs = [dict(outcome=outcome, scope=scope, model=model_name, term=name, coefficient=float(v),
                   standard_error=float(np.sqrt(cov[i, i]))) for i, (name, v) in enumerate(zip(model.exog_names, beta))]
    return d, vectors, beta, cov, info, coefs


def summary(v, beta, cov):
    estimate = float(v @ beta)
    variance = float(v @ cov @ v)
    assert variance >= -1e-9
    se = np.sqrt(max(0., variance))
    low, high = estimate - stats.norm.ppf(.975)*se, estimate + stats.norm.ppf(.975)*se
    return dict(log_estimate=estimate, log_ci_low=low, log_ci_high=high, standard_error=se,
                estimate=float(np.expm1(estimate)), ci_low=float(np.expm1(low)), ci_high=float(np.expm1(high)),
                ratio=float(np.exp(estimate)), ratio_low=float(np.exp(low)), ratio_high=float(np.exp(high)))


def profile_outputs(d, vectors, beta, cov, info, cfg):
    levels, changes, contrasts, shapes = [], [], [], []
    fields = {k: info[k] for k in ['outcome', 'scope', 'model', 'n_initial', 'n_patients', 'n_observations']}
    n_cell = d.groupby(['treatment', 'week']).size()
    all_vectors = dict(vectors)
    for arm in ARMS:
        if (arm, 4) not in all_vectors:
            assert (arm, 0) in vectors and (arm, 12) in vectors
            for shape, fraction in cfg['interpolation_sensitivity'].items():
                v = (1-fraction)*vectors[arm, 0]+fraction*vectors[arm, 12]
                z = dict(**fields, treatment=arm, week=4, kind='assumption_based_interpolation',
                    time_shape=shape, fraction=fraction, n_observed_cell=0, **summary(v, beta, cov))
                shapes.append(z)
            all_vectors[arm, 4] = (2/3)*vectors[arm, 0]+(1/3)*vectors[arm, 12]
    for (arm, week), v in sorted(all_vectors.items()):
        kind = 'estimated_from_observed_visits' if (arm, week) in vectors else 'assumption_based_interpolation'
        values = summary(v, beta, cov)
        levels.append(dict(**fields, treatment=arm, week=week, kind=kind,
                          n_observed_cell=int(n_cell.get((arm, week), 0)), **values))
        delta = v-vectors[arm, 0]
        changes.append(dict(**fields, treatment=arm, week=week, kind=kind,
            n_observed_cell=int(n_cell.get((arm, week), 0)), **summary(delta, beta, cov)))
    for week in [12, 24]:
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            a, b = ARMS[i], ARMS[j]
            v = (vectors[a, week]-vectors[a, 0])-(vectors[b, week]-vectors[b, 0])
            z = summary(v, beta, cov)
            p = 2*stats.norm.sf(abs(z['log_estimate']/z['standard_error']))
            contrasts.append(dict(**fields, week=week, comparison=f'{a} - {b}',
                interpretation='Difference in within-group log(1+outcome) change from own baseline',
                p=float(p), **z))
    adjusted = multipletests([r['p'] for r in contrasts], method='holm')[1]
    for r, p in zip(contrasts, adjusted):
        r['p_holm'] = float(p)
        r['family_size'] = 6
    return levels, changes, contrasts, shapes


def main(root=ROOT):
    root = Path(root)
    cfg_path = root/'config/lab_display_20261112.json'
    cfg = json.loads(cfg_path.read_text())
    out, report = root/'results/tables/update_20261112/labs', root/'reports/update_20261112/labs'
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    source = root/cfg['source']
    protected = [source, root/'config/unified_bsa_three_v1_20261017.json',
                 root/'results/tables/update_20261017/clinical/clinical_trajectory_adjusted.csv',
                 root/'results/tables/update_20261025/ige/mi_log_changes.csv',
                 root/'results/tables/update_20261025/ige/mi_adjusted_contrasts.csv']
    before = {str(p.relative_to(root)):digest(p) for p in protected}
    c, b = load(root, cfg)
    counts, patterns = coverage(c, b, cfg)
    counts.to_csv(out/'coverage_by_centre.csv', index=False)
    patterns.to_csv(out/'observation_patterns.csv', index=False)
    levels, changes, contrasts, shapes, diagnostics, coefficients = [], [], [], [], [], []
    for outcome, item in cfg['outcomes'].items():
        for scope, model_name in [('all_163', 'shared_centre'), ('CAMS', 'single_centre'), ('all_163', 'centre_by_visit')]:
            d, vectors, beta, cov, info, coefs = fit_profile(c, b, outcome, item, scope, model_name)
            a, e, f, g = profile_outputs(d, vectors, beta, cov, info, cfg)
            levels.extend(a); changes.extend(e); contrasts.extend(f); shapes.extend(g)
            diagnostics.append(info); coefficients.extend(coefs)
            print(outcome, scope, model_name, info['n_patients'], info['n_observations'], 'converged', flush=True)
    for name, rows in [('longitudinal_levels', levels), ('longitudinal_changes', changes),
                       ('between_treatment_changes', contrasts), ('time_shape_sensitivity', shapes),
                       ('model_coefficients', coefficients)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    pd.DataFrame(diagnostics).drop(columns='warnings').to_csv(out/'model_fit_diagnostics.csv', index=False)
    df = pd.DataFrame(changes)
    old = pd.read_csv(root/'results/tables/update_20261025/ige/available_longitudinal_changes.csv')
    check = []
    for scope, model in [('all_163', 'shared_centre'), ('CAMS', 'single_centre')]:
        a = df[df.outcome.eq('IgE') & df.scope.eq(scope) & df.model.eq(model) & df.week.isin([12, 24])]
        a = a.assign(window='W0_W'+a.week.astype(str)).merge(old[old.scope.eq(scope)], on=['treatment', 'window'], suffixes=('_new', '_old'), validate='one_to_one')
        check.append(dict(scope=scope, n=len(a), max_absolute_log_change_difference=float(abs(a.log_estimate-a.estimate_old).max()),
                          max_absolute_standard_error_difference=float(abs(a.standard_error-(a.ci_high_old-a.ci_low_old)/(2*1.96)).max())))
    pd.DataFrame(check).to_csv(out/'prior_model_reproduction.csv', index=False)
    after = {str(p.relative_to(root)):digest(p) for p in protected}
    assert before == after
    assert counts[counts.outcome.eq('IgE') & counts.week.eq(4)].n_observed.sum() == 0
    assert counts[counts.outcome.eq('EOS') & counts.treatment.eq('dupilumab') & counts.week.eq(4)].n_observed.sum() == 0
    numeric = pd.DataFrame(levels).select_dtypes('number')
    assert np.isfinite(numeric).all().all()
    assert max(x['max_absolute_log_change_difference'] for x in check) < 1e-5
    record = dict(status='PASS', version=cfg['version'], protected_inputs_unchanged=before==after,
        n_clinical=163, observed_record_values_written=False, clinical_validation_truth_created=False,
        interpolated_W4_used_in_significance_tests=False, patient_outcome_rows_exported=False,
        interpolation_requires_temporal_shape_assumption=True,
        prior_IgE_model_reproduction=check, model_diagnostics=diagnostics,
        source_sha256=before, config_sha256=digest(cfg_path),
        outputs_sha256={p.name:digest(p) for p in sorted(out.glob('*.csv'))})
    (report/'verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['status', 'n_clinical', 'protected_inputs_unchanged']}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.root)
