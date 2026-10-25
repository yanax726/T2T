"""Compare simple EASI input sets on identical patient-held-out observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import lstsq
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


MARKERS = ['log2_CCL17', 'log2_CCL26', 'log2_IL-5']
MODEL_INPUTS = {
    'ccl17': ['log2_CCL17'],
    'bsa_ccl17': ['bsa', 'log2_CCL17'],
    'serum3': MARKERS,
    'bsa': ['bsa'],
    'bsa_serum3': ['bsa'] + MARKERS,
    'itch_serum3': ['pp_nrs'] + MARKERS,
    'bsa_itch': ['bsa', 'pp_nrs'],
    'bsa_serum3_itch': ['bsa'] + MARKERS + ['pp_nrs'],
    'bsa_serum3_dlqi': ['bsa'] + MARKERS + ['dlqi'],
    'bsa_itch_dlqi': ['bsa', 'pp_nrs', 'dlqi'],
    'bsa_serum3_itch_dlqi': ['bsa'] + MARKERS + ['pp_nrs', 'dlqi'],
    'bsa_area': ['bsa'],
    'bsa_serum3_area': ['bsa'] + MARKERS,
    'without_CCL17': ['bsa', 'log2_CCL26', 'log2_IL-5'],
    'without_CCL26': ['bsa', 'log2_CCL17', 'log2_IL-5'],
    'without_IL5': ['bsa', 'log2_CCL17', 'log2_CCL26'],
}
LABELS = {
    'ccl17': 'CCL17 alone', 'bsa_ccl17': 'BSA + CCL17',
    'serum3': 'Three serum markers', 'bsa': 'BSA alone',
    'bsa_serum3': 'BSA + three markers', 'itch_serum3': 'PP-NRS + three markers',
    'bsa_itch': 'BSA + PP-NRS', 'bsa_serum3_itch': 'BSA + three markers + PP-NRS',
    'bsa_serum3_dlqi': 'BSA + three markers + adult DLQI',
    'bsa_itch_dlqi': 'BSA + PP-NRS + adult DLQI',
    'bsa_serum3_itch_dlqi': 'BSA + three markers + PP-NRS + adult DLQI',
    'bsa_area': 'BSA alone; fixed area structure',
    'bsa_serum3_area': 'BSA + three markers; fixed area structure',
    'without_CCL17': 'BSA + CCL26 + IL-5; fixed area structure',
    'without_CCL26': 'BSA + CCL17 + IL-5; fixed area structure',
    'without_IL5': 'BSA + CCL17 + CCL26; fixed area structure',
}
METRICS = ['MAE', 'RMSE', 'bias', 'rho', 'CCC', 'R2', 'calibration_slope',
           'calibration_intercept', 'within1', 'within2']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structures(model):
    if model.endswith('_area') or model.startswith('without_'):
        return ['area']
    return ['linear', 'quadratic', 'area'] if 'bsa' in MODEL_INPUTS[model] else ['linear']


class Estimator:
    def __init__(self, frame, model, structure, alpha):
        self.model, self.structure, self.alpha = model, structure, float(alpha)
        self.features = MODEL_INPUTS[model]
        self.initial = StandardScaler().fit(frame[self.features])
        self.zero = structure == 'area'
        self.scale = StandardScaler(with_mean=not self.zero)
        x = self.scale.fit_transform(self.design(frame))
        y = frame.observed.to_numpy(float)
        w = 1 / frame.patient_uid.map(frame.patient_uid.value_counts()).to_numpy()
        self.weight = w / w.mean()
        xm = np.zeros(x.shape[1]) if self.zero else np.average(x, axis=0, weights=self.weight)
        ym = 0.0 if self.zero else np.average(y, weights=self.weight)
        matrix = np.vstack([(x-xm)*np.sqrt(self.weight)[:, None], np.sqrt(alpha)*np.eye(x.shape[1])])
        target = np.r_[(y-ym)*np.sqrt(self.weight), np.zeros(x.shape[1])]
        self.beta = lstsq(matrix, target, lapack_driver='gelsd')[0]
        self.intercept = ym - np.einsum('i,i->', xm, self.beta)

    def design(self, frame):
        x = frame[self.features].to_numpy(float)
        z = self.initial.transform(frame[self.features])
        if self.structure == 'area':
            return np.column_stack([x[:, 0], x[:, :1]*z[:, 1:]])
        if self.structure == 'quadratic':
            return np.column_stack([z, frame.bsa.to_numpy(float)**2])
        return z

    def raw(self, frame):
        return self.intercept + np.einsum('ij,j->i', self.scale.transform(self.design(frame)), self.beta, optimize=False)

    def predict(self, frame):
        return np.clip(self.raw(frame), 0, 72)

    def formula(self):
        beta = self.beta/self.scale.scale_
        origin = self.intercept - np.einsum('i,i->', beta, self.scale.mean_) if self.scale.with_mean else self.intercept
        if self.zero:
            slopes = beta[1:]/self.initial.scale_[1:]
            constant = beta[0]-np.einsum('i,i->', slopes, self.initial.mean_[1:])
            return dict(expression='clip(BSA * (constant + sum(coefficient * input)), 0, 72)',
                        bsa_factor=float(constant), interactions=dict(zip(self.features[1:], map(float, slopes))))
        slopes = beta[:len(self.features)]/self.initial.scale_
        coefficients = dict(zip(self.features, map(float, slopes)))
        if self.structure == 'quadratic':
            coefficients['bsa_squared'] = float(beta[-1])
        return dict(expression='clip(intercept + sum(coefficient * input), 0, 72)',
                    intercept=float(origin-np.einsum('i,i->', slopes, self.initial.mean_)), coefficients=coefficients)


def select(frame, model, cfg):
    ids = np.array(sorted(frame.patient_uid.unique()))
    scores = []
    for structure in structures(model):
        for alpha in cfg['ridge_alpha']:
            errors = []
            for tr, te in KFold(cfg['inner_folds'], shuffle=True, random_state=cfg['seed']).split(ids):
                train, test = frame[frame.patient_uid.isin(ids[tr])], frame[frame.patient_uid.isin(ids[te])]
                assert set(train.patient_uid).isdisjoint(test.patient_uid)
                fit = Estimator(train, model, structure, alpha)
                d = pd.DataFrame(dict(patient_uid=test.patient_uid, error=abs(test.observed-fit.predict(test))))
                errors.extend(d.groupby('patient_uid').error.mean().tolist())
            scores.append(dict(structure=structure, alpha=alpha, inner_patient_MAE=float(np.mean(errors))))
    best = min(scores, key=lambda d: d['inner_patient_MAE'])
    return Estimator(frame, model, best['structure'], best['alpha']), scores


def metric_arrays(y, p):
    """Each row is a patient resample; repeats are the middle axis of p."""
    yy = y[:, None, :]
    e = p-yy
    ym, pm = yy.mean(axis=-1), p.mean(axis=-1)
    yc, pc = yy-ym[..., None], p-pm[..., None]
    vy, vp = (yc**2).mean(axis=-1), (pc**2).mean(axis=-1)
    cov = (yc*pc).mean(axis=-1)
    ry, rp = rankdata(yy, axis=-1), rankdata(p, axis=-1)
    ry, rp = ry-ry.mean(axis=-1, keepdims=True), rp-rp.mean(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = (ry*rp).sum(axis=-1)/np.sqrt((ry**2).sum(axis=-1)*(rp**2).sum(axis=-1))
        slope = cov/vp
        values = dict(MAE=np.mean(abs(e), axis=-1), RMSE=np.sqrt(np.mean(e**2, axis=-1)),
                      bias=e.mean(axis=-1), rho=rho, CCC=2*cov/(vy+vp+(ym-pm)**2),
                      R2=1-np.mean(e**2, axis=-1)/vy, calibration_slope=slope,
                      calibration_intercept=ym-slope*pm, within1=(abs(e)<=1).mean(axis=-1),
                      within2=(abs(e)<=2).mean(axis=-1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return {k: np.nanmean(v, axis=1) for k, v in values.items()}


def aligned_arrays(frame):
    pivot = frame.pivot(index='patient_uid', columns='repeat', values='prediction').sort_index()
    y = frame.drop_duplicates('patient_uid').set_index('patient_uid').loc[pivot.index, 'observed'].to_numpy(float)
    assert not pivot.isna().any().any()
    assert frame.groupby('patient_uid')['observed'].nunique().max() == 1
    return pivot.index.to_numpy(), y, pivot.to_numpy(float).T


def bootstrap_metrics(frame, cfg, draws=None):
    ids, y, p = aligned_arrays(frame)
    point = metric_arrays(y[None, :], p[None, :, :])
    n = len(ids)
    out = {k: float(v[0]) for k, v in point.items()}
    out['n_patients'] = n
    if n >= 5:
        if draws is None:
            draws = np.random.default_rng(cfg['seed']).integers(n, size=(cfg['bootstrap_replicates'], n))
        boot = metric_arrays(y[draws], np.transpose(p[:, draws], (1, 0, 2)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for k, v in boot.items():
                out[k+'_lo'], out[k+'_hi'] = np.nanquantile(v, [.025, .975])
                out[k+'_bootstrap_valid'] = int(np.isfinite(v).sum())
        return out, boot
    out.update({k+s: np.nan for k in METRICS for s in ['_lo', '_hi']})
    out.update({k+'_bootstrap_valid': 0 for k in METRICS})
    return out, None


def summarize(pred, cfg):
    rows = []
    for (cohort, model, week), data in pred.groupby(['cohort', 'model', 'visit_week']):
        subsets = [('overall', 'all', data)]
        for name, mask in [('EASI_le7', data.observed.le(7)), ('EASI_7_to21', data.observed.gt(7)&data.observed.le(21)), ('EASI_gt21', data.observed.gt(21))]:
            if mask.any(): subsets.append(('severity', name, data[mask]))
        for kind, group, sub in subsets:
            values, _ = bootstrap_metrics(sub, cfg)
            rows.append(dict(cohort=cohort, model=model, model_label=LABELS[model], week=week,
                             group_type=kind, group=group, **values))
    return pd.DataFrame(rows)


def comparisons(pred, cfg):
    rows = []
    for cohort, pairs in cfg['paired_contrasts'].items():
        for candidate, reference in pairs:
            for week in cfg['evaluation_weeks']:
                a = pred[pred.cohort.eq(cohort)&pred.model.eq(candidate)&pred.visit_week.eq(week)]
                b = pred[pred.cohort.eq(cohort)&pred.model.eq(reference)&pred.visit_week.eq(week)]
                ia, ya, pa = aligned_arrays(a); ib, yb, pb = aligned_arrays(b)
                assert np.array_equal(ia, ib) and np.array_equal(ya, yb)
                va, ba = bootstrap_metrics(a, cfg); vb, bb = bootstrap_metrics(b, cfg)
                for key in METRICS:
                    diff = va[key]-vb[key]
                    lo, hi = np.nanquantile(ba[key]-bb[key], [.025, .975])
                    rows.append(dict(cohort=cohort, candidate=candidate, reference=reference, week=week,
                                     metric=key, difference=diff, ci_low=lo, ci_high=hi, n_patients=len(ia),
                                     direction='candidate minus reference; lower MAE/RMSE is better'))
    return pd.DataFrame(rows)


def change_summary(pred, cfg):
    rows = []
    for (cohort, model), sub in pred.groupby(['cohort', 'model']):
        for start, end in [(0, 12), (0, 24), (12, 24)]:
            a, b = sub[sub.visit_week.eq(start)], sub[sub.visit_week.eq(end)]
            z = b.merge(a[['patient_uid', 'repeat', 'fold', 'observed', 'prediction']],
                        on=['patient_uid', 'repeat', 'fold'], suffixes=('', '_start'), validate='one_to_one')
            z['observed'] = z.observed-z.observed_start
            z['prediction'] = z.prediction-z.prediction_start
            values, _ = bootstrap_metrics(z, cfg)
            rows.append(dict(cohort=cohort, model=model, window=f'W{start}_W{end}', **values))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--config', default='config/model_ablation_20261025.json')
    args = parser.parse_args(); root = args.root.resolve()
    cp = root/args.config; cfg = json.loads(cp.read_text()); out = root/cfg['output_path']; out.mkdir(parents=True, exist_ok=True)
    reports = root/cfg['report_path']; reports.mkdir(parents=True, exist_ok=True)
    source = root/cfg['input_path']; data = pd.read_csv(source)
    frozen_path = root/cfg['frozen_predictions_path']; frozen = pd.read_csv(frozen_path)
    assert len(data)==316 and data.patient_uid.nunique()==160
    assert not data.duplicated(['patient_uid', 'visit_week']).any()
    assert data.age.ge(12).all() and data.observed.between(0,72).all()
    for key, bounds in [('pp_nrs',(0,10)), ('dlqi',(0,30)), ('bsa',(0,100))]:
        v = data[key].dropna(); assert v.between(*bounds).all()
    anchors = np.array(sorted(data.patient_uid.unique()))
    outputs, audits, selections, cohorts, formulas, checks, coefficients = [], [], [], [], {}, [], []
    for cohort, spec in cfg['cohorts'].items():
        keep = data.age.ge(spec['minimum_age']) & np.isfinite(data[spec['required_inputs']]).all(axis=1)
        d = data[keep].copy()
        for week in cfg['evaluation_weeks']:
            z = d[d.visit_week.eq(week)]
            cohorts.append(dict(cohort=cohort, week=week, n_patients=z.patient_uid.nunique(), n_visits=len(z),
                                excluded_from_core=int((data.visit_week.eq(week)&~keep).sum()),
                                n_development_patients=d[d.visit_week.isin(cfg['development_weeks'])].patient_uid.nunique()))
        for repeat in range(cfg['outer_repeats']):
            for fold, (tr, te) in enumerate(KFold(cfg['outer_folds'], shuffle=True, random_state=cfg['seed']+repeat).split(anchors)):
                train = d[d.patient_uid.isin(anchors[tr]) & d.visit_week.isin(cfg['development_weeks'])]
                test = d[d.patient_uid.isin(anchors[te])]
                assert set(train.patient_uid).isdisjoint(test.patient_uid)
                for model in spec['models']:
                    fit, scores = select(train, model, cfg)
                    row = test[['patient_uid','visit_week','observed','treatment','center']].copy()
                    row['prediction'] = fit.predict(test)
                    row['cohort'], row['model'], row['repeat'], row['fold'] = cohort, model, repeat, fold
                    outputs.append(row)
                    formula = fit.formula()
                    terms = ({'bsa': formula['bsa_factor'], **{'bsa_x_'+k: v for k,v in formula['interactions'].items()}}
                             if 'bsa_factor' in formula else {'intercept': formula['intercept'], **formula['coefficients']})
                    coefficients.extend(dict(cohort=cohort, model=model, repeat=repeat, fold=fold, structure=fit.structure,
                                             term=term, coefficient=value) for term,value in terms.items())
                    audits.append(dict(cohort=cohort, model=model, repeat=repeat, fold=fold, selected_structure=fit.structure,
                                       alpha=fit.alpha, n_training_patients=train.patient_uid.nunique(), n_training_visits=len(train),
                                       n_test_patients=test.patient_uid.nunique(), n_test_visits=len(test), patient_overlap=0,
                                       max_training_week=int(train.visit_week.max()), inputs='|'.join(fit.features)))
                    selections.extend(dict(cohort=cohort, model=model, repeat=repeat, fold=fold, **s) for s in scores)
                print(f'{cohort}: repeat {repeat+1}, fold {fold+1}', flush=True)
        dev = d[d.visit_week.isin(cfg['development_weeks'])]
        for model in spec['models']:
            fit, _ = select(dev, model, cfg)
            x = fit.scale.transform(fit.design(dev))
            ref = Ridge(alpha=fit.alpha, fit_intercept=not fit.zero, solver='svd').fit(x, dev.observed, sample_weight=fit.weight)
            diff = np.max(abs(fit.raw(d)-ref.predict(fit.scale.transform(fit.design(d)))))
            assert diff < 1e-8
            checks.append(dict(cohort=cohort, model=model, sklearn_max_abs=float(diff)))
            formulas[cohort+'__'+model] = dict(inputs=fit.features, selected_structure=fit.structure, alpha=fit.alpha,
                                              formula=fit.formula(), n_training_patients=int(dev.patient_uid.nunique()),
                                              n_training_visits=len(dev), status='Exploratory refit, not a new frozen clinical calculator')
    pred = pd.concat(outputs, ignore_index=True)
    exact_truth = np.array([[0.,1.,2.,3.,4.]])
    exact_check = metric_arrays(exact_truth, exact_truth[:,None,:])
    assert exact_check['MAE'][0] == 0 and abs(exact_check['CCC'][0]-1)<1e-12
    two_repeats = np.stack([exact_truth+2, exact_truth-2], axis=1)
    assert metric_arrays(exact_truth, two_repeats)['MAE'][0] == 2
    assert abs(metric_arrays(exact_truth, two_repeats)['bias'][0]) < 1e-12
    assert not pred.duplicated(['cohort','model','repeat','patient_uid','visit_week']).any()
    exact = []
    for model, old in [('serum3','serum_linear'),('bsa_serum3','unified_selected'),('bsa_area','bsa_area'),('bsa_serum3_area','unified_area_interaction')]:
        new = pred[pred.cohort.eq('core_all') & pred.model.eq(model)]
        z = new.merge(frozen[frozen.model.eq(old)][['patient_uid','visit_week','repeat','fold','prediction']],
                      on=['patient_uid','visit_week','repeat','fold'], suffixes=('', '_old'), validate='one_to_one')
        assert len(z)==632
        diff = float(abs(z.prediction-z.prediction_old).max()); assert diff < 1e-8
        exact.append(dict(model=model, frozen_model=old, max_abs_prediction_difference=diff))
    pred.to_csv(out/'predictions_private.csv', index=False)
    pd.DataFrame(audits).to_csv(out/'fit_audit.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'inner_selection.csv', index=False)
    pd.DataFrame(cohorts).to_csv(out/'cohort_counts.csv', index=False)
    coefficient_frame = pd.DataFrame(coefficients)
    coefficient_frame.to_csv(out/'fold_coefficients.csv', index=False)
    coefficient_frame.groupby(['cohort','model','structure','term']).coefficient.agg(
        n_fits='size', median='median', minimum='min', maximum='max',
        positive_fits=lambda x: int((x>0).sum()), negative_fits=lambda x: int((x<0).sum())
    ).reset_index().to_csv(out/'coefficient_stability.csv', index=False)
    pd.DataFrame(audits).groupby(['cohort','model','selected_structure','alpha']).size().rename('n_outer_fits').reset_index().to_csv(out/'selection_frequencies.csv', index=False)
    pd.DataFrame([dict(model=k, label=LABELS[k], inputs='|'.join(v), structures='|'.join(structures(k)), no_treatment_or_history_inputs=True) for k,v in MODEL_INPUTS.items()]).to_csv(out/'feature_sets.csv', index=False)
    (out/'exploratory_formulas.json').write_text(json.dumps(formulas, indent=2)+'\n')
    print('All held-out predictions generated; computing patient-bootstrap intervals.', flush=True)
    metrics = summarize(pred, cfg); metrics.to_csv(out/'metrics.csv', index=False)
    overview = metrics[metrics.group_type.eq('overall')].copy(); overview.to_csv(out/'overall_metrics.csv', index=False)
    overview[overview.cohort.eq('itch_matched')].to_csv(out/'main_input_comparison.csv', index=False)
    comparisons(pred, cfg).to_csv(out/'paired_comparisons.csv', index=False)
    change_summary(pred, cfg).to_csv(out/'change_metrics.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'numerical_verification.csv', index=False)
    pd.DataFrame(exact).to_csv(out/'frozen_prediction_reproduction.csv', index=False)
    report = dict(status='PASS', patient_outer_folds_unchanged=True, patient_overlap=0,
                  outcome_imputed=False, clinical_inputs_imputed=False, W24_used_in_fit_or_selection=False,
                  primary_frozen_files_modified=False, metrics_scored_before_repeat_averaging=True,
                  old_predictions_reproduced=exact, identical_patients_within_comparison=True,
                  conditional_bootstrap_intervals=True, exact_prediction_metric_check=True, opposite_repeat_error_check=True, n_predictive_rows=len(pred),
                  input_files={str(p.relative_to(root)):digest(p) for p in [source,frozen_path,cp,Path(__file__)]},
                  software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    (reports/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=dict(**report,outputs=[dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(overview[['cohort','model','week','n_patients','MAE','RMSE','rho','CCC']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
