"""Week-12 serum measurements predicting observed week-24 EASI75."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from . import itch_prediction_20261112 as shared


ROOT = Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[3])).resolve()


def frame(root, cfg):
    c = pd.read_csv(root / cfg['sources'][0])
    c = c[c.age.ge(12) & ~c.dose_ambiguous].copy()
    assert c.patient_uid.nunique() == 163
    assert not c.duplicated(['patient_uid', 'visit_week']).any()
    baseline_cols = ['patient_uid', 'age', 'sex_male', 'bmi', 'center', 'treatment', 'easi_total']
    b = c[c.visit_week.eq(0)][baseline_cols].rename(columns={'easi_total': 'baseline_easi'})
    current_cols = ['patient_uid', 'easi_total', 'pp_nrs', 'iga', 'bsa', 'dlqi']
    d = b.merge(c[c.visit_week.eq(12)][current_cols], on='patient_uid', validate='one_to_one')
    future = c[c.visit_week.eq(24)][['patient_uid', 'easi_total']].rename(columns={'easi_total': 'outcome_easi'})
    d = d.merge(future, on='patient_uid', validate='one_to_one')
    d = d[d.baseline_easi.gt(0) & d.outcome_easi.notna()].copy()
    assert d.baseline_easi.between(0, 72).all() and d.outcome_easi.between(0, 72).all()
    d.loc[d.age.lt(18), 'dlqi'] = np.nan
    m = pd.read_csv(root / cfg['sources'][1])
    assert not m.duplicated(['patient_uid', 'visit_week']).any()
    d = d.merge(m[m.visit_week.eq(12)][['patient_uid'] + cfg['markers']],
                on='patient_uid', how='left', validate='one_to_one')
    d['target'] = d.outcome_easi.le(.25 * d.baseline_easi).astype(int)
    counts = [dict(population='all_clinically_evaluable', n=len(d),
                   events=int(d.target.sum()), nonevents=int((1-d.target).sum()))]
    d = d.sort_values('patient_uid').reset_index(drop=True)
    populations = {}
    for name, valid in [('complete_three_markers', d[cfg['markers']].notna().all(axis=1)),
                        ('available_serum_imputation_sensitivity', d[cfg['markers']].notna().any(axis=1))]:
        z = d[valid].copy().reset_index(drop=True)
        assert np.isfinite(z[cfg['markers']].to_numpy()[z[cfg['markers']].notna()]).all()
        count = dict(population=name, n=len(z), events=int(z.target.sum()), nonevents=int((1-z.target).sum()))
        assert {k: count[k] for k in ['n', 'events', 'nonevents']} == cfg['populations'][name]
        populations[name] = z
        counts.append(count)
    return populations, pd.DataFrame(counts)


def fit(d, cfg, population, models, out):
    rows, audits, coefficients = [], [], []
    y = d.target.to_numpy(int)
    for repeat in range(cfg['outer_repeats']):
        outer = StratifiedKFold(cfg['outer_folds'], shuffle=True, random_state=cfg['seed']+repeat)
        for fold, (tr, te) in enumerate(outer.split(d, y)):
            train_ids, test_ids = set(d.iloc[tr].patient_uid), set(d.iloc[te].patient_uid)
            assert train_ids.isdisjoint(test_ids)
            inner = StratifiedKFold(cfg['inner_folds'], shuffle=True,
                                    random_state=cfg['seed']+100+repeat*cfg['outer_folds']+fold)
            splits = list(inner.split(d.iloc[tr], y[tr]))
            for a, b in splits:
                assert set(d.iloc[tr].iloc[a].patient_uid).isdisjoint(d.iloc[tr].iloc[b].patient_uid)
                assert len(np.unique(y[tr][a])) == len(np.unique(y[tr][b])) == 2
            for name, columns in models.items():
                assert 'outcome_easi' not in columns and 'target' not in columns
                search = GridSearchCV(shared.estimator(columns), {'model__C': cfg['logistic_C']},
                                      cv=splits, scoring='neg_log_loss', n_jobs=1, error_score='raise')
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    search.fit(d.iloc[tr], y[tr])
                    p = search.predict_proba(d.iloc[te])[:, 1]
                assert all(np.isfinite(v).all() for k, v in search.cv_results_.items()
                           if k.endswith('_test_score'))
                assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
                best = search.best_estimator_
                classifier = best.named_steps['model']
                assert np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all()
                assert int(classifier.n_iter_[0]) < classifier.max_iter
                for k, i in enumerate(te):
                    rows.append(dict(population=population, model=name, repeat=repeat, fold=fold,
                                     patient_uid=d.iloc[i].patient_uid, observed=int(y[i]),
                                     prediction=float(p[k]), training_event_rate=float(y[tr].mean())))
                for feature, beta in zip(best.named_steps['prep'].get_feature_names_out(), classifier.coef_.ravel()):
                    coefficients.append(dict(population=population, model=name, repeat=repeat,
                                             fold=fold, feature=feature, standardized_coefficient=float(beta)))
                audits.append(dict(population=population, model=name, repeat=repeat, fold=fold,
                    train_n=len(tr), test_n=len(te), train_events=int(y[tr].sum()), test_events=int(y[te].sum()),
                    C=search.best_params_['model__C'], solver_iterations=int(classifier.n_iter_[0]),
                    solver_max_iterations=int(classifier.max_iter),
                    train_ids='|'.join(sorted(train_ids)), test_ids='|'.join(sorted(test_ids)),
                    inner_train_ids=json.dumps([d.iloc[tr].iloc[a].patient_uid.tolist() for a, b in splits]),
                    inner_test_ids=json.dumps([d.iloc[tr].iloc[b].patient_uid.tolist() for a, b in splits])))
            print(population, 'repeat', repeat+1, 'fold', fold+1, flush=True)
    pred, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    assert not pred.duplicated(['patient_uid', 'model', 'repeat']).any()
    assert pred.groupby(['patient_uid', 'model']).size().eq(cfg['outer_repeats']).all()
    restricted = out/'restricted'
    restricted.mkdir(exist_ok=True, parents=True)
    pred.to_csv(restricted/(population+'_oof.csv'), index=False)
    audit.to_csv(restricted/(population+'_cv_membership.csv'), index=False)
    audit.drop(columns=['train_ids', 'test_ids', 'inner_train_ids', 'inner_test_ids']).to_csv(
        out/(population+'_fit_summary.csv'), index=False)
    pd.DataFrame(coefficients).to_csv(out/(population+'_fold_coefficients.csv'), index=False)
    return pred


def run():
    cfg_path = ROOT/'config/landmark_three_20261114.json'
    cfg = json.loads(cfg_path.read_text())
    report = ROOT/'reports/update_20261114/prognosis_integration/landmark'
    assert json.loads((report/'protocol_before_fit.json').read_text()) == cfg
    before = cfg['source_sha256']
    assert all(shared.sha(ROOT/p) == value for p, value in before.items())
    out = ROOT/'results/tables/update_20261114/landmark_three'
    out.mkdir(parents=True, exist_ok=True)
    populations, counts = frame(ROOT, cfg)
    counts.to_csv(out/'cohort_counts.csv', index=False)
    models = {'three_markers': cfg['markers'], 'clinical': cfg['clinical_inputs'],
              'clinical_three_markers': cfg['clinical_inputs'] + cfg['markers']}
    old_models = shared.MODELS
    outputs = {name: [] for name in ['metrics', 'paired_differences', 'repeat_metrics',
                                     'calibration', 'roc_curves', 'null_reference']}
    try:
        shared.MODELS = models
        for name, d in populations.items():
            pred = fit(d, cfg, name, models, out)
            for key, table in zip(outputs, shared.evaluate(pred, cfg, name)):
                outputs[key].append(table)
    finally:
        shared.MODELS = old_models
    for key, tables in outputs.items():
        pd.concat(tables, ignore_index=True).to_csv(out/(key+'.csv'), index=False)
    assert all(shared.sha(ROOT/p) == value for p, value in before.items())
    receipt = dict(status='PASS', protocol_saved_before_fit=True, n_clinical=163,
        cohort_counts=counts.to_dict('records'), outer_fits=150, outer_patient_crossing=0,
        observed_outcomes_only=True, predictor_times_at_or_before_W12=True,
        same_patients_and_splits_across_models=True, endpoint='W24 EASI75 relative to W0',
        no_new_panel_selection=True, primary_evaluation='mean_repetition',
        protected_inputs_unchanged=True, source_sha256=before, config_sha256=shared.sha(cfg_path),
        code_sha256=shared.sha(Path(__file__)),
        outputs_sha256={p.name: shared.sha(p) for p in sorted(out.glob('*.csv'))},
        limitations=cfg['limitations'])
    (report/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    a = pd.concat(outputs['metrics']).query("metric == 'AUROC' and evaluation == 'mean_repetition'")
    print(a[['population', 'model', 'estimate', 'ci_low', 'ci_high', 'n', 'events']].to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
