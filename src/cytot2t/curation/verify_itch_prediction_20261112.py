"""Independent aggregate replay of saved baseline itch predictions and splits."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / 'results/tables/update_20261112/itch_prediction'
OUT = ROOT / 'reports/update_20261112/itch_prediction/independent_prediction_audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(y, p):
    return dict(AUROC=roc_auc_score(y, p),
                AUPRC=average_precision_score(y, p),
                Brier=brier_score_loss(y, p))


def auc_by_patient_counts(y, p, counts):
    """Replay bootstrap AUC via weighted positive/negative pair comparisons."""
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    difference = p[positive, None] - p[None, negative]
    concordance = (difference > 0).astype(float) + .5 * (difference == 0)
    wp, wn = counts[:, positive], counts[:, negative]
    numerator = np.einsum('bi,ij,bj->b', wp, concordance, wn, optimize=False)
    denominator = wp.sum(axis=1) * wn.sum(axis=1)
    return np.divide(numerator, denominator,
                     out=np.full(len(counts), np.nan), where=denominator > 0)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / 'config/itch_prediction_20261112.json'
    cfg = json.loads(cfg_path.read_text())
    sources = [cfg_path, TABLES / 'metrics.csv', TABLES / 'paired_differences.csv',
               TABLES / 'repeat_metrics.csv',
               ROOT / 'src/cytot2t/curation/itch_prediction_20261112.py']
    raw = pd.read_csv(ROOT / cfg['sources'][0])
    mol = pd.read_csv(ROOT / cfg['sources'][1])
    clinical = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    baseline = clinical[clinical.visit_week.eq(0)].copy()
    assert len(baseline) == 163 and baseline.patient_uid.is_unique
    baseline.loc[baseline.age.lt(18), 'dlqi'] = np.nan
    later = clinical[clinical.visit_week.eq(12)][['patient_uid', 'pp_nrs']]
    baseline = baseline.merge(later.rename(columns={'pp_nrs': 'W12_itch'}),
                              on='patient_uid', how='left', validate='one_to_one')
    baseline = baseline[baseline.pp_nrs.ge(4) & baseline.W12_itch.notna()]
    baseline = baseline.merge(mol[mol.visit_week.eq(0)][['patient_uid'] + cfg['markers']],
                              on='patient_uid', how='left', validate='one_to_one')
    baseline['y'] = baseline.pp_nrs.sub(baseline.W12_itch).ge(4).astype(int)
    frames = {
        'complete_three_markers': baseline[baseline[cfg['markers']].notna().all(axis=1)],
        'available_serum_imputation_sensitivity': baseline[baseline[cfg['markers']].notna().any(axis=1)],
    }
    saved_metrics = pd.read_csv(TABLES / 'metrics.csv')
    saved_repeats = pd.read_csv(TABLES / 'repeat_metrics.csv')
    saved_deltas = pd.read_csv(TABLES / 'paired_differences.csv')
    replay, delta_replay, fold_count, inner_count = [], [], 0, 0
    point_errors, ci_errors = [], []
    for population, frame in frames.items():
        frame = frame.sort_values('patient_uid').reset_index(drop=True)
        ids = frame.patient_uid.tolist()
        y = frame.y.to_numpy(int)
        expected_n, expected_events = ((102, 88) if population == 'complete_three_markers' else (105, 89))
        assert (len(frame), int(y.sum())) == (expected_n, expected_events)
        pred_path = TABLES / 'restricted' / (population + '_oof.csv')
        split_path = TABLES / 'restricted' / (population + '_cv_membership.csv')
        sources.extend([pred_path, split_path])
        pred, splits = pd.read_csv(pred_path), pd.read_csv(split_path)
        assert set(pred.patient_uid) == set(ids)
        assert not pred.duplicated(['patient_uid', 'model', 'repeat']).any()
        assert pred.groupby(['patient_uid', 'model']).size().eq(cfg['outer_repeats']).all()
        assert pred.groupby('patient_uid')['observed'].nunique().eq(1).all()
        assert np.array_equal(pred.groupby('patient_uid')['observed'].first().reindex(ids), y)
        assert pred.prediction.between(0, 1, inclusive='neither').all()
        for row in splits.itertuples():
            train, test = set(row.train_ids.split('|')), set(row.test_ids.split('|'))
            assert train.isdisjoint(test) and train | test == set(ids)
            assert (len(train), len(test)) == (row.train_n, row.test_n)
            expected_train, expected_test = list(StratifiedKFold(
                cfg['outer_folds'], shuffle=True,
                random_state=cfg['seed'] + row.repeat).split(frame, y))[row.fold]
            assert train == set(frame.iloc[expected_train].patient_uid)
            assert test == set(frame.iloc[expected_test].patient_uid)
            assert int(frame[frame.patient_uid.isin(train)].y.sum()) == row.train_events
            assert int(frame[frame.patient_uid.isin(test)].y.sum()) == row.test_events
            observed_test = pred[pred.model.eq(row.model) & pred['repeat'].eq(row.repeat)
                                 & pred.fold.eq(row.fold)]
            assert set(observed_test.patient_uid) == test
            assert np.allclose(observed_test.training_event_rate, row.train_events / row.train_n)
            inner_train, inner_test = json.loads(row.inner_train_ids), json.loads(row.inner_test_ids)
            assert len(inner_train) == len(inner_test) == cfg['inner_folds']
            valid_ids = []
            for a, b in zip(inner_train, inner_test):
                a, b = set(a), set(b)
                assert a.isdisjoint(b) and a | b == train
                assert a.isdisjoint(test) and b.isdisjoint(test)
                valid_ids.extend(b)
                inner_count += 1
            assert len(valid_ids) == len(set(valid_ids)) == len(train)
            fold_count += 1
        for _, same_fold in splits.groupby(['repeat', 'fold']):
            assert same_fold.train_ids.nunique() == same_fold.test_ids.nunique() == 1
        draws = np.random.default_rng(cfg['seed'] + 731).integers(
            len(y), size=(cfg['bootstrap_iterations'], len(y)))
        counts = np.array([np.bincount(index, minlength=len(y)) for index in draws])
        points, bootstrap = {}, {}
        for model in cfg['candidates']:
            matrix = pred[pred.model.eq(model)].pivot(index='patient_uid', columns='repeat',
                                                     values='prediction').reindex(ids).to_numpy()
            rep = []
            auc, brier = [], []
            for repeat in range(cfg['outer_repeats']):
                p = matrix[:, repeat]
                metric = scores(y, p)
                rep.append(metric)
                reference = saved_repeats[saved_repeats.population.eq(population)
                                          & saved_repeats.model.eq(model)
                                          & saved_repeats['repeat'].eq(repeat)].iloc[0]
                for name, value in metric.items():
                    error = abs(value - reference[name])
                    point_errors.append(error)
                    assert error < 1e-12
                    replay.append(dict(population=population, evaluation='single_repetition',
                                       model=model, repeat=repeat, metric=name,
                                       independent_estimate=value, saved_estimate=reference[name]))
                auc.append(auc_by_patient_counts(y, p, counts))
                brier.append(np.einsum('bi,i->b', counts, (p-y)**2) / len(y))
            points[('mean_repetition', model)] = {key: np.mean([r[key] for r in rep]) for key in rep[0]}
            bootstrap[('mean_repetition', model)] = dict(AUROC=np.mean(auc, axis=0),
                                                         Brier=np.mean(brier, axis=0))
            average = matrix.mean(axis=1)
            points[('averaged_patient_probability', model)] = scores(y, average)
            bootstrap[('averaged_patient_probability', model)] = dict(
                AUROC=auc_by_patient_counts(y, average, counts),
                Brier=np.einsum('bi,i->b', counts, (average-y)**2)/len(y))
        for (evaluation, model), values in points.items():
            for metric, value in values.items():
                reference = saved_metrics[saved_metrics.population.eq(population)
                                          & saved_metrics.evaluation.eq(evaluation)
                                          & saved_metrics.model.eq(model)
                                          & saved_metrics.metric.eq(metric)].iloc[0]
                assert (reference['n'], reference.events) == (len(y), int(y.sum()))
                point_errors.append(abs(value-reference.estimate))
                assert abs(value-reference.estimate) < 1e-12
                row = dict(population=population, evaluation=evaluation, model=model, metric=metric,
                           independent_estimate=value, saved_estimate=reference.estimate)
                if metric in ['AUROC', 'Brier']:
                    low, high = np.nanquantile(bootstrap[(evaluation, model)][metric], [.025, .975])
                    ci_errors.extend([abs(low-reference.ci_low), abs(high-reference.ci_high)])
                    assert abs(low-reference.ci_low) < 1e-12 and abs(high-reference.ci_high) < 1e-12
                    row.update(independent_ci_low=low, independent_ci_high=high)
                replay.append(row)
        for evaluation in ['mean_repetition', 'averaged_patient_probability']:
            for model in ['three_markers', 'clinical_three_markers']:
                for metric in ['AUROC', 'Brier']:
                    diff = bootstrap[(evaluation, model)][metric] - bootstrap[(evaluation, 'clinical')][metric]
                    low, high = np.nanquantile(diff, [.025, .975])
                    value = points[(evaluation, model)][metric] - points[(evaluation, 'clinical')][metric]
                    reference = saved_deltas[saved_deltas.population.eq(population)
                                             & saved_deltas.evaluation.eq(evaluation)
                                             & saved_deltas.model.eq(model)
                                             & saved_deltas.metric.eq(metric)].iloc[0]
                    assert max(abs(value-reference.difference), abs(low-reference.ci_low),
                               abs(high-reference.ci_high)) < 1e-12
                    ci_errors.extend([abs(low-reference.ci_low), abs(high-reference.ci_high)])
                    delta_replay.append(dict(population=population, evaluation=evaluation, model=model,
                                             reference='clinical', metric=metric, independent_difference=value,
                                             independent_ci_low=low, independent_ci_high=high,
                                             saved_difference=reference.difference,
                                             saved_ci_low=reference.ci_low, saved_ci_high=reference.ci_high))
    # Read the actual model-builder configuration without fitting anything.
    from cytot2t.curation import itch_prediction_20261112 as builder
    primary, sensitivity = builder.frame()
    for frame in [primary, sensitivity]:
        assert frame.loc[frame.age.lt(18), 'dlqi'].isna().all()
    assert list(builder.MARKERS) == cfg['markers']
    assert list(builder.CLINICAL) == cfg['clinical_inputs']
    forbidden = {'target', 'itch12', 'W12_itch', 'patient_uid', 'visit_week', 'response'}
    for name, columns in builder.MODELS.items():
        assert not forbidden.intersection(columns)
        actual = builder.estimator(columns).named_steps['prep']
        selected = [c for _, _, cols in actual.transformers for c in cols]
        assert set(selected) == set(columns) and actual.remainder == 'drop'
    pd.DataFrame(replay).to_csv(OUT/'metric_replay.csv', index=False)
    pd.DataFrame(delta_replay).to_csv(OUT/'paired_bootstrap_replay.csv', index=False)
    report = dict(status='PASS', model_fits_checked=fold_count,
                  inner_partitions_checked=inner_count, patient_train_test_crossing=0,
                  outer_split_reproduction=True, paired_models_use_identical_patient_splits=True,
                  metrics_replayed=['AUROC', 'AUPRC', 'Brier'],
                  evaluations_replayed=['single_repetition', 'mean_repetition', 'averaged_patient_probability'],
                  population_counts={'complete_three_markers': {'n':102, 'responders':88, 'nonresponders':14},
                                     'available_serum_imputation_sensitivity': {'n':105, 'responders':89, 'nonresponders':16}},
                  bootstrap_iterations=cfg['bootstrap_iterations'],
                  bootstrap_replayed=['AUROC and Brier intervals for all three models and both evaluations',
                                      'Combined-minus-clinical and molecular-only-minus-clinical AUROC/Brier intervals'],
                  independent_bootstrap_method='Positive-negative pairwise concordance weighted by patient resample multiplicities; direct weighted Brier losses',
                  maximum_point_metric_error=float(max(point_errors)), maximum_interval_replay_error=float(max(ci_errors)),
                  baseline_only_predictor_columns=True, prediction_input_target_or_future_column=False,
                  adolescent_DLQI_mask_verified=True,
                  preprocessing_boundary='Pipeline imputer/scaler and ColumnTransformer are passed inside inner GridSearchCV; outer test values used only in predict_proba',
                  no_new_fitting=True, input_values_modified=False, participant_rows_exported=False,
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    (OUT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','model_fits_checked','inner_partitions_checked',
                                         'maximum_point_metric_error','maximum_interval_replay_error']}, indent=2))


if __name__ == '__main__':
    run()
