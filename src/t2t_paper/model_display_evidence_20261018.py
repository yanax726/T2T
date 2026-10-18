"""Aggregate monitoring, uncertainty and matched-formula evidence from saved OOF predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODEL = 'unified_selected'
SEED = 20261002
BOOTSTRAPS = 2000
METRICS = ['MAE', 'RMSE', 'bias', 'rho', 'CCC', 'within1', 'within2']
KEYS = ['patient_uid', 'repeat', 'fold']


def scores(y, p):
    error = p - y
    ry, rp = rankdata(y), rankdata(p)
    rho = np.corrcoef(ry, rp)[0, 1] if np.std(ry) > 0 and np.std(rp) > 0 else np.nan
    den = np.var(y) + np.var(p) + (np.mean(y) - np.mean(p)) ** 2
    ccc = 2 * np.mean((y - y.mean()) * (p - p.mean())) / den if den > 0 else np.nan
    return np.array([np.mean(abs(error)), np.sqrt(np.mean(error**2)), np.mean(error), rho, ccc,
                     np.mean(abs(error) <= 1), np.mean(abs(error) <= 2)])


def repeated_scores(y, predictions):
    values = np.asarray([scores(y, pred) for pred in predictions])
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count, out=np.full(len(METRICS), np.nan), where=count > 0)


def arrays(frame, column='prediction'):
    pivot = frame.pivot(index='patient_uid', columns='repeat', values=column).sort_index()
    assert pivot.notna().all().all(), 'Every patient must have all repetitions in this comparison.'
    observed = frame.groupby('patient_uid')['observed'].agg(['first', 'nunique']).reindex(pivot.index)
    assert observed['nunique'].eq(1).all()
    return observed['first'].to_numpy(), pivot.to_numpy().T


def patient_bootstrap(frame, comparator=None, seed=SEED, replicates=BOOTSTRAPS):
    y, p = arrays(frame)
    q = arrays(frame, comparator)[1] if comparator else None
    estimate = repeated_scores(y, p)
    if q is not None:
        estimate -= repeated_scores(y, q)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        idx = rng.integers(0, len(y), len(y))
        value = repeated_scores(y[idx], p[:, idx])
        if q is not None:
            value -= repeated_scores(y[idx], q[:, idx])
        draws.append(value)
    draws = np.asarray(draws)
    lo, hi = np.nanquantile(draws, [.025, .975], axis=0)
    return [dict(metric=name, estimate=estimate[j], ci_low=lo[j], ci_high=hi[j],
                 n_patients=len(y), n_repeats=p.shape[0], bootstrap_replicates=replicates,
                 valid_bootstrap_replicates=int(np.isfinite(draws[:, j]).sum()))
            for j, name in enumerate(METRICS)]


def paired_visits(predictions, start, end):
    a = predictions[predictions.visit_week.eq(start)]
    b = predictions[predictions.visit_week.eq(end)]
    return b.merge(a[KEYS + ['observed', 'prediction']], on=KEYS, validate='one_to_one',
                   suffixes=('', '_start'))


def confusion_rows(pair, start, end):
    rows = []
    for rep, group in pair.groupby('repeat'):
        eligible = group[group.observed_start.gt(0) & group.prediction_start.gt(0)]
        for threshold in [.5, .75, .9]:
            truth = 1 - eligible.observed / eligible.observed_start >= threshold
            estimate = 1 - eligible.prediction / eligible.prediction_start >= threshold
            for actual in [0, 1]:
                n_actual = int(truth.eq(bool(actual)).sum())
                for predicted in [0, 1]:
                    count = int((truth.eq(bool(actual)) & estimate.eq(bool(predicted))).sum())
                    rows.append(dict(window=f'W{start}_W{end}', repeat=int(rep),
                                     endpoint=f'EASI{int(threshold * 100)}', observed_response=actual,
                                     predicted_response=predicted, count=count, n_observed_class=n_actual,
                                     row_proportion=count / n_actual if n_actual else np.nan,
                                     overall_proportion=count / len(eligible) if len(eligible) else np.nan,
                                     n_pairs=len(group), n_evaluable=len(eligible),
                                     n_excluded=len(group) - len(eligible),
                                     n_excluded_observed_baseline_nonpositive=int(group.observed_start.le(0).sum()),
                                     n_excluded_predicted_baseline_nonpositive=int(group.prediction_start.le(0).sum())))
    return rows


def ecdf(frame, grid, replicates=BOOTSTRAPS, seed=SEED):
    y, p = arrays(frame)
    indicators = (abs(p - y[None, :])[:, :, None] <= grid[None, None, :]).mean(axis=0)
    # Average threshold indicators, not predictions or signed errors, across repeats.
    estimate = indicators.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (replicates, len(y)))
    boot = indicators[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    return pd.DataFrame(dict(error_threshold=grid, proportion_within=estimate, ci_low=lo, ci_high=hi,
                             n_patients=len(y), n_repeats=p.shape[0], bootstrap_replicates=replicates))


def response_uncertainty(pair, window, replicates=BOOTSTRAPS):
    ids = sorted(pair.patient_uid.unique())
    values = {column: pair.pivot(index='patient_uid', columns='repeat', values=column).reindex(ids).to_numpy().T
              for column in ['observed', 'observed_start', 'prediction', 'prediction_start']}
    observed, baseline, prediction, predicted_baseline = [values[c] for c in ['observed', 'observed_start', 'prediction', 'prediction_start']]
    valid = (baseline > 0) & (predicted_baseline > 0)
    observed_improvement = 1 - np.divide(observed, baseline, out=np.full_like(observed, np.nan), where=baseline > 0)
    predicted_improvement = 1 - np.divide(prediction, predicted_baseline, out=np.full_like(prediction, np.nan), where=predicted_baseline > 0)
    result = []
    for threshold in [.5, .75, .9]:
        truth = observed_improvement >= threshold
        estimate = predicted_improvement >= threshold

        def classify(index):
            metrics = []
            for t, p, use in zip(truth[:, index], estimate[:, index], valid[:, index]):
                t, p = t[use], p[use]
                metrics.append([p[t].mean() if t.any() else np.nan,
                                (~p[~t]).mean() if (~t).any() else np.nan,
                                (p == t).mean() if len(t) else np.nan])
            metrics = np.asarray(metrics); n = np.isfinite(metrics).sum(axis=0)
            return np.divide(np.nansum(metrics, axis=0), n, out=np.full(3, np.nan), where=n > 0)

        point = classify(np.arange(len(ids)))
        rng = np.random.default_rng(SEED)
        draws = np.array([classify(rng.integers(0, len(ids), len(ids))) for _ in range(replicates)])
        lo, hi = np.nanquantile(draws, [.025, .975], axis=0)
        for j, metric in enumerate(['sensitivity', 'specificity', 'agreement']):
            result.append(dict(window=window, endpoint=f'EASI{int(threshold * 100)}', metric=metric,
                               estimate=point[j], ci_low=lo[j], ci_high=hi[j],
                               n_patients=len(ids), n_evaluable_min=int(valid.sum(1).min()),
                               n_evaluable_max=int(valid.sum(1).max()), n_repeats=valid.shape[0],
                               bootstrap_replicates=replicates, valid_bootstrap_replicates=int(np.isfinite(draws[:, j]).sum())))
    return result


def run(root, replicates=BOOTSTRAPS):
    root = Path(root)
    model = root / 'results/tables/update_20261017/model'
    out = root / 'results/tables/update_20261017/redesign_20261018'
    report = root / 'reports/update_20261017/redesign_20261018'
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    paths = [model / 'predictions.csv', model / 'fit_audit_private.csv',
             model / 'published_matched_predictions_private.csv', model / 'paired_change_metrics.csv',
             model / 'improvement_classification.csv', model / 'overall_metrics.csv']
    all_predictions = pd.read_csv(paths[0])
    predictions = all_predictions[all_predictions.model.eq(MODEL)]
    assert not predictions.duplicated(KEYS + ['visit_week']).any()
    assert predictions.groupby(['patient_uid', 'repeat']).fold.nunique().eq(1).all()
    fits = pd.read_csv(paths[1]); fits = fits[fits.model.eq(MODEL)]
    for row in fits.itertuples():
        assert set(row.train_ids.split('|')).isdisjoint(row.test_ids.split('|'))
        assert row.max_training_week <= 12
        heldout = predictions[predictions['repeat'].eq(row.repeat) & predictions.fold.eq(row.fold)]
        assert set(heldout.patient_uid) <= set(row.test_ids.split('|'))
    bins, per_repeat, summary, confusion, response_intervals = [], [], [], [], []
    for start, end in [(0, 12), (0, 24), (12, 24)]:
        pair = paired_visits(predictions, start, end)
        changes = pair.copy()
        changes['observed'] = pair.observed - pair.observed_start
        changes['prediction'] = pair.prediction - pair.prediction_start
        window = f'W{start}_W{end}'
        display = changes.assign(observed_change_bin=changes.observed.round(),
                                 predicted_change_bin=changes.prediction.round())
        counted = display.groupby(['repeat', 'observed_change_bin', 'predicted_change_bin']).size().reset_index(name='count')
        counted['window'] = window
        counted['definition'] = 'end minus start; negative is improvement'
        bins.append(counted)
        for rep, group in changes.groupby('repeat'):
            per_repeat.append(dict(window=window, repeat=rep, n_patients=len(group),
                                   **dict(zip(METRICS, scores(group.observed.to_numpy(), group.prediction.to_numpy())))))
        summary.extend(dict(window=window, **r) for r in patient_bootstrap(changes, replicates=replicates))
        if start == 0:
            confusion.extend(confusion_rows(pair, start, end))
            response_intervals.extend(response_uncertainty(pair, window, replicates=replicates))
    bins = pd.concat(bins, ignore_index=True)
    confusion = pd.DataFrame(confusion)
    confusion_summary = confusion.groupby(['window', 'endpoint', 'observed_response', 'predicted_response'], as_index=False).agg(
        mean_row_proportion=('row_proportion', 'mean'), mean_overall_proportion=('overall_proportion', 'mean'),
        n_evaluable_min=('n_evaluable', 'min'), n_evaluable_max=('n_evaluable', 'max'),
        n_observed_class_min=('n_observed_class', 'min'), n_observed_class_max=('n_observed_class', 'max'))
    summary = pd.DataFrame(summary)
    old = pd.read_csv(paths[3]); old = old[old.model.eq(MODEL)].set_index('window')
    differences = []
    for row in summary.itertuples():
        difference = abs(row.estimate - old.loc[row.window, row.metric])
        assert difference < 1e-12
        differences.append(dict(check='monitoring_' + row.metric, window=row.window, max_abs_difference=difference))
    old_conf = pd.read_csv(paths[4]); old_conf = old_conf[old_conf.model.eq(MODEL)].set_index(['window', 'endpoint'])
    for (window, endpoint), group in confusion.groupby(['window', 'endpoint']):
        rep_metrics = []
        for rep, part in group.groupby('repeat'):
            cell = part.set_index(['observed_response', 'predicted_response'])
            rep_metrics.append(dict(sensitivity=cell.loc[(1, 1), 'row_proportion'],
                                    specificity=cell.loc[(0, 0), 'row_proportion'],
                                    agreement=cell.loc[(1, 1), 'overall_proportion'] + cell.loc[(0, 0), 'overall_proportion']))
        average = pd.DataFrame(rep_metrics).mean()
        for key, value in average.items():
            delta = abs(value - old_conf.loc[(window, endpoint), key]); assert delta < 1e-12
            differences.append(dict(check='classification_' + key, window=window, endpoint=endpoint, max_abs_difference=delta))
    uncertainty, ecdfs = [], []
    for week, group in predictions.groupby('visit_week'):
        uncertainty.extend(dict(week=week, **r) for r in patient_bootstrap(group, replicates=replicates))
        maximum = np.ceil(abs(group.prediction - group.observed).max())
        grid = np.unique(np.r_[np.arange(0, 10.01, .25), np.arange(11, max(12, maximum) + 1)])
        e = ecdf(group, grid, replicates=replicates); e['week'] = week; ecdfs.append(e)
    matched = pd.read_csv(paths[2]); matched = matched[matched.model.eq(MODEL)].copy()
    paired, paired_repeats, band_counts, matched_controls, matched_intervals = [], [], [], [], []
    eligible_keys = matched[['patient_uid', 'visit_week']].drop_duplicates()
    for name in [MODEL, 'bsa_area', 'serum_linear']:
        control = all_predictions[all_predictions.model.eq(name)].merge(
            eligible_keys, on=['patient_uid', 'visit_week'], validate='many_to_one')
        for week, group in control.groupby('visit_week'):
            y, p = arrays(group)
            matched_controls.append(dict(model=name, week=week, n_patients=len(y), n_repeats=p.shape[0],
                                         **dict(zip(METRICS, repeated_scores(y, p)))))
            matched_intervals.extend(dict(model=name, week=week, **r)
                                     for r in patient_bootstrap(group, replicates=replicates))
    for variant in ['raw', 'clipped_0_72']:
        matched['comparator'] = matched.published_prediction if variant == 'raw' else matched.published_prediction.clip(0, 72)
        for week, group in matched.groupby('visit_week'):
            y, p = arrays(group, 'comparator')
            matched_controls.append(dict(model='published_' + variant, week=week, n_patients=len(y), n_repeats=p.shape[0],
                                         **dict(zip(METRICS, repeated_scores(y, p)))))
            marginal = group.assign(prediction=group.comparator)
            matched_intervals.extend(dict(model='published_' + variant, week=week, **r)
                                     for r in patient_bootstrap(marginal, replicates=replicates))
            paired.extend(dict(week=week, published_variant=variant, comparison='current minus published', **r)
                          for r in patient_bootstrap(group, 'comparator', replicates=replicates))
            for rep, part in group.groupby('repeat'):
                diff = scores(part.observed.to_numpy(), part.prediction.to_numpy()) - scores(part.observed.to_numpy(), part.comparator.to_numpy())
                paired_repeats.append(dict(week=week, repeat=rep, published_variant=variant,
                                           n_patients=len(part), **dict(zip(METRICS, diff))))
            first = group[group['repeat'].eq(0)]
            for method, column in [('current', 'prediction'), ('published', 'comparator')]:
                observed = np.digitize(first.observed, [7, 21, 50], right=True)
                estimated = np.digitize(first[column], [7, 21, 50], right=True)
                for actual in range(4):
                    for predicted in range(4):
                        nclass = int((observed == actual).sum())
                        count = int(((observed == actual) & (estimated == predicted)).sum())
                        band_counts.append(dict(week=week, published_variant=variant, method=method,
                                                repeat=0, observed_band=actual, predicted_band=predicted,
                                                count=count, n_observed_band=nclass, n_evaluable=len(first),
                                                row_proportion=count/nclass if nclass else np.nan))
    tables = {'monitoring_change_bins': bins, 'monitoring_change_bins_repeat0': bins[bins['repeat'].eq(0)],
              'monitoring_change_metrics_per_repeat': per_repeat, 'monitoring_change_uncertainty': summary,
              'monitoring_confusion_counts': confusion, 'monitoring_confusion_repeat0': confusion[confusion['repeat'].eq(0)],
              'monitoring_confusion_mean_proportions': confusion_summary,
              'monitoring_response_uncertainty': response_intervals,
              'prediction_uncertainty': uncertainty, 'prediction_error_ecdf': pd.concat(ecdfs, ignore_index=True),
              'published_paired_difference_uncertainty': paired,
              'published_paired_difference_per_repeat': paired_repeats,
              'published_matched_controls_points': matched_controls,
              'published_matched_method_uncertainty': matched_intervals,
              'published_severity_confusion_repeat0': band_counts,
              'model_display_reconciliation': differences}
    public = root / 'manuscript/source_data'; public.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        frame = pd.DataFrame(table)
        frame.to_csv(out / (name + '.csv'), index=False)
        frame.to_csv(public / (name + '.csv'), index=False)
    pd.concat(ecdfs, ignore_index=True).rename(columns={'error_threshold': 'absolute_error',
        'proportion_within': 'proportion'}).to_csv(public / 'error_ecdf.csv', index=False)
    manifest = dict(status='PASS', seed=SEED, bootstrap_replicates=replicates,
                    inputs=[dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    patient_held_out_audit_passed=True, no_averaging_predictions_before_scoring=True,
                    monitoring_points_match_existing_primary_metrics=True,
                    source_tables=[str((out / (key + '.csv')).relative_to(root)) for key in tables],
                    interval_scope='Percentile patient bootstrap with the same sampled patients in every repeat; compute each repeat metric then average. Conditional on saved held-out predictions, not a refit/bootstrap of model selection.',
                    ECDF_band='Pointwise 95% intervals, not a simultaneous confidence band.',
                    publication_comparison='Identical eligible samples; original raw and clipped formula sensitivities. Different inputs and unconfirmed IL2RA/sIL-2R assay bridge prevent attribution to algorithm or population ethnicity.')
    (report / 'model_display_evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (report / 'model_display_methods.md').write_text(
        '# Monitoring display and uncertainty\n\nUse repeat 0 for scatter coordinates and integer confusion matrices. '
        'Every point represents one paired patient in one held-out repetition. Changes are later minus earlier; negative values are improvement. '
        'Do not connect the three intervals as consecutive steps: they overlap in people and W0→W24 is its own estimand. '
        'Observed response uses observed baseline and follow-up EASI; estimated response uses estimated baseline and follow-up EASI. '
        'EASI50/75/90 require both baseline denominators to be positive. Counts and excluded denominators are provided per repeat. '
        'The primary table evaluates each repetition separately, then averages the metrics. '
        'A display of averaged row percentages is also supplied, but it must not be labelled as integer patient counts.\n\n'
        'Bootstrap intervals resample patients, carrying all saved predictions for that patient together across repeats. '
        'The procedure quantifies conditional held-out prediction uncertainty and does not refit model selection. '
        'The ECDF is the average of within-repeat fractions below an absolute-error threshold. Its intervals are pointwise. '
        'The published-formula comparison is paired at the patient and visit level; both raw and [0,72]-clipped comparisons are retained. '
        'A negative MAE difference favours the current estimator, but BSA access and unresolved assay equivalence mean it is not a fair isolated comparison of algorithms.\n')
    print(json.dumps(dict(status='PASS', tables=len(tables), bootstrap_replicates=replicates,
                         max_reconciliation_difference=max(r['max_abs_difference'] for r in differences)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])))
    parser.add_argument('--bootstrap-replicates', type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    run(args.root, args.bootstrap_replicates)
