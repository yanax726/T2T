"""Prepare public plotting aggregates without invoking any figure renderer."""
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats


def combined_moments(frame):
    rows = []
    for (target, week), group in frame.groupby(['target', 'week'], sort=True):
        valid = group[group.n.gt(0)]
        n = int(valid.n.sum())
        mean = float(np.average(valid['mean'], weights=valid.n)) if n else np.nan
        within = np.where(valid.n.gt(1), (valid.n - 1) * valid.sd.pow(2), 0.)
        ss = float(np.sum(within + valid.n * (valid['mean'] - mean).pow(2))) if n else np.nan
        sd = np.sqrt(ss / (n - 1)) if n > 1 else np.nan
        radius = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(target=target, week=int(week), n_initial=int(group.n_initial.sum()),
                         n=n, n_missing=int(group.n_missing.sum()), mean=mean, sd=sd,
                         ci_low=mean-radius, ci_high=mean+radius, n_treatment_groups=int(len(valid)),
                         method='Combined group moments; mean with t 95% CI; observed initial-treatment mixture'))
    return pd.DataFrame(rows)


def run(root):
    root = Path(root)
    public = root / 'manuscript/source_data'; public.mkdir(parents=True, exist_ok=True)
    clinical = pd.read_csv(root / 'results/tables/update_20261017/clinical/clinical_observed.csv')
    combined_moments(clinical).to_csv(public / 'clinical_overall.csv', index=False)
    private = root / 'results/tables/update_20261017/model/predictions.csv'
    names = ['model_plot_bins.csv', 'model_agreement_bins.csv', 'model_agreement_summary.csv']
    if private.exists():
        predictions = pd.read_csv(private)
        first = predictions[predictions.model.eq('unified_selected') & predictions['repeat'].eq(0)].copy()
        first['observed_bin'] = first.observed.round()
        first['predicted_bin'] = first.prediction.round()
        first.groupby(['visit_week', 'observed_bin', 'predicted_bin']).size().reset_index(name='count').to_csv(public / names[0], index=False)
        first['mean_bin'] = ((first.observed + first.prediction) / 2).round()
        first['difference_bin'] = ((first.prediction - first.observed) * 2).round() / 2
        first.groupby(['visit_week', 'mean_bin', 'difference_bin']).size().reset_index(name='count').to_csv(public / names[1], index=False)
        rows = []
        for week, group in first.groupby('visit_week'):
            difference = group.prediction - group.observed
            bias = difference.mean(); sd = difference.std(ddof=1)
            rows.append(dict(week=int(week), n=len(group), bias=bias,
                             LoA_low=bias - 1.96 * sd, LoA_high=bias + 1.96 * sd,
                             minimum=difference.min(), maximum=difference.max(), display_repeat=0))
        pd.DataFrame(rows).to_csv(public / names[2], index=False)
    else:
        for name in names:
            if not (public / name).is_file():
                raise FileNotFoundError('Public reproduction needs supplied aggregate: ' + str(public / name))
    metrics = pd.read_csv(root / 'results/tables/update_20261017/model/overall_metrics.csv')
    counts = metrics[metrics.model.eq('unified_selected')].set_index('week').n_patients
    for name in names[:2]:
        actual = pd.read_csv(public / name).groupby('visit_week')['count'].sum()
        assert actual.reindex(counts.index).eq(counts).all()
    summary = pd.read_csv(public / names[2]).set_index('week')
    assert summary.n.reindex(counts.index).eq(counts).all() and summary.display_repeat.eq(0).all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])))
    run(parser.parse_args().root)
