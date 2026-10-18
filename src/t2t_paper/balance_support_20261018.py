"""Export baseline balance, propensity overlap and analysis-specific support."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
BASE = ['easi_total', 'bmi', 'age', 'sex_male']


def effective_n(weights):
    w = np.asarray(weights, dtype=float)
    return float(w.sum()**2 / np.sum(w**2)) if len(w) and np.sum(w**2) > 0 else np.nan


def standardized_differences(frame, design, weight, label, sample):
    rows = []
    for one, two in itertools.combinations(ARMS, 2):
        a, b = frame.treatment.eq(one).to_numpy(), frame.treatment.eq(two).to_numpy()
        for name in design:
            x = design[name].to_numpy(float)
            denominator = np.sqrt((x[a].var(ddof=1) + x[b].var(ddof=1)) / 2) if a.sum() > 1 and b.sum() > 1 else np.nan
            difference = (np.average(x[a], weights=weight[a]) - np.average(x[b], weights=weight[b])) if a.any() and b.any() else np.nan
            smd = difference / denominator if denominator > 0 else np.nan
            rows.append(dict(sample=sample, analysis=label, group1=one, group2=two,
                             variable=name, n1=int(a.sum()), n2=int(b.sum()), SMD=smd,
                             absolute_SMD=abs(smd), pooled_unweighted_SD=denominator))
    return rows


def run(root):
    root = Path(root)
    out = root / 'results/tables/update_20261017/redesign_20261018'
    report = root / 'reports/update_20261017/redesign_20261018'
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    clinical_dir = root / 'results/tables/update_20261017/clinical'
    paths = [root / 'results/tables/update_20260925/clinical.csv',
             root / 'results/tables/redevelopment/molecular.csv',
             clinical_dir / 'restricted/overlap_patient_weights.csv',
             clinical_dir / 'baseline_balance.csv', clinical_dir / 'molecular_model_diagnostics.csv']
    c = pd.read_csv(paths[0]); c = c[c.age.ge(12) & ~c.dose_ambiguous].copy()
    baseline = c[c.visit_week.eq(0)].copy().reset_index(drop=True)
    X = baseline[BASE].astype(float).copy(); X['HUASHAN'] = baseline.center.eq('HUASHAN').astype(float)
    fit = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)).fit(X, baseline.treatment)
    probabilities = fit.predict_proba(X)
    classes = list(fit[-1].classes_)
    own = probabilities[np.arange(len(baseline)), [classes.index(t) for t in baseline.treatment]]
    weight = (1 / np.sum(1 / probabilities, axis=1)) / own
    previous = pd.read_csv(paths[2]).set_index('patient_uid').reindex(baseline.patient_uid)
    assert np.allclose(weight, previous.weight, rtol=0, atol=1e-12)
    assert np.allclose(own, previous.prob_assigned, rtol=0, atol=1e-12)
    baseline['weight'] = weight; baseline['assigned_probability'] = own
    X['CAMS'] = baseline.center.eq('CAMS').astype(float)
    balances = standardized_differences(baseline, X, np.ones(len(baseline)), 'unweighted', 'clinical_baseline')
    balances += standardized_differences(baseline, X, weight, 'overlap_weighted', 'clinical_baseline')
    existing = pd.read_csv(paths[3])
    new = pd.DataFrame(balances).merge(existing, on=['analysis', 'group1', 'group2', 'variable'], suffixes=('', '_existing'))
    assert len(new) == len(existing) and np.allclose(new.SMD, new.SMD_existing, rtol=0, atol=1e-12)
    hist = []
    for probability_arm in ARMS:
        p = probabilities[:, classes.index(probability_arm)]
        bin_number = np.minimum((p / .05).astype(int), 19)
        for treatment in ARMS:
            group = baseline.treatment.eq(treatment).to_numpy()
            for i in range(20):
                use = group & (bin_number == i)
                hist.append(dict(probability_arm=probability_arm, treatment=treatment,
                                 bin_low=i*.05, bin_high=(i+1)*.05, n=int(use.sum()),
                                 n_group=int(group.sum()), fraction=use.sum()/group.sum(),
                                 overlap_weight_sum=weight[use].sum(),
                                 weighted_fraction=weight[use].sum()/weight[group].sum()))
    support = []

    def add_support(sample, frame, weighted, model_use, marker='', start=np.nan, end=np.nan):
        unique = frame.drop_duplicates('patient_uid')
        for center in ['all', 'CAMS', 'HUASHAN']:
            sub = unique if center == 'all' else unique[unique.center.eq(center)]
            for arm in ARMS:
                g = sub[sub.treatment.eq(arm)]
                weights = g.weight.to_numpy() if weighted else np.ones(len(g))
                support.append(dict(sample=sample, marker=marker, start=start, end=end,
                                    treatment=arm, center=center, n_patients=len(g),
                                    ESS=effective_n(weights), sum_weights=weights.sum(),
                                    weighting='overlap_weighted' if weighted else 'unweighted',
                                    actual_analysis_use=model_use))

    add_support('clinical_baseline', baseline, True, 'Initial-treatment overlap comparisons')
    for week in [12, 24]:
        IDs = set(c.loc[c.visit_week.eq(week) & c.easi_total.notna(), 'patient_uid'])
        subset = baseline[baseline.patient_uid.isin(IDs)]
        xx = X.loc[subset.index]
        for label, weights in [('unweighted', np.ones(len(subset))), ('overlap_weighted', subset.weight.to_numpy())]:
            balances += standardized_differences(subset, xx, weights, label, f'EASI_observed_W{week}')
        add_support(f'EASI_observed_W{week}', subset, True, 'Observed EASI subset; baseline weights are retained', end=week)
    m = pd.read_csv(paths[1])
    serum = c[c.visit_week.isin([0, 12, 24])].merge(m, on=['patient_uid', 'visit_week'], validate='one_to_one')
    diagnostic = pd.read_csv(paths[4])
    molecular_balance = []
    for marker in [col[5:] for col in m if col.startswith('log2_')]:
        col = 'log2_' + marker
        for start, end in [(0, 12), (12, 24), (0, 24)]:
            pairs = serum[serum.visit_week.eq(end)][['patient_uid', col]].merge(
                serum[serum.visit_week.eq(start)][['patient_uid', col]], on='patient_uid',
                validate='one_to_one', suffixes=('_end', '_start')).dropna()
            subset = baseline.merge(pairs, on='patient_uid', validate='one_to_one').dropna(subset=BASE + ['center'])
            expected = diagnostic[diagnostic.marker.eq(marker) & diagnostic.start.eq(start) &
                                  diagnostic.end.eq(end) & diagnostic.scope.eq('all_centers') & diagnostic.analysis.eq('primary')]
            assert len(expected) == 1 and len(subset) == int(expected.n.iloc[0])
            sample = f'molecular_{marker}_W{start}_W{end}'
            add_support(sample, subset, False, 'Molecular ANCOVA uses unweighted HC3 regression, not propensity weighting', marker, start, end)
            design = subset[BASE].astype(float).copy()
            design['HUASHAN'] = subset.center.eq('HUASHAN').astype(float)
            design['start_marker'] = subset[col + '_start'].to_numpy()
            molecular_balance += standardized_differences(subset, design, np.ones(len(subset)), 'unweighted', sample)
    covariates = [
        dict(component='baseline_overlap_weights', outcome='initial treatment propensity',
             adjustment='baseline EASI; BMI; age; sex; center (HUASHAN indicator)',
             weight='generalized overlap: h(x)/e_t(x), h(x)=1/sum_t[1/e_t(x)]',
             uncertainty='weights re-estimated in original endpoint bootstrap', unit='patient'),
        dict(component='clinical_continuous', outcome='follow-up clinical score',
             adjustment='same clinical outcome at baseline; age; BMI; sex; center; treatment×visit',
             weight='baseline generalized overlap weights; unweighted outcome-adjusted sensitivity retained',
             uncertainty='patient-cluster robust t interval', unit='patient with repeated visits'),
        dict(component='molecular_changes', outcome='log2 concentration at end minus start',
             adjustment='start-marker concentration; W0 EASI; age; BMI; sex; center; initial treatment',
             weight='none; HC3 least squares, with assay-batch sensitivity',
             uncertainty='HC3 t interval; minimum support, rank and leverage checks', unit='one paired change per patient'),
        dict(component='EASI_estimator', outcome='current EASI', adjustment='current BSA; log2 CCL17; log2 CCL26; log2 IL-5',
             weight='inverse number of development visits per patient in fitting',
             uncertainty='patient-held-out outer repeats; no treatment labels', unit='patient-held-out observations')]
    tables = {'baseline_balance_full': balances, 'propensity_overlap_bins': hist,
              'analysis_effective_sample_sizes': support, 'molecular_sample_balance': molecular_balance,
              'analysis_covariate_inventory': covariates}
    public = root / 'manuscript/source_data'; public.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        frame.to_csv(out / (name + '.csv'), index=False)
        frame.to_csv(public / (name + '.csv'), index=False)
    by = pd.DataFrame(balances).query("sample == 'clinical_baseline'").groupby('analysis').absolute_SMD.agg(['max', lambda x: int(x.gt(.1).sum())])
    by.columns = ['max_abs_SMD', 'n_above_0_1_including_redundant_center_indicator']
    manifest = dict(status='PASS', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    inputs=[dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                    propensity_and_weights_reconstruct_saved_values=True,
                    all_original_baseline_SMD_reconstructed=True,
                    molecular_model_sample_counts_reconcile=True,
                    baseline_balance_summary=by.reset_index().to_dict('records'),
                    interpretation='Balance improved but some absolute SMDs exceed 0.1. This is an observational overlap population, not proof that confounding is removed. Categorical center has two complementary indicators; use one in compact balance plots.',
                    molecular_note='Molecular effects were covariate-adjusted without propensity weights; molecular ESS therefore equals the unweighted paired n. No fictional after-weighting balance is shown.')
    (report / 'balance_support_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])))
    run(parser.parse_args().root)
