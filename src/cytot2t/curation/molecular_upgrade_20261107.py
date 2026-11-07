"""Bounded blood-only learning of EASI levels, paired changes and ordinal auxiliaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import solve, svd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cytot2t.curation.model_ablation_20261025 import (
    METRICS, aligned_arrays, bootstrap_metrics,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patient_weights(frame):
    w = 1.0 / frame.patient_uid.map(frame.patient_uid.value_counts()).to_numpy(float)
    return w / w.mean()


class RBFMap:
    """Nyström features without the platform BLAS distance-product warning."""
    def __init__(self, x, gamma, components, seed):
        self.gamma = gamma
        indices = np.random.RandomState(seed).permutation(len(x))[:components]
        self.components = x[indices]
        u, s, v = svd(self.kernel(self.components))
        self.normalization = np.einsum('ij,jk->ik', u/np.sqrt(np.maximum(s, 1e-12)), v, optimize=False)

    def kernel(self, x):
        return np.exp(-self.gamma*cdist(x, self.components, metric='sqeuclidean'))

    def transform(self, x):
        result = np.einsum('ij,kj->ik', self.kernel(x), self.normalization, optimize=False)
        assert np.isfinite(result).all()
        return result


class Prepared:
    def __init__(self, frame, features, gamma, cfg):
        self.features = list(features)
        x = frame[self.features].to_numpy(float)
        self.median = np.nanmedian(x, axis=0)
        if not np.isfinite(self.median).all():
            raise ValueError('An assay has no observed training value')
        x = np.where(np.isfinite(x), x, self.median)
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.basis = None
        self.basis_scaler = None
        if gamma:
            self.basis = RBFMap(z, float(gamma)/len(features),
                                min(cfg['nystroem_components'], len(frame)), cfg['seed'])
            z = self.basis.transform(z)
            self.basis_scaler = StandardScaler().fit(z)
            z = self.basis_scaler.transform(z)
        self.z = z
        self.training_ids = set(frame.patient_uid)

    def transform(self, frame):
        x = frame[self.features].to_numpy(float)
        z = self.scaler.transform(np.where(np.isfinite(x), x, self.median))
        if self.basis is not None:
            z = self.basis_scaler.transform(self.basis.transform(z))
        return z


class Fit:
    def __init__(self, frame, prep, candidate):
        self.prep, self.candidate = prep, dict(candidate)
        z = prep.z
        y = frame.observed.to_numpy(float)
        w = patient_weights(frame)
        xm, ym = np.average(z, axis=0, weights=w), float(np.average(y, weights=w))
        xc, yc = z-xm, y-ym
        a = np.einsum('ni,n,nj->ij', xc, w, xc, optimize=False)
        b = np.einsum('ni,n,n->i', xc, w, yc, optimize=False)
        self.pair_count = 0
        if candidate['family'] == 'paired_rbf' and candidate['pair_lambda'] > 0:
            idx = frame[['patient_uid', 'visit_week']].copy().reset_index(drop=True)
            idx['position'] = np.arange(len(idx))
            pivot = idx.pivot(index='patient_uid', columns='visit_week', values='position')
            pairs = pivot.reindex(columns=[0, 12]).dropna().astype(int)
            self.pair_count = len(pairs)
            if not len(pairs):
                raise ValueError('No within-patient development pairs')
            dz = z[pairs[12].to_numpy()]-z[pairs[0].to_numpy()]
            dy = y[pairs[12].to_numpy()]-y[pairs[0].to_numpy()]
            weight = float(candidate['pair_lambda'])*len(frame)/len(pairs)
            a = a + weight*np.einsum('ni,nj->ij', dz, dz, optimize=False)
            b = b + weight*np.einsum('ni,n->i', dz, dy, optimize=False)
        penalty = float(candidate['alpha'])*np.eye(z.shape[1])
        if candidate['family'] == 'ordinal_rrr':
            ys = max(float(np.sqrt(np.average(yc**2, weights=w))), 1e-8)
            iga = frame.iga.to_numpy(float)
            if not np.isfinite(iga).all():
                raise ValueError('Auxiliary IGA absent; no clinical label imputation')
            ordinal = np.column_stack([iga >= cut for cut in range(1, 6)]).astype(float)
            om = np.average(ordinal, axis=0, weights=w)
            os = np.sqrt(np.average((ordinal-om)**2, axis=0, weights=w))
            os[os < 1e-8] = 1
            target = np.column_stack([yc/ys, (ordinal-om)/os/np.sqrt(5)])
            coef = solve(a+penalty, np.einsum('ni,n,nj->ij', xc, w, target, optimize=False), assume_a='pos')
            fitted = np.einsum('ij,jk->ik', xc, coef, optimize=False)*np.sqrt(w)[:, None]
            _, _, vt = np.linalg.svd(fitted, full_matrices=False)
            v = vt[:int(candidate['rank'])].T
            self.beta = np.einsum('ij,jk,k->i', coef, v, v[0], optimize=False)*ys
        else:
            self.beta = solve(a+penalty, b, assume_a='pos')
        self.intercept = ym-float(np.einsum('i,i->', xm, self.beta, optimize=False))
        assert np.isfinite(self.beta).all() and np.isfinite(self.intercept)

    def raw(self, frame):
        prediction = self.intercept + np.einsum('ij,j->i', self.prep.transform(frame), self.beta, optimize=False)
        assert np.isfinite(prediction).all()
        return prediction

    def predict(self, frame):
        return np.clip(self.raw(frame), 0, 72)


def candidates(cfg):
    out = []
    for alpha in cfg['alpha']:
        out.append(dict(family='ridge', alpha=alpha, gamma=0, pair_lambda=0, rank=0))
    for gamma in cfg['rbf_gamma_factors']:
        for alpha in [1, 10, 100]:
            for lam in cfg['pair_lambda']:
                out.append(dict(family='paired_rbf', alpha=alpha, gamma=gamma, pair_lambda=lam, rank=0))
    for alpha in [1, 10, 100]:
        for rank in cfg['rrr_ranks']:
            out.append(dict(family='ordinal_rrr', alpha=alpha, gamma=0, pair_lambda=0, rank=rank))
    return out


def select(frame, features, cfg):
    cs = candidates(cfg)
    ids = np.array(sorted(frame.patient_uid.unique()))
    folds = []
    for a, b in KFold(cfg['inner_folds'], shuffle=True, random_state=cfg['seed']).split(ids):
        tr = frame[frame.patient_uid.isin(ids[a])].reset_index(drop=True)
        te = frame[frame.patient_uid.isin(ids[b])].reset_index(drop=True)
        assert set(tr.patient_uid).isdisjoint(te.patient_uid)
        preps = {g: Prepared(tr, features, g, cfg) for g in [0]+cfg['rbf_gamma_factors']}
        folds.append((tr, te, preps))
    scored = []
    for i, c in enumerate(cs):
        errors = []
        for tr, te, preps in folds:
            fit = Fit(tr, preps[c['gamma']], c)
            e = pd.DataFrame({'patient_uid':te.patient_uid, 'error':abs(te.observed-fit.predict(te))})
            errors.extend(e.groupby('patient_uid').error.mean())
        scored.append(dict(candidate=i, **c, inner_patient_MAE=float(np.mean(errors))))
    scores = pd.DataFrame(scored)
    pools = {'selected':scores}
    pools.update({f:scores[scores.family.eq(f)] for f in ['ridge','paired_rbf','ordinal_rrr']})
    chosen = {name:int(g.loc[g.inner_patient_MAE.idxmin(), 'candidate']) for name,g in pools.items()}
    preps = {g:Prepared(frame, features, g, cfg) for g in [0]+cfg['rbf_gamma_factors']}
    models = {name:Fit(frame, preps[cs[i]['gamma']], cs[i]) for name,i in chosen.items()}
    return models, scores, chosen


def evaluate(pred, cfg):
    results = []
    for (cohort, model, week), g in pred.groupby(['cohort','model','visit_week']):
        subsets = [('overall','all',g)]
        for label, mask in [('EASI_le7', g.observed.le(7)),
                            ('EASI_7_to21',g.observed.gt(7)&g.observed.le(21)),
                            ('EASI_gt21',g.observed.gt(21))]:
            if mask.any():
                subsets.append(('severity',label,g[mask]))
        for group_var in ['treatment','center','complete_19_analytes']:
            subsets.extend((group_var,str(name),z) for name,z in g.groupby(group_var))
        for kind, name, z in subsets:
            point,_ = bootstrap_metrics(z,cfg)
            results.append(dict(cohort=cohort,model=model,week=week,group_type=kind,group=name,**point))
    return pd.DataFrame(results)


def pairwise(pred, cfg):
    out = []
    pairs = [('three_selected','three_ridge_reference'),
             ('full19_ridge','three_ridge_reference'),
             ('full19_paired_rbf','full19_ridge'),
             ('full19_ordinal_rrr','full19_ridge'),
             ('full19_selected','three_ridge_reference'),
             ('full19_selected','full19_ridge')]
    for cohort in pred.cohort.unique():
        p = pred[pred.cohort.eq(cohort)]
        for a,b in pairs:
            if not {a,b}.issubset(set(p.model)):
                continue
            for week in cfg['evaluation_weeks']:
                aa=p[p.model.eq(a)&p.visit_week.eq(week)]
                bb=p[p.model.eq(b)&p.visit_week.eq(week)]
                ia,ya,_=aligned_arrays(aa);ib,yb,_=aligned_arrays(bb)
                assert np.array_equal(ia,ib) and np.array_equal(ya,yb)
                draws=np.random.default_rng(cfg['seed']).integers(len(ia),size=(cfg['bootstrap_replicates'],len(ia)))
                va,ba=bootstrap_metrics(aa,cfg,draws);vb,bb=bootstrap_metrics(bb,cfg,draws)
                for metric in METRICS:
                    lo,hi=np.nanquantile(ba[metric]-bb[metric],[.025,.975])
                    out.append(dict(cohort=cohort,week=week,candidate=a,reference=b,metric=metric,
                                    difference=va[metric]-vb[metric],ci_low=lo,ci_high=hi,n_patients=len(ia)))
    return pd.DataFrame(out)


def paired_changes(pred, cfg):
    rows=[]
    for (cohort,model),g in pred.groupby(['cohort','model']):
        for start,end in [(0,12),(12,24),(0,24)]:
            a=g[g.visit_week.eq(start)]
            b=g[g.visit_week.eq(end)]
            pair=a.merge(b,on=['patient_uid','repeat','fold'],validate='one_to_one',suffixes=('_start','_end'))
            assert not pair.empty
            d=pair[['patient_uid','repeat']].copy()
            d['observed']=pair.observed_end-pair.observed_start
            d['prediction']=pair.prediction_end-pair.prediction_start
            stat,_=bootstrap_metrics(d,cfg)
            rows.append(dict(cohort=cohort,model=model,start_week=start,end_week=end,**stat))
    return pd.DataFrame(rows)


def tests(cfg):
    rng=np.random.default_rng(987)
    n=50
    f=pd.DataFrame({'patient_uid':np.repeat(np.arange(n),2),'visit_week':np.tile([0,12],n),
                    'a':rng.normal(size=n*2),'b':rng.normal(size=n*2)})
    f['observed']=20+2*f.a-.5*f.b+rng.normal(size=2*n)
    f['iga']=np.digitize(f.observed,[17,18,19,20,21])
    prep=Prepared(f,['a','b'],0,cfg)
    c=dict(family='ridge',alpha=10,gamma=0,pair_lambda=0,rank=0)
    fit=Fit(f,prep,c)
    reference=Ridge(alpha=10).fit(prep.z,f.observed,sample_weight=patient_weights(f))
    assert np.max(abs(fit.raw(f)-reference.predict(prep.z)))<1e-10
    rrr=Fit(f,prep,dict(c,family='ordinal_rrr',rank=6))
    assert np.max(abs(fit.raw(f)-rrr.raw(f)))<1e-10
    shuffled=f.copy();shuffled['observed']=1000;shuffled['iga']=-100;shuffled['visit_week']=999;shuffled['bsa']=999
    assert np.array_equal(fit.predict(f),fit.predict(shuffled))
    pairfit=Fit(f,prep,dict(c,family='paired_rbf',pair_lambda=1000))
    delta=lambda p:np.diff(p.reshape(n,2),axis=1).ravel()
    assert np.mean((delta(pairfit.raw(f))-delta(f.observed.to_numpy()))**2)<=np.mean((delta(fit.raw(f))-delta(f.observed.to_numpy()))**2)+1e-8
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter('always', RuntimeWarning)
        expected=Nystroem(kernel='rbf',gamma=.2,n_components=20,random_state=cfg['seed']).fit_transform(prep.z)
    actual=RBFMap(prep.z,.2,20,cfg['seed']).transform(prep.z)
    assert np.max(abs(expected-actual))<1e-7
    return dict(weighted_ridge_matches_sklearn=True,full_rank_joint_model_reduces_to_ridge=True,
                prediction_ignores_all_clinical_and_time_columns=True,pair_training_penalty_behaves_as_defined=True,
                direct_distance_nystroem_matches_sklearn=True,
                numerical_reference_runtime_warnings=sorted({str(w.message) for w in reference_warnings}))


def historical_inventory(root):
    records = [
      ('20260908', 'Sparse ridge / elastic net / small panel', 'src/cytot2t/redevelopment/models.py', 'results/tables/redevelopment/four_target_summary.csv'),
      ('20260912', 'PLS low rank, spline, shallow boosting, soft severity gate', 'src/cytot2t/refinement/models.py', 'results/tables/refinement/performance.csv'),
      ('20260913', 'TabPFN small-panel and broad-panel learning', 'src/cytot2t/frontier/severity.py', 'config/frontier/severity_protocol.json'),
      ('20260920', 'RBF, pathway panels, clinical history and context calibration', 'src/cytot2t/score_reproduction/molecular_refinement.py', 'results/tables/score_reproduction/molecular_refinement/paired_comparisons.csv'),
      ('20261002', 'Blood-only pooled W0/W12 ridge, RBF and random forest', 'src/cytot2t/curation/serum_pooled_w0_w12_audit.py', 'results/tables/clinical_alignment/serum_pooled_metrics.csv'),
      ('20261002', 'BSA nonlinear models and severity refinement', 'src/cytot2t/curation/stage_refinement_20261002.py', 'results/tables/update_20261002/stage_refinement/input.csv'),
      ('20261025', 'Matched input ablation and leave-one-marker-out', 'src/cytot2t/curation/model_ablation_20261025.py', 'results/tables/update_20261025/ablation/overall_metrics.csv'),
      ('20261101', 'Unchanged molecular equations on OMIX and matched p-EASI comparison', 'src/cytot2t/curation/external_molecular_benchmark_20261101.py', 'reports/external_validation_20261101/benchmark_conclusion.md'),
    ]
    return pd.DataFrame([dict(date=date,method=method,source=source,evidence=evidence,
                              source_exists=(root/source).exists(),evidence_exists=(root/evidence).exists(),
                              comparability='Different earlier populations/protocols; not directly ranked against the present matched comparison')
                         for date,method,source,evidence in records])


def independent_metric_audit(pred, stats):
    discrepancies=[]
    for model in ['three_selected','full19_selected']:
        for week in [0,12,24]:
            g=pred[pred.cohort.eq('core')&pred.model.eq(model)&pred.visit_week.eq(week)]
            direct=[]
            for _,r in g.groupby('repeat'):
                y,p=r.observed.to_numpy(),r.prediction.to_numpy()
                cov=np.mean((y-y.mean())*(p-p.mean()))
                direct.append([np.mean(abs(y-p)),spearmanr(y,p).statistic,
                               2*cov/(np.var(y)+np.var(p)+(y.mean()-p.mean())**2)])
            expected=stats[stats.cohort.eq('core')&stats.model.eq(model)&stats.week.eq(week)&stats.group_type.eq('overall')]
            discrepancy=float(np.max(abs(np.mean(direct,axis=0)-expected[['MAE','rho','CCC']].to_numpy()[0])))
            assert discrepancy<1e-12
            discrepancies.append(discrepancy)
    return max(discrepancies)


def run(root):
    started=time.monotonic()
    cp=root/'config/molecular_upgrade_20261107.json';cfg=json.loads(cp.read_text())
    for path,h in cfg['protected_sha256'].items():
        assert digest(root/path)==h, path
    cfg_hash=digest(cp)
    out=root/cfg['output_path'];report=root/cfg['report_path']
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    checks=tests(cfg)
    core=pd.read_csv(root/cfg['input_path'])
    assert len(core)==316 and core.patient_uid.nunique()==160
    ambiguous_treatment_ids=set(core.loc[core.dose_ambiguous.astype(bool),'patient_uid'])
    registry=pd.read_csv(root/cfg['registry_path'])
    full=['log2_'+x for x in registry.loc[registry.discovery_eligible,'analyte']]
    assert len(full)==19
    clinical=pd.read_csv(root/'results/tables/redevelopment/clinical.csv')
    molecular=pd.read_csv(root/'results/tables/redevelopment/molecular.csv')
    extended=clinical.merge(molecular,on=['patient_uid','visit_week'],validate='one_to_one').rename(columns={'easi_total':'observed'})
    extended=extended[extended.age.ge(12)&extended.observed.notna()&extended.visit_week.isin([0,12,24])&extended[full].notna().sum(axis=1).ge(8)].copy()
    keys=['patient_uid','visit_week']
    assert not extended.duplicated(keys).any()
    anchors=np.array(sorted(core.patient_uid.unique()))
    old=pd.read_csv(root/'results/tables/update_20261025/ablation/predictions_private.csv')
    old=old[old.cohort.eq('core_all')]
    outputs=[];audit=[];selection=[];coverage=[];foldmap=[]
    references=old[old.model.isin(['serum3','bsa','bsa_serum3'])].copy()
    references['model']=references.model.map({'serum3':'three_ridge_reference','bsa':'bsa_reference','bsa_serum3':'hybrid_frozen_reference'})
    references['cohort']='core';outputs.append(references)
    for repeat in range(cfg['outer_repeats']):
        assignments={}
        for fold,(_,te) in enumerate(KFold(cfg['outer_folds'],shuffle=True,random_state=cfg['seed']+repeat).split(anchors)):
            assignments.update({x:fold for x in anchors[te]})
        for uid in sorted(set(extended.patient_uid)-set(anchors)):
            assignments[uid]=int(hashlib.sha256(f'{uid}:{repeat}:{cfg["seed"]}'.encode()).hexdigest(),16)%cfg['outer_folds']
        foldmap.extend(dict(patient_uid=u,repeat=repeat,fold=f,added_patient=u not in set(anchors)) for u,f in assignments.items())
        for cohort,data,panels in [('core',core,{'three':cfg['core_panel'],'full19':full}),('extended',extended,{'full19':full})]:
            for fold in range(cfg['outer_folds']):
                testids={u for u,f in assignments.items() if f==fold}
                train=data[~data.patient_uid.isin(testids)&data.visit_week.isin(cfg['development_weeks'])].reset_index(drop=True)
                test=data[data.patient_uid.isin(testids)].reset_index(drop=True)
                assert set(train.patient_uid).isdisjoint(test.patient_uid)
                assert train.visit_week.isin([0,12]).all()
                for panel,features in panels.items():
                    fitted,scores,chosen=select(train,features,cfg)
                    selection.append(scores.assign(cohort=cohort,panel=panel,repeat=repeat,fold=fold))
                    names=['selected','ridge'] if panel=='three' else list(fitted)
                    for name in names:
                        fit=fitted[name]
                        row=test[['patient_uid','visit_week','observed','treatment','center']].copy()
                        row['prediction']=fit.predict(test);row['raw_prediction']=fit.raw(test)
                        row['cohort']=cohort;row['model']=panel+'_'+name;row['repeat']=repeat;row['fold']=fold
                        outputs.append(row)
                        audit.append(dict(cohort=cohort,panel=panel,model=row.model.iloc[0],repeat=repeat,fold=fold,
                                          n_train_patients=train.patient_uid.nunique(),n_train_visits=len(train),
                                          training_weeks='|'.join(map(str,sorted(train.visit_week.unique()))),
                                          n_test_patients=test.patient_uid.nunique(),n_test_visits=len(test),
                                          train_ids='|'.join(sorted(set(train.patient_uid))),test_ids='|'.join(sorted(set(test.patient_uid))),
                                          molecular_features='|'.join(features),n_training_pairs=fit.pair_count,
                                          selected_candidate=json.dumps(fit.candidate,sort_keys=True)))
                    print(f'{cohort} repeat={repeat} fold={fold} panel={panel} selected={fitted["selected"].candidate}',flush=True)
    pred=pd.concat(outputs,ignore_index=True)
    assert not pred.duplicated(['cohort','model','repeat']+keys).any()
    assert np.isfinite(pred.prediction).all() and pred.prediction.between(0,72).all()
    left=pred[pred.cohort.eq('core')&pred.model.eq('three_ridge')]
    right=pred[pred.cohort.eq('core')&pred.model.eq('three_ridge_reference')]
    cmp=left.merge(right,on=keys+['repeat','fold'],validate='one_to_one',suffixes=('_new','_old'))
    discrepancy=float(abs(cmp.prediction_new-cmp.prediction_old).max())
    assert len(cmp)==632 and discrepancy<1e-8
    checks['existing_three_marker_reference_reproduced_max_abs_diff']=discrepancy
    matched=pred[pred.cohort.eq('extended')].merge(core[keys],on=keys,validate='many_to_one')
    matched['cohort']='extended_training_core_evaluation'
    pred=pd.concat([pred,matched],ignore_index=True)
    assay_flags=molecular[keys].copy()
    assay_flags['complete_19_analytes']=molecular[full].notna().all(axis=1)
    pred=pred.merge(assay_flags,on=keys,validate='many_to_one')
    assert pred.complete_19_analytes.notna().all()
    pred.loc[pred.patient_uid.isin(ambiguous_treatment_ids),'treatment']='unclassified_initial_dose'
    for cohort,data in [('core',core),('extended',extended)]:
        for week,g in data.groupby('visit_week'):
            coverage.append(dict(cohort=cohort,week=week,n_patients=g.patient_uid.nunique(),n_visits=len(g),
                                 n_complete_full19=int(g[full].notna().all(axis=1).sum()),n_EASI_gt21=int(g.observed.gt(21).sum())))
    pred.to_csv(out/'predictions_PRIVATE.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'fit_audit_PRIVATE.csv',index=False)
    pd.DataFrame(foldmap).to_csv(out/'fold_map_PRIVATE.csv',index=False)
    pd.concat(selection,ignore_index=True).to_csv(out/'inner_candidates.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    historical_inventory(root).to_csv(out/'historical_method_inventory.csv',index=False)
    print('All predictions saved; patient-bootstrap summaries.',flush=True)
    stats=evaluate(pred,cfg);stats.to_csv(out/'metrics.csv',index=False)
    stats[stats.group_type.eq('overall')].to_csv(out/'overall_metrics.csv',index=False)
    label_map={
        'three_ridge_reference':'Three markers: prior ridge',
        'three_selected':'Three markers: nested selection',
        'full19_ridge':'19 markers: ridge',
        'full19_paired_rbf':'19 markers: paired-change RBF',
        'full19_ordinal_rrr':'19 markers: ordinal-IGA-assisted reduced rank',
        'full19_selected':'19 markers: nested selection',
    }
    concise=stats[stats.cohort.eq('core')&stats.group_type.eq('overall')&stats.model.isin(label_map)].copy()
    concise['method']=concise.model.map(label_map)
    concise[['method','week','n_patients','MAE','MAE_lo','MAE_hi','RMSE','rho','CCC','bias',
             'calibration_slope','calibration_intercept','within1','within2']].to_csv(out/'manuscript_comparison.csv',index=False)
    a=pd.DataFrame(audit)
    selected=a[a.model.str.endswith('selected')].copy()
    chosen=pd.concat([selected[['cohort','model']].reset_index(drop=True),
                      pd.DataFrame(selected.selected_candidate.map(json.loads).tolist())],axis=1)
    chosen.value_counts().rename('outer_folds').reset_index().to_csv(out/'selection_counts.csv',index=False)
    differences=pairwise(pred,cfg);differences.to_csv(out/'paired_differences.csv',index=False)
    paired_changes(pred,cfg).to_csv(out/'paired_change_metrics.csv',index=False)
    checks['independent_scipy_numpy_metric_max_abs_difference']=independent_metric_audit(pred,stats)
    for path,h in cfg['protected_sha256'].items():
        assert digest(root/path)==h, path
    assert digest(cp)==cfg_hash
    checks.update(patient_outer_folds_preserved=True,training_test_patient_overlap=0,
                  W24_used_for_selection=False,no_clinical_deployment_input=True,no_outcome_imputation=True,
                  clinical_IGA_training_auxiliary_only=True,predictions_scored_before_repeat_average=True,
                  prior_main_formula_unchanged=True,earlier_adaptive_exploration_not_covered_by_validation=True,
                  ambiguous_initial_dose_patients_labelled_unclassified=len(ambiguous_treatment_ids))
    manifest=dict(status='PASS_EXPLORATORY_COMPARISON',date='2026-11-07',checks=checks,
                  runtime_seconds=time.monotonic()-started,inputs=cfg['protected_sha256'],
                  config_sha256=cfg_hash,code_sha256=digest(Path(__file__)),
                  software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                  scope='Nested patient procedure comparison; no new clinical formula released; W24 previously examined same cohort',
                  outputs={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    (report/'verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(stats[stats.group_type.eq('overall')][['cohort','model','week','n_patients','MAE','MAE_lo','MAE_hi','rho','CCC']].to_string(index=False))
    print(differences[differences.metric.eq('MAE')].to_string(index=False))
    print('PASS',round(manifest['runtime_seconds'],1),'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args();run(args.root.resolve())
