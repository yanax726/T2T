"""Baseline three-marker prediction of measured week-12 itch improvement."""
from pathlib import Path
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[3])).resolve()
OUT = ROOT/'results/tables/update_20261112/itch_prediction'
REPORT = ROOT/'reports/update_20261112/itch_prediction'
MARKERS = ['log2_CCL17', 'log2_CCL26', 'log2_IL-5']
CLINICAL = ['age', 'sex_male', 'bmi', 'center', 'treatment',
            'easi_total', 'pp_nrs', 'iga', 'bsa', 'dlqi']
MODELS = {'three_markers': MARKERS, 'clinical': CLINICAL,
          'clinical_three_markers': CLINICAL+MARKERS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame():
    raw = pd.read_csv(ROOT/'results/tables/update_20260925/clinical.csv')
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    assert c.patient_uid.nunique() == 163
    assert not c.duplicated(['patient_uid', 'visit_week']).any()
    b = c[c.visit_week.eq(0)].copy()
    b.loc[b.age.lt(18), 'dlqi'] = np.nan
    later = c[c.visit_week.eq(12)][['patient_uid', 'pp_nrs']].rename(columns={'pp_nrs':'itch12'})
    b = b.merge(later, on='patient_uid', how='left', validate='one_to_one')
    b = b[b.pp_nrs.ge(4) & b.itch12.notna()].copy()
    assert len(b) == 112 and b.pp_nrs.between(0,10).all() and b.itch12.between(0,10).all()
    m = pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv')
    b = b.merge(m[m.visit_week.eq(0)][['patient_uid']+MARKERS], on='patient_uid',
                how='left', validate='one_to_one')
    b['target'] = (b.pp_nrs-b.itch12 >= 4).astype(int)
    b = b.sort_values('patient_uid').reset_index(drop=True)
    primary = b[b[MARKERS].notna().all(axis=1)].copy().reset_index(drop=True)
    sensitivity = b[b[MARKERS].notna().any(axis=1)].copy().reset_index(drop=True)
    assert (len(primary), int(primary.target.sum())) == (102,88)
    assert (len(sensitivity), int(sensitivity.target.sum())) == (105,89)
    return primary, sensitivity


def estimator(columns):
    cats = [c for c in columns if c in ['center','treatment']]
    nums = [c for c in columns if c not in cats]
    branches = [('numeric', Pipeline([
        ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler())]),nums)]
    if cats:
        branches.append(('categorical',Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
    return Pipeline([('prep',ColumnTransformer(branches,remainder='drop')),
                     ('model',LogisticRegression(C=1.,penalty='l2',solver='lbfgs',
                                                 max_iter=3000,class_weight=None))])


def fit_cohort(d, cfg, population):
    rows, audits, coefficients = [], [], []
    y = d.target.to_numpy(int)
    for repeat in range(cfg['outer_repeats']):
        outer = StratifiedKFold(cfg['outer_folds'], shuffle=True, random_state=cfg['seed']+repeat)
        for fold,(tr,te) in enumerate(outer.split(d,y)):
            train_ids, test_ids = set(d.iloc[tr].patient_uid), set(d.iloc[te].patient_uid)
            assert train_ids.isdisjoint(test_ids)
            inner = StratifiedKFold(cfg['inner_folds'],shuffle=True,
                                    random_state=cfg['seed']+100+repeat*cfg['outer_folds']+fold)
            splits = list(inner.split(d.iloc[tr],y[tr]))
            for a,b in splits:
                assert set(d.iloc[tr].iloc[a].patient_uid).isdisjoint(d.iloc[tr].iloc[b].patient_uid)
            prior = float(y[tr].mean())
            for name,columns in MODELS.items():
                search = GridSearchCV(estimator(columns),{'model__C':cfg['logistic_C']},
                                      cv=splits,scoring='neg_log_loss',n_jobs=1,error_score='raise')
                # Accelerate matmul warnings were checked against non-BLAS arithmetic.
                # Finite scores/parameters/probabilities and solver limits remain gates.
                with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
                    search.fit(d.iloc[tr],y[tr])
                    p = search.predict_proba(d.iloc[te])[:,1]
                assert all(np.isfinite(values).all() for key,values in search.cv_results_.items()
                           if key.endswith('_test_score'))
                assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
                for k,i in enumerate(te):
                    rows.append(dict(population=population,model=name,repeat=repeat,fold=fold,
                                     patient_uid=d.iloc[i].patient_uid,observed=int(y[i]),
                                     prediction=float(p[k]),training_event_rate=prior))
                best = search.best_estimator_
                assert np.isfinite(best.named_steps['model'].coef_).all()
                assert np.isfinite(best.named_steps['model'].intercept_).all()
                assert int(best.named_steps['model'].n_iter_[0]) < 3000
                names = best.named_steps['prep'].get_feature_names_out()
                beta = best.named_steps['model'].coef_.ravel()
                for feature,value in zip(names,beta):
                    coefficients.append(dict(population=population,model=name,repeat=repeat,
                                             fold=fold,feature=feature,standardized_coefficient=value))
                audits.append(dict(population=population,model=name,repeat=repeat,fold=fold,
                                   train_n=len(tr),test_n=len(te),train_events=int(y[tr].sum()),
                                   test_events=int(y[te].sum()),C=search.best_params_['model__C'],
                                   solver_iterations=int(best.named_steps['model'].n_iter_[0]),
                                   train_ids='|'.join(sorted(train_ids)),test_ids='|'.join(sorted(test_ids)),
                                   inner_train_ids=json.dumps([d.iloc[tr].iloc[a].patient_uid.tolist() for a,b in splits]),
                                   inner_test_ids=json.dumps([d.iloc[tr].iloc[b].patient_uid.tolist() for a,b in splits])))
            print(population,'repeat',repeat+1,'fold',fold+1,flush=True)
    pred = pd.DataFrame(rows)
    assert not pred.duplicated(['patient_uid','model','repeat']).any()
    assert pred.groupby(['patient_uid','model']).size().eq(cfg['outer_repeats']).all()
    private = OUT/'restricted'; private.mkdir(exist_ok=True,parents=True)
    pred.to_csv(private/(population+'_oof.csv'),index=False)
    pd.DataFrame(audits).to_csv(private/(population+'_cv_membership.csv'),index=False)
    pd.DataFrame(coefficients).to_csv(OUT/(population+'_fold_coefficients.csv'),index=False)
    public_audit = pd.DataFrame(audits).drop(columns=['train_ids','test_ids','inner_train_ids','inner_test_ids'])
    public_audit.to_csv(OUT/(population+'_fit_summary.csv'),index=False)
    return pred


def metric_values(y,p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    return {'AUROC':roc_auc_score(y,p),'AUPRC':average_precision_score(y,p),
            'Brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),
            'mean_prediction':float(p.mean())}


def bootstrap_values(y,p,indices):
    yy=y[indices]; pp=np.clip(p[indices],1e-7,1-1e-7)
    n1=yy.sum(axis=1); n0=len(y)-n1
    valid=(n1>0)&(n0>0)
    auc=np.full(len(indices),np.nan)
    ranks=rankdata(pp,axis=1,method='average')
    auc[valid]=((ranks*yy).sum(axis=1)-n1*(n1+1)/2)[valid]/(n1*n0)[valid]
    order=np.argsort(-pp,axis=1,kind='stable')
    sy=np.take_along_axis(yy,order,axis=1); sp=np.take_along_axis(pp,order,axis=1)
    precision=np.cumsum(sy,axis=1)/np.arange(1,len(y)+1)
    group_end=np.column_stack([sp[:,:-1]!=sp[:,1:],np.ones(len(indices),bool)])
    ends=np.where(group_end,np.arange(len(y)),len(y))
    group_last=np.minimum.accumulate(ends[:,::-1],axis=1)[:,::-1]
    ap=np.full(len(indices),np.nan)
    ap[valid]=(sy*np.take_along_axis(precision,group_last,axis=1)).sum(axis=1)[valid]/n1[valid]
    return {'AUROC':auc,'AUPRC':ap,'Brier':np.mean((pp-yy)**2,axis=1),
            'log_loss':np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp),axis=1),
            'mean_prediction':np.mean(pp,axis=1)}


def evaluate(pred,cfg,population):
    ids=sorted(pred.patient_uid.unique()); repeats=sorted(pred['repeat'].unique())
    y=pred.groupby('patient_uid')['observed'].first().reindex(ids).to_numpy(int)
    indices=np.random.default_rng(cfg['seed']+731).integers(len(ids),size=(cfg['bootstrap_iterations'],len(ids)))
    points,boots,repeat_rows,calibration,curves={},{},[],[],[]
    grid=np.linspace(0,1,101)
    for name in MODELS:
        pp=pred[pred.model.eq(name)].pivot(index='patient_uid',columns='repeat',values='prediction').reindex(ids).to_numpy()
        rep_points=[]; rep_boot=[]
        for j,repeat in enumerate(repeats):
            values=metric_values(y,pp[:,j]);rep_points.append(values)
            repeat_rows.append(dict(population=population,model=name,repeat=repeat,n=len(y),events=int(y.sum()),**values))
            rep_boot.append(bootstrap_values(y,pp[:,j],indices))
            # Check tie-aware vectorized bootstrap against sklearn itself.
            for i in range(3):
                k=indices[i]
                if len(np.unique(y[k]))==2:
                    assert abs(rep_boot[-1]['AUROC'][i]-roc_auc_score(y[k],pp[k,j]))<1e-12
                    assert abs(rep_boot[-1]['AUPRC'][i]-average_precision_score(y[k],pp[k,j]))<1e-12
            fpr,tpr,_=roc_curve(y,pp[:,j])
            for x,t in zip(grid,np.interp(grid,fpr,tpr)):
                curves.append(dict(population=population,model=name,repeat=repeat,FPR=x,TPR=t))
            p=np.clip(pp[:,j],1e-7,1-1e-7);z=logit(p)
            row=dict(population=population,model=name,repeat=repeat,observed_rate=float(y.mean()),
                     predicted_rate=float(p.mean()),calibration_in_large=np.nan,calibration_slope=np.nan)
            with warnings.catch_warnings(record=True) as warning_list:
                try:
                    offset=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=z).fit()
                    fit=sm.GLM(y,sm.add_constant(z),family=sm.families.Binomial()).fit()
                    if offset.converged and fit.converged:
                        row.update(calibration_in_large=float(offset.params[0]),calibration_slope=float(fit.params[1]),status='converged')
                    else:row['status']='not_converged'
                except Exception as exc:row['status']=type(exc).__name__
                row['warning_count']=len(warning_list)
            calibration.append(row)
        points[('mean_repetition',name)]={k:float(np.mean([x[k] for x in rep_points])) for k in rep_points[0]}
        boots[('mean_repetition',name)]={k:np.mean([x[k] for x in rep_boot],axis=0) for k in rep_boot[0]}
        points[('averaged_patient_probability',name)]=metric_values(y,pp.mean(axis=1))
        boots[('averaged_patient_probability',name)]=bootstrap_values(y,pp.mean(axis=1),indices)
    rows,differences=[],[]
    for (evaluation,name),values in points.items():
        for metric,value in values.items():
            low,high=np.nanquantile(boots[(evaluation,name)][metric],[.025,.975])
            rows.append(dict(population=population,evaluation=evaluation,model=name,metric=metric,
                             estimate=value,ci_low=low,ci_high=high,n=len(y),events=int(y.sum()),
                             nonevents=int(len(y)-y.sum()),event_rate=float(y.mean())))
            if name!='clinical':
                delta=boots[(evaluation,name)][metric]-boots[(evaluation,'clinical')][metric]
                low,high=np.nanquantile(delta,[.025,.975])
                differences.append(dict(population=population,evaluation=evaluation,model=name,
                                        reference='clinical',metric=metric,
                                        difference=value-points[(evaluation,'clinical')][metric],
                                        ci_low=low,ci_high=high,n=len(y),events=int(y.sum())))
    priors=pred[pred.model.eq('clinical')].pivot(index='patient_uid',columns='repeat',values='training_event_rate').reindex(ids).to_numpy()
    null=[]
    for j,repeat in enumerate(repeats):
        null.append(dict(population=population,repeat=repeat,n=len(y),events=int(y.sum()),
                         Brier=brier_score_loss(y,priors[:,j]),log_loss=log_loss(y,priors[:,j],labels=[0,1]),
                         prevalence_AUPRC_reference=float(y.mean())))
    return (pd.DataFrame(rows),pd.DataFrame(differences),pd.DataFrame(repeat_rows),
            pd.DataFrame(calibration),pd.DataFrame(curves),pd.DataFrame(null))


def run():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/itch_prediction_20261112.json').read_text())
    inputs=cfg['sources']+['config/itch_prediction_20261112.json']
    hashes={p:sha(ROOT/p) for p in inputs}
    declared={x['path']:x['sha256'] for x in json.loads(
        (ROOT/'config/reproduction_inputs.json').read_text())['inputs']}
    for path in cfg['sources']:
        if hashes[path] != declared.get(path):
            raise ValueError('Restricted input does not match its declared snapshot: '+path)
    primary,sensitivity=frame()
    outputs={name:[] for name in ['metrics','paired_differences','repeat_metrics','calibration','roc_curves','null_reference']}
    for label,data in [('complete_three_markers',primary),('available_serum_imputation_sensitivity',sensitivity)]:
        pred=fit_cohort(data,cfg,label)
        for key,df in zip(outputs,evaluate(pred,cfg,label)):outputs[key].append(df)
    for key,frames in outputs.items():pd.concat(frames,ignore_index=True).to_csv(OUT/(key+'.csv'),index=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    receipt=dict(status='PASS',primary_n=102,primary_events=88,primary_nonevents=14,
                 sensitivity_n=105,sensitivity_events=89,candidates=list(MODELS),
                 outer_repetitions=5,outer_folds=5,inner_folds=3,fit_patient_crossing=0,
                 baseline_only_inputs=True,imputed_response_values=0,
                 source_values_changed=False,model_and_preprocessing_refit_in_training_folds=True,
                 primary_evaluation='mean_repetition',historical_comparable_evaluation='averaged_patient_probability',
                 bootstrap='2000 paired patient resamples of saved cross-validated predictions, conditional only',
                 input_sha256=hashes,package_boundary='Patient predictions and split membership stay under restricted/')
    (REPORT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    main=pd.concat(outputs['metrics']).query("population == 'complete_three_markers' and metric == 'AUROC'")
    print(main[['evaluation','model','estimate','ci_low','ci_high','n','events']].to_string(index=False),flush=True)


if __name__=='__main__':
    run()
