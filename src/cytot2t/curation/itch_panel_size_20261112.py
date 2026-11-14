"""Bounded, nested comparison of serum panel sizes for W12 itch response."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import itch_prediction_20261112 as prior


ROOT = prior.ROOT
OUT = ROOT/'results/tables/update_20261112/itch_panel_size'
REPORT = ROOT/'reports/update_20261112/itch_panel_size'
SIZES = [3,4,5,6,19]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimator(markers,clinical):
    molecular=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                        ('scale',StandardScaler()),('select',SelectKBest(f_classif,k=3))])
    blocks=[('molecules',molecular,markers)]
    if clinical:
        blocks.insert(0,('clinical',prior.estimator(prior.CLINICAL).named_steps['prep'],prior.CLINICAL))
    return Pipeline([('prep',ColumnTransformer(blocks,remainder='drop')),
                     ('model',LogisticRegression(solver='lbfgs',penalty='l2',
                                                 class_weight=None,max_iter=3000))])


def choose(grid,compact=False):
    """Choose C within k; optionally apply an inner-only compact 1SE rule."""
    best=[]
    for k,g in grid.groupby('k'):
        z=g.sort_values(['mean_loss','C'],kind='stable').iloc[0]
        best.append(z)
    best=pd.DataFrame(best)
    if not compact:
        return best
    best=best[best.k.le(6)]
    minimum=best.sort_values(['mean_loss','k','C'],kind='stable').iloc[0]
    threshold=minimum.mean_loss+minimum.SE
    selected=best[best.mean_loss.le(threshold+1e-12)].sort_values(['k','mean_loss','C'],kind='stable').iloc[0]
    return selected,float(threshold),int(minimum.k)


def fit_population(d,markers,cfg,population):
    base_path=ROOT/'results/tables/update_20261112/itch_prediction/restricted'/f'{population}_oof.csv'
    base=pd.read_csv(base_path)
    base.model=base.model.map({'clinical':'clinical','three_markers':'fixed_three_serum',
                              'clinical_three_markers':'clinical_fixed_three'})
    base['selected_k']=base.model.map({'clinical':0,'fixed_three_serum':3,'clinical_fixed_three':3})
    rows=base.to_dict('records'); audits=[]; selections=[]; inner_rows=[]; coverage=[]
    y=d.target.to_numpy(int)
    for repeat in range(cfg['outer_repeats']):
        outer=StratifiedKFold(cfg['outer_folds'],shuffle=True,random_state=cfg['seed']+repeat)
        for fold,(tr,te) in enumerate(outer.split(d,y)):
            train_ids=set(d.iloc[tr].patient_uid);test_ids=set(d.iloc[te].patient_uid)
            assert train_ids.isdisjoint(test_ids)
            existing=base[base['repeat'].eq(repeat)&base.fold.eq(fold)]
            assert set(existing.patient_uid)==test_ids
            assert existing.groupby('patient_uid')['observed'].first().to_dict()==dict(zip(d.iloc[te].patient_uid,y[te]))
            inner=StratifiedKFold(cfg['inner_folds'],shuffle=True,
                                  random_state=cfg['seed']+100+repeat*cfg['outer_folds']+fold)
            splits=list(inner.split(d.iloc[tr],y[tr]))
            for a,b in splits:
                assert set(d.iloc[tr].iloc[a].patient_uid).isdisjoint(d.iloc[tr].iloc[b].patient_uid)
            for branch,clinical in [('serum',False),('clinical_serum',True)]:
                search=GridSearchCV(estimator(markers,clinical),
                    {'prep__molecules__select__k':SIZES,'model__C':cfg['C_grid']},
                    cv=splits,scoring='neg_log_loss',n_jobs=1,error_score='raise',refit=False)
                with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
                    search.fit(d.iloc[tr],y[tr])
                assert all(np.isfinite(v).all() for key,v in search.cv_results_.items()
                           if key.endswith('_test_score'))
                grid=[]
                for i,params in enumerate(search.cv_results_['params']):
                    losses=np.array([-search.cv_results_[f'split{j}_test_score'][i]
                                     for j in range(cfg['inner_folds'])])
                    row=dict(k=int(params['prep__molecules__select__k']),C=float(params['model__C']),
                             mean_loss=float(losses.mean()),SE=float(losses.std(ddof=1)/np.sqrt(len(losses))),
                             **{f'inner{j}_loss':float(v) for j,v in enumerate(losses)})
                    grid.append(row)
                    inner_rows.append(dict(population=population,branch=branch,repeat=repeat,fold=fold,**row))
                grid=pd.DataFrame(grid);best=choose(grid);compact,threshold,minimum_k=choose(grid,True)
                cache={}
                for row in best.itertuples():
                    k=int(row.k);c=float(row.C)
                    model=estimator(markers,clinical).set_params(prep__molecules__select__k=k,model__C=c)
                    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
                        model.fit(d.iloc[tr],y[tr]);p=model.predict_proba(d.iloc[te])[:,1]
                    assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
                    assert np.isfinite(model['model'].coef_).all() and np.isfinite(model['model'].intercept_).all()
                    assert model['model'].n_iter_[0]<3000
                    selector=model['prep'].named_transformers_['molecules']['select']
                    selected=np.array(markers)[selector.get_support()].tolist()
                    assert len(selected)==k
                    cache[k]=(p,selected,c,int(model['model'].n_iter_[0]))
                variants=[(f'{branch}_ranked{k}',k) for k in SIZES]+[(f'{branch}_compact',int(compact.k))]
                for name,k in variants:
                    p,selected,c,n_iter=cache[k]
                    observed=d.iloc[te][selected].notna().to_numpy()
                    for j,index in enumerate(te):
                        rows.append(dict(population=population,model=name,repeat=repeat,fold=fold,
                                         patient_uid=d.iloc[index].patient_uid,observed=int(y[index]),
                                         prediction=float(p[j]),training_event_rate=float(y[tr].mean()),selected_k=k))
                    for marker in markers:
                        selections.append(dict(population=population,model=name,repeat=repeat,fold=fold,
                                               selected_k=k,marker=marker.removeprefix('log2_'),selected=marker in selected))
                    coverage.append(dict(population=population,model=name,repeat=repeat,fold=fold,
                                         selected_k=k,test_n=len(te),test_events=int(y[te].sum()),
                                         complete_selected_input_n=int(observed.all(axis=1).sum()),
                                         imputed_marker_values=int((~observed).sum()),
                                         total_marker_values=int(observed.size)))
                    audits.append(dict(population=population,model=name,repeat=repeat,fold=fold,k=k,C=c,
                                       solver_iterations=n_iter,train_n=len(tr),test_n=len(te),
                                       train_events=int(y[tr].sum()),test_events=int(y[te].sum()),
                                       one_SE_threshold=threshold if name.endswith('_compact') else np.nan,
                                       inner_minimum_k=minimum_k if name.endswith('_compact') else np.nan,
                                       selected_markers='|'.join(selected),
                                       train_ids='|'.join(sorted(train_ids)),test_ids='|'.join(sorted(test_ids)),
                                       inner_train_ids=json.dumps([d.iloc[tr].iloc[a].patient_uid.tolist() for a,b in splits]),
                                       inner_test_ids=json.dumps([d.iloc[tr].iloc[b].patient_uid.tolist() for a,b in splits])))
            print(population,'repeat',repeat+1,'fold',fold+1,flush=True)
    pred=pd.DataFrame(rows)
    assert not pred.duplicated(['patient_uid','model','repeat']).any()
    assert pred.groupby(['patient_uid','model']).size().eq(cfg['outer_repeats']).all()
    private=OUT/'restricted';private.mkdir(parents=True,exist_ok=True)
    pred.to_csv(private/(population+'_oof.csv'),index=False)
    a=pd.DataFrame(audits);a.to_csv(private/(population+'_membership.csv'),index=False)
    a.drop(columns=['train_ids','test_ids','inner_train_ids','inner_test_ids']).to_csv(OUT/(population+'_fit_summary.csv'),index=False)
    pd.DataFrame(inner_rows).to_csv(OUT/(population+'_inner_grid.csv'),index=False)
    pd.DataFrame(coverage).to_csv(OUT/(population+'_observation_coverage.csv'),index=False)
    s=pd.DataFrame(selections)
    s.groupby(['population','model','marker']).selected.agg(['sum','count','mean']).reset_index().rename(
        columns={'sum':'selected_fits','count':'outer_fits','mean':'selection_fraction'}).to_csv(
            OUT/(population+'_marker_stability.csv'),index=False)
    s[['population','model','repeat','fold','selected_k']].drop_duplicates().groupby(
        ['population','model','selected_k']).size().rename('fits').reset_index().to_csv(
            OUT/(population+'_panel_size_frequency.csv'),index=False)
    return pred


def evaluate(pred,cfg,population):
    ids=sorted(pred.patient_uid.unique());y=pred.groupby('patient_uid')['observed'].first().reindex(ids).to_numpy(int)
    indices=np.random.default_rng(cfg['seed']+731).integers(len(ids),size=(cfg['bootstrap'],len(ids)))
    points={};boots={};repetition=[]
    for model,g in pred.groupby('model',sort=True):
        p=g.pivot(index='patient_uid',columns='repeat',values='prediction').reindex(ids).to_numpy()
        values=[prior.metric_values(y,p[:,j]) for j in range(cfg['outer_repeats'])]
        bb=[prior.bootstrap_values(y,p[:,j],indices) for j in range(cfg['outer_repeats'])]
        for j,v in enumerate(values):repetition.append(dict(population=population,model=model,repeat=j,n=len(y),events=int(y.sum()),**v))
        points[('mean_repetition',model)]={m:float(np.mean([v[m] for v in values])) for m in values[0]}
        boots[('mean_repetition',model)]={m:np.mean([v[m] for v in bb],axis=0) for m in bb[0]}
        points[('averaged_patient_probability',model)]=prior.metric_values(y,p.mean(axis=1))
        boots[('averaged_patient_probability',model)]=prior.bootstrap_values(y,p.mean(axis=1),indices)
    metrics=[];delta=[]
    for (evaluation,model),point in points.items():
        for metric,value in point.items():
            low,high=np.nanquantile(boots[(evaluation,model)][metric],[.025,.975])
            metrics.append(dict(population=population,evaluation=evaluation,model=model,metric=metric,
                                estimate=value,ci_low=low,ci_high=high,n=len(y),events=int(y.sum()),
                                nonevents=int(len(y)-y.sum())))
        refs=['clinical','fixed_three_serum']
        if model.startswith('serum_'):refs+=['serum_ranked3','serum_ranked19']
        if model.startswith('clinical_serum_'):refs+=['clinical_serum_ranked3','clinical_fixed_three','clinical_serum_ranked19']
        for reference in dict.fromkeys(refs):
            if model==reference:continue
            for metric in point:
                d=boots[(evaluation,model)][metric]-boots[(evaluation,reference)][metric]
                low,high=np.nanquantile(d,[.025,.975])
                delta.append(dict(population=population,evaluation=evaluation,model=model,reference=reference,
                                  metric=metric,difference=point[metric]-points[(evaluation,reference)][metric],
                                  ci_low=low,ci_high=high,n=len(y),events=int(y.sum())))
    return pd.DataFrame(metrics),pd.DataFrame(delta),pd.DataFrame(repetition)


def run(replay_only=False):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/itch_panel_size_20261112.json').read_text())
    assert SIZES==cfg['panel_sizes']
    guarded=cfg['protected_sources']+['config/itch_panel_size_20261112.json']+[
        'results/tables/update_20261112/itch_prediction/restricted/'+p+'_oof.csv'
        for p in ['complete_three_markers','available_serum_imputation_sensitivity']]
    hashes={p:sha(ROOT/p) for p in guarded}
    declared={x['path']:x['sha256'] for x in json.loads((ROOT/'config/reproduction_inputs.json').read_text())['inputs']}
    for p in ['results/tables/update_20260925/clinical.csv','results/tables/redevelopment/molecular.csv']:
        assert hashes[p]==declared[p]
    registry=pd.read_csv(ROOT/'config/marker_registry.csv')
    markers=['log2_'+m for m in registry[registry.discovery_eligible].analyte]
    assert len(markers)==19 and not set(cfg['excluded_markers'])&set(m[5:] for m in markers)
    molecular=pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv')
    results=[[],[],[]]
    for population,d in zip(['complete_three_markers','available_serum_imputation_sensitivity'],prior.frame()):
        d=d.drop(columns=prior.MARKERS).merge(molecular[molecular.visit_week.eq(0)][['patient_uid']+markers],
                                              on='patient_uid',validate='one_to_one')
        d=d.sort_values('patient_uid').reset_index(drop=True)
        assert len(d)==cfg['populations'][population]['n']
        if replay_only:
            pred=pd.read_csv(OUT/'restricted'/(population+'_oof.csv'))
            assert set(pred.patient_uid)==set(d.patient_uid)
        else:
            pred=fit_population(d,markers,cfg,population)
        for dest,table in zip(results,evaluate(pred,cfg,population)):dest.append(table)
    for key,tables in zip(['metrics','paired_differences','repeat_metrics'],results):
        pd.concat(tables,ignore_index=True).to_csv(OUT/(key+'.csv'),index=False)
    main=pd.concat(results[0],ignore_index=True)
    summary=[]
    for population in cfg['populations']:
        subset=main[main.population.eq(population)&main.evaluation.eq('mean_repetition')]
        for label,serum,combined in [('fixed_EASI_three','fixed_three_serum','clinical_fixed_three')]+[
                (f'itch_ranked{k}',f'serum_ranked{k}',f'clinical_serum_ranked{k}') for k in SIZES]+[
                ('inner_selected_compact','serum_compact','clinical_serum_compact')]:
            row=dict(population=population,panel=label,**cfg['populations'][population])
            for prefix,model in [('serum_only',serum),('clinical_plus_serum',combined)]:
                z=subset[subset.model.eq(model)].set_index('metric');a=z.loc['AUROC']
                row.update({prefix+'_AUC':a.estimate,prefix+'_AUC_low':a.ci_low,
                            prefix+'_AUC_high':a.ci_high,prefix+'_Brier':z.loc['Brier','estimate']})
            summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'panel_size_summary.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    receipt=dict(status='PASS',input_sha256=hashes,markers=markers,populations=cfg['populations'],
                 no_patient_train_test_crossing=True,baseline_only=True,no_outcomes_imputed=True,
                 selector_inside_inner_pipeline=True,nested_compact_selection='inner-logloss1SE only',
                 outer_metric_selection_used=False,final_fixed_marker_panel_independently_validated=False,
                 fitted_in_this_invocation=not replay_only)
    (REPORT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.concat(results[0]).query("population=='complete_three_markers' and evaluation=='mean_repetition' and metric=='AUROC'")[
        ['model','estimate','ci_low','ci_high']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-only',action='store_true',help='Evaluate saved predictions without fitting')
    run(parser.parse_args().replay_only)
