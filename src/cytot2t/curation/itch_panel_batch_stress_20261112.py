"""Fixed assay-file holdout diagnostic for baseline itch-response prediction."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import itch_prediction_20261112 as prior
from . import itch_panel_size_20261112 as panels


ROOT = prior.ROOT
OUT = ROOT/'results/tables/update_20261112/itch_panel_size/batch_stress'
REPORT = ROOT/'reports/update_20261112/itch_panel_size/batch_stress'


def build_frames(markers):
    molecular = pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv')
    assay = pd.read_csv(ROOT/'results/tables/redevelopment/assay_long.csv')
    assay = assay[assay.visit_week.eq(0) & assay.analyte.eq('IL-4')]
    assert assay.patient_uid.is_unique
    aliases = {name:f'assay_file_{i+1}' for i,name in
               enumerate(sorted(assay.assay_file.dropna().unique()))}
    frames = {}
    for population,d in zip(['complete_three_markers',
                              'available_serum_imputation_sensitivity'],prior.frame()):
        d = d.drop(columns=prior.MARKERS).merge(
            molecular[molecular.visit_week.eq(0)][['patient_uid']+markers],
            on='patient_uid',validate='one_to_one')
        d = d.merge(assay[['patient_uid','assay_file']],on='patient_uid',validate='one_to_one')
        d['_assay_file_group'] = d.assay_file.map(aliases)
        assert d._assay_file_group.notna().all() and d._assay_file_group.nunique()==4
        frames[population] = d.sort_values('patient_uid').reset_index(drop=True)
    return frames,aliases


def search_grid(d,train,inner,model,markers,cfg):
    clinical = model in ['clinical','clinical_fixed_three','clinical_serum_compact']
    compact = model.endswith('_compact')
    if compact:
        estimator = panels.estimator(markers,clinical)
        parameters = {'prep__molecules__select__k':cfg['compact_sizes'],
                      'model__C':cfg['C_grid']}
    else:
        columns = (prior.CLINICAL if clinical else [])
        if model != 'clinical':columns=columns+prior.MARKERS
        assert '_assay_file_group' not in columns and 'assay_file' not in columns
        estimator = prior.estimator(columns)
        parameters = {'model__C':cfg['C_grid']}
    search = GridSearchCV(estimator,parameters,cv=inner,scoring='neg_log_loss',
                          n_jobs=1,error_score='raise',refit=False)
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        search.fit(d.iloc[train],d.iloc[train].target.to_numpy(int))
    assert all(np.isfinite(v).all() for key,v in search.cv_results_.items()
               if key.endswith('_test_score'))
    rows=[]
    for i,params in enumerate(search.cv_results_['params']):
        loss=np.array([-search.cv_results_[f'split{j}_test_score'][i]
                       for j in range(cfg['inner_folds'])])
        rows.append(dict(k=int(params.get('prep__molecules__select__k',0 if model=='clinical' else 3)),
                         C=float(params['model__C']),mean_loss=float(loss.mean()),
                         SE=float(loss.std(ddof=1)/np.sqrt(len(loss))),
                         **{f'inner{j}_loss':float(x) for j,x in enumerate(loss)}))
    grid=pd.DataFrame(rows)
    if compact:
        chosen,threshold,minimum_k=panels.choose(grid,compact=True)
        estimator.set_params(prep__molecules__select__k=int(chosen.k),model__C=float(chosen.C))
    else:
        chosen=grid.sort_values(['mean_loss','C'],kind='stable').iloc[0]
        threshold,minimum_k=np.nan,np.nan
        estimator.set_params(model__C=float(chosen.C))
    return estimator,chosen,threshold,minimum_k,grid


def fit_population(d,markers,cfg,population):
    predictions=[];membership=[];summaries=[];grids=[];coverage=[]
    y=d.target.to_numpy(int)
    for group in sorted(d._assay_file_group.unique()):
        test=np.flatnonzero(d._assay_file_group.eq(group))
        train=np.flatnonzero(d._assay_file_group.ne(group))
        assert set(d.iloc[train].patient_uid).isdisjoint(d.iloc[test].patient_uid)
        assert group not in set(d.iloc[train]._assay_file_group)
        assert np.bincount(y[train],minlength=2).min()>=cfg['inner_folds']
        splitter=StratifiedKFold(cfg['inner_folds'],shuffle=True,
                                 random_state=cfg['seed']+300+int(group.rsplit('_',1)[1]))
        inner=list(splitter.split(d.iloc[train],y[train]))
        for a,b in inner:
            assert set(d.iloc[train].iloc[a].patient_uid).isdisjoint(d.iloc[train].iloc[b].patient_uid)
            assert not set(d.iloc[train].iloc[b].patient_uid)&set(d.iloc[test].patient_uid)
        for model in cfg['models']:
            estimator,choice,threshold,minimum_k,grid=search_grid(d,train,inner,model,markers,cfg)
            with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
                estimator.fit(d.iloc[train],y[train])
                p=estimator.predict_proba(d.iloc[test])[:,1]
            assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
            assert np.isfinite(estimator['model'].coef_).all()
            assert np.isfinite(estimator['model'].intercept_).all()
            assert int(estimator['model'].n_iter_[0])<3000
            if model.endswith('_compact'):
                selector=estimator['prep'].named_transformers_['molecules']['select']
                selected=np.array(markers)[selector.get_support()].tolist()
            else:selected=[] if model=='clinical' else list(prior.MARKERS)
            assert len(selected)==int(choice.k)
            info=dict(population=population,holdout_assay_file=group,model=model,
                      train_n=len(train),train_responders=int(y[train].sum()),
                      train_nonresponders=int(len(train)-y[train].sum()),test_n=len(test),
                      test_responders=int(y[test].sum()),test_nonresponders=int(len(test)-y[test].sum()),
                      selected_k=int(choice.k),C=float(choice.C),
                      solver_iterations=int(estimator['model'].n_iter_[0]),
                      one_SE_threshold=threshold,inner_minimum_k=minimum_k,
                      selected_markers='|'.join(m[5:] for m in selected))
            summaries.append(info)
            membership.append(dict(**info,train_ids='|'.join(sorted(d.iloc[train].patient_uid)),
                                   test_ids='|'.join(sorted(d.iloc[test].patient_uid)),
                                   inner_train_ids=json.dumps([d.iloc[train].iloc[a].patient_uid.tolist() for a,b in inner]),
                                   inner_test_ids=json.dumps([d.iloc[train].iloc[b].patient_uid.tolist() for a,b in inner])))
            for j,index in enumerate(test):
                predictions.append(dict(population=population,holdout_assay_file=group,model=model,
                                        patient_uid=d.iloc[index].patient_uid,observed=int(y[index]),
                                        prediction=float(p[j]),training_event_rate=float(y[train].mean())))
            for row in grid.to_dict('records'):
                grids.append(dict(population=population,holdout_assay_file=group,model=model,**row))
            observed=d.iloc[test][selected].notna().to_numpy(dtype=bool)
            coverage.append(dict(population=population,holdout_assay_file=group,model=model,
                                 test_n=len(test),selected_k=len(selected),
                                 complete_selected_input_n=int(observed.all(axis=1).sum()),
                                 imputed_marker_values=int((~observed).sum()),total_marker_values=int(observed.size)))
        print(population,group,'complete',flush=True)
    pred=pd.DataFrame(predictions)
    assert not pred.duplicated(['patient_uid','model']).any()
    assert pred.groupby('model').size().eq(len(d)).all()
    private=OUT/'restricted';private.mkdir(parents=True,exist_ok=True)
    pred.to_csv(private/(population+'_predictions.csv'),index=False)
    pd.DataFrame(membership).to_csv(private/(population+'_membership.csv'),index=False)
    pd.DataFrame(summaries).to_csv(OUT/(population+'_fit_summary.csv'),index=False)
    pd.DataFrame(grids).to_csv(OUT/(population+'_inner_grid.csv'),index=False)
    pd.DataFrame(coverage).to_csv(OUT/(population+'_observation_coverage.csv'),index=False)
    return pred


def point_metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    values=dict(AUROC=np.nan,AUPRC=np.nan,Brier=brier_score_loss(y,p),
                log_loss=log_loss(y,p,labels=[0,1]),mean_prediction=float(p.mean()))
    if len(np.unique(y))==2:
        values.update(AUROC=roc_auc_score(y,p),AUPRC=average_precision_score(y,p))
    return values


def evaluate(pred,cfg,population):
    ids=sorted(pred.patient_uid.unique())
    first=pred[pred.model.eq('clinical')].set_index('patient_uid').reindex(ids)
    y=first.observed.to_numpy(int);groups=first.holdout_assay_file.to_numpy()
    rng=np.random.default_rng(cfg['seed']+cfg['bootstrap_seed_offset'])
    parts=[]
    for group in sorted(np.unique(groups)):
        idx=np.flatnonzero(groups==group)
        parts.append(rng.choice(idx,size=(cfg['bootstrap_iterations'],len(idx)),replace=True))
    draws=np.column_stack(parts)
    metrics=[];per_group=[];points={};boots={}
    for model,g in pred.groupby('model',sort=True):
        g=g.set_index('patient_uid').reindex(ids)
        assert np.array_equal(g.observed,y) and np.array_equal(g.holdout_assay_file,groups)
        p=g.prediction.to_numpy();values=point_metrics(y,p)
        boot=prior.bootstrap_values(y,p,draws)
        points[model]=values;boots[model]=boot
        for metric,value in values.items():
            low,high=np.nanquantile(boot[metric],[.025,.975])
            metrics.append(dict(population=population,model=model,metric=metric,estimate=value,
                                ci_low=low,ci_high=high,n=len(y),responders=int(y.sum()),
                                nonresponders=int(len(y)-y.sum()),source_file_groups=4,
                                interval_scope='conditional_patient_bootstrap_within_heldout_file'))
        for group in sorted(np.unique(groups)):
            z=g[g.holdout_assay_file.eq(group)];yy=z.observed.to_numpy(int)
            values=point_metrics(yy,z.prediction.to_numpy())
            row=dict(population=population,holdout_assay_file=group,model=model,n=len(z),
                     responders=int(yy.sum()),nonresponders=int(len(z)-yy.sum()),
                     AUC_status='estimable' if len(np.unique(yy))==2 else 'NE_single_response_class',
                     training_event_rate=float(z.training_event_rate.iloc[0]),
                     training_prior_Brier=brier_score_loss(yy,z.training_event_rate.to_numpy()),**values)
            per_group.append(row)
    differences=[]
    for contrast in cfg['comparisons']:
        model,reference=contrast['model'],contrast['reference']
        for metric,value in points[model].items():
            delta=boots[model][metric]-boots[reference][metric]
            low,high=np.nanquantile(delta,[.025,.975])
            differences.append(dict(population=population,model=model,reference=reference,metric=metric,
                                    difference=value-points[reference][metric],ci_low=low,ci_high=high,n=len(y)))
    return pd.DataFrame(metrics),pd.DataFrame(per_group),pd.DataFrame(differences)


def run():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'config/itch_panel_batch_stress_20261112.json';cfg=json.loads(path.read_text())
    hashes={p:panels.sha(ROOT/p) for p in cfg['protected_sources']+[str(path.relative_to(ROOT))]}
    declared={x['path']:x['sha256'] for x in json.loads(
        (ROOT/'config/reproduction_inputs.json').read_text())['inputs']}
    for p in ['results/tables/update_20260925/clinical.csv','results/tables/redevelopment/molecular.csv',
              'results/tables/redevelopment/assay_long.csv']:
        assert hashes[p]==declared[p]
    registry=pd.read_csv(ROOT/'config/marker_registry.csv')
    markers=['log2_'+m for m in registry[registry.discovery_eligible].analyte]
    assert len(markers)==19 and '_assay_file_group' not in prior.CLINICAL
    frames,aliases=build_frames(markers)
    private=OUT/'restricted';private.mkdir(parents=True,exist_ok=True)
    (private/'source_file_aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
    results=[[],[],[]];descriptions=[]
    for population,d in frames.items():
        assert (len(d),int(d.target.sum()))==(cfg['populations'][population]['n'],cfg['populations'][population]['responders'])
        for group,g in d.groupby('_assay_file_group'):
            descriptions.append(dict(population=population,holdout_assay_file=group,n=len(g),
                                     responders=int(g.target.sum()),nonresponders=int(len(g)-g.target.sum()),
                                     centers='|'.join(sorted(g.center.unique())),
                                     n_dupi=int(g.treatment.eq('dupilumab').sum()),
                                     n_abro100=int(g.treatment.eq('abrocitinib_100mg').sum()),
                                     n_abro200=int(g.treatment.eq('abrocitinib_200mg').sum())))
        pred=fit_population(d,markers,cfg,population)
        for dest,table in zip(results,evaluate(pred,cfg,population)):dest.append(table)
    for name,tables in zip(['aggregate_metrics','assay_file_metrics','paired_differences'],results):
        pd.concat(tables,ignore_index=True).to_csv(OUT/(name+'.csv'),index=False)
    pd.DataFrame(descriptions).to_csv(OUT/'assay_file_case_mix.csv',index=False)
    assert all(panels.sha(ROOT/p)==h for p,h in hashes.items())
    report=dict(status='PASS',input_sha256=hashes,models=cfg['models'],source_file_groups=4,
                populations=cfg['populations'],outer_fits=40,patient_train_test_overlap=0,
                heldout_file_seen_in_training=False,assay_file_used_as_predictor=False,
                outcomes_imputed=0,main_panel_choice_rule_changed=False,
                single_class_AUC='NE; Brier/logloss retained',
                interpretation='Joint assay-file/case-mix transport stress, not external validation or causal batch adjustment',
                interval_scope=cfg['aggregate_interval'])
    (REPORT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    m=pd.concat(results[0]);print(m[m.metric.isin(['AUROC','Brier'])][
        ['population','model','metric','estimate','ci_low','ci_high']].to_string(index=False),flush=True)


if __name__=='__main__':
    run()
