"""Independent replay of compact-panel predictions, selection and uncertainty."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT/'results/tables/update_20261112/itch_panel_size'
OUT = ROOT/'reports/update_20261112/itch_panel_size/independent_audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(markers):
    raw=pd.read_csv(ROOT/'results/tables/update_20260925/clinical.csv')
    c=raw[raw.age.ge(12)&~raw.dose_ambiguous]
    base=c[c.visit_week.eq(0)].copy()
    assert len(base)==163 and base.patient_uid.is_unique
    base.loc[base.age.lt(18),'dlqi']=np.nan
    later=c[c.visit_week.eq(12)][['patient_uid','pp_nrs']].rename(columns={'pp_nrs':'followup_itch'})
    base=base.merge(later,on='patient_uid',how='left',validate='one_to_one')
    base=base[base.pp_nrs.ge(4)&base.followup_itch.notna()].copy()
    mol=pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv')
    base=base.merge(mol[mol.visit_week.eq(0)][['patient_uid']+markers],on='patient_uid',
                    how='left',validate='one_to_one')
    base['target']=base.pp_nrs.sub(base.followup_itch).ge(4).astype(int)
    fixed=['log2_CCL17','log2_CCL26','log2_IL-5']
    return {'complete_three_markers':base[base[fixed].notna().all(axis=1)].sort_values('patient_uid').reset_index(drop=True),
            'available_serum_imputation_sensitivity':base[base[fixed].notna().any(axis=1)].sort_values('patient_uid').reset_index(drop=True)}


def auc_counts(y,p,counts):
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    difference=p[pos,None]-p[None,neg]
    concordance=(difference>0).astype(float)+.5*(difference==0)
    wp,wn=counts[:,pos],counts[:,neg]
    numerator=np.einsum('bi,ij,bj->b',wp,concordance,wn,optimize=False)
    denominator=wp.sum(axis=1)*wn.sum(axis=1)
    return np.divide(numerator,denominator,out=np.full(len(counts),np.nan),where=denominator>0)


def metric_values(y,p):
    return dict(AUROC=roc_auc_score(y,p),AUPRC=average_precision_score(y,p),Brier=brier_score_loss(y,p))


def top_markers(train,markers,k):
    x=train[markers].to_numpy(float)
    medians=np.nanmedian(x,axis=0)
    x=np.where(np.isnan(x),medians,x)
    mean=x.mean(axis=0);sd=x.std(axis=0,ddof=0)
    x=(x-mean)/np.where(sd>0,sd,1.)
    y=train.target.to_numpy(int)
    total_mean=x.mean(axis=0)
    between=np.zeros(len(markers));within=np.zeros(len(markers))
    for level in [0,1]:
        z=x[y==level];group_mean=z.mean(axis=0)
        between+=len(z)*(group_mean-total_mean)**2
        within+=np.einsum('ij,ij->j',z-group_mean,z-group_mean,optimize=False)
    f=between/(within/(len(x)-2))
    chosen=np.argsort(np.nan_to_num(f,nan=-np.inf),kind='mergesort')[-k:]
    return [markers[j] for j in sorted(chosen)]


def expected_choices(grid):
    best={}
    for k,g in grid.groupby('k'):
        # Recompute mean and sample standard error from the saved inner losses.
        a=g.copy()
        losses=a[['inner0_loss','inner1_loss','inner2_loss']].to_numpy()
        mean=losses.mean(axis=1);se=losses.std(axis=1,ddof=1)/np.sqrt(3)
        assert np.allclose(mean,a.mean_loss,rtol=0,atol=1e-12)
        assert np.allclose(se,a.SE,rtol=0,atol=1e-12)
        a['loss_replay']=mean;a['SE_replay']=se
        row=a.sort_values(['loss_replay','C'],kind='mergesort').iloc[0]
        best[int(k)]=row
    compact=[row for k,row in best.items() if k<=6]
    minimum=min(compact,key=lambda r:(r.loss_replay,int(r.k),r.C))
    threshold=float(minimum.loss_replay+minimum.SE_replay)
    candidates=[r for r in compact if r.loss_replay<=threshold+1e-12]
    chosen=min(candidates,key=lambda r:(int(r.k),r.loss_replay,r.C))
    return best,chosen,threshold,int(minimum.k)


def batch_descriptions(all_frames):
    assay=pd.read_csv(ROOT/'results/tables/redevelopment/assay_long.csv')
    assay=assay[assay.visit_week.eq(0)&assay.analyte.eq('IL-4')]
    assert not assay.patient_uid.duplicated().any()
    files=sorted(assay.assay_file.dropna().unique())
    labels={name:f'assay_file_{i+1}' for i,name in enumerate(files)}
    rows=[]
    for population,d in all_frames.items():
        a=d[['patient_uid','center','treatment','target']].merge(
            assay[['patient_uid','assay_file','state','log2_concentration']],on='patient_uid',validate='one_to_one')
        a['assay_batch']=a.assay_file.map(labels)
        for scope,keys in [('center',['center']),('assay_file',['assay_batch']),
                           ('center_and_assay',['center','assay_batch']),
                           ('center_and_response',['center','target']),
                           ('assay_and_response',['assay_batch','target']),
                           ('assay_and_treatment',['assay_batch','treatment'])]:
            for key,g in a.groupby(keys,dropna=False):
                key=key if isinstance(key,tuple) else (key,)
                x=g.log2_concentration.dropna()
                row=dict(population=population,scope=scope,n=len(g),responders=int(g.target.sum()),
                         nonresponders=int((1-g.target).sum()),n_observed=len(x),distinct=int(x.nunique()),
                         log2_median=float(x.median()),log2_q25=float(x.quantile(.25)),
                         log2_q75=float(x.quantile(.75)),log2_min=float(x.min()),log2_max=float(x.max()),
                         dominant_fraction=float(x.value_counts(normalize=True).max()))
                row.update(dict(zip(keys,key)));rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'IL4_center_assay_distribution.csv',index=False)
    return len(files)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg_path=ROOT/'config/itch_panel_size_20261112.json'
    cfg=json.loads(cfg_path.read_text())
    registry=pd.read_csv(ROOT/'config/marker_registry.csv')
    markers=['log2_'+m for m in registry[registry.discovery_eligible].analyte]
    assert len(markers)==19
    data=frames(markers)
    saved_metrics=pd.read_csv(TABLES/'metrics.csv')
    saved_deltas=pd.read_csv(TABLES/'paired_differences.csv')
    saved_repeats=pd.read_csv(TABLES/'repeat_metrics.csv')
    sources=[cfg_path,ROOT/'config/marker_registry.csv',TABLES/'metrics.csv',TABLES/'paired_differences.csv',
             TABLES/'repeat_metrics.csv',ROOT/'src/cytot2t/curation/itch_panel_size_20261112.py',
             ROOT/'results/tables/update_20260925/clinical.csv',ROOT/'results/tables/redevelopment/molecular.csv']
    point_errors=[];ci_errors=[];replay=[];delta_replay=[];selection_replay=[]
    outer_records=0;inner_records=0;fscore_replays=0;best_choices=0
    expected_models={'clinical','fixed_three_serum','clinical_fixed_three'}|{
        f'{branch}_{suffix}' for branch in ['serum','clinical_serum']
        for suffix in ['ranked3','ranked4','ranked5','ranked6','ranked19','compact']}
    for population,d in data.items():
        ids=d.patient_uid.tolist();y=d.target.to_numpy(int)
        assert (len(d),int(y.sum()))==((102,88) if population=='complete_three_markers' else (105,89))
        pred_path=TABLES/'restricted'/f'{population}_oof.csv'
        split_path=TABLES/'restricted'/f'{population}_membership.csv'
        grid_path=TABLES/f'{population}_inner_grid.csv'
        sources.extend([pred_path,split_path,grid_path])
        pred,splits,grid=[pd.read_csv(p) for p in [pred_path,split_path,grid_path]]
        assert set(pred.model)==expected_models and set(pred.patient_uid)==set(ids)
        assert not pred.duplicated(['patient_uid','model','repeat']).any()
        assert pred.groupby(['patient_uid','model']).size().eq(5).all()
        assert np.array_equal(pred.groupby('patient_uid')['observed'].first().reindex(ids),y)
        assert pred.groupby('patient_uid')['observed'].nunique().eq(1).all()
        assert pred.prediction.between(0,1,inclusive='neither').all()
        for (repeat,fold),same in pred.groupby(['repeat','fold']):
            assert all(set(h.patient_uid)==set(same.patient_uid) for _,h in same.groupby('model'))
        choices={}
        selected_sets={}
        for (branch,repeat,fold),g in grid.groupby(['branch','repeat','fold']):
            assert len(g)==15 and set(g.k)=={3,4,5,6,19}
            choices[(branch,repeat,fold)]=expected_choices(g)
            best_choices+=len(g)
        for row in splits.itertuples():
            train,test=set(row.train_ids.split('|')),set(row.test_ids.split('|'))
            assert train.isdisjoint(test) and train|test==set(ids)
            tr,te=list(StratifiedKFold(5,shuffle=True,random_state=cfg['seed']+row.repeat).split(d,y))[row.fold]
            assert train==set(d.iloc[tr].patient_uid) and test==set(d.iloc[te].patient_uid)
            assert (row.train_n,row.test_n)==(len(tr),len(te))
            assert (row.train_events,row.test_events)==(int(y[tr].sum()),int(y[te].sum()))
            a,b=json.loads(row.inner_train_ids),json.loads(row.inner_test_ids)
            expected=list(StratifiedKFold(3,shuffle=True,random_state=cfg['seed']+100+row.repeat*5+row.fold).split(d.iloc[tr],y[tr]))
            for aa,bb,(ia,ib) in zip(a,b,expected):
                aa,bb=set(aa),set(bb)
                assert aa.isdisjoint(bb) and aa|bb==train and not (aa|bb)&test
                assert aa==set(d.iloc[tr].iloc[ia].patient_uid) and bb==set(d.iloc[tr].iloc[ib].patient_uid)
                inner_records+=1
            outer_records+=1
            branch='clinical_serum' if row.model.startswith('clinical_serum') else 'serum'
            best,compact,threshold,minimum_k=choices[(branch,row.repeat,row.fold)]
            if row.model.endswith('_compact'):
                assert row.k==int(compact.k) and row.C==float(compact.C)
                assert np.isclose(row.one_SE_threshold,threshold,rtol=0,atol=1e-12)
                assert row.inner_minimum_k==minimum_k
            else:assert row.C==float(best[int(row.k)].C)
            key=(row.repeat,row.fold,int(row.k))
            if key not in selected_sets:selected_sets[key]=top_markers(d.iloc[tr],markers,int(row.k))
            expected_selected=selected_sets[key]
            assert row.selected_markers.split('|')==expected_selected
            fscore_replays+=1
            selection_replay.append(dict(population=population,model=row.model,repeat=row.repeat,fold=row.fold,
                                         k=int(row.k),C=float(row.C),selection_reproduced=True,
                                         selected_markers='|'.join(m[5:] for m in expected_selected)))
        # The three old controls must be copied exactly, not refitted under a new ranking protocol.
        old=pd.read_csv(ROOT/'results/tables/update_20261112/itch_prediction/restricted'/f'{population}_oof.csv')
        old.model=old.model.map({'clinical':'clinical','three_markers':'fixed_three_serum','clinical_three_markers':'clinical_fixed_three'})
        keys=['patient_uid','model','repeat','fold']
        z=pred[pred.model.isin(old.model.unique())].merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
        assert len(z)==len(old) and np.array_equal(z.prediction_new,z.prediction_old)
        assert np.array_equal(z.observed_new,z.observed_old)
        draws=np.random.default_rng(cfg['seed']+731).integers(len(y),size=(cfg['bootstrap'],len(y)))
        counts=np.array([np.bincount(x,minlength=len(y)) for x in draws])
        points={};boots={}
        for model,g in pred.groupby('model'):
            p=g.pivot(index='patient_uid',columns='repeat',values='prediction').reindex(ids).to_numpy()
            reps=[];aucs=[];briers=[]
            for repeat in range(5):
                v=metric_values(y,p[:,repeat]);reps.append(v)
                ref=saved_repeats[saved_repeats.population.eq(population)&saved_repeats.model.eq(model)&saved_repeats['repeat'].eq(repeat)].iloc[0]
                for m,x in v.items():
                    point_errors.append(abs(x-ref[m]));assert abs(x-ref[m])<1e-12
                aucs.append(auc_counts(y,p[:,repeat],counts))
                briers.append(np.einsum('bi,i->b',counts,(p[:,repeat]-y)**2,optimize=False)/len(y))
            points[('mean_repetition',model)]={m:np.mean([r[m] for r in reps]) for m in reps[0]}
            boots[('mean_repetition',model)]=dict(AUROC=np.mean(aucs,axis=0),Brier=np.mean(briers,axis=0))
            mean=p.mean(axis=1)
            points[('averaged_patient_probability',model)]=metric_values(y,mean)
            boots[('averaged_patient_probability',model)]=dict(AUROC=auc_counts(y,mean,counts),
                Brier=np.einsum('bi,i->b',counts,(mean-y)**2,optimize=False)/len(y))
        for (evaluation,model),v in points.items():
            for metric,value in v.items():
                ref=saved_metrics[saved_metrics.population.eq(population)&saved_metrics.evaluation.eq(evaluation)&saved_metrics.model.eq(model)&saved_metrics.metric.eq(metric)].iloc[0]
                err=abs(value-ref.estimate);point_errors.append(err);assert err<1e-12
                row=dict(population=population,evaluation=evaluation,model=model,metric=metric,independent_estimate=value,saved_estimate=ref.estimate)
                if metric in ['AUROC','Brier']:
                    low,high=np.nanquantile(boots[(evaluation,model)][metric],[.025,.975])
                    err=max(abs(low-ref.ci_low),abs(high-ref.ci_high));ci_errors.append(err);assert err<1e-12
                    row.update(independent_ci_low=low,independent_ci_high=high)
                replay.append(row)
        for ref in saved_deltas[saved_deltas.population.eq(population)&saved_deltas.metric.isin(['AUROC','Brier'])].itertuples():
            evaluation,model,reference,metric=ref.evaluation,ref.model,ref.reference,ref.metric
            value=points[(evaluation,model)][metric]-points[(evaluation,reference)][metric]
            delta=boots[(evaluation,model)][metric]-boots[(evaluation,reference)][metric]
            low,high=np.nanquantile(delta,[.025,.975])
            err=max(abs(value-ref.difference),abs(low-ref.ci_low),abs(high-ref.ci_high))
            ci_errors.append(err);assert err<1e-12
            delta_replay.append(dict(population=population,evaluation=evaluation,model=model,reference=reference,metric=metric,
                                     independent_difference=value,independent_ci_low=low,independent_ci_high=high))
    # Confirm selection remains inside the estimator supplied to GridSearchCV.
    from cytot2t.curation import itch_panel_size_20261112 as builder
    for clinical in [False,True]:
        pipeline=builder.estimator(markers,clinical)
        prep=pipeline.named_steps['prep']
        molecular=[x for x in prep.transformers if x[0]=='molecules'][0]
        assert molecular[2]==markers
        assert list(molecular[1].named_steps)==['impute','scale','select'] and prep.remainder=='drop'
    pd.DataFrame(replay).to_csv(OUT/'metric_replay.csv',index=False)
    pd.DataFrame(delta_replay).to_csv(OUT/'paired_bootstrap_replay.csv',index=False)
    selection_table=pd.DataFrame(selection_replay)
    selection_table.to_csv(OUT/'selection_replay.csv',index=False)
    compositions=[]
    for (population,model),g in selection_table.groupby(['population','model']):
        counts=g.selected_markers.value_counts()
        compositions.append(dict(population=population,model=model,outer_model_fits=len(g),
                                 unique_marker_sets=g.selected_markers.nunique(),
                                 selected_sizes='|'.join(str(x) for x in sorted(g.k.unique())),
                                 most_common_set=counts.index[0],most_common_set_fits=int(counts.iloc[0])))
    pd.DataFrame(compositions).to_csv(OUT/'panel_composition_counts.csv',index=False)
    batch_files=batch_descriptions(data)
    sources.append(ROOT/'results/tables/redevelopment/assay_long.csv')
    report=dict(status='PASS',population_counts={p:dict(n=len(d),events=int(d.target.sum())) for p,d in data.items()},
                models_per_population=15,OOF_rows_per_patient_model=5,new_outer_model_records=outer_records,
                stored_inner_partitions_checked=inner_records,independent_outer_splits=50,independent_inner_partitions=150,
                patient_crossing=0,topk_training_ANOVA_replays=fscore_replays,inner_grid_rows_replayed=best_choices,
                inner_SE_definition='Sample SD(ddof=1) divided by sqrt(3)',one_SE_and_C_choices_reproduced=True,
                fixed_three_controls_copied_exactly=True,selector_inside_inner_pipeline=True,
                point_metric_max_error=max(point_errors),bootstrap_interval_max_error=max(ci_errors),
                independent_bootstrap_method='Patient-count-weighted positive/negative concordance and direct Brier losses',
                bootstrap_replays=len(delta_replay),IL4_assay_files=batch_files,
                IL4_file_grouping_boundary='Assay-file groups are observed report-file groups, not independently verified plate or technical batch identifiers.',
                IL4_file_case_mix_sensitivity=True,
                IL4_selection_frequency_not_biomarker_validation=True,
                new_models_fitted=0,input_values_modified=False,patient_rows_exported=False,
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','new_outer_model_records','stored_inner_partitions_checked',
                        'topk_training_ANOVA_replays','point_metric_max_error','bootstrap_interval_max_error']},indent=2))


if __name__=='__main__':
    run()
