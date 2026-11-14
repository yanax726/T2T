"""Independently replay the saved W12-to-W24 three-marker analysis."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[3])).resolve()
TABLES = ROOT / 'results/tables/update_20261114/landmark_three'
OUT = ROOT / 'reports/update_20261114/prognosis_integration/landmark/independent_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return {'AUROC': roc_auc_score(y, p), 'AUPRC': average_precision_score(y, p),
            'Brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p),
            'mean_prediction': float(np.mean(p))}


def auc_counts(y, p, counts):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diff = p[pos, None] - p[None, neg]
    concordance = (diff > 0).astype(float) + .5 * (diff == 0)
    a, b = counts[:, pos], counts[:, neg]
    denominator = a.sum(axis=1) * b.sum(axis=1)
    numerator = np.einsum('bi,ij,bj->b', a, concordance, b, optimize=False)
    return np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)



def bootstrap_scores(y, p, counts):
    p = np.clip(p, 1e-7, 1-1e-7)
    order = np.argsort(-p, kind='stable')
    ordered_p = p[order]
    starts = np.r_[0, np.flatnonzero(ordered_p[:-1] != ordered_p[1:]) + 1]
    ends = np.r_[starts[1:]-1, len(y)-1]
    w = counts[:, order]
    wp = w*y[order]
    cumulative_positive = np.cumsum(wp, axis=1)[:, ends]
    cumulative_total = np.cumsum(w, axis=1)[:, ends]
    precision = np.divide(cumulative_positive, cumulative_total,
                          out=np.zeros_like(cumulative_positive, dtype=float), where=cumulative_total > 0)
    group_positive = np.add.reduceat(wp, starts, axis=1)
    npositive = np.einsum('bi,i->b',counts,y,optimize=False)
    valid = (npositive > 0) & (npositive < len(y))
    ap = np.divide(np.sum(precision*group_positive,axis=1),npositive,
                   out=np.full(len(counts),np.nan),where=valid)
    direct = lambda loss: np.einsum('bi,i->b',counts,loss,optimize=False)/len(y)
    return {'AUROC':auc_counts(y,p,counts),'AUPRC':ap,'Brier':direct((p-y)**2),
            'log_loss':direct(-y*np.log(p)-(1-y)*np.log1p(-p)),'mean_prediction':direct(p)}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT/'config/landmark_three_20261114.json'
    cfg = json.loads(cfg_path.read_text())
    assert cfg == json.loads((OUT.parent/'protocol_before_fit.json').read_text())
    sources = [cfg_path, Path(__file__), ROOT/'src/cytot2t/curation/landmark_three_20261114.py',
               ROOT/'src/cytot2t/curation/itch_prediction_20261112.py',
               TABLES/'metrics.csv', TABLES/'paired_differences.csv', TABLES/'repeat_metrics.csv']
    c = pd.read_csv(ROOT/cfg['sources'][0])
    m = pd.read_csv(ROOT/cfg['sources'][1])
    sources.extend(ROOT/p for p in cfg['sources'])
    c = c[c.age.ge(12) & ~c.dose_ambiguous].copy()
    assert c.patient_uid.nunique() == 163 and not c.duplicated(['patient_uid','visit_week']).any()
    assert not m.duplicated(['patient_uid','visit_week']).any()
    b = c[c.visit_week.eq(0)][['patient_uid','age','sex_male','bmi','center','treatment','easi_total']]
    b = b.rename(columns={'easi_total':'baseline_easi'})
    z = b.merge(c[c.visit_week.eq(12)][['patient_uid','easi_total','pp_nrs','iga','bsa','dlqi']],
                on='patient_uid',validate='one_to_one')
    z = z.merge(c[c.visit_week.eq(24)][['patient_uid','easi_total']].rename(columns={'easi_total':'outcome_easi'}),
                on='patient_uid',validate='one_to_one')
    z = z[z.baseline_easi.gt(0) & z.outcome_easi.notna()].copy()
    z['target'] = z.outcome_easi.le(.25*z.baseline_easi).astype(int)
    assert (len(z), int(z.target.sum())) == (93,77)
    z.loc[z.age.lt(18),'dlqi'] = np.nan
    z = z.merge(m[m.visit_week.eq(12)][['patient_uid']+cfg['markers']],on='patient_uid',how='left',validate='one_to_one')
    frames = {'complete_three_markers':z[z[cfg['markers']].notna().all(axis=1)],
              'available_serum_imputation_sensitivity':z[z[cfg['markers']].notna().any(axis=1)]}
    metrics, repeats, deltas = (pd.read_csv(TABLES/(p+'.csv')) for p in ['metrics','repeat_metrics','paired_differences'])
    replay, interval_replay, delta_replay, split_counts = [], [], [], []
    point_errors, interval_errors = [], []
    for population, f in frames.items():
        f = f.sort_values('patient_uid').reset_index(drop=True)
        ids, y = f.patient_uid.tolist(), f.target.to_numpy(int)
        expected = cfg['populations'][population]
        assert (len(f),int(y.sum()),int((1-y).sum())) == (expected['n'],expected['events'],expected['nonevents'])
        assert f.loc[f.age.lt(18),'dlqi'].isna().all()
        pred_path = TABLES/'restricted'/(population+'_oof.csv')
        split_path = TABLES/'restricted'/(population+'_cv_membership.csv')
        sources.extend([pred_path,split_path])
        pred, partitions = pd.read_csv(pred_path),pd.read_csv(split_path)
        assert set(pred.model) == set(cfg['candidates']) and set(pred.patient_uid) == set(ids)
        assert not pred.duplicated(['patient_uid','model','repeat']).any()
        assert pred.groupby(['patient_uid','model']).size().eq(cfg['outer_repeats']).all()
        assert pred.groupby('patient_uid')['observed'].nunique().eq(1).all()
        assert np.array_equal(pred.groupby('patient_uid')['observed'].first().reindex(ids),y)
        assert pred.prediction.between(0,1,inclusive='neither').all()
        ninner = 0
        for row in partitions.itertuples():
            train,test=set(row.train_ids.split('|')),set(row.test_ids.split('|'))
            assert train.isdisjoint(test) and train|test==set(ids)
            outer=list(StratifiedKFold(cfg['outer_folds'],shuffle=True,random_state=cfg['seed']+row.repeat).split(f,y))
            tr,te=outer[row.fold]
            assert train==set(f.iloc[tr].patient_uid) and test==set(f.iloc[te].patient_uid)
            assert (row.train_n,row.test_n,row.train_events,row.test_events)==(len(tr),len(te),int(y[tr].sum()),int(y[te].sum()))
            pp=pred[pred.model.eq(row.model)&pred['repeat'].eq(row.repeat)&pred.fold.eq(row.fold)]
            assert set(pp.patient_uid)==test and np.allclose(pp.training_event_rate,y[tr].mean())
            a_saved,b_saved=json.loads(row.inner_train_ids),json.loads(row.inner_test_ids)
            inner=list(StratifiedKFold(cfg['inner_folds'],shuffle=True,
                random_state=cfg['seed']+100+row.repeat*cfg['outer_folds']+row.fold).split(f.iloc[tr],y[tr]))
            assert len(inner)==len(a_saved)==len(b_saved)
            for (ia,ib),a,b in zip(inner,a_saved,b_saved):
                a,b=set(a),set(b)
                assert a.isdisjoint(b) and a|b==train and a.isdisjoint(test) and b.isdisjoint(test)
                assert a==set(f.iloc[tr].iloc[ia].patient_uid) and b==set(f.iloc[tr].iloc[ib].patient_uid)
                ninner+=1
            assert row.C in cfg['logistic_C'] and row.solver_iterations<row.solver_max_iterations
        assert partitions.groupby(['repeat','fold']).train_ids.nunique().eq(1).all()
        assert partitions.groupby(['repeat','fold']).test_ids.nunique().eq(1).all()
        split_counts.append({'population':population,'n':len(y),'events':int(y.sum()),'nonevents':int((1-y).sum()),
                             'outer_model_records':len(partitions),'inner_partition_records':ninner})
        indices=np.random.default_rng(cfg['seed']+731).integers(len(y),size=(cfg['bootstrap_iterations'],len(y)))
        counts=np.array([np.bincount(index,minlength=len(y)) for index in indices])
        points,boots={},{}
        for model in cfg['candidates']:
            mat=pred[pred.model.eq(model)].pivot(index='patient_uid',columns='repeat',values='prediction').reindex(ids).to_numpy()
            point_reps,boot_reps=[],[]
            for repeat in range(cfg['outer_repeats']):
                p=mat[:,repeat]; vals=scores(y,p);point_reps.append(vals)
                ref=repeats[repeats.population.eq(population)&repeats.model.eq(model)&repeats['repeat'].eq(repeat)].iloc[0]
                for key,value in vals.items():
                    error=abs(value-ref[key]);point_errors.append(error);assert error<1e-12
                    replay.append({'population':population,'evaluation':'single_repetition','model':model,
                                   'repeat':repeat,'metric':key,'independent':value,'saved':ref[key]})
                boot_reps.append(bootstrap_scores(y,p,counts))
            points['mean_repetition',model]={key:np.mean([r[key] for r in point_reps]) for key in point_reps[0]}
            boots['mean_repetition',model]={key:np.mean([r[key] for r in boot_reps],axis=0) for key in boot_reps[0]}
            avg=mat.mean(axis=1)
            points['averaged_patient_probability',model]=scores(y,avg)
            boots['averaged_patient_probability',model]=bootstrap_scores(y,avg,counts)
        for (evaluation,model),vals in points.items():
            for key,value in vals.items():
                ref=metrics[metrics.population.eq(population)&metrics.evaluation.eq(evaluation)&metrics.model.eq(model)&metrics.metric.eq(key)].iloc[0]
                error=abs(value-ref.estimate);point_errors.append(error);assert error<1e-12
                replay.append({'population':population,'evaluation':evaluation,'model':model,'metric':key,'independent':value,'saved':ref.estimate})
                if key in boots[evaluation,model]:
                    low,high=np.nanquantile(boots[evaluation,model][key],[.025,.975])
                    error=max(abs(low-ref.ci_low),abs(high-ref.ci_high));interval_errors.append(error);assert error<1e-12
                    interval_replay.append({'population':population,'evaluation':evaluation,'model':model,'metric':key,
                                            'independent_low':low,'independent_high':high,'saved_low':ref.ci_low,'saved_high':ref.ci_high})
        for row in deltas[deltas.population.eq(population)].itertuples():
            if row.metric not in boots[row.evaluation,row.model]:continue
            key=(row.evaluation,row.model);ref=(row.evaluation,row.reference)
            value=points[key][row.metric]-points[ref][row.metric]
            low,high=np.nanquantile(boots[key][row.metric]-boots[ref][row.metric],[.025,.975])
            error=max(abs(value-row.difference),abs(low-row.ci_low),abs(high-row.ci_high))
            interval_errors.append(error);assert error<1e-12
            delta_replay.append({'population':population,'evaluation':row.evaluation,'model':row.model,'reference':row.reference,
                                 'metric':row.metric,'independent':value,'independent_low':low,'independent_high':high,
                                 'saved':row.difference,'saved_low':row.ci_low,'saved_high':row.ci_high})
    # Inspect the actual preprocessor used by the builder; do not refit anything.
    from cytot2t.curation import itch_prediction_20261112 as shared
    candidates={'three_markers':cfg['markers'],'clinical':cfg['clinical_inputs'],
                'clinical_three_markers':cfg['clinical_inputs']+cfg['markers']}
    for model,cols in candidates.items():
        assert not {'outcome_easi','target','patient_uid','visit_week'}.intersection(cols)
        prep=shared.estimator(cols).named_steps['prep']
        selected=[c for _,_,cc in prep.transformers for c in cc]
        assert selected==cols or set(selected)==set(cols)
        assert prep.remainder=='drop'
    assert cfg['landmark_week']==12 and cfg['outcome_week']==24
    assert all(sha(ROOT/p)==v for p,v in cfg['source_sha256'].items())
    pd.DataFrame(replay).to_csv(OUT/'metric_replay.csv',index=False)
    pd.DataFrame(interval_replay).to_csv(OUT/'interval_replay.csv',index=False)
    pd.DataFrame(delta_replay).to_csv(OUT/'paired_difference_replay.csv',index=False)
    record={'status':'PASS','no_new_fitting':True,'participant_rows_exported':False,
            'raw_clinical_patients':163,'clinically_evaluable':{'n':93,'events':77,'nonevents':16},
            'cohorts_and_partitions':split_counts,'outer_model_records_checked':sum(r['outer_model_records'] for r in split_counts),
            'inner_partition_records_checked':sum(r['inner_partition_records'] for r in split_counts),
            'patient_crossing':0,'matched_model_splits':True,'raw_W24_EASI75_labels_reconstructed':True,
            'W12_marker_inputs_only':True,'adolescent_DLQI_mask':True,'W24_predictor_columns':False,
            'bootstrap_iterations':cfg['bootstrap_iterations'],'bootstrap_method':'Independent patient-count-weighted concordance, tied score-group precision/recall, and direct weighted Brier/log-loss/predicted-rate means',
            'interval_metrics_replayed':['AUROC','AUPRC','Brier','log_loss','mean_prediction'],'max_point_error':float(max(point_errors)),
            'max_interval_error':float(max(interval_errors)),
            'scope':'Current fixed-three W12 concentrations predicting W24 EASI75 relative to W0, not W12-to-W24 improvement or contemporaneous EASI',
            'historical_landmark_not_same_population':'Historical n68=60/8 included one under12; current n68=59/9 uses current eligibility and W12 three-marker availability',
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['status','outer_model_records_checked','inner_partition_records_checked','max_point_error','max_interval_error']},indent=2))


if __name__=='__main__':
    run()
