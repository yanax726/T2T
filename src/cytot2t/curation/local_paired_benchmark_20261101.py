"""Compare saved local predictions with p-EASI on identical observed visits."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from cytot2t.curation.model_ablation_20261025 import bootstrap_metrics, aligned_arrays, METRICS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def published(frame, precise):
    t = frame['log2_CCL17'].to_numpy()*np.log10(2)
    il22 = np.exp2(frame['log2_IL-22'].to_numpy())
    receptor = np.exp2(frame.log2_IL2RA.to_numpy())
    if precise:
        baseline = -36.1180+18.4941*t+.008942*il22-.00946*receptor
        treated = -5.8183+4.0392*t+.002739*il22-.00284*receptor
    else:
        baseline = -36.12+18.49*t+.009*il22-.009*receptor
        treated = -5.82+4.04*t+.003*il22-.003*receptor
    return np.where(frame.visit_week.eq(0), baseline, treated)


def run(root):
    cfgpath=root/'config/local_paired_benchmark_20261101.json'
    cfg=json.loads(cfgpath.read_text())
    paths=[cfgpath,root/cfg['input'],root/cfg['saved_predictions'],root/'config/unified_bsa_three_v1_20261017.json',root/'results/tables/update_20261025/ablation/exploratory_formulas.json']
    before={str(p.relative_to(root)):sha(p) for p in paths}
    out=root/'results/tables/external_validation_20261101';out.mkdir(parents=True,exist_ok=True)
    report=root/'reports/external_validation_20261101';report.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(root/cfg['input'])
    eligible=data[np.isfinite(data[cfg['matched_requirements']]).all(axis=1)].copy()
    assert not eligible.duplicated(['patient_uid','visit_week']).any()
    predictions=pd.read_csv(root/cfg['saved_predictions'])
    predictions=predictions[predictions.cohort.eq(cfg['cohort']) & predictions.model.isin(cfg['model_ids'])].copy()
    keys=['patient_uid','visit_week']
    match=predictions.merge(eligible[keys],on=keys,validate='many_to_one')
    assert not match.duplicated(keys+['model','repeat']).any()
    template=match[match.model.eq('serum3')].merge(eligible[keys+['log2_CCL17','log2_IL-22','log2_IL2RA']],on=keys,validate='many_to_one')
    rows=[match]
    for precise in [True,False]:
        raw=published(template,precise)
        for clip in [False,True]:
            row=template.copy();row['prediction']=np.clip(raw,0,72) if clip else raw
            row['model']='peasi_'+('precise' if precise else 'rounded')+('_clip' if clip else '_raw')
            rows.append(row[match.columns])
    allp=pd.concat(rows,ignore_index=True)
    assert not allp.duplicated(keys+['model','repeat']).any()
    existing=pd.read_csv(root/'results/tables/update_20261017/model/published_matched_predictions_private.csv')
    existing=existing[existing.model.eq('unified_selected')]
    check=allp[allp.model.eq('peasi_rounded_raw')].merge(existing[keys+['repeat','published_prediction']],on=keys+['repeat'],validate='one_to_one')
    assert len(check)==170 and np.max(abs(check.prediction-check.published_prediction))<1e-12
    original=pd.read_csv(root/'results/tables/update_20261017/model/predictions.csv')
    x=allp[allp.model.eq('serum3')].merge(original[original.model.eq('serum_linear')][keys+['repeat','fold','prediction']],on=keys+['repeat','fold'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(x)==170 and np.max(abs(x.prediction_new-x.prediction_old))<1e-8
    stats=[]; paired=[]
    for week,g in allp.groupby('visit_week'):
        for model,sub in g.groupby('model'):
            summary,_=bootstrap_metrics(sub,cfg)
            stats.append(dict(week=int(week),model=model,**summary))
        for a,b in cfg['comparisons']:
            left=g[g.model.eq(a)];right=g[g.model.eq(b)]
            ia,ya,pa=aligned_arrays(left);ib,yb,pb=aligned_arrays(right)
            assert np.array_equal(ia,ib) and np.array_equal(ya,yb)
            draws=np.random.default_rng(cfg['seed']).integers(len(ia),size=(cfg['bootstrap_replicates'],len(ia)))
            va,ba=bootstrap_metrics(left,cfg,draws);vb,bb=bootstrap_metrics(right,cfg,draws)
            for metric in METRICS:
                lo,hi=np.nanquantile(ba[metric]-bb[metric],[.025,.975])
                paired.append(dict(week=int(week),candidate=a,reference=b,metric=metric,difference=va[metric]-vb[metric],ci_low=lo,ci_high=hi,n_patients=len(ia),direction='candidate minus reference; lower MAE/RMSE is better'))
    allp.to_csv(out/'local_matched_predictions_private.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'local_matched_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'local_paired_differences.csv',index=False)
    assert all(sha(root/p)==h for p,h in before.items())
    metadata=dict(status='PASS',no_refitting=True,inputs_unchanged=True,primary_formula_unchanged=True,matched_counts=eligible.groupby('visit_week').size().to_dict(),n_distinct_matched_patients=int(eligible.patient_uid.nunique()),repeat_scored_before_averaging=True,paired_patient_bootstrap=True,rounded_formula_reproduces_prior_run=True,serum_predictions_reproduce_prior_run=True,inputs=before,limits=cfg['limits'])
    (report/'local_verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(pd.DataFrame(stats)[['week','model','n_patients','MAE','MAE_lo','MAE_hi','rho','CCC']].to_string(index=False))
    print(pd.DataFrame(paired).query("metric == 'MAE'").to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);args=parser.parse_args();run(args.root.resolve())
