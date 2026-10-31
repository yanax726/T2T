"""Reconcile laboratory displays and treatment claims against observed records."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from statsmodels.stats.multitest import multipletests
from patsy import dmatrices, build_design_matrices

ROOT = Path(__file__).resolve().parents[3]
ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
LABS = ['ige_numeric', 'eo_count_numeric']


def run(root=ROOT):
    root = Path(root)
    out = root/'results/tables/update_20261031'
    report = root/'reports/review_20261031/checks'
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    source = root/'results/tables/update_20260925/clinical.csv'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = pd.read_csv(source)
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    assert c.patient_uid.nunique() == 163 and not c.duplicated(['patient_uid','visit_week']).any()
    old = root/'results/tables/update_20261017/clinical'
    observed = pd.read_csv(old/'clinical_observed.csv')
    paired = pd.read_csv(old/'clinical_paired_changes.csv')
    checks, coverage, changes, patterns = [], [], [], []
    for target in LABS:
        for scope, cohort in [('all',c), *[(x,g) for x,g in c.groupby('center')]]:
            for tx, g in cohort.groupby('treatment'):
                wide = g.pivot(index='patient_uid',columns='visit_week',values=target)
                for week in [0,4,12,24]:
                    v = wide[week].dropna()
                    coverage.append(dict(target=target,scope=scope,treatment=tx,week=week,n_initial=len(wide),n_observed=len(v),n_missing=len(wide)-len(v)))
                    if scope=='all' and not (target=='ige_numeric' and week==4):
                        saved=observed[observed.target.eq(target)&observed.treatment.eq(tx)&observed.week.eq(week)].iloc[0]
                        assert len(v)==saved.n
                        if len(v):
                            err=max(abs(v.mean()-saved['mean']),abs(v.median()-saved['median']))
                            assert err<1e-9
                            checks.append(dict(check='observed_summary',target=target,treatment=tx,week=week,max_abs_error=err))
                for start,end in [(0,12),(0,24),(12,24)]:
                    h=wide[[start,end]].dropna();delta=h[end]-h[start];n=len(h)
                    if n<2:continue
                    radius=stats.t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)
                    changes.append(dict(target=target,scope=scope,treatment=tx,start=start,end=end,n=n,mean_change=delta.mean(),ci_low=delta.mean()-radius,ci_high=delta.mean()+radius))
                    if scope=='all':
                        saved=paired[paired.target.eq(target)&paired.treatment.eq(tx)&paired.start.eq(start)&paired.end.eq(end)].iloc[0]
                        err=max(abs(delta.mean()-saved.mean_change),abs(delta.mean()-radius-saved.ci_low),abs(delta.mean()+radius-saved.ci_high))
                        assert n==saved.n and err<1e-9
                        checks.append(dict(check='paired_change',target=target,treatment=tx,week=end,max_abs_error=err))
                for week in [12,24]:
                    for available in [True,False]:
                        b=wide.loc[wide[week].notna().eq(available),0].dropna()
                        patterns.append(dict(target=target,scope=scope,treatment=tx,followup_week=week,followup_observed=available,n_baseline=len(b),baseline_median=b.median(),baseline_q25=b.quantile(.25),baseline_q75=b.quantile(.75)))
    weights=pd.read_csv(old/'restricted/overlap_patient_weights.csv').set_index('patient_uid').weight
    baseline=c[c.visit_week.eq(0)].set_index('patient_uid')
    saved_means=pd.read_csv(old/'clinical_trajectory_adjusted.csv')
    saved_contrasts=pd.read_csv(old/'clinical_trajectory_contrasts.csv')
    independent=[]
    time_changes=[]
    for target in ['easi_total','pp_nrs',*LABS]:
        weeks=[2,4,12,24] if target=='pp_nrs' else ([12,24] if target=='ige_numeric' else [4,12,24])
        d=c[c.visit_week.isin(weeks)].copy()
        d['baseline']=d.patient_uid.map(baseline[target]);d['weight']=d.patient_uid.map(weights)
        d=d.dropna(subset=[target,'baseline','weight','age','bmi','sex_male','center'])
        d['y']=np.log1p(d[target]) if target in LABS else d[target]
        if target in LABS:d['baseline']=np.log1p(d.baseline)
        Y,X=dmatrices('y ~ baseline + C(treatment)*C(visit_week) + age + bmi + sex_male + C(center)',d,return_type='dataframe')
        fit=sm.WLS(Y.iloc[:,0],X,weights=d.weight).fit()
        G=d.patient_uid.nunique();rank=np.linalg.matrix_rank(X.to_numpy()*np.sqrt(d.weight.to_numpy())[:,None])
        cov=cov_cluster(fit,d.patient_uid,use_correction=False)*G/(G-1)*(len(d)-1)/(len(d)-rank)
        ref=d.drop_duplicates('patient_uid');vectors={}
        q=saved_means[saved_means.target.eq(target)]
        for row in q.itertuples():
            matrix=np.asarray(build_design_matrices([X.design_info],ref.assign(treatment=row.treatment,visit_week=row.week))[0])
            v=np.average(matrix,axis=0,weights=ref.weight);vectors[(row.week,row.treatment)]=v
            effect=float(v@fit.params);se=np.sqrt(v@cov@v);rad=stats.t.ppf(.975,G-1)*se
            err=max(abs(effect-row.adjusted_mean_scale),abs(effect-rad-row.ci_low),abs(effect+rad-row.ci_high))
            assert err<1e-8,(target,row.week,err)
            independent.append(dict(target=target,week=row.week,comparison=row.treatment,quantity='adjusted_level_and_interval',max_abs_error=err,n_patients=G))
        q=saved_contrasts[saved_contrasts.target.eq(target)&saved_contrasts.family.eq('three_pairwise')]
        for row in q.itertuples():
            v=vectors[(row.week,row.group1)]-vectors[(row.week,row.group2)]
            effect=float(v@fit.params);se=np.sqrt(v@cov@v);rad=stats.t.ppf(.975,G-1)*se
            err=max(abs(effect-row.effect),abs(effect-rad-row.ci_low),abs(effect+rad-row.ci_high))
            assert err<1e-8
            independent.append(dict(target=target,week=row.week,comparison=row.comparison,quantity='adjusted_contrast_and_interval',max_abs_error=err,n_patients=G))
        if target in ['easi_total','pp_nrs']:
            early = 2 if target == 'pp_nrs' else 4
            for end in [12,24]:
                family=[]
                for first,second in [(ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])]:
                    start_contrast=vectors[(early,first)]-vectors[(early,second)]
                    end_contrast=vectors[(end,first)]-vectors[(end,second)]
                    v=end_contrast-start_contrast
                    effect=float(v@fit.params);se=float(np.sqrt(v@cov@v))
                    rad=stats.t.ppf(.975,G-1)*se
                    family.append(dict(target=target,start=early,end=end,group1=first,group2=second,
                                       early_difference=float(start_contrast@fit.params),later_difference=float(end_contrast@fit.params),
                                       change_in_difference=effect,ci_low=effect-rad,ci_high=effect+rad,
                                       p=2*stats.t.sf(abs(effect/se),G-1),n_model_patients=G,
                                       estimand='Later minus early adjusted group difference; fixed baseline weights; observed endpoint population'))
                corrected=multipletests([x['p'] for x in family],method='holm')[1]
                for row,pvalue in zip(family,corrected):row['p_holm']=float(pvalue)
                time_changes.extend(family)
    pd.DataFrame(coverage).to_csv(out/'laboratory_coverage.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'laboratory_paired_changes.csv',index=False)
    pd.DataFrame(patterns).to_csv(out/'laboratory_observation_patterns.csv',index=False)
    pd.DataFrame(independent).to_csv(out/'independent_regression_checks.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'observed_pair_checks.csv',index=False)
    claims=saved_contrasts[saved_contrasts.target.isin(['easi_total','pp_nrs'])&saved_contrasts.week.isin([2,12,24])&saved_contrasts.family.eq('three_pairwise')]
    claims.to_csv(out/'clinical_claim_contrasts.csv',index=False)
    pd.DataFrame(time_changes).to_csv(out/'clinical_contrast_time_changes.csv',index=False)
    assert source_hash==hashlib.sha256(source.read_bytes()).hexdigest()
    result=dict(status='PASS',source_sha256=source_hash,n_patients=163,observed_pair_checks=len(checks),independent_regression_checks=len(independent),maximum_regression_error=max(x['max_abs_error'] for x in independent),observed_values_changed=False,models_changed=False,imputed_values_created=False,software={'statsmodels':sm.__version__},output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))})
    (report/'numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','n_patients','observed_pair_checks','independent_regression_checks','maximum_regression_error']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();run(a.root)
