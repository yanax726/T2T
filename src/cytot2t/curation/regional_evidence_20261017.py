"""Rebuild regional EASI burden and paired treatment contrasts from the clinical snapshot."""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from patsy import build_design_matrices, dmatrices
from statsmodels.stats.multitest import multipletests

ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
REGIONS = {'head_neck': 7.2, 'upper_limb': 14.4, 'trunk': 21.6, 'lower_limb': 28.8}
WINDOWS = [(0, 12), (12, 24), (0, 24)]
SCOPES = ['overall'] + ARMS + ['pooled_abrocitinib']


def subset(d, scope):
    return d if scope == 'overall' else d[d.treatment.isin(ARMS[1:])] if scope == 'pooled_abrocitinib' else d[d.treatment.eq(scope)]


def summarize(x):
    x = pd.Series(x).dropna(); n = len(x)
    mean = x.mean(); se = x.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
    half = stats.t.ppf(.975, n-1)*se if n >= 5 else np.nan
    return dict(n=n, mean=mean, sd=x.std(ddof=1), median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75), ci_low=mean-half, ci_high=mean+half, status='estimated' if n >= 5 else 'descriptive_n_lt5')


def bh(d, keys):
    d['q'] = np.nan
    for _, idx in d.groupby(keys, dropna=False).groups.items():
        p = d.loc[idx, 'p'].dropna()
        if len(p): d.loc[p.index, 'q'] = multipletests(p, method='fdr_bh')[1]
    return d


def run(root):
    root = Path(root).resolve(); source = root/'results/tables/update_20260925/clinical.csv'
    out = root/'results/tables/update_20261017/regional'; out.mkdir(parents=True, exist_ok=True)
    private = out/'restricted'; private.mkdir(exist_ok=True)
    raw = pd.read_csv(source)
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous & raw.treatment.isin(ARMS)].copy()
    assert not c.duplicated(['patient_uid','visit_week']).any()
    cols = ['easi_'+r for r in REGIONS]
    complete = c[cols].notna().all(axis=1) & c.easi_total.notna()
    total_valid = (c[cols].sum(axis=1, min_count=4)-c.easi_total).abs().le(.01)
    bounds = pd.concat([c['easi_'+r].between(0,m) for r,m in REGIONS.items()],axis=1).all(axis=1)
    reliable = complete & total_valid & bounds
    assert (reliable == c.regional_reliable).all(), 'The explicit region audit differs from the curated flag.'
    c['regional_eligible'] = reliable
    baseline = c[c.visit_week.eq(0)][['patient_uid','easi_total']].rename(columns={'easi_total':'baseline_easi'})
    d = c[reliable & c.visit_week.isin([0,12,24])].merge(baseline,on='patient_uid',validate='many_to_one')
    audit = c[c.visit_week.isin([0,12,24])].groupby(['visit_week','treatment']).agg(n_initial=('patient_uid','size'),n_recorded_total=('easi_total','count'),n_reliable=('regional_eligible','sum')).reset_index()
    audit['excluded_or_missing'] = audit.n_initial-audit.n_reliable
    audit.to_csv(out/'regional_availability.csv',index=False)
    burden = []; changes = []; pairframes = []
    for region,maximum in REGIONS.items():
        col='easi_'+region
        for scope in SCOPES:
            for week in [0,12,24]:
                h=subset(d[d.visit_week.eq(week)],scope)
                burden.append(dict(region=region,scope=scope,week=week,theoretical_max=maximum,**summarize(h[col]/maximum)))
        for start,end in WINDOWS:
            a=d[d.visit_week.eq(start)][['patient_uid','treatment','center','age','sex_male','bmi','baseline_easi',col]].rename(columns={col:'start_region'})
            b=d[d.visit_week.eq(end)][['patient_uid',col]].rename(columns={col:'end_region'})
            pairs=a.merge(b,on='patient_uid',validate='one_to_one')
            pairs['region']=region; pairs['start']=start; pairs['end']=end
            pairs['raw_improvement']=pairs.start_region-pairs.end_region
            pairs['pct_improvement']=100*pairs.raw_improvement/pairs.start_region.where(pairs.start_region.gt(0))
            pairs['normalized_improvement']=pairs.raw_improvement/maximum
            pairframes.append(pairs)
            for scope in SCOPES:
                h=subset(pairs,scope)
                for outcome in ['raw_improvement','pct_improvement','normalized_improvement']:
                    changes.append(dict(region=region,scope=scope,start=start,end=end,outcome=outcome,theoretical_max=maximum,n_complete_pairs=len(h),n_zero_start=int(h.start_region.eq(0).sum()),**summarize(h[outcome])))
    pd.DataFrame(burden).to_csv(out/'regional_normalized_burden.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'regional_paired_improvement.csv',index=False)
    pairs=pd.concat(pairframes,ignore_index=True);pairs.to_csv(private/'regional_improvement_source.csv',index=False)
    contrasts=[]
    for (region,start,end),g in pairs.groupby(['region','start','end']):
        for outcome in ['raw_improvement','pct_improvement']:
            h=g.dropna(subset=[outcome,'start_region','baseline_easi','age','sex_male','bmi','center']).copy()
            mix=(h.treatment.eq(ARMS[1]).sum()/h.treatment.isin(ARMS[1:]).sum())
            comparisons=[('D - A100',ARMS[0],ARMS[1]),('D - A200',ARMS[0],ARMS[2]),('A200 - A100',ARMS[2],ARMS[1]),('D - pooled A',ARMS[0],'pooled_abrocitinib')]
            formula=f'{outcome} ~ C(treatment) + start_region + baseline_easi + age + sex_male + bmi + C(center)'
            Y,design=dmatrices(formula,h,return_type='dataframe'); X=design.to_numpy(); y=Y.iloc[:,0].to_numpy()
            u,s,vt=svd(X,full_matrices=False); rank=int((s>s[0]*1e-10).sum()); V=vt[:rank].T
            beta=np.einsum('ij,j->i',V,np.einsum('ij,i->j',u[:,:rank],y)/s[:rank])
            residual=y-np.einsum('ij,j->i',X,beta); bread=np.einsum('ik,jk,k->ij',V,V,1/s[:rank]**2)
            leverage=np.square(u[:,:rank]).sum(axis=1); scores=X*(residual/(1-leverage))[:,None]
            covariance=np.einsum('ij,jk,lk->il',bread,np.einsum('ki,kj->ij',scores,scores),bread)
            means={}
            for arm in ARMS:
                hh=h.copy();hh['treatment']=arm
                means[arm]=np.asarray(build_design_matrices([design.design_info],hh)[0]).mean(axis=0)
            means['pooled_abrocitinib']=mix*means[ARMS[1]]+(1-mix)*means[ARMS[2]]
            for label,left,right in comparisons:
                a=subset(h,left)[outcome];b=subset(h,right)[outcome]
                info=dict(region=region,start=start,end=end,outcome=outcome,comparison=label,n_left=len(a),n_right=len(b),n_model=len(h),pooled_A100_fraction=mix,model_rank=rank,max_leverage=float(leverage.max()))
                diff=a.mean()-b.mean();se2=a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)
                df=se2**2/((a.var(ddof=1)/len(a))**2/(len(a)-1)+(b.var(ddof=1)/len(b))**2/(len(b)-1))
                half=stats.t.ppf(.975,df)*np.sqrt(se2)
                p=stats.ttest_ind(a,b,equal_var=False).pvalue if min(len(a),len(b))>=5 else np.nan
                contrasts.append(dict(**info,analysis='observed_Welch',effect=diff,ci_low=diff-half if np.isfinite(p) else np.nan,ci_high=diff+half if np.isfinite(p) else np.nan,p=p,status='estimated' if np.isfinite(p) else 'insufficient_support'))
                L=means[left]-means[right];estimable=np.linalg.norm(L-np.einsum('ik,jk,j->i',V,V,L))<1e-8
                enough=h.groupby('treatment').size().reindex(ARMS,fill_value=0).min()>=5 and len(h)-rank>=8 and leverage.max()<.95
                if estimable and enough:
                    effect=float(np.einsum('i,i->',L,beta)); se=np.sqrt(max(0.,float(np.einsum('i,ij,j->',L,covariance,L))))
                    df=len(h)-rank; half=stats.t.ppf(.975,df)*se; p=2*stats.t.sf(abs(effect/se),df)
                    contrasts.append(dict(**info,analysis='adjusted_HC3',effect=effect,ci_low=effect-half,ci_high=effect+half,p=p,status='estimated'))
                else:
                    contrasts.append(dict(**info,analysis='adjusted_HC3',effect=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,status='not_estimable_or_sparse'))
    contrast=bh(pd.DataFrame(contrasts),['analysis','outcome'])
    contrast.to_csv(out/'regional_treatment_contrasts.csv',index=False)
    # Whole-body itch is kept separate; no regional itch was measured.
    itch=[]
    for start,end in WINDOWS:
        a=c[c.visit_week.eq(start)][['patient_uid','treatment','pp_nrs']].rename(columns={'pp_nrs':'start_score'})
        b=c[c.visit_week.eq(end)][['patient_uid','pp_nrs']].rename(columns={'pp_nrs':'end_score'})
        pp=a.merge(b,on='patient_uid',validate='one_to_one').dropna(subset=['start_score','end_score'])
        for scope in SCOPES:
            h=subset(pp,scope)
            itch.append(dict(scope=scope,start=start,end=end,**summarize(h.start_score-h.end_score)))
    pd.DataFrame(itch).to_csv(out/'whole_body_PP_NRS_improvement.csv',index=False)
    manifest=dict(source=str(source.relative_to(root)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_patients=c.patient_uid.nunique(),n_reliable_patients=d.patient_uid.nunique(),n_reliable_visits=len(d),definition='Region EASI / theoretical maximum; paired percentage improvement only when starting region > 0',raw_rebuild=False,outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(os.environ.get('T2T_ROOT',Path(__file__).resolve().parents[3])))
    args=parser.parse_args();print(json.dumps(run(args.root),indent=2))
