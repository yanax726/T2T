"""Doctor-requested descriptive evidence, separated by treatment after W0.

These are associations and observed response contrasts, never predictive models.
Original/refinement inputs are read-only. Families are fixed before computation.
"""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/tables/clinical_alignment'
REP = ROOT / 'reports/clinical_alignment'
OLD = ROOT / 'results/tables/redevelopment'
GROUPS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
WINDOWS = [(0, 12), (0, 24), (12, 24)]
TARGETS = ['easi_total', 'iga', 'pp_nrs', 'dlqi', 'bsa']
REGIONS = {'head_neck':7.2, 'upper_limb':14.4, 'trunk':21.6, 'lower_limb':28.8}
SEED = 2026091307


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bh(d, family='family'):
    d['q'] = np.nan
    for _, idx in d.groupby(family).groups.items():
        good = d.loc[idx, 'p'].notna()
        available = d.loc[idx].index[good]
        if len(available):
            d.loc[available,'q'] = multipletests(d.loc[available,'p'], method='fdr_bh')[1]
    return d


def correlation(x, y, rng, minimum=8):
    x, y = np.asarray(x,float), np.asarray(y,float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    r = dict(n=n, rho=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan,
             status='insufficient_support' if n < minimum else 'constant_input')
    if n < minimum or np.ptp(x)==0 or np.ptp(y)==0:
        return r
    rho, p = stats.spearmanr(x,y)
    idx = rng.integers(0,n,(1000,n))
    a = stats.rankdata(x[idx],axis=1); b = stats.rankdata(y[idx],axis=1)
    a -= a.mean(axis=1)[:,None]; b -= b.mean(axis=1)[:,None]
    den = np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    vals = np.divide((a*b).sum(axis=1),den,out=np.full(1000,np.nan),where=den>0)
    lo, hi = np.nanquantile(vals,[.025,.975])
    return dict(n=n,rho=float(rho),p=float(p),ci_low=lo,ci_high=hi,status='ESTIMATED_ASSOCIATION')


def response_difference(x_response, x_nonresponse, rng):
    a, b = np.asarray(x_response,float), np.asarray(x_nonresponse,float)
    rec = dict(n_response=len(a), n_nonresponse=len(b), mean_response=np.mean(a) if len(a) else np.nan,
               mean_nonresponse=np.mean(b) if len(b) else np.nan, effect=np.nan,
               ci_low=np.nan,ci_high=np.nan,p=np.nan,status='insufficient_response_support')
    if len(a): rec['effect'] = rec['mean_response']-rec['mean_nonresponse']
    if min(len(a),len(b)) < 3:
        return rec
    idxa=rng.integers(0,len(a),(1000,len(a))); idxb=rng.integers(0,len(b),(1000,len(b)))
    delta=a[idxa].mean(axis=1)-b[idxb].mean(axis=1)
    rec.update(ci_low=np.quantile(delta,.025),ci_high=np.quantile(delta,.975),
               p=stats.mannwhitneyu(a,b,alternative='two-sided',method='auto').pvalue,
               status='ESTIMATED_DESCRIPTIVE')
    return rec


def run(all_markers=False, dlqi_min_age=17):
    OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
    inputs = [OLD/'clinical.csv', OLD/'molecular.csv', ROOT/'config/marker_registry.csv']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    protocol = dict(version='20260913_doctor_stratified_v1', minimum_correlation_n=8,
        treatment_rule='W0 may pool; every post-treatment association is separated by initial treatment; ambiguous initial dose excluded from stratified analyses.',
        population=f'Age >=12; recorded DLQI >={dlqi_min_age}; regional rows require reliable sums. Original values are never imputed.',
        families={'changes_all':855,'regional_baseline':76,'regional_post_treatment':456,
                  'regional_interaction':152,'response_contrasts':342},
        details='BH separately within each listed family over estimable P. Missing tests retain NA. All P and Q retained; CIs are nominal patient bootstrap. Response differences are mean log2 changes, P from two-sided Mann–Whitney, not inverse-CI tests. IGA delta is descriptive recorded-category change, not an interval-scale replacement. No p-based panel selection.',
        response_rules='EASI75/90 relative to W0, only W0→12/W0→24; IGA01 is recorded follow-up category <=1. EASI depth uses mutually exclusive <50,[50,75),[75,90),>=90%.',
        source_hashes=hashes)
    if all_markers:
        protocol['version']='20260920_all23_extension'
        protocol['families']={k:int(v*23/19) for k,v in protocol['families'].items()}
    (REP/'doctor_descriptive_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    c=pd.read_csv(inputs[0]); m=pd.read_csv(inputs[1]); registry=pd.read_csv(inputs[2])
    markers=registry.analyte.tolist() if all_markers else registry.loc[registry.discovery_eligible,'analyte'].tolist()
    count=len(markers)
    assert not c.duplicated(['patient_uid','visit_week']).any()
    assert not m.duplicated(['patient_uid','visit_week']).any()
    d=c.merge(m,on=['patient_uid','visit_week'],validate='one_to_one')
    d=d[d.age.ge(12)&d.visit_week.isin([0,12,24])].copy()
    assert d.dose_ambiguous.dtype == bool
    work=d[~d.dose_ambiguous].copy()
    rng=np.random.default_rng(SEED)
    changes=[]; source=[]; depth=[]; responses=[]; coverage=[]
    for a,b in WINDOWS:
        pair=work[work.visit_week.eq(a)].merge(work[work.visit_week.eq(b)],on='patient_uid',suffixes=('_start','_end'),validate='one_to_one')
        assert pair.treatment_start.equals(pair.treatment_end)
        for tx in GROUPS:
            group=pair[pair.treatment_start.eq(tx)]
            coverage.append(dict(start=a,end=b,treatment=tx,n_molecular_pairs=len(group)))
            for marker in markers:
                mc=['log2_'+marker+'_start','log2_'+marker+'_end']
                for target in TARGETS:
                    cols=mc+[target+'_start',target+'_end']
                    g=group.dropna(subset=cols).copy()
                    if target=='dlqi': g=g[g.age_start.ge(dlqi_min_age)]
                    x=g[mc[1]]-g[mc[0]]; y=g[target+'_end']-g[target+'_start']
                    changes.append(dict(marker=marker,outcome=target,treatment=tx,start=a,end=b,family='changes_all',**correlation(x,y,rng)))
                    ss=pd.DataFrame(dict(patient_uid=g.patient_uid,treatment=tx,marker=marker,outcome=target,start=a,end=b,
                                         delta_marker=x,delta_clinical=y));source.append(ss)
                for target in ['easi_total','iga']:
                    g=group.dropna(subset=mc+[target+'_start',target+'_end']).copy()
                    if target=='easi_total': g=g[g.easi_total_start.gt(0)]
                    if target=='easi_total':
                        imp=100*(g.easi_total_start-g.easi_total_end)/g.easi_total_start
                        bands=pd.cut(imp,[-np.inf,50,75,90,np.inf],right=False)
                    else:
                        imp=g.iga_start-g.iga_end
                        bands=pd.cut(imp,[-np.inf,0,1,2,np.inf],right=False)
                    g['band']=bands; g['delta_marker']=g[mc[1]]-g[mc[0]]
                    for band,h in g.groupby('band',observed=False):
                        n=len(h); mean=h.delta_marker.mean();se=h.delta_marker.std()/np.sqrt(n) if n else np.nan
                        ci=stats.t.ppf(.975,n-1)*se if n>1 else np.nan
                        depth.append(dict(marker=marker,outcome=target,treatment=tx,start=a,end=b,band=str(band),n=n,mean_change=mean,ci_low=mean-ci,ci_high=mean+ci,small_cell=n<5))
                if a==0:
                    for endpoint,target in [('EASI75','easi_total'),('EASI90','easi_total'),('IGA01','iga')]:
                        g=group.dropna(subset=mc+[target+'_start',target+'_end']).copy()
                        if target=='easi_total':
                            g=g[g.easi_total_start.gt(0)]
                            resp=100*(g.easi_total_start-g.easi_total_end)/g.easi_total_start >= int(endpoint[4:])
                        else: resp=g.iga_end.le(1)
                        x=g[mc[1]]-g[mc[0]]
                        responses.append(dict(marker=marker,endpoint=endpoint,treatment=tx,start=a,end=b,n=len(g),family='response_contrasts',**response_difference(x[resp],x[~resp],rng)))
    change=bh(pd.DataFrame(changes)); response=bh(pd.DataFrame(responses))
    assert len(change)==count*45 and len(response)==count*18 and len(depth)==count*72
    change.to_csv(OUT/'doctor_stratified_change_associations.csv',index=False)
    pd.concat(source).to_csv(OUT/'doctor_stratified_change_source.csv',index=False)
    pd.DataFrame(depth).to_csv(OUT/'doctor_stratified_response_depth.csv',index=False)
    response.to_csv(OUT/'doctor_stratified_response_contrasts.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'doctor_stratified_pair_coverage.csv',index=False)

    regional=[]; interactions=[]; slopes=[]
    base=d[d.visit_week.eq(0)&d.regional_reliable]
    post=work[work.visit_week.isin([12,24])&work.regional_reliable]
    strata=[('baseline_pooled',0,base)]+[(tx,week,post[post.treatment.eq(tx)&post.visit_week.eq(week)]) for tx in GROUPS for week in [12,24]]
    for tx,week,g in strata:
        for marker in markers:
            for region in REGIONS:
                regional.append(dict(marker=marker,region=region,treatment=tx,visit_week=week,family='regional_baseline' if week==0 else 'regional_post_treatment',**correlation(g['log2_'+marker],g['easi_'+region],rng)))
    interaction_strata=[('baseline_pooled',base)]+[(tx,post[post.treatment.eq(tx)]) for tx in GROUPS]
    for tx,g in interaction_strata:
        long=[]
        for region,mx in REGIONS.items():
            h=g.copy();h['region']=region;h['regional_burden']=h['easi_'+region]/mx;h['rest_burden']=(h.easi_total-h['easi_'+region])/(72-mx);long.append(h)
        ll=pd.concat(long,ignore_index=True)
        for marker in markers:
            for adjustment in ['none','rest_burden']:
                gg=ll.rename(columns={'log2_'+marker:'marker'}).dropna(subset=['marker','regional_burden','age','sex_male','bmi','rest_burden'])
                rec=dict(marker=marker,treatment=tx,adjustment=adjustment,n_patients=gg.patient_uid.nunique(),n_rows=len(gg),family='regional_interaction',p=np.nan,status='insufficient_support')
                if rec['n_patients']>=20 and gg.marker.nunique()>1:
                    formula='regional_burden ~ marker*C(region)+C(visit_week)+C(center)+age+sex_male+bmi'+(' + rest_burden' if adjustment=='rest_burden' else '')
                    try:
                        with warnings.catch_warnings(record=True) as warns:
                            warnings.simplefilter('always')
                            f=smf.ols(formula,gg).fit(cov_type='cluster',cov_kwds={'groups':gg.patient_uid},use_t=True)
                            terms=[x for x in f.params.index if 'marker:C(region)' in x]
                            R=np.zeros((len(terms),len(f.params)))
                            for j,name in enumerate(terms):R[j,list(f.params.index).index(name)]=1
                            sub=R@np.asarray(f.cov_params())@R.T
                            if np.linalg.matrix_rank(sub)!=len(terms) or not np.isfinite(sub).all():raise ValueError('invalid interaction covariance')
                            rec.update(p=float(f.wald_test(R,scalar=True).pvalue),status='ESTIMATED_INTERACTION',formula=formula,warning=';'.join(str(w.message) for w in warns),
                                       n_parameters=len(f.params),design_rank=int(np.linalg.matrix_rank(f.model.exog)),max_row_leverage=float(f.get_influence().hat_matrix_diag.max()))
                            for region in REGIONS:
                                v=np.zeros(len(f.params));v[list(f.params.index).index('marker')]=1
                                term=f'marker:C(region)[T.{region}]'
                                if term in f.params.index:v[list(f.params.index).index(term)]=1
                                effect=float(v@f.params);variance=float(v@np.asarray(f.cov_params())@v)
                                se=np.sqrt(max(variance,0));crit=stats.t.ppf(.975,rec['n_patients']-1)
                                slopes.append(dict(marker=marker,treatment=tx,adjustment=adjustment,region=region,n_patients=rec['n_patients'],
                                                   effect=effect,se=se,ci_low=effect-crit*se,ci_high=effect+crit*se,
                                                   unit='regional contribution divided by theoretical maximum per 1 log2 serum concentration',
                                                   CI='patient-cluster covariance; t with clusters minus 1 df; exploratory marginal slope'))
                    except Exception as e:rec.update(status='FAILED_MODEL',failure=str(e))
                interactions.append(rec)
    regional=bh(pd.DataFrame(regional));interactions=bh(pd.DataFrame(interactions))
    assert len(regional)==count*28 and len(interactions)==count*8
    regional.to_csv(OUT/'doctor_stratified_regional_correlations.csv',index=False)
    interactions.to_csv(OUT/'doctor_stratified_regional_interactions.csv',index=False)
    pd.DataFrame(slopes).to_csv(OUT/'doctor_stratified_regional_slopes.csv',index=False)
    for path, old_hash in hashes.items():assert sha(ROOT/path)==old_hash
    verdict=dict(changes=dict(rows=len(change),estimated=int(change.p.notna().sum()),q_lt_05=int(change.q.lt(.05).sum())),
                 response=dict(rows=len(response),estimated=int(response.p.notna().sum()),q_lt_05=int(response.q.lt(.05).sum())),
                 regional=dict(rows=len(regional),estimated=int(regional.p.notna().sum()),q_lt_05=int(regional.q.lt(.05).sum())),
                 regional_interactions=dict(rows=len(interactions),estimated=int(interactions.p.notna().sum()),q_lt_05=int(interactions.q.lt(.05).sum())),
                 original_inputs_unchanged=True,unit='patient, independently paired within each treatment and interval',clinical_validation=False)
    (REP/'doctor_descriptive_verification.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps(verdict))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run()
