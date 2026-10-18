"""Baseline overlap diagnostics and W12/W24 response associations, initial strategies."""
from pathlib import Path
import itertools
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/tables/update_20261002';SEED=20261002
ARMS=['dupilumab','abrocitinib_100mg','abrocitinib_200mg']

def weights(b):
    cols=['easi_total','bmi','age','sex_male'];X=b[cols].astype(float).copy();X['HUASHAN']=b.center.eq('HUASHAN').astype(float)
    fit=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)).fit(X,b.treatment)
    p=fit.predict_proba(X);classes=list(fit[-1].classes_);own=np.array([classes.index(x) for x in b.treatment]);pown=p[np.arange(len(b)),own]
    h=1/np.sum(1/p,axis=1);w=h/pown
    return w,pown,X

def balance(b,w,X,label):
    out=[]
    for tx1,tx2 in itertools.combinations(ARMS,2):
      for v in X:
        one=b.treatment.eq(tx1).to_numpy();two=b.treatment.eq(tx2).to_numpy();x=X[v].to_numpy()
        m1=np.average(x[one],weights=w[one]);m2=np.average(x[two],weights=w[two]);sd=np.sqrt((x[one].var(ddof=1)+x[two].var(ddof=1))/2)
        out.append(dict(analysis=label,group1=tx1,group2=tx2,variable=v,SMD=(m1-m2)/sd if sd else np.nan))
    return out

def run():
    c=pd.read_csv(ROOT/'results/tables/update_20260925/clinical.csv');b=c[c.visit_week.eq(0)&c.age.ge(12)&~c.dose_ambiguous].copy().reset_index(drop=True)
    w,p,X=weights(b);b['weight']=w;b['prob_assigned']=p;b.to_csv(OUT/'overlap_patient_weights.csv',index=False)
    rows=balance(b,np.ones(len(b)),X,'unweighted')+balance(b,w,X,'overlap_weighted')
    pd.DataFrame(rows).to_csv(OUT/'overlap_balance.csv',index=False)
    ess=[]
    for (tx,cx),g in b.groupby(['treatment','center']):ess.append(dict(treatment=tx,center=cx,n=len(g),ESS=g.weight.sum()**2/(g.weight**2).sum(),min_propensity=g.prob_assigned.min(),median_propensity=g.prob_assigned.median()))
    pd.DataFrame(ess).to_csv(OUT/'overlap_support.csv',index=False)
    endpoints=pd.read_csv(OUT/'endpoint_patient_values.csv');out=[];comparison=[];rng=np.random.default_rng(SEED)
    for week in [12,24]:
     for target in ['EASI75','EASI90','LDA_EASI_branch','vLDA_EASI_branch']:
      y=endpoints[endpoints.week.eq(week)&endpoints.target.eq(target)].set_index('patient_uid').value
      a=b.copy();a['y']=a.patient_uid.map(y).astype(float)
      means={};boots={tx:[] for tx in ARMS}
      for tx in ARMS:
       g=a[a.treatment.eq(tx)];z=g[g.y.notna()];means[tx]=np.average(z.y,weights=z.weight)
       num=(z.y*z.weight).sum();total=g.weight.sum();missing=g.loc[g.y.isna(),'weight'].sum()
       out.append(dict(week=week,target=target,treatment=tx,n_group=len(g),n_observed=len(z),observed_weighted_rate=means[tx],ESS=z.weight.sum()**2/(z.weight**2).sum(),missing_weight_fraction=missing/total,lower_bound=num/total,upper_bound=(num+missing)/total))
      for rep in range(300):
       z=a.iloc[rng.integers(0,len(a),len(a))].copy().reset_index(drop=True)
       if z.treatment.nunique()<3:continue
       ww,_,_=weights(z);z['weight']=ww
       for tx in ARMS:
        g=z[z.treatment.eq(tx)&z.y.notna()];boots[tx].append(np.average(g.y,weights=g.weight) if len(g) else np.nan)
      for tx1,tx2 in itertools.combinations(ARMS,2):
       delta=np.array(boots[tx1])-np.array(boots[tx2]);delta=delta[np.isfinite(delta)];estimate=means[tx1]-means[tx2];se=delta.std(ddof=1)
       comparison.append(dict(week=week,target=target,group1=tx1,group2=tx2,risk_difference=estimate,ci_low=np.quantile(delta,.025),ci_high=np.quantile(delta,.975),p_normal_bootstrapSE=2*norm.sf(abs(estimate/se)) if se>0 else np.nan,n_bootstrap=len(delta),estimand='observed outcomes in baseline overlap population; initial strategies; exploratory association'))
      print(week,target,flush=True)
    comp=pd.DataFrame(comparison);comp['p_holm']=comp.groupby(['week','target']).p_normal_bootstrapSE.transform(lambda x:multipletests(x,method='holm')[1]);comp.to_csv(OUT/'overlap_response_contrasts.csv',index=False)
    pd.DataFrame(out).to_csv(OUT/'overlap_response_rates.csv',index=False)
    # Prespecified fallback for incomplete three-arm balance: pairwise overlap
    # targets a different comparable population for each pair, without discarding cases.
    pp=[];bal=[];observed_balance=[];pair_weights=[]
    for tx1,tx2 in itertools.combinations(ARMS,2):
      a=b[b.treatment.isin([tx1,tx2])].copy().reset_index(drop=True)
      def pairw(z):
        X=z[['easi_total','bmi','age','sex_male']].astype(float).copy();X['HUASHAN']=z.center.eq('HUASHAN').astype(float)
        y=z.treatment.eq(tx1).astype(int)
        fit=make_pipeline(StandardScaler(),LogisticRegression(C=1000,max_iter=2000)).fit(X,y)
        prob=fit.predict_proba(X)[:,1];return np.where(y==1,1-prob,prob),X
      wa,X=pairw(a);a['pw']=wa
      pair_weights.append(a[['patient_uid','treatment','pw']].assign(group1=tx1,group2=tx2))
      for v in X:
        x=a.treatment.eq(tx1);z=a.treatment.eq(tx2);sd=np.sqrt((X.loc[x,v].var()+X.loc[z,v].var())/2)
        bal.append(dict(group1=tx1,group2=tx2,variable=v,SMD=(np.average(X.loc[x,v],weights=wa[x])-np.average(X.loc[z,v],weights=wa[z]))/sd if sd else np.nan))
      for week in [12,24]:
        y=endpoints[endpoints.week.eq(week)&endpoints.target.eq('EASI75')].set_index('patient_uid').value;a['y']=a.patient_uid.map(y).astype(float)
        for v in X:
          x=a.treatment.eq(tx1)&a.y.notna();z=a.treatment.eq(tx2)&a.y.notna();sd=np.sqrt((X.loc[x,v].var()+X.loc[z,v].var())/2)
          observed_balance.append(dict(week=week,group1=tx1,group2=tx2,variable=v,SMD=(np.average(X.loc[x,v],weights=wa[x])-np.average(X.loc[z,v],weights=wa[z]))/sd if sd else np.nan))
        def rate(z,tx):
          g=z[z.treatment.eq(tx)&z.y.notna()];return np.average(g.y,weights=g.pw) if len(g) else np.nan
        dif=rate(a,tx1)-rate(a,tx2);boots=[]
        for rep in range(300):
          z=a.iloc[rng.integers(0,len(a),len(a))].copy().reset_index(drop=True)
          if z.treatment.nunique()<2:continue
          z['pw'],_=pairw(z);boots.append(rate(z,tx1)-rate(z,tx2))
        lo,hi=np.nanquantile(boots,[.025,.975])
        pp.append(dict(week=week,group1=tx1,group2=tx2,n=len(a),rate1=rate(a,tx1),rate2=rate(a,tx2),risk_difference=dif,ci_low=lo,ci_high=hi,interpretation='pair-specific baseline-overlap observed EASI75; associations, not fixed-dose effects'))
    pd.DataFrame(pp).to_csv(OUT/'pairwise_overlap_easi75.csv',index=False);pd.DataFrame(bal).to_csv(OUT/'pairwise_overlap_balance.csv',index=False)
    pd.DataFrame(observed_balance).to_csv(OUT/'pairwise_overlap_observed_balance.csv',index=False)
    pd.concat(pair_weights).to_csv(OUT/'pairwise_overlap_weights.csv',index=False)

if __name__=='__main__':run()
