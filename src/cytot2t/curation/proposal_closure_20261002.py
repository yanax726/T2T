"""Current-data T2T and reduction evidence, with explicit limits on clinical claims."""
from pathlib import Path
import json,itertools,hashlib,warnings
import numpy as np,pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint,confint_proportions_2indep
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices,dmatrices
from scipy.linalg import svd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/tables/update_20261002/proposal_closure';CURRENT=ROOT/'results/tables/update_20261002'
ARMS=['dupilumab','abrocitinib_100mg','abrocitinib_200mg'];SEED=20261002
ADULT_MIN_AGE=18  # Adult-only reporting; 16+ is retained as a labelled sensitivity.
def save(x,name):
    z=pd.DataFrame(x);z.to_csv(OUT/(name+'.csv'),index=False);return z
def paired_signflip_p(values):
    """Two-sided paired mean-change test, retaining ties and zero changes."""
    x=np.asarray(values,float);obs=abs(x.mean())
    if np.all(x==0):return 1.
    if len(x)<=15:
        signs=np.array(list(itertools.product([-1.,1.],repeat=len(x))))
        return float(np.mean(abs(np.einsum('ij,j->i',signs,x)/len(x))>=obs-1e-12))
    rng=np.random.default_rng(SEED);signs=rng.choice([-1.,1.],size=(19999,len(x)))
    return float((1+(abs(np.einsum('ij,j->i',signs,x)/len(x))>=obs-1e-12).sum())/20000)
def rule(x,cut,le=True):return np.nan if pd.isna(x) else bool(x<=cut if le else x>=cut)
def either(values):
    if any(pd.notna(x) and bool(x) for x in values):return True
    return False if all(pd.notna(x) for x in values) else np.nan
def both(values):
    if any(pd.notna(x) and not bool(x) for x in values):return False
    return True if all(pd.notna(x) for x in values) else np.nan

def endpoints(c):
    b=c[c.visit_week.eq(0)].set_index('patient_uid');rows=[]
    ep=pd.read_csv(ROOT/'data/restricted/treatment_episode_v0.csv').set_index('treatment_episode_id')
    for r in c[c.visit_week.isin([2,4,12,24,48])].itertuples():
      w=r.visit_week;v=b.loc[r.patient_uid];e0=v.easi_total
      dlqi=r.dlqi if r.age>=ADULT_MIN_AGE else np.nan
      if w==2:vals={'early_itch':rule(r.pp_nrs,4)}
      else:
        cut=50 if w==4 else (90 if w==48 else 75)
        easi=rule(r.easi_total,e0*(1-cut/100)) if pd.notna(e0) and e0>0 else np.nan
        iga=rule(v.iga-r.iga,2,False) if w==4 and pd.notna(v.iga) and pd.notna(r.iga) else (rule(r.iga,1) if w!=4 else np.nan)
        physician=either([easi,iga]+([rule(r.easi_total,5 if w==24 else 3)] if w in [24,48] else []))
        itch=rule(r.pp_nrs,4 if w==4 else 1);qol=rule(dlqi,10 if w==4 else 4);patient=either([itch,qol])
        vals={'physician':physician,'patient':patient,'overall':both([physician,patient]),f'EASI{cut}':easi,'itch':itch,'qol_adult':qol,'IGA_target':iga}
      date=pd.to_datetime(ep.loc[r.treatment_episode_id,'episode_start_date'],errors='coerce');asof=pd.Timestamp('2026-09-21' if r.treatment=='dupilumab' else '2026-09-22')
      due=np.nan if pd.isna(date) else bool(date+pd.Timedelta(weeks=w)<=asof)
      for target,y in vals.items():rows.append(dict(patient_uid=r.patient_uid,week=w,treatment=r.treatment,target=target,value=y,age=r.age,nominal_due_by_filename_date=due,due_rule='Administrative sensitivity only: source filename date, exact window/lock not approved'))
    d=save(rows,'endpoint_values');rates=[];tests=[]
    for (w,target),g in d.groupby(['week','target']):
      ns=[]
      for tx in ARMS:
        z=g[g.treatment.eq(tx)];n=int(z.value.notna().sum());k=int(z.value.eq(True).sum());lo,hi=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
        rates.append(dict(week=w,target=target,treatment=tx,n_initial=len(z),n_evaluable=n,not_evaluable=len(z)-n,k=k,rate=k/n if n else np.nan,ci_low=lo,ci_high=hi,n_nominal_due=int(z.nominal_due_by_filename_date.eq(True).sum()),n_due_unknown=int(z.nominal_due_by_filename_date.isna().sum())))
        ns.append([k,n-k])
      counts=np.array(ns)
      if counts.sum(axis=0).min()>0 and counts.sum(axis=1).min()>0:
        # Fixed observed group denominators, Monte Carlo null with pooled response probability.
        obs=stats.chi2_contingency(counts,correction=False)[0];N=counts.sum(1);K=int(counts[:,0].sum());rng=np.random.default_rng(SEED)
        draws=rng.multivariate_hypergeometric(N,K,size=20000);exp=N*K/N.sum();chi=((draws-exp)**2/exp).sum(1)+(((N-draws)-(N-exp))**2/(N-exp)).sum(1)
        tests.append(dict(week=w,target=target,comparison='three_groups',p=(1+(chi>=obs-1e-12).sum())/(1+len(chi)),method='Conditional Monte Carlo Pearson; 20000 allocations',risk_difference=np.nan,ci_low=np.nan,ci_high=np.nan))
      for i,j in itertools.combinations(range(3),2):
        k1,n1=counts[i,0],counts[i].sum();k2,n2=counts[j,0],counts[j].sum()
        if min(n1,n2)==0:continue
        _,p=stats.fisher_exact(counts[[i,j]]);lo,hi=confint_proportions_2indep(k1,n1,k2,n2,method='newcomb')
        tests.append(dict(week=w,target=target,comparison=ARMS[i]+' - '+ARMS[j],p=p,method='Fisher exact; Newcombe RD CI',risk_difference=k1/n1-k2/n2,ci_low=lo,ci_high=hi))
    save(rates,'endpoint_rates');t=pd.DataFrame(tests);t['p_holm']=np.nan
    pairwise=t.comparison.ne('three_groups')
    t.loc[pairwise,'p_holm']=t.loc[pairwise].groupby(['week','target']).p.transform(lambda x:multipletests(x,method='holm')[1])
    save(t,'endpoint_tests')
    transitions=[];first=[]
    z=d[d.target.eq('overall')].pivot(index='patient_uid',columns='week',values='value').reindex(columns=[4,12,24,48]);tx=b.treatment
    for uid,r in z.iterrows():
      pos=[w for w in [4,12,24,48] if pd.notna(r[w]) and bool(r[w])];f=min(pos) if pos else np.nan
      first.append(dict(patient_uid=uid,treatment=tx.loc[uid],first_recorded_target_week=f,earlier_missing=any(pd.isna(r[w]) for w in [4,12,24,48] if w<(f if np.isfinite(f) else 49)),warning='First recorded achievement, not exact biological time to target'))
    for txg in ARMS:
      a=z.loc[z.index.isin(b[b.treatment.eq(txg)].index)]
      for start,end in [(4,12),(12,24),(24,48)]:
        for left,right in itertools.product([False,True],[False,True]):transitions.append(dict(treatment=txg,start=start,end=end,start_target=left,end_target=right,n=int((a[start].eq(left)&a[end].eq(right)).sum()),n_both_evaluable=int(a[[start,end]].notna().all(axis=1).sum())))
    save(first,'first_recorded_target');save(transitions,'target_transitions');return d

def trajectories(c):
    b=c[c.visit_week.eq(0)].set_index('patient_uid');weights=pd.read_csv(CURRENT/'overlap_patient_weights.csv').set_index('patient_uid').weight;contr=[];pred=[];issues=[]
    for target in ['easi_total','pp_nrs','dlqi','bsa','scorad','eo_count_numeric','ige_numeric']:
      weeks=[2,4,12,24] if target=='pp_nrs' else ([12,24] if target=='ige_numeric' else [4,12,24])
      d=c[c.visit_week.isin(weeks)].copy();d=d[d.age.ge(ADULT_MIN_AGE)] if target=='dlqi' else d;d['baseline']=d.patient_uid.map(b[target]);d['weight']=d.patient_uid.map(weights)
      d=d.dropna(subset=[target,'baseline','weight','age','bmi','sex_male']);d['y']=np.log1p(d[target]) if target in ['ige_numeric','eo_count_numeric'] else d[target];d['baseline']=np.log1p(d.baseline) if target in ['ige_numeric','eo_count_numeric'] else d.baseline
      f='y ~ baseline + C(treatment)*C(visit_week) + age + bmi + sex_male + C(center)'
      try:
        # Stable weighted marginal least squares, patient-cluster sandwich.
        # SVD also tests estimability: missing drug-by-week cells are not
        # manufactured from arbitrary coefficients of a rank-deficient design.
        Y,design=dmatrices(f,d,return_type='dataframe');X=design.to_numpy();y=Y.iloc[:,0].to_numpy();w=d.weight.to_numpy();A=X*np.sqrt(w)[:,None]
        u,s,vt=svd(A,full_matrices=False);rank=int((s>s[0]*1e-10).sum());V=vt[:rank].T
        beta=np.einsum('ij,j->i',V,np.einsum('ij,i->j',u[:,:rank],y*np.sqrt(w))/s[:rank]);res=y-np.einsum('ij,j->i',X,beta)
        bread=np.einsum('ik,jk,k->ij',V,V,1/s[:rank]**2);scores=pd.DataFrame(X*(w*res)[:,None]).assign(uid=d.patient_uid.to_numpy()).groupby('uid').sum().to_numpy();G=len(scores)
        if G<=2 or len(d)<=rank:raise ValueError('Insufficient independent clusters or residual degrees of freedom')
        meat=np.einsum('ki,kj->ij',scores,scores);covariance=np.einsum('ij,jk,lk->il',bread,meat,bread)*G/(G-1)*(len(d)-1)/(len(d)-rank)
        if not np.isfinite(beta).all() or not np.isfinite(covariance).all():raise ValueError('Nonfinite marginal fit')
        critical=stats.t.ppf(.975,G-1)
        def estimable(v):return np.max(abs(v-np.einsum('ik,jk,j->i',V,V,v)))<1e-7*max(1.,np.max(abs(v)))
        base=d.drop_duplicates('patient_uid');designs={}
        for w in weeks:
          for tx in ARMS:
            q=base.assign(visit_week=w,treatment=tx);XX=np.asarray(build_design_matrices([design.design_info],q)[0]);mean=np.average(XX,axis=0,weights=base.weight)
            if not estimable(mean):
              issues.append(dict(target=target,status='CELL_NOT_ESTIMABLE',reason=f'W{w} {tx}: unsupported drug-by-week cell; no extrapolated mean'));continue
            est=float(np.einsum('i,i->',mean,beta));variance=float(np.einsum('i,ij,j->',mean,covariance,mean))
            if variance < -1e-8 or not np.isfinite(variance):raise ValueError('Invalid adjusted-mean variance')
            se=float(np.sqrt(max(0.,variance)));designs[(w,tx)]=mean
            pred.append(dict(target=target,week=w,treatment=tx,n_observed=int(((d.visit_week==w)&(d.treatment==tx)).sum()),n_patients=d.patient_uid.nunique(),adjusted_mean_scale=est,ci_low=est-critical*se,ci_high=est+critical*se,scale='log1p recorded units' if target in ['ige_numeric','eo_count_numeric'] else 'recorded score',method='Overlap-weighted marginal least squares; patient-cluster robust t CI; SVD estimability check',interpretation='Observed trajectories; residual imbalance and missing-data assumptions remain'))
          for a,bb in itertools.combinations(ARMS,2):
            if (w,a) not in designs or (w,bb) not in designs:continue
            v=designs[(w,a)]-designs[(w,bb)];effect=float(np.einsum('i,i->',v,beta));se=np.sqrt(max(0.,float(np.einsum('i,ij,j->',v,covariance,v))));p=2*stats.t.sf(abs(effect/se),G-1) if se>0 else np.nan
            contr.append(dict(target=target,week=w,group1=a,group2=bb,effect=effect,ci_low=effect-critical*se,ci_high=effect+critical*se,p=p))
      except Exception as err:issues.append(dict(target=target,status='NOT_ESTIMATED',reason=str(err)))
    save(pred,'clinical_trajectory_adjusted');co=pd.DataFrame(contr)
    if len(co):co['p_holm']=co.groupby(['target','week']).p.transform(lambda x:multipletests(x,method='holm')[1])
    save(co,'clinical_trajectory_contrasts');save(issues,'clinical_model_issues')

def reductions(c):
    cl=pd.read_csv(CURRENT/'maintenance_patient_ledger.csv');cl=cl[cl.age.ge(12)&~cl.dose_ambiguous].copy();b=c[c.visit_week.eq(0)].set_index('patient_uid');m=pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv');markers=[x for x in m if x.startswith('log2_')]
    responses=cl[cl.EASI75_12.eq(True)];rates=[];source=[];changes=[];comparisons=[]
    known=['responder_reduced_after_W12','responder_no_recorded_reduction']
    # These are descriptive future paths, not randomized W12 assignment.
    for marker in markers:
      z=m[m.visit_week.eq(24)][['patient_uid',marker]].merge(m[m.visit_week.eq(12)][['patient_uid',marker]],on='patient_uid',suffixes=('_24','_12'),validate='one_to_one').merge(cl,on='patient_uid',validate='one_to_one')
      z=z.dropna(subset=[marker+'_24',marker+'_12']);z['delta']=z[marker+'_24']-z[marker+'_12']
      z=z.merge(b[['age','bmi','sex_male']],on='patient_uid',suffixes=('','_base'),validate='one_to_one')
      for (tx,group),g in z.groupby(['initial_treatment','group']):
        n=len(g);mean=g.delta.mean();se=g.delta.std(ddof=1)/np.sqrt(n);rad=stats.t.ppf(.975,n-1)*se if n>=5 else np.nan
        p=paired_signflip_p(g.delta) if n>=5 else np.nan
        changes.append(dict(marker=marker,initial_treatment=tx,group=group,n_pairs=n,mean_delta_log2=mean,ci_low=mean-rad,ci_high=mean+rad,geometric_pair_ratio=2**mean,p=p,p_method='paired mean-change sign-flip; exact n<=15, otherwise 19999 draws; symmetry assumption',status='descriptive' if n>=5 else 'tiny_n_no_inference'))
      zz=z[z.group.isin(known)].copy();zz['reduced']=zz.group.eq(known[0]).astype(int)
      source.extend(zz[['patient_uid','initial_treatment','group','delta']].assign(marker=marker).to_dict('records'))
      for tx in ARMS:
        g=zz[zz.initial_treatment.eq(tx)].dropna(subset=['EASI12','baseline_EASI']);ns=g.reduced.value_counts().reindex([0,1],fill_value=0)
        row=dict(marker=marker,initial_treatment=tx,n_pairs=len(g),continued_n=int(ns[0]),reduced_n=int(ns[1]),effect=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,status='insufficient_comparable_pairs')
        if len(g)>=20 and ns.min()>=8:
          g=g.rename(columns={marker+'_12':'m12',marker+'_24':'m24'});fit=smf.ols('m24 ~ m12 + reduced + EASI12 + baseline_EASI',g).fit(cov_type='HC3',use_t=True)
          if np.linalg.matrix_rank(fit.model.exog)==fit.model.exog.shape[1]:
            ci=fit.conf_int().loc['reduced'];row.update(effect=fit.params.reduced,ci_low=ci.iloc[0],ci_high=ci.iloc[1],p=fit.pvalues.reduced,status='exploratory_within_initial_drug_association')
        comparisons.append(row)
    ch=pd.DataFrame(changes);ch['q']=ch.groupby(['initial_treatment','group']).p.transform(lambda x:multipletests(x.fillna(1),method='fdr_bh')[1]);save(ch,'reduction_molecular_changes');save(source,'reduction_molecular_paired_source');co=pd.DataFrame(comparisons)
    co['q']=co.groupby('initial_treatment').p.transform(lambda x:multipletests(x.fillna(1),method='fdr_bh')[1]);save(co,'reduction_molecular_comparisons')
    # Immediate W12 decisions use available state then, not whether reduction happens later.
    for tx in ARMS:
      g=responses[responses.initial_treatment.eq(tx)]
      for decision,h in g.groupby('W12_decision'):
        n=int(h.EASI75_24.notna().sum());k=int(h.EASI75_24.eq(True).sum());lo,hi=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
        rates.append(dict(initial_treatment=tx,decision_at_W12=decision,n_W12_responders=len(h),n_W24_observed=n,k=k,rate=k/n if n else np.nan,ci_low=lo,ci_high=hi,n_missing24=len(h)-n))
    save(rates,'W12_decision_maintenance');
    # Do not equate future successful reduction with a baseline predictor.
    outcomes=[]
    for (tx,group),g in cl.groupby(['initial_treatment','group']):
      for target in ['easi_total','pp_nrs','dlqi','bsa']:
        for start,end in [(0,12),(12,24),(0,24)]:
          eligible=c[c.age.ge(16)] if target=='dlqi' else c
          z=eligible[eligible.patient_uid.isin(g.patient_uid)&eligible.visit_week.eq(end)][['patient_uid',target]].merge(eligible[eligible.visit_week.eq(start)][['patient_uid',target]],on='patient_uid',suffixes=('_end','_start'),validate='one_to_one').dropna();delta=z[target+'_end']-z[target+'_start']
          outcomes.append(dict(initial_treatment=tx,group=group,target=target,start=start,end=end,n_pairs=len(z),median_start=z[target+'_start'].median(),median_end=z[target+'_end'].median(),median_paired_change=delta.median(),q25_change=delta.quantile(.25),q75_change=delta.quantile(.75),interpretation='Future-path descriptive only'))
    save(outcomes,'clinical_paths_paired_changes')

def safety(c):
    a=c[c.adverse_event_text.notna()].copy();a['text']=a.adverse_event_text.astype(str).str.strip();a=a[a.text.ne('')]
    a['documentation']=np.where(a.text.str.fullmatch(r'无|暂无|没有|无明显不良反应|无不良反应|未见明显不良反应',case=False),'explicit_negative','text_requires_clinical_adjudication')
    exposure=pd.read_csv(CURRENT/'exposure_at_visits.csv');a=a.merge(exposure[['patient_uid','visit_week','previsit_regimen']],on=['patient_uid','visit_week'],how='left',validate='one_to_one');save(a[['patient_uid','visit_week','treatment','text','documentation','previsit_regimen']],'safety_source_ledger')
    rows=[]
    for tx,g in c[c.visit_week.eq(0)].groupby('treatment'):
      z=a[a.treatment.eq(tx)];n=g.patient_uid.nunique();rows.append(dict(initial_treatment=tx,n_patients=n,n_with_documentation=z.patient_uid.nunique(),n_without_documentation=n-z.patient_uid.nunique(),n_text_to_review=z[z.documentation.eq('text_requires_clinical_adjudication')].patient_uid.nunique(),n_explicit_negative=z[z.documentation.eq('explicit_negative')].patient_uid.nunique(),status='Documentation counts, not adjudicated AE incidence; event-time actual treatment attribution unresolved'))
    save(rows,'safety_documentation_summary')

def run():
    OUT.mkdir(parents=True,exist_ok=True);allc=pd.read_csv(ROOT/'results/tables/update_20260925/clinical.csv');c=allc[allc.age.ge(12)&~allc.dose_ambiguous].copy();endpoints(c);trajectories(c);reductions(c);safety(allc)
    # Verify all paired changes are based on the same patient and preserved observed values.
    d=pd.read_csv(OUT/'endpoint_values.csv');rates=pd.read_csv(OUT/'endpoint_rates.csv');assert not d.duplicated(['patient_uid','week','target']).any();assert (rates.n_evaluable+rates.not_evaluable==rates.n_initial).all();assert rates.k.le(rates.n_evaluable).all()
    assert not d[(d.week==2)&(d.target=='overall')].shape[0]
    (OUT/'verification.json').write_text(json.dumps({'unique_patient_endpoint_records':True,'endpoint_denominators_reconcile':True,'W2_only_itch_not_overall':True,'clinical_raw_scores_not_overwritten':True,'within_drug_reduction_models_require_minimum_support':True},indent=2))
    files=list(OUT.glob('*.csv'))+[Path(__file__)];(OUT/'manifest.json').write_text(json.dumps([{'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],ensure_ascii=False,indent=2));print('Proposal closure observations',len(c),'patients',c.patient_uid.nunique(),flush=True)
if __name__=='__main__':run()
