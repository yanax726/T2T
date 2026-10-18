"""Source-local exposure parsing and study-specific molecular/clinical analyses."""
import re,json,itertools,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/tables/update_20261002'
OLD=ROOT/'results/tables/update_20260925';ARMS=['dupilumab','abrocitinib_100mg','abrocitinib_200mg'];SEED=20261002

def save(x,name):
    d=pd.DataFrame(x);d.to_csv(OUT/(name+'.csv'),index=False,quoting=1);return d

def interval_days(text,drug):
    f=re.search(r'q(?:(\d+(?:\.\d+)?)(w|h|d)|(od|d))',text)
    if f:
        if f[3]:return (2. if f[3]=='od' else 1.),'explicit'
        return float(f[1])*{'w':7,'h':1/24,'d':1}[f[2]],'explicit'
    if drug!='dupilumab':return 1.,'daily_assumed_from_regimen_context'
    return np.nan,'interval_not_stated'

def parse(source):
    s=re.sub(r'\s+','',str(source.regimen).lower()).replace('周','w').replace('；',';').replace('，',';').replace('至今','后')
    segments=[]
    # Clauses retain local ambiguity; a change after week 48 does not invalidate W0-W24.
    for clause in s.split(';'):
        pat=r'(\d+(?:\.\d+)?)w?[-–](\d+(?:\.\d+)?)w(?:及以后|后)?(\d+)mg'
        found=[]
        for q in re.finditer(pat,clause):
            end=np.inf if '及以后' in q[0] else float(q[2])
            found.append((q.start(),q.end(),float(q[1]),end,float(q[3])))
        for q in re.finditer(r'(?<![\d-])(\d+)w(?:后|开始|以后)?(\d+)mg',clause):
            if any(i<=q.start()<j for i,j,_,_,_ in found):continue
            found.append((q.start(),q.end(),float(q[1]),np.inf,float(q[2])))
        found.sort()
        for i,(startpos,endpos,st,en,dose) in enumerate(found):
            tail=clause[endpos:found[i+1][0] if i+1<len(found) else len(clause)]
            days,evidence=interval_days(tail,source.treatment)
            uncertain=bool(re.search('基本|左右|交替|尝试|反复|痒时|按需',clause))
            # A dated later comment cannot alter earlier unambiguous scheduled exposure.
            later=re.search(r'(\d+)w(?:时)?尝试',tail)
            if later and float(later[1])>24:uncertain=False
            if 'qod/q' in tail or re.search(r'mg[/\\]\d+mg',clause):uncertain=True
            segments.append(dict(patient_uid=source.patient_uid,start_week=st,end_week=en,dose_mg=dose,interval_days=days,intensity=dose/days if np.isfinite(days) else np.nan,ambiguous=uncertain,frequency_evidence=evidence,source_clause=clause))
    return sorted(segments,key=lambda z:z['start_week'])

def source_week(text):
    q=re.search(r'(\d+(?:\.\d+)?)\s*(?:w|周)',str(text).lower())
    return float(q[1]) if q else np.nan

def run_exposure(c):
    # Legacy export contains a literal CR in free text, not a new patient row.
    src=pd.read_csv(OLD/'exposure_source.csv',lineterminator='\n').fillna('');base=c[c.visit_week.eq(0)].set_index('patient_uid')
    assert len(src)==170 and src.patient_uid.isin(base.index).all()
    src.to_csv(OUT/'exposure_source_repaired.csv',index=False,quoting=1)
    allseg=[];states=[];events=[];issues=[];classifications=[]
    for r in src.itertuples():
        uid=r.patient_uid;seg=parse(r);allseg+=seg;patient=c[c.patient_uid.eq(uid)].set_index('visit_week')
        sw=source_week(r.stop_time);rw=source_week(r.restart) if re.match(r'^\d+\s*(?:w|周)',r.restart) else np.nan
        rel=re.search(r'停药(\d+)\s*(?:w|周)',r.restart)
        if rel and np.isfinite(sw):rw=sw+float(rel[1])
        # Undated/complex attempts in an additional clause invalidate the relevant
        # exposure window even when the nominal regimen itself is explicit.
        uncertain_from=np.inf
        for clause in re.split('[;；，]',r.regimen.lower().replace('周','w')):
            if re.search('尝试|反复|左右|基本|痒时|交替',clause):
                dates=[float(v) for v in re.findall(r'(\d+)\s*w',clause)]
                if dates:uncertain_from=min(uncertain_from,min(dates))
                elif not any(clause.strip() in z['source_clause'] for z in seg):
                    uncertain_from=min(uncertain_from,max([z['start_week'] for z in seg],default=0))
        initial=[z for z in seg if z['start_week']==0]
        expected=100 if r.treatment=='abrocitinib_100mg' else 200
        conflict=bool(initial and r.treatment!='dupilumab' and initial[0]['dose_mg']!=expected)
        if conflict:issues.append(dict(patient_uid=uid,issue='initial dose conflicts with source',source=r.regimen))
        if r.stop=='是' and not np.isfinite(sw):issues.append(dict(patient_uid=uid,issue='stop timing unknown',source=r.stop_time))
        def state(t):
            if t>=uncertain_from:return 'unknown',np.nan
            if r.stop=='是' and not np.isfinite(sw):return 'unknown_stop_timing',np.nan
            if np.isfinite(sw) and sw<t and (not np.isfinite(rw) or t<=rw):
                # Zero only denotes original-drug stopped; switched-drug dose never encoded as zero.
                return 'stopped_or_switched',np.nan
            hit=[z for z in seg if z['start_week']<=t<z['end_week']]
            if len(hit)!=1 or hit[0]['ambiguous'] or not np.isfinite(hit[0]['intensity']):return 'unknown',np.nan
            z=hit[0]
            return f"{z['dose_mg']:g}mg/q{z['interval_days']:g}days",z['intensity']
        for w in [0,2,4,12,16,20,24]:
            s,intensity=state(0 if w==0 else w-1e-6)
            after,ai=state(w+1e-6)
            states.append(dict(patient_uid=uid,center=base.loc[uid,'center'],initial_treatment=r.treatment,visit_week=w,previsit_regimen=s,previsit_intensity=intensity,postvisit_regimen=after,postvisit_intensity=ai,initial_conflict=conflict,boundary_assumption='same-week change after assessment; ordering unconfirmed' if s!=after else 'none',easi=patient.loc[w,'easi_total'] if w in patient.index else np.nan))
        pe=[]
        for z in seg:
            t=z['start_week']
            if t<=0 or t>24:continue
            before,bi=state(t-1e-6);after,ai=state(t+1e-6)
            if not np.isfinite(bi) or not np.isfinite(ai) or ai==bi:continue
            event=dict(patient_uid=uid,week=t,event='reduction' if ai<bi else 'escalation',before=before,after=after,source=z['source_clause']);pe.append(event);events.append(event)
        for event,t in [('stop_or_switch',sw),('restart',rw)]:
            if np.isfinite(t) and 0<t<=24:events.append(dict(patient_uid=uid,week=t,event=event,before='',after='',source=r.stop_time if event=='stop_or_switch' else r.restart))
        b0=patient.loc[0,'easi_total'];v12=patient.loc[12,'easi_total'];v24=patient.loc[24,'easi_total']
        resp12=np.nan if pd.isna(v12) or pd.isna(b0) or b0<=0 else bool(v12<=.25*b0)
        resp24=np.nan if pd.isna(v24) or pd.isna(b0) or b0<=0 else bool(v24<=.25*b0)
        reductions=[z for z in pe if z['event']=='reduction' and 12<=z['week']<24]
        first=min([z['week'] for z in reductions],default=np.nan)
        reversal=any(z['event']=='escalation' and z['week']>first and z['week']<24 for z in pe) if np.isfinite(first) else False
        # Only observations after the first reduction can be reduction follow-up.
        mid=patient.loc[patient.index.isin([16,20]) & (patient.index>=first),'easi_total'].dropna() if np.isfinite(first) else pd.Series(dtype=float)
        loss_mid=bool((mid>.25*b0).any()) if len(mid) else False
        coverage=all(state(w)[0] not in ['unknown','unknown_stop_timing','stopped_or_switched'] for w in np.arange(12.01,24,.1))
        if pd.isna(resp12):group='W12_missing'
        elif not resp12:group='W12_not_EASI75'
        elif conflict:group='responder_exposure_conflict'
        elif np.isfinite(first):group='responder_reduced_after_W12'
        elif np.isfinite(sw) and 12<=sw<24:group='responder_stopped_or_switched'
        elif coverage:group='responder_no_recorded_reduction'
        else:group='responder_exposure_unknown'
        outcome='W24_missing' if pd.isna(resp24) else ('W24_EASI75' if resp24 else 'W24_lost_EASI75')
        # Groups describe paths. Later reduction is never recoded as a W12 treatment decision.
        before,bi=state(12-1e-6);after,ai=state(12+1e-6)
        decision='unknown' if conflict or not np.isfinite(ai) or not np.isfinite(bi) else ('reduce_at12' if ai<bi else ('increase_at12' if ai>bi else 'continue_at12'))
        classifications.append(dict(patient_uid=uid,center=base.loc[uid,'center'],initial_treatment=r.treatment,age=base.loc[uid,'age'],dose_ambiguous=base.loc[uid,'dose_ambiguous'],baseline_EASI=b0,EASI12=v12,EASI24=v24,EASI75_12=resp12,EASI75_24=resp24,group=group,outcome=outcome,W12_decision=decision,first_reduction_week=first,reduction_events_12_to24=len(reductions),intensity_changes_0_to24=len(pe),reescalated_by24=reversal,recorded_interim_target_loss=loss_mid,n_interim_easi=len(mid),source_regimen=r.regimen,source_stop=r.stop_time,source_restart=r.restart,stop_week=sw,initial_conflict=conflict,continuous_control='not established from intermittent visits'))
    st=save(states,'exposure_at_visits');save(allseg,'exposure_segments');save(events,'exposure_events');save(issues,'exposure_issues');cl=save(classifications,'maintenance_patient_ledger')
    old=pd.read_csv(OLD/'exposure_at_visits.csv');change=st.merge(old[['patient_uid','visit_week','previsit_regimen']],on=['patient_uid','visit_week'],suffixes=('_new','_old'))
    change=change[change.previsit_regimen_old.str.contains('unknown')&~change.previsit_regimen_new.str.contains('unknown')]
    save(change,'recovered_exposure_records')
    eligible=cl[cl.age.ge(12)&~cl.dose_ambiguous];rates=[]
    for kind in ['group','W12_decision']:
      for (tx,grp),g in eligible.groupby(['initial_treatment',kind]):
        if kind=='W12_decision':g=g[g.EASI75_12.eq(True)]
        n=g.EASI75_24.notna().sum();k=int(g.EASI75_24.eq(True).sum());lo,hi=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
        rates.append(dict(classification=kind,initial_treatment=tx,group=grp,n_group=len(g),n_W24_observed=n,n_W24_missing=len(g)-n,n_EASI75=k,rate=k/n if n else np.nan,ci_low=lo,ci_high=hi,best_case_bound=(k+len(g)-n)/len(g) if len(g) else np.nan,worst_case_bound=k/len(g) if len(g) else np.nan,postbaseline_grouping=kind=='group'))
    save(rates,'maintenance_rates')
    save(eligible.groupby(['initial_treatment','group','outcome']).size().rename('n').reset_index(),'maintenance_transitions')
    red=eligible[eligible.group.eq('responder_reduced_after_W12')].copy()
    red['weeks_after_reduction']=24-red.first_reduction_week
    def maintenance_category(r):
        if pd.notna(r.stop_week) and r.first_reduction_week<=r.stop_week<24:return '减药后停药或换药，需核对原因'
        if r.reescalated_by24:return '减药后有再次加量记录'
        if r.recorded_interim_target_loss or r.outcome=='W24_lost_EASI75':return '记录访视失去EASI75'
        if r.outcome=='W24_missing':return 'W24缺失，不能判断维持'
        if r.weeks_after_reduction<8:return '减药后观察不足8周'
        return 'W24仍达标且无已记录失标或加量；连续维持待核'
    red['maintenance_category']=red.apply(maintenance_category,axis=1)
    save(red,'reduction_maintenance_classification')
    return cl

def clinical(c):
    c=c[c.age.ge(12)&~c.dose_ambiguous].copy();b=c[c.visit_week.eq(0)].set_index('patient_uid');rows=[]
    for w in [0,4,12,24]:
     g=c[c.visit_week.eq(w)]
     for r in g.itertuples():
      easi=r.easi_total;itch=r.pp_nrs;e0=b.loc[r.patient_uid,'easi_total']
      vals={'LDA_EASI_branch':np.nan if pd.isna(easi) or pd.isna(itch) else easi<=7 and itch<=4,
       'vLDA_EASI_branch':np.nan if pd.isna(easi) or pd.isna(itch) else easi<=3 and itch<=1,
       'point_clearance_not_ODCC':np.nan if pd.isna(easi) or pd.isna(itch) else easi==0 and itch<=1}
      if w>0:vals.update({f'EASI{cut}':np.nan if pd.isna(easi) or e0<=0 else easi<=e0*(1-cut/100) for cut in [50,75,90]})
      for target,y in vals.items():rows.append(dict(patient_uid=r.patient_uid,week=w,treatment=r.treatment,center=r.center,target=target,value=y))
    d=save(rows,'endpoint_patient_values');rates=[]
    for (w,tx,target),g in d.groupby(['week','treatment','target']):
     n=int(g.value.notna().sum());k=int(g.value.eq(True).sum());lo,hi=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
     rates.append(dict(week=w,treatment=tx,target=target,n_total=len(g),n_observed=n,n_missing=len(g)-n,k=k,rate=k/n if n else np.nan,ci_low=lo,ci_high=hi,worst_case_bound=k/len(g),best_case_bound=(k+len(g)-n)/len(g)))
    save(rates,'endpoint_rates')
    # Compare clinical outcomes with measured baseline confounding adjusted.
    fits=[];balance=[]
    for a,bb in itertools.combinations(ARMS,2):
     base=b[b.treatment.isin([a,bb])]
     for var in ['easi_total','pp_nrs','bmi','age','sex_male']:
      x=base[base.treatment.eq(a)][var].dropna();z=base[base.treatment.eq(bb)][var].dropna();sd=np.sqrt((x.var()+z.var())/2)
      balance.append(dict(group1=a,group2=bb,variable=var,n1=len(x),n2=len(z),mean1=x.mean(),mean2=z.mean(),SMD=(x.mean()-z.mean())/sd if sd>0 else np.nan))
    save(balance,'baseline_balance')
    for w in [12,24]:
     for target in ['easi_total','pp_nrs']:
      a=c[c.visit_week.eq(w)].merge(b[[target,'easi_total','bmi','age','sex_male']].loc[:,lambda x:~x.columns.duplicated()],on='patient_uid',suffixes=('','_base'))
      a=a.dropna(subset=[target,target+'_base','easi_total_base','bmi','age','sex_male','center'])
      f=f'{target} ~ C(treatment) + {target}_base + bmi + age + sex_male + C(center)'+(' + easi_total_base' if target!='easi_total' else '')
      fit=smf.ols(f,a).fit(cov_type='HC3',use_t=True)
      for tx1,tx2 in itertools.combinations(ARMS,2):
       from patsy import build_design_matrices
       d1=a.assign(treatment=tx1);d2=a.assign(treatment=tx2)
       x1=np.asarray(build_design_matrices([fit.model.data.design_info],d1)[0]);x2=np.asarray(build_design_matrices([fit.model.data.design_info],d2)[0]);con=(x1-x2).mean(axis=0)
       t=fit.t_test(con);ci=t.conf_int().ravel()
       fits.append(dict(week=w,target=target,group1=tx1,group2=tx2,n=len(a),difference=float(t.effect.item()),ci_low=ci[0],ci_high=ci[1],p=float(t.pvalue),estimand='initial-treatment adjusted association',formula=f))
    save(fits,'clinical_adjusted_contrasts')
    return d

def molecular(c):
    m=pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv');d=c[c.age.ge(12)&~c.dose_ambiguous&c.visit_week.isin([0,12,24])].merge(m,on=['patient_uid','visit_week'],validate='one_to_one')
    markers=[x for x in m if x.startswith('log2_')];dist=[];corr=[];paired=[];induction=[]
    for x in markers:
     for (w,tx),g in d.groupby(['visit_week','treatment']):
      z=g[x].dropna();dist.append(dict(marker=x,week=w,treatment=tx,n=len(z),median=z.median(),q25=z.quantile(.25),q75=z.quantile(.75),min=z.min(),max=z.max(),unique=z.nunique()))
      z=g[[x,'easi_total']].dropna()
      if len(z)>=8 and z[x].nunique()>3 and z.easi_total.nunique()>3:
       rho,p=stats.spearmanr(z[x],z.easi_total);corr.append(dict(marker=x,week=w,treatment=tx,n=len(z),rho=rho,p=p))
     for w in [0,12,24]:
      z=d[d.visit_week.eq(w)][[x,'easi_total']].dropna()
      if len(z)>=8 and z[x].nunique()>3:
       rho,p=stats.spearmanr(z[x],z.easi_total);corr.append(dict(marker=x,week=w,treatment='all',n=len(z),rho=rho,p=p))
     for start,end in [(0,12),(12,24),(0,24)]:
      z=d[d.visit_week.eq(end)].merge(d[d.visit_week.eq(start)][['patient_uid',x,'easi_total']],on='patient_uid',suffixes=('','_start')).dropna(subset=[x,x+'_start','easi_total','easi_total_start'])
      for tx,g in [('all',z)]+list(z.groupby('treatment')):
       if len(g)>=8 and (g[x]-g[x+'_start']).nunique()>3:
        rho,p=stats.spearmanr(g[x]-g[x+'_start'],g.easi_total-g.easi_total_start);paired.append(dict(marker=x,start=start,end=end,treatment=tx,n=len(g),rho=rho,p=p,mean_log2_change=(g[x]-g[x+'_start']).mean()))
     z=d[d.visit_week.eq(12)].merge(d[d.visit_week.eq(0)][['patient_uid',x,'easi_total']],on='patient_uid',suffixes=('','_base')).rename(columns={x:'m12',x+'_base':'m0'}).dropna(subset=['m12','m0','easi_total_base','bmi','age','sex_male'])
     if len(z)<30 or z.treatment.value_counts().min()<8:continue
     f='m12 ~ C(treatment) + m0 + easi_total_base + bmi + age + sex_male + C(center)'
     fit=smf.ols(f,z).fit(cov_type='HC3',use_t=True)
     if np.linalg.matrix_rank(fit.model.exog)<fit.model.exog.shape[1]:continue
     from patsy import build_design_matrices
     for a,b in itertools.combinations(ARMS,2):
      x1=np.asarray(build_design_matrices([fit.model.data.design_info],z.assign(treatment=a))[0]);x2=np.asarray(build_design_matrices([fit.model.data.design_info],z.assign(treatment=b))[0]);t=fit.t_test((x1-x2).mean(axis=0));ci=t.conf_int().ravel()
      induction.append(dict(marker=x,group1=a,group2=b,n=len(z),adjusted_log2_difference=float(t.effect.item()),ci_low=ci[0],ci_high=ci[1],p=float(t.pvalue),formula=f))
    save(dist,'molecular_distributions');
    for rows,name,keys in [(corr,'molecular_easi_correlations',['week','treatment']),(paired,'paired_molecular_easi_changes',['start','end','treatment']),(induction,'molecular_induction_adjusted',['group1','group2'])]:
     df=pd.DataFrame(rows)
     if len(df):df['q']=df.groupby(keys).p.transform(lambda x:multipletests(x,method='fdr_bh')[1])
     save(df,name)
    assay=pd.read_csv(ROOT/'results/tables/redevelopment/assay_long.csv')
    save(assay.groupby(['analyte','visit_week','state']).patient_uid.nunique().rename('n_patients').reset_index(),'assay_availability')
    return d

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(OLD/'clinical.csv');cl=run_exposure(c);clinical(c);molecular(c)
    print('Patients',c.patient_uid.nunique(),'eligible',c[c.age.ge(12)&~c.dose_ambiguous].patient_uid.nunique(),flush=True)
    print('Recovered dose records',len(pd.read_csv(OUT/'recovered_exposure_records.csv')),flush=True)
    print(pd.read_csv(OUT/'maintenance_rates.csv').to_string(index=False),flush=True)
    assert len(cl)==170 and cl.patient_uid.nunique()==170

if __name__=='__main__':run()
