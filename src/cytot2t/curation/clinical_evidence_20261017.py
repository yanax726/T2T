"""Rebuild the clinical and serum evidence used in the October manuscript.

The analysis unit is the patient. Initial treatment defines comparisons;
recorded changes in treatment describe later paths rather than random groups.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from patsy import build_design_matrices, dmatrices
from scipy.linalg import svd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from threadpoolctl import threadpool_limits

from cytot2t.curation import proposal_closure_20261002 as previous
from cytot2t.curation.update_20261002_overlap import weights, balance
from cytot2t.clinical_alignment import doctor_evidence

ROOT = Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[3])).resolve()
OUT = ROOT / 'results/tables/update_20261017/clinical'
REPORT = ROOT / 'reports/update_20261017/clinical'
PRIV = OUT / 'restricted'
SEED = 20261017
ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
WINDOWS = [(0, 12), (12, 24), (0, 24)]
CLINICAL = ['easi_total', 'pp_nrs', 'dlqi', 'bsa', 'scorad', 'eo_count_numeric', 'ige_numeric']


def save(data, name, private=False):
    frame = pd.DataFrame(data)
    frame.to_csv((PRIV if private else OUT) / (name + '.csv'), index=False)
    return frame


def adjust_p(frame, keys, method='fdr_bh', name='q'):
    frame[name] = np.nan
    labels = frame[keys].fillna('unspecified').astype(str).agg('|'.join, axis=1)
    for _, idx in labels.groupby(labels).groups.items():
        finite = frame.loc[idx, 'p'].dropna()
        if len(finite):
            frame.loc[finite.index, name] = multipletests(finite, method=method)[1]
    return frame


def rule(value, threshold, direction='le'):
    if pd.isna(value):
        return np.nan
    return float(value <= threshold if direction == 'le' else value >= threshold)


def or_rule(values):
    if any(pd.notna(v) and v == 1 for v in values):
        return 1.
    return 0. if all(pd.notna(v) for v in values) else np.nan


def and_rule(values):
    if any(pd.notna(v) and v == 0 for v in values):
        return 0.
    return 1. if all(pd.notna(v) for v in values) else np.nan


def build_endpoints(clinical):
    baseline = clinical[clinical.visit_week.eq(0)].set_index('patient_uid')
    rows = []
    for r in clinical[clinical.visit_week.isin([0, 2, 4, 12, 24, 48])].itertuples():
        b = baseline.loc[r.patient_uid]
        if r.visit_week == 2:
            values = {'PP_NRS_le4': rule(r.pp_nrs, 4)}
        else:
            values = {'PP_NRS_le4': rule(r.pp_nrs, 4), 'PP_NRS_01': rule(r.pp_nrs, 1),
                      'IGA_01': rule(r.iga, 1), 'EASI_le5': rule(r.easi_total, 5),
                      'EASI_le3': rule(r.easi_total, 3)}
            if r.age >= 18:
                values.update(DLQI_le10=rule(r.dlqi, 10), DLQI_le4=rule(r.dlqi, 4))
            if r.visit_week:
                for cut in [50, 75, 90]:
                    values[f'EASI{cut}'] = rule(r.easi_total, b.easi_total * (1 - cut / 100)) if pd.notna(b.easi_total) and b.easi_total > 0 else np.nan
                values['IGA_drop2'] = rule(b.iga - r.iga, 2, 'ge')
                values['PP_NRS_drop4'] = rule(b.pp_nrs - r.pp_nrs, 4, 'ge') if b.pp_nrs >= 4 else np.nan
                cut = 50 if r.visit_week == 4 else (90 if r.visit_week == 48 else 75)
                physician_parts = [values[f'EASI{cut}'], values['IGA_drop2'] if r.visit_week == 4 else values['IGA_01']]
                if r.visit_week in [24, 48]:
                    physician_parts.append(values['EASI_le5'] if r.visit_week == 24 else values['EASI_le3'])
                patient_parts = [values['PP_NRS_le4'] if r.visit_week == 4 else values['PP_NRS_01'],
                                 values.get('DLQI_le10' if r.visit_week == 4 else 'DLQI_le4', np.nan)]
                values['physician'] = or_rule(physician_parts)
                values['patient'] = or_rule(patient_parts)
                values['overall'] = and_rule([values['physician'], values['patient']])
        for target, value in values.items():
            rows.append(dict(patient_uid=r.patient_uid, week=r.visit_week, target=target,
                             treatment=r.treatment, value=value, age=r.age,
                             exploratory_week48=r.visit_week == 48))
    return save(rows, 'endpoint_values', private=True)


def endpoint_summaries(values, baseline):
    rates, tests = [], []
    for (week, target), g in values.groupby(['week', 'target']):
        counts = []
        for arm in ARMS:
            h = g[g.treatment.eq(arm)]
            n, k = int(h.value.notna().sum()), int(h.value.eq(1).sum())
            lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
            rates.append(dict(week=week, target=target, treatment=arm, n_initial=len(h), n_evaluable=n,
                              k=k, not_evaluable=len(h)-n, rate=k/n if n else np.nan, ci_low=lo, ci_high=hi,
                              lower_missing_bound=k/len(h) if len(h) else np.nan,
                              upper_missing_bound=(k+len(h)-n)/len(h) if len(h) else np.nan))
            counts.append([k, n-k])
        counts = np.asarray(counts)
        if week == 0:
            continue
        if counts.sum(0).min() > 0 and counts.sum(1).min() > 0:
            obs = stats.chi2_contingency(counts, correction=False)[0]
            n = counts.sum(1); k = int(counts[:, 0].sum()); expected = n*k/n.sum()
            rng = np.random.default_rng(SEED)
            draws = rng.multivariate_hypergeometric(n, k, size=49999)
            chi = ((draws-expected)**2/expected).sum(1)+((n-draws-n+expected)**2/(n-expected)).sum(1)
            p = (1+(chi >= obs-1e-12).sum())/50000
            tests.append(dict(week=week, target=target, comparison='three_groups', p=p, effect=np.nan,
                              ci_low=np.nan, ci_high=np.nan, statistic=obs, family='three_groups', method='Conditional Monte Carlo Pearson; 49999 draws'))
        for i, j in itertools.combinations(range(3), 2):
            for combined in [False, True] if (i, j) == (0, 1) else [False]:
                table = np.vstack([counts[0], counts[1:].sum(0)]) if combined else counts[[i, j]]
                k1, n1 = table[0, 0], table[0].sum(); k2, n2 = table[1, 0], table[1].sum()
                if min(n1, n2) == 0:
                    continue
                lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method='newcomb')
                label = 'dupilumab - pooled_abrocitinib' if combined else ARMS[i]+' - '+ARMS[j]
                tests.append(dict(week=week, target=target, comparison=label, p=stats.fisher_exact(table).pvalue,
                                  effect=k1/n1-k2/n2, ci_low=lo, ci_high=hi, n1=n1, n2=n2,
                                  method='Fisher exact; Newcombe risk difference CI', family='pooled_supplement' if combined else 'three_pairwise'))
    r = save(rates, 'endpoint_rates')
    t = pd.DataFrame(tests); t['p_holm'] = np.nan
    ix = t.family.eq('three_pairwise')
    t.loc[ix, 'p_holm'] = t.loc[ix].groupby(['week', 'target']).p.transform(lambda p: multipletests(p, method='holm')[1])
    t = adjust_p(t, ['family'], name='q_exploratory')
    save(t, 'endpoint_tests')
    return r


def weighted_endpoints(values, baseline, n_boot):
    b = baseline.reset_index(drop=True).copy(); w, _, _ = weights(b)
    keys = values[['week', 'target']].drop_duplicates().sort_values(['week', 'target'])
    tuples = list(keys.itertuples(index=False, name=None)); colmap = {x:i for i,x in enumerate(tuples)}
    Y = np.full((len(b), len(tuples)), np.nan); eligible = np.zeros_like(Y, dtype=bool)
    rowmap = dict(zip(b.patient_uid, range(len(b))))
    for r in values.itertuples():
        i, j = rowmap[r.patient_uid], colmap[(r.week, r.target)]
        Y[i,j] = r.value; eligible[i,j] = True
    arm_indices = [np.flatnonzero(b.treatment.eq(a)) for a in ARMS]
    def means(idx, ww):
        yy = Y[idx]; valid = np.isfinite(yy)
        numerator = np.nansum(yy*ww[:,None], axis=0)
        denominator = np.sum(valid*ww[:,None], axis=0)
        return np.divide(numerator, denominator, out=np.full(len(tuples), np.nan), where=denominator>0)
    central = np.vstack([means(idx, w[idx]) for idx in arm_indices])
    boot = np.full((n_boot, 3, len(tuples)), np.nan); rng = np.random.default_rng(SEED)
    for rep in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in arm_indices])
        bb = b.iloc[idx].reset_index(drop=True); ww, _, _ = weights(bb)
        for k, arm in enumerate(ARMS):
            group = np.flatnonzero(bb.treatment.eq(arm)); boot[rep,k] = means(idx[group], ww[group])
    np.savez_compressed(PRIV/'endpoint_weight_bootstrap.npz', rates=boot, central=central)
    rows, contrasts = [], []
    mix = len(arm_indices[1])/(len(arm_indices[1])+len(arm_indices[2]))
    for j, (week, target) in enumerate(tuples):
        for k, arm in enumerate(ARMS):
            idx = arm_indices[k]; valid = np.isfinite(Y[idx,j]); zz = w[idx][valid]
            lo, hi = np.nanquantile(boot[:,k,j], [.025,.975])
            total = w[idx][eligible[idx,j]].sum(); missing = w[idx][eligible[idx,j] & ~valid].sum()
            numerator = np.nansum(Y[idx,j]*w[idx])
            rows.append(dict(week=week,target=target,treatment=arm,n_initial=int(eligible[idx,j].sum()),n_evaluable=int(valid.sum()),
                             weighted_rate=central[k,j],ci_low=lo,ci_high=hi,ESS=zz.sum()**2/(zz**2).sum() if len(zz) else np.nan,
                             lower_missing_bound=numerator/total if total else np.nan,
                             upper_missing_bound=(numerator+missing)/total if total else np.nan,
                             n_bootstrap=n_boot,method='Three-arm overlap weights; patient bootstrap re-estimates propensity model'))
        if week == 0:
            continue
        for i,k in itertools.combinations(range(3),2):
            delta = boot[:,i,j]-boot[:,k,j]; effect = central[i,j]-central[k,j]
            finite = delta[np.isfinite(delta)]; lo,hi = np.quantile(finite,[.025,.975]); se=np.std(finite,ddof=1)
            contrasts.append(dict(week=week,target=target,comparison=ARMS[i]+' - '+ARMS[k],effect=effect,ci_low=lo,ci_high=hi,
                                  p=2*stats.norm.sf(abs(effect/se)) if se>0 else np.nan,family='three_pairwise',n_bootstrap=len(finite)))
        delta=boot[:,0,j]-(mix*boot[:,1,j]+(1-mix)*boot[:,2,j]); effect=central[0,j]-(mix*central[1,j]+(1-mix)*central[2,j])
        finite=delta[np.isfinite(delta)];lo,hi=np.quantile(finite,[.025,.975]);se=np.std(finite,ddof=1)
        contrasts.append(dict(week=week,target=target,comparison='dupilumab - pooled_abrocitinib',effect=effect,ci_low=lo,ci_high=hi,
                              p=2*stats.norm.sf(abs(effect/se)) if se>0 else np.nan,family='pooled_supplement',n_bootstrap=len(finite),A100_mixture_weight=mix))
        dif = np.column_stack([boot[:,0,j]-boot[:,1,j],boot[:,0,j]-boot[:,2,j]])
        dif = dif[np.isfinite(dif).all(1)]; cov = np.cov(dif,rowvar=False)
        vector=np.array([central[0,j]-central[1,j],central[0,j]-central[2,j]])
        if np.linalg.matrix_rank(cov)==2:
            stat=float(vector@np.linalg.solve(cov,vector))
            contrasts.append(dict(week=week,target=target,comparison='three_groups',effect=np.nan,ci_low=np.nan,ci_high=np.nan,
                                  p=stats.chi2.sf(stat,2),family='three_groups',statistic=stat,n_bootstrap=len(dif)))
    save(rows,'endpoint_weighted_rates'); ct=pd.DataFrame(contrasts);ct['p_holm']=np.nan
    ix=ct.family.eq('three_pairwise');ct.loc[ix,'p_holm']=ct.loc[ix].groupby(['week','target']).p.transform(lambda p:multipletests(p,method='holm')[1])
    save(adjust_p(ct,['family'],name='q_exploratory'),'endpoint_weighted_contrasts')


def continuous_summaries(c):
    rows, paired, composition, triplets = [], [], [], []
    for target in CLINICAL:
        visits=[0,2,4,12,24] if target=='pp_nrs' else ([0,12,24] if target=='ige_numeric' else [0,4,12,24])
        d=c[c.age.ge(18)] if target=='dlqi' else c
        for (arm,week),g in d[d.visit_week.isin(visits)].groupby(['treatment','visit_week']):
            x=g[target].dropna();rows.append(dict(target=target,treatment=arm,week=week,n_initial=len(g),n=len(x),n_missing=len(g)-len(x),
                                                  mean=x.mean(),sd=x.std(),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75)))
        for arm,g in d.groupby('treatment'):
            z=g.pivot(index='patient_uid',columns='visit_week',values=target)
            complete=z.reindex(columns=[0,12,24]).dropna();n_complete=len(complete)
            for week in [0,12,24]:
                x=complete[week].to_numpy();boot=np.median(np.random.default_rng(SEED).choice(x,(2000,n_complete)),axis=1) if n_complete else [np.nan]
                triplets.append(dict(target=target,treatment=arm,week=week,n=n_complete,median=np.median(x) if n_complete else np.nan,ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
            for start,end in WINDOWS:
                h=z[[start,end]].dropna();change=h[end]-h[start];n=len(h)
                radius=stats.t.ppf(.975,n-1)*change.std()/np.sqrt(n) if n>=5 else np.nan
                paired.append(dict(target=target,treatment=arm,start=start,end=end,n=n,mean_start=h[start].mean(),mean_end=h[end].mean(),
                                   median_start=h[start].median(),median_end=h[end].median(),mean_change=change.mean(),median_change=change.median(),
                                   ci_low=change.mean()-radius,ci_high=change.mean()+radius,
                                   p=previous.paired_signflip_p(change) if n>=5 else np.nan,method='Patient-paired mean change; t CI; paired sign-flip P'))
    for (arm,week),g in c[c.visit_week.isin([0,4,12,24])].groupby(['treatment','visit_week']):
        n=g.iga.notna().sum()
        for category in range(6):
            k=g.iga.eq(category).sum();composition.append(dict(treatment=arm,week=week,category=category,n=int(n),k=int(k),proportion=k/n if n else np.nan))
    save(rows,'clinical_observed');save(adjust_p(pd.DataFrame(paired),['target']),'clinical_paired_changes');save(composition,'IGA_composition')
    save(triplets,'clinical_fixed_triplets')


def continuous_adjusted(c):
    baseline=c[c.visit_week.eq(0)].set_index('patient_uid')
    baseline_weights=pd.read_csv(PRIV/'overlap_patient_weights.csv').set_index('patient_uid').weight
    means=[];contrasts=[];issues=[]
    mix=(baseline.treatment.eq(ARMS[1]).sum()/baseline.treatment.isin(ARMS[1:]).sum())
    for target in CLINICAL:
        weeks=[2,4,12,24] if target=='pp_nrs' else ([12,24] if target=='ige_numeric' else [4,12,24])
        d=c[c.visit_week.isin(weeks)].copy();d=d[d.age.ge(18)] if target=='dlqi' else d
        d['baseline']=d.patient_uid.map(baseline[target]);d['weight']=d.patient_uid.map(baseline_weights)
        d=d.dropna(subset=[target,'baseline','weight','age','bmi','sex_male','center'])
        d['y']=np.log1p(d[target]) if target in ['ige_numeric','eo_count_numeric'] else d[target]
        if target in ['ige_numeric','eo_count_numeric']:d['baseline']=np.log1p(d.baseline)
        formula='y ~ baseline + C(treatment)*C(visit_week) + age + bmi + sex_male + C(center)'
        for analysis in ['overlap_weighted','unweighted_outcome_adjusted']:
            Y,design=dmatrices(formula,d,return_type='dataframe');X=design.to_numpy();y=Y.iloc[:,0].to_numpy()
            ww=d.weight.to_numpy() if analysis=='overlap_weighted' else np.ones(len(d))
            u,s,vt=svd(X*np.sqrt(ww)[:,None],full_matrices=False);rank=int((s>s[0]*1e-10).sum());V=vt[:rank].T
            beta=np.einsum('ij,j->i',V,np.einsum('ij,i->j',u[:,:rank],y*np.sqrt(ww))/s[:rank])
            residual=y-np.einsum('ij,j->i',X,beta);bread=np.einsum('ik,jk,k->ij',V,V,1/s[:rank]**2)
            scores=pd.DataFrame(X*(ww*residual)[:,None]).assign(uid=d.patient_uid.to_numpy()).groupby('uid').sum().to_numpy();G=len(scores)
            covariance=np.einsum('ij,jk,lk->il',bread,np.einsum('ki,kj->ij',scores,scores),bread)*G/(G-1)*(len(d)-1)/(len(d)-rank)
            critical=stats.t.ppf(.975,G-1);reference=d.drop_duplicates('patient_uid')
            refw=reference.weight if analysis=='overlap_weighted' else np.ones(len(reference));vectors={}
            for week in weeks:
                for arm in ARMS:
                    matrix=np.asarray(build_design_matrices([design.design_info],reference.assign(visit_week=week,treatment=arm))[0]);vector=np.average(matrix,axis=0,weights=refw)
                    supported=np.max(abs(vector-np.einsum('ik,jk,j->i',V,V,vector)))<1e-7*max(1.,np.max(abs(vector)))
                    if not supported:
                        issues.append(dict(target=target,week=week,treatment=arm,analysis=analysis,status='cell_not_estimable'));continue
                    effect=float(np.einsum('i,i->',vector,beta));se=np.sqrt(max(0.,float(np.einsum('i,ij,j->',vector,covariance,vector))))
                    vectors[(week,arm)]=vector
                    means.append(dict(target=target,week=week,treatment=arm,analysis=analysis,n_observed=int((d.visit_week.eq(week)&d.treatment.eq(arm)).sum()),n_patients=G,
                                      adjusted_mean_scale=effect,ci_low=effect-critical*se,ci_high=effect+critical*se,
                                      scale='log1p recorded units' if target in ['ige_numeric','eo_count_numeric'] else 'recorded score',
                                      method='Marginal least squares; patient-cluster robust t CI; common reference distribution'))
                differences=[]
                for i,j in itertools.combinations(range(3),2):
                    if (week,ARMS[i]) not in vectors or (week,ARMS[j]) not in vectors:continue
                    differences.append((ARMS[i]+' - '+ARMS[j],vectors[(week,ARMS[i])]-vectors[(week,ARMS[j])],'three_pairwise'))
                if all((week,a) in vectors for a in ARMS):
                    differences.append(('dupilumab - pooled_abrocitinib',vectors[(week,ARMS[0])]-mix*vectors[(week,ARMS[1])]-(1-mix)*vectors[(week,ARMS[2])],'pooled_supplement'))
                    L=np.array([vectors[(week,ARMS[0])]-vectors[(week,a)] for a in ARMS[1:]])
                    effects=np.einsum('ij,j->i',L,beta);cov=np.einsum('ij,jk,lk->il',L,covariance,L)
                    if np.linalg.matrix_rank(cov)==2:
                        fstat=float(effects@np.linalg.solve(cov,effects)/2)
                        contrasts.append(dict(target=target,week=week,analysis=analysis,comparison='three_groups',group1='three_groups',group2='',effect=np.nan,ci_low=np.nan,ci_high=np.nan,
                                              p=stats.f.sf(fstat,2,G-1),family='three_groups',statistic=fstat,n_patients=G))
                for label,vector,family in differences:
                    effect=float(np.einsum('i,i->',vector,beta));se=np.sqrt(max(0.,float(np.einsum('i,ij,j->',vector,covariance,vector))))
                    parts=label.split(' - ')
                    contrasts.append(dict(target=target,week=week,analysis=analysis,comparison=label,group1=parts[0],group2=parts[1],effect=effect,ci_low=effect-critical*se,ci_high=effect+critical*se,
                                          p=2*stats.t.sf(abs(effect/se),G-1) if se>0 else np.nan,family=family,n_patients=G,A100_mixture_weight=mix if family=='pooled_supplement' else np.nan))
    estimates=pd.DataFrame(means);tests=pd.DataFrame(contrasts);tests['p_holm']=np.nan
    ix=tests.family.eq('three_pairwise');tests.loc[ix,'p_holm']=tests.loc[ix].groupby(['target','week','analysis']).p.transform(lambda p:multipletests(p,method='holm')[1])
    tests=adjust_p(tests,['analysis','family'],name='q_exploratory')
    save(estimates[estimates.analysis.eq('overlap_weighted')],'clinical_trajectory_adjusted')
    save(tests[tests.analysis.eq('overlap_weighted')],'clinical_trajectory_contrasts')
    save(estimates,'clinical_trajectory_all_analyses');save(tests,'clinical_trajectory_all_contrasts');save(issues,'clinical_model_issues')


def molecular_evidence(c):
    m=pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv')
    registry=pd.read_csv(ROOT/'config/marker_registry.csv');markers=registry.analyte.tolist()
    d=c[c.visit_week.isin([0,12,24])].merge(m,on=['patient_uid','visit_week'],validate='one_to_one')
    d=d.merge(c[c.visit_week.eq(0)][['patient_uid','easi_total']].rename(columns={'easi_total':'baseline_easi'}),on='patient_uid',validate='many_to_one')
    assay=pd.read_csv(ROOT/'results/tables/redevelopment/assay_long.csv')
    assay=assay[assay.patient_uid.isin(c.patient_uid.unique())]
    availability=assay.groupby(['analyte','visit_week','state']).patient_uid.nunique().rename('n').reset_index()
    save(availability,'assay_availability')
    distributions=[];within=[];between=[];diagnostics=[]
    for marker in markers:
        col='log2_'+marker
        for (week,arm),g in d.groupby(['visit_week','treatment']):
            x=g[col].dropna();distributions.append(dict(marker=marker,week=week,treatment=arm,n=len(x),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75),
                                                      mean=x.mean(),sd=x.std(),min=x.min(),max=x.max(),unique=x.nunique()))
        for start,end in WINDOWS:
            z=d[d.visit_week.eq(end)].merge(d[d.visit_week.eq(start)][['patient_uid',col]],on='patient_uid',suffixes=('','_start'),validate='one_to_one')
            z=z.rename(columns={col:'end_marker',col+'_start':'start_marker'}).dropna(subset=['start_marker','end_marker'])
            z['delta']=z.end_marker-z.start_marker
            for arm in ARMS:
                x=z[z.treatment.eq(arm)].delta;n=len(x);rad=stats.t.ppf(.975,n-1)*x.std()/np.sqrt(n) if n>=5 else np.nan
                within.append(dict(marker=marker,treatment=arm,start=start,end=end,n=n,effect=x.mean(),ci_low=x.mean()-rad,ci_high=x.mean()+rad,
                                   geometric_pair_ratio=2**x.mean() if n else np.nan,p=previous.paired_signflip_p(x) if n>=5 else np.nan,
                                   qc_tier=registry.set_index('analyte').loc[marker,'qc_tier'],status='estimated' if n>=5 else 'descriptive_small_n'))
            for scope in ['all_centers','CAMS','HUASHAN']:
                h=z.copy() if scope=='all_centers' else z[z.center.eq(scope)].copy()
                h=h.dropna(subset=['start_marker','baseline_easi','age','bmi','sex_male','center']);counts=h.treatment.value_counts().reindex(ARMS,fill_value=0)
                forms={'primary':'delta ~ C(treatment) + start_marker + baseline_easi + age + bmi + sex_male'+(' + C(center)' if scope=='all_centers' else '')}
                if scope=='all_centers':
                    batches=assay[assay.analyte.eq(marker)&assay.visit_week.eq(end)][['patient_uid','assay_file']].rename(columns={'assay_file':'batch_end'})
                    h=h.merge(batches,on='patient_uid',validate='one_to_one')
                    forms['assay_batch_sensitivity']=forms['primary']+' + C(batch_end)'
                for analysis,formula in forms.items():
                    fit=None;status='insufficient_pair_support'
                    if counts.min()>=5:
                        try:
                            fit=smf.ols(formula,h).fit(cov_type='HC3',use_t=True)
                            if fit.df_resid<8:raise ValueError('insufficient_residual_df')
                            if analysis=='primary' and np.linalg.matrix_rank(fit.model.exog)<fit.model.exog.shape[1]:raise ValueError('rank_deficient')
                            if fit.get_influence().hat_matrix_diag.max()>.95:raise ValueError('excessive_leverage')
                            status='estimated_HC3'
                        except Exception as exc:fit=None;status=str(exc)
                    diagnostics.append(dict(marker=marker,start=start,end=end,scope=scope,analysis=analysis,n=len(h),n_D=int(counts.iloc[0]),n_A100=int(counts.iloc[1]),n_A200=int(counts.iloc[2]),status=status,formula=formula))
                    mix=counts.iloc[1]/counts.iloc[1:].sum() if counts.iloc[1:].sum() else np.nan
                    compare=[('D - A100',0,1),('D - A200',0,2),('A200 - A100',2,1),('D - pooled A',0,None)]
                    for label,left,right in compare:
                        row=dict(marker=marker,start=start,end=end,scope=scope,analysis=analysis,comparison=label,n=len(h),n_D=int(counts.iloc[0]),n_A100=int(counts.iloc[1]),n_A200=int(counts.iloc[2]),
                                 effect=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,status=status,pooled_A100_weight=mix)
                        if fit is not None:
                            vectors=[np.asarray(build_design_matrices([fit.model.data.design_info],h.assign(treatment=arm))[0]).mean(0) for arm in ARMS]
                            vector=vectors[left]-(mix*vectors[1]+(1-mix)*vectors[2] if right is None else vectors[right])
                            _,singular,vectors_t=svd(fit.model.exog,full_matrices=False)
                            rank=int((singular>singular[0]*1e-10).sum());space=vectors_t[:rank].T
                            residual=vector-np.einsum('ik,jk,j->i',space,space,vector)
                            if np.max(abs(residual))>1e-7*max(1.,np.max(abs(vector))):
                                row['status']='contrast_not_estimable_with_batch'
                            else:
                                test=fit.t_test(vector);lo,hi=np.ravel(test.conf_int());row.update(effect=float(np.ravel(test.effect)[0]),ci_low=lo,ci_high=hi,p=float(test.pvalue))
                        between.append(row)
    save(distributions,'molecular_distributions')
    w=pd.DataFrame(within);w['family']='all_23_markers_three_drugs_three_windows';save(adjust_p(w,['family']),'molecular_paired_changes')
    b=adjust_p(pd.DataFrame(between),['scope','analysis']);save(b,'molecular_adjusted_contrasts');save(diagnostics,'molecular_model_diagnostics')
    return markers


def additional_doctor_evidence(c):
    # The archived caller ignored its current clinical_path for these analyses.
    # Supply the current, eligible snapshot explicitly to the shared implementation.
    staged=PRIV/'doctor_inputs';staged.mkdir(exist_ok=True)
    c.to_csv(staged/'clinical.csv',index=False)
    shutil.copy2(ROOT/'results/tables/redevelopment/molecular.csv',staged/'molecular.csv')
    doctor_evidence.OLD=staged;doctor_evidence.OUT=OUT;doctor_evidence.REP=REPORT
    doctor_evidence.run(all_markers=True,dlqi_min_age=18)
    shutil.move(str(OUT/'doctor_stratified_change_source.csv'),str(PRIV/'doctor_stratified_change_source.csv'))
    corr=pd.read_csv(OUT/'doctor_stratified_change_associations.csv');save(corr,'molecular_clinical_change_correlations')
    regional=pd.read_csv(OUT/'doctor_stratified_regional_correlations.csv');save(regional[regional.visit_week.eq(0)],'molecular_regional_baseline_correlations')


def molecular_triplets_and_global(c):
    m=pd.read_csv(ROOT/'results/tables/redevelopment/molecular.csv')
    d=c[c.visit_week.isin([0,12,24])].merge(m,on=['patient_uid','visit_week'],validate='one_to_one')
    base=c[c.visit_week.eq(0)].set_index('patient_uid');d['baseline_easi']=d.patient_uid.map(base.easi_total)
    markers=[col for col in m if col.startswith('log2_')];rows=[];global_tests=[];baseline_correlations=[]
    rng=np.random.default_rng(SEED)
    for col in markers:
        for arm,g in d.groupby('treatment'):
            z=g.pivot(index='patient_uid',columns='visit_week',values=col).reindex(columns=[0,12,24]).dropna()
            n=len(z)
            for week in [0,12,24]:
                x=(z[week]-z[0]).to_numpy();mean=np.mean(x) if n else np.nan
                radius=stats.t.ppf(.975,n-1)*np.std(x,ddof=1)/np.sqrt(n) if n>=5 else np.nan
                rows.append(dict(marker=col[5:],treatment=arm,week=week,n=n,effect=mean,ci_low=mean-radius,ci_high=mean+radius,
                                 geometric_pair_ratio=2**mean if n else np.nan,status='estimated' if n>=5 else 'descriptive_small_n',
                                 definition='All three visits observed for this marker; mean log2 concentration minus own W0 value; identical patients at each point'))
        for arm,g in [('baseline_pooled',d[d.visit_week.eq(0)])]+list(d[d.visit_week.eq(0)].groupby('treatment')):
            for target in ['easi_total','pp_nrs','dlqi','iga']:
                h=g[g.age.ge(18)] if target=='dlqi' else g
                baseline_correlations.append(dict(marker=col[5:],treatment=arm,target=target,family='pooled_baseline' if arm=='baseline_pooled' else 'stratified_baseline',
                                                  **doctor_evidence.correlation(h[col],h[target],rng)))
        h=d.rename(columns={col:'molecule'}).dropna(subset=['molecule','age','bmi','sex_male','baseline_easi'])
        rec=dict(marker=col[5:],n_patients=h.patient_uid.nunique(),n_visits=len(h),p=np.nan,status='insufficient_support')
        if rec['n_patients']>=30 and h.groupby(['treatment','visit_week']).size().reindex(pd.MultiIndex.from_product([ARMS,[0,12,24]]),fill_value=0).min()>=5:
            formula='molecule ~ C(visit_week)*C(treatment) + baseline_easi + age + bmi + sex_male + C(center)'
            try:
                fit=smf.ols(formula,h).fit(cov_type='cluster',cov_kwds={'groups':h.patient_uid},use_t=True)
                if np.linalg.matrix_rank(fit.model.exog)<fit.model.exog.shape[1]:raise ValueError('rank_deficient')
                terms=[x for x in fit.params.index if 'C(visit_week)' in x and ':C(treatment)' in x]
                L=np.zeros((len(terms),len(fit.params)))
                for j,term in enumerate(terms):L[j,list(fit.params.index).index(term)]=1
                if len(terms)!=4 or np.linalg.matrix_rank(L@fit.cov_params()@L.T)!=4:raise ValueError('interaction_not_estimable')
                test=fit.wald_test(L,scalar=True);rec.update(p=float(test.pvalue),statistic=float(test.statistic),df_num=4,df_denom=rec['n_patients']-1,
                                                          status='estimated_patient_cluster_F',formula=formula)
            except Exception as exc:rec['status']=str(exc)
        global_tests.append(rec)
    save(rows,'molecular_three_visit_paired')
    tests=pd.DataFrame(global_tests);tests['family']='23_molecule_time_by_treatment_interactions';save(adjust_p(tests,['family']),'molecular_global_interactions')
    save(adjust_p(pd.DataFrame(baseline_correlations),['family']),'molecular_baseline_clinical_correlations')


def support_and_early_response(c,values):
    b=c[c.visit_week.eq(0)].copy();weights_table=pd.read_csv(PRIV/'overlap_patient_weights.csv').set_index('patient_uid')
    w=b.patient_uid.map(weights_table.weight).to_numpy();_,_,X=weights(b)
    balance_rows=[];ess=[]
    for (week,target),g in values.groupby(['week','target']):
        observed=set(g.loc[g.value.notna(),'patient_uid']);mask=b.patient_uid.isin(observed).to_numpy()
        if all((mask&b.treatment.eq(a).to_numpy()).sum()>=2 for a in ARMS):
            res=balance(b.loc[mask],w[mask],X.loc[mask],'overlap_observed_subset')
            balance_rows.extend([dict(week=week,target=target,**r) for r in res])
        for arm in ARMS:
            use=mask&b.treatment.eq(arm).to_numpy();ww=w[use]
            ess.append(dict(week=week,target=target,treatment=arm,n_observed=int(use.sum()),ESS=ww.sum()**2/(ww**2).sum() if len(ww) else np.nan))
    save(balance_rows,'endpoint_observed_balance');save(ess,'endpoint_effective_sample_sizes')
    future=values[values.week.eq(24)&values.target.eq('overall')][['patient_uid','value']].rename(columns={'value':'W24_overall'})
    early=[]
    for week,target in [(2,'PP_NRS_le4'),(4,'overall'),(12,'overall'),(12,'EASI75')]:
        h=values[values.week.eq(week)&values.target.eq(target)].merge(future,on='patient_uid',validate='one_to_one').dropna(subset=['value','W24_overall'])
        for arm,g in h.groupby('treatment'):
            table=np.array([[int(((g.value==state)&(g.W24_overall==outcome)).sum()) for outcome in [1,0]] for state in [1,0]])
            p=stats.fisher_exact(table).pvalue if table.sum(1).min()>0 else np.nan
            for state in [1,0]:
                z=g[g.value.eq(state)];n=len(z);k=int(z.W24_overall.eq(1).sum());lo,hi=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
                early.append(dict(early_week=week,early_target=target,treatment=arm,early_achieved=state,n=n,k_W24_overall=k,rate=k/n if n else np.nan,ci_low=lo,ci_high=hi,p=p,
                                  interpretation='Observed early/late association; not a validated prediction model'))
    save(early,'early_target_W24_outcomes')


def safety_documentation_detail(c):
    h=c[c.adverse_event_text.notna()].copy();h['source_text']=h.adverse_event_text.astype(str).str.strip();h=h[h.source_text.ne('')]
    h['explicit_negative']=h.source_text.str.fullmatch(r'无|暂无|没有|无明显不良反应|无不良反应|未见明显不良反应',case=False)
    rows=[]
    for (arm,text),g in h.groupby(['treatment','source_text']):
        rows.append(dict(treatment=arm,source_text=text,n_patients=g.patient_uid.nunique(),n_visit_records=len(g),repeated_patient_text_records=len(g)-g.patient_uid.nunique(),
                         explicit_negative=bool(g.explicit_negative.iloc[0]),MedDRA_PT='not_adjudicated',severity='not_recorded_as_structured_grade',causality='not_adjudicated',
                         interpretation='Documentation counts; repeated text cannot identify distinct events'))
    save(rows,'safety_text_inventory',private=True)


def nominal_followup_eligibility(c):
    episodes=pd.read_csv(ROOT/'data/restricted/treatment_episode_v0.csv').set_index('treatment_episode_id')
    rows=[]
    for r in c[c.visit_week.isin([2,4,12,24,48])].itertuples():
        date=pd.to_datetime(episodes.loc[r.treatment_episode_id,'episode_start_date'],errors='coerce')
        asof=pd.Timestamp('2026-09-21' if r.treatment=='dupilumab' else '2026-09-22')
        due=np.nan if pd.isna(date) else bool(date+pd.Timedelta(weeks=r.visit_week)<=asof)
        rows.append(dict(treatment=r.treatment,week=r.visit_week,due=due))
    d=pd.DataFrame(rows);summary=[]
    for (arm,week),g in d.groupby(['treatment','week']):
        summary.append(dict(treatment=arm,week=week,n_initial=len(g),n_nominal_due=int(g.due.eq(True).sum()),n_nominal_not_due=int(g.due.eq(False).sum()),n_date_unknown=int(g.due.isna().sum()),
                            interpretation='Administrative sensitivity using registry filename date, not an approved database lock or visit window'))
    save(summary,'nominal_followup_eligibility')


def cohort_inventory(raw,c):
    b=c[c.visit_week.eq(0)].copy();rows=[]
    b['age_group']=pd.cut(b.age,[11,17,64,np.inf],labels=['12-17','18-64','65+'])
    b['EASI_severity']=pd.cut(b.easi_total,[-np.inf,7,21,50,np.inf],labels=['0-7','>7-21','>21-50','>50'])
    for variable in ['center','sex_male','age_group','EASI_severity']:
        for (arm,category),g in b.groupby(['treatment',variable],observed=False):
            rows.append(dict(treatment=arm,variable=variable,category=category,n=len(g)))
    save(rows,'baseline_group_counts')
    initial=raw[raw.visit_week.eq(0)]
    save([dict(step='all_current_patients',n=initial.patient_uid.nunique()),dict(step='age_under_12_excluded',n=int(initial.age.lt(12).sum())),
          dict(step='age_at_least_12',n=int(initial.age.ge(12).sum())),dict(step='ambiguous_initial_dose_excluded',n=int((initial.age.ge(12)&initial.dose_ambiguous).sum())),
          dict(step='clinical_main_population',n=len(b))],'cohort_flow')
    rows=[]
    for target,maximum in [('easi_total',72),('pp_nrs',10),('dlqi',30),('bsa',100),('scorad',103),('iga',4)]:
        x=c[target].dropna();rows.append(dict(target=target,n_values=len(x),minimum=x.min(),maximum=x.max(),nominal_maximum=maximum,
                                             outside_nominal_range_n=int(((x<0)|(x>maximum)).sum()),handling='Source category 5 retained; CRF scale confirmation required' if target=='iga' else 'No outcome imputation or clipping'))
    save(rows,'clinical_score_range_audit')
    save(c[c.ige_numeric.eq(5000)].groupby(['treatment','visit_week']).size().rename('n_exactly_5000').reset_index(),'IgE_5000_record_counts')


def treatment_changes(c):
    ledger=pd.read_csv(ROOT/'results/tables/update_20261002/maintenance_patient_ledger.csv')
    ledger=ledger[ledger.patient_uid.isin(c.patient_uid.unique())].copy();save(ledger,'maintenance_patient_ledger',private=True)
    exposure=pd.read_csv(ROOT/'results/tables/update_20261002/exposure_at_visits.csv')
    exposure=exposure[exposure.patient_uid.isin(c.patient_uid.unique())&exposure.visit_week.isin([0,12,24])]
    exposure['boundary_unknown']=exposure.boundary_assumption.ne('none')
    save(exposure.groupby(['visit_week','initial_treatment','previsit_regimen']).agg(n=('patient_uid','size'),boundary_unknown_n=('boundary_unknown','sum')).reset_index(),'actual_exposure_counts')
    rows=[]
    for classification in ['group','W12_decision']:
        for (arm,group),g in ledger.groupby(['initial_treatment',classification]):
            if classification=='W12_decision':g=g[g.EASI75_12.eq(True)]
            n=int(g.EASI75_24.notna().sum());k=int(g.EASI75_24.eq(True).sum());lo,hi=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
            rows.append(dict(classification=classification,treatment=arm,group=group,n_group=len(g),n_evaluable=n,k=k,rate=k/n if n else np.nan,ci_low=lo,ci_high=hi,n_missing=len(g)-n,
                             lower_missing_bound=k/len(g) if len(g) else np.nan,upper_missing_bound=(k+len(g)-n)/len(g) if len(g) else np.nan))
    save(rows,'maintenance_rates')
    save(ledger.groupby(['initial_treatment','group','outcome']).size().rename('n').reset_index(),'maintenance_transitions')
    source=pd.read_csv(ROOT/'results/tables/update_20261002/reduction_maintenance_classification.csv');save(source[source.patient_uid.isin(c.patient_uid.unique())],'reduction_maintenance_classification',private=True)
    previous.OUT=OUT;previous.safety(c)
    shutil.move(str(OUT/'safety_source_ledger.csv'),str(PRIV/'safety_source_ledger.csv'))
    # Exposure states above come from the repaired October parser, not September's obsolete states.
    paths=[]
    for (arm,path),g in ledger.groupby(['initial_treatment','group']):
        for target in ['easi_total','pp_nrs','dlqi','bsa']:
            d=c[c.patient_uid.isin(g.patient_uid)];d=d[d.age.ge(18)] if target=='dlqi' else d
            z=d.pivot(index='patient_uid',columns='visit_week',values=target).reindex(columns=[0,12,24]).dropna()
            for week in [0,12,24]:
                x=z[week].to_numpy();n=len(x);boot=np.median(np.random.default_rng(SEED).choice(x,(2000,n)),axis=1) if n else [np.nan]
                paths.append(dict(treatment=arm,path=path,target=target,week=week,n=n,median=np.median(x) if n else np.nan,ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),interpretation='Same observed patients at all three visits; later treatment path is descriptive'))
    save(paths,'clinical_paths_fixed_triplets')


def run(n_boot=1000):
    for directory in [OUT,REPORT,PRIV]:directory.mkdir(parents=True,exist_ok=True)
    previous.ROOT=ROOT;doctor_evidence.ROOT=ROOT
    raw=pd.read_csv(ROOT/'results/tables/update_20260925/clinical.csv')
    c=raw[raw.age.ge(12)&~raw.dose_ambiguous].copy();b=c[c.visit_week.eq(0)].copy()
    assert c.patient_uid.nunique()==163 and not c.duplicated(['patient_uid','visit_week']).any()
    cohort_inventory(raw,c)
    w,prob,x=weights(b);save(b.assign(weight=w,prob_assigned=prob),'overlap_patient_weights',private=True)
    save(balance(b,np.ones(len(b)),x,'unweighted')+balance(b,w,x,'overlap_weighted'),'baseline_balance')
    features=[]
    for arm,g in b.groupby('treatment'):
        for variable in ['age','bmi','sex_male','easi_total','pp_nrs','iga','dlqi','bsa','scorad','eo_count_numeric','ige_numeric']:
            h=g[g.age.ge(18)] if variable=='dlqi' else g;xx=h[variable].dropna()
            features.append(dict(treatment=arm,variable=variable,n_initial=len(h),n=len(xx),mean=xx.mean(),sd=xx.std(),median=xx.median(),q25=xx.quantile(.25),q75=xx.quantile(.75)))
    save(features,'baseline_characteristics')
    values=build_endpoints(c);rates=endpoint_summaries(values,b);print('Endpoint tables ready',flush=True)
    continuous_summaries(c)
    previous.OUT=OUT;previous.CURRENT=ROOT/'results/tables/update_20261002';previous.ADULT_MIN_AGE=18;continuous_adjusted(c)
    print('Clinical trajectories ready',flush=True)
    markers=molecular_evidence(c);print('All molecular windows ready',flush=True)
    additional_doctor_evidence(c);print('Doctor association tables ready',flush=True)
    molecular_triplets_and_global(c);support_and_early_response(c,values)
    treatment_changes(c);print('Treatment paths ready',flush=True)
    safety_documentation_detail(c)
    nominal_followup_eligibility(c)
    weighted_endpoints(values,b,n_boot);print('Weighted response comparisons ready',flush=True)
    reconciled=[]
    for name,cohort in [('legacy_including_ambiguous_dose',raw[raw.age.ge(12)]),('current_unambiguous_dose',c)]:
        h=cohort[cohort.visit_week.eq(2)&cohort.treatment.eq('abrocitinib_100mg')];n=h.pp_nrs.notna().sum();k=h.pp_nrs.le(4).sum()
        reconciled.append(dict(analysis=name,n_initial=len(h),n_evaluable=int(n),k=int(k),rate=k/n,missing=len(h)-n))
    save(reconciled,'W2_100mg_reconciliation')
    check=rates[(rates.week==2)&rates.target.eq('PP_NRS_le4')&rates.treatment.eq('abrocitinib_100mg')].iloc[0]
    assert check.k==34 and check.n_evaluable==37
    assert (rates.k<=rates.n_evaluable).all() and (rates.n_initial==rates.n_evaluable+rates.not_evaluable).all()
    assert len(markers)==23
    (REPORT/'verification.json').write_text(json.dumps(dict(patients=163,patient_visits=len(c),W2_100mg_k=34,W2_100mg_n=37,
        source_scores_unchanged=True,missing_scores_not_treated_as_failures=True,DLQI_age_min=18,
        independent_unit='patient',all_23_markers=True,n_bootstrap=n_boot,initial_group_preserved_after_regimen_change=True),indent=2))
    files=[Path(__file__),ROOT/'results/tables/update_20260925/clinical.csv',ROOT/'results/tables/redevelopment/molecular.csv',ROOT/'results/tables/redevelopment/assay_long.csv',ROOT/'config/marker_registry.csv']+sorted(OUT.glob('*.csv'))
    manifest=[]
    for f in files:
        try:label=str(f.relative_to(ROOT))
        except ValueError:label='src/cytot2t/curation/'+f.name
        manifest.append(dict(path=label,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (REPORT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=1000);parser.add_argument('--root',type=Path);args=parser.parse_args()
    if args.root:
        ROOT=args.root.resolve();OUT=ROOT/'results/tables/update_20261017/clinical';REPORT=ROOT/'reports/update_20261017/clinical';PRIV=OUT/'restricted'
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        run(args.bootstrap)
