"""Post hoc IgE missingness analysis without changing observed clinical records.

Imputation is patient-wide. Measured values stay fixed; repeated visits are
never counted as independent patients. The primary sensitivity population is
CAMS, where every treatment has measured IgE at both follow-up visits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq
from statsmodels.regression.mixed_linear_model import MixedLM
from threadpoolctl import threadpool_limits

ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
WEEKS = [0, 12, 24]
WINDOWS = [(0, 12), (0, 24), (12, 24)]
ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(rows, path):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def wide_data(root):
    source = root / 'results/tables/update_20260925/clinical.csv'
    raw = pd.read_csv(source)
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    assert c.patient_uid.nunique() == 163
    assert not c.duplicated(['patient_uid', 'visit_week']).any()
    b = c[c.visit_week.eq(0)].set_index('patient_uid').sort_index()
    assert b.groupby('treatment').size().to_dict() == dict(zip(sorted(ARMS), [56, 42, 65]))
    w = b[['treatment', 'center', 'age', 'bmi', 'sex_male', 'easi_total']].copy()
    for var, prefix in [('ige_numeric', 'ige'), ('easi_total', 'easi')]:
        pivot = c.pivot(index='patient_uid', columns='visit_week', values=var)
        for week in WEEKS:
            w[f'{prefix}{week}'] = pivot[week]
    assert w[['age', 'bmi', 'sex_male', 'easi0']].notna().all().all()
    assert (w[[f'ige{x}' for x in WEEKS]].stack() > 0).all()
    weight_path = root / 'results/tables/update_20261017/clinical/restricted/overlap_patient_weights.csv'
    w['overlap_weight'] = pd.read_csv(weight_path).set_index('patient_uid').weight
    return c, w, [source, weight_path]


def descriptive(c, w, out):
    coverage, patterns, composition, paired, levels = [], [], [], [], []
    for (arm, center), group in c.groupby(['treatment', 'center']):
        for week, g in group.groupby('visit_week'):
            x = g.ige_numeric.dropna()
            coverage.append(dict(treatment=arm, center=center, week=week, n_initial=g.patient_uid.nunique(),
                                 n_observed=len(x), n_missing=len(g)-len(x), n_exactly5000=int(x.eq(5000).sum()),
                                 median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75)))
    for (arm, center), g in w.groupby(['treatment', 'center']):
        pats = g[[f'ige{x}' for x in WEEKS]].notna().astype(int).astype(str).agg(''.join, axis=1).value_counts()
        for pattern, n in pats.items():
            patterns.append(dict(treatment=arm, center=center, observed_W0_W12_W24=pattern, n=int(n)))
    for scope, data in [('all_163', w), ('CAMS', w[w.center.eq('CAMS')])]:
        for arm, g in data.groupby('treatment'):
            for week in [12, 24]:
                for observed in [True, False]:
                    h = g[g[f'ige{week}'].notna().eq(observed)]
                    x = h.ige0.dropna()
                    composition.append(dict(scope=scope, treatment=arm, followup_week=week,
                        followup_IgE_observed=observed, n_group=len(h), n_baseline_IgE=len(x),
                        baseline_IgE_median=x.median(), baseline_IgE_q25=x.quantile(.25),
                        baseline_IgE_q75=x.quantile(.75), baseline_EASI_median=h.easi0.median(),
                        age_median=h.age.median(), BMI_median=h.bmi.median()))
            complete = g.dropna(subset=['ige0', 'ige12', 'ige24'])
            for method, h in [('available_case', g), ('fixed_three_visit', complete)]:
                for week in WEEKS:
                    x = h[f'ige{week}'].dropna(); y = np.log1p(x); n = len(y)
                    mean, var = y.mean(), y.var(ddof=1)/n
                    radius = stats.t.ppf(.975, n-1)*np.sqrt(var)
                    levels.append(dict(scope=scope, treatment=arm, week=week, method=method, n=n,
                         log_mean=mean, log_ci_low=mean-radius, log_ci_high=mean+radius,
                         estimate=np.expm1(mean), ci_low=np.expm1(mean-radius), ci_high=np.expm1(mean+radius),
                         raw_median=x.median(), raw_q25=x.quantile(.25), raw_q75=x.quantile(.75)))
            for method, h in [('all_available_pairs', g), ('fixed_three_visit', complete)]:
                for start, end in WINDOWS:
                    z = h[[f'ige{start}', f'ige{end}']].dropna(); n = len(z)
                    delta = np.log1p(z.iloc[:, 1])-np.log1p(z.iloc[:, 0])
                    mean, se = delta.mean(), delta.std()/np.sqrt(n)
                    radius = stats.t.ppf(.975,n-1)*se
                    paired.append(dict(scope=scope, treatment=arm, window=f'W{start}_W{end}', method=method,
                         n=n, estimate=mean, ci_low=mean-radius, ci_high=mean+radius, se=se,
                         ratio=np.exp(mean), ratio_low=np.exp(mean-radius), ratio_high=np.exp(mean+radius),
                         median_start=z.iloc[:,0].median(), median_end=z.iloc[:,1].median(),
                         p=2*stats.t.sf(abs(mean/se), n-1)))
    save(coverage, out/'coverage_by_center.csv'); save(patterns, out/'missing_patterns.csv')
    save(composition, out/'baseline_by_followup_observation.csv')
    save(paired, out/'paired_log_changes.csv'); save(levels, out/'observed_log_levels.csv')
    return coverage


def prepare_imputation(w, method):
    incomplete = np.log1p(w[['ige0', 'ige12', 'ige24', 'easi12', 'easi24']].to_numpy(float))
    if method == 'normal_log':
        incomplete[:,:3] = np.log(w[['ige0','ige12','ige24']].to_numpy(float))
    complete = np.column_stack([w.age, w.bmi, w.sex_male, np.log1p(w.easi0),
                                 w.treatment.eq(ARMS[1]), w.treatment.eq(ARMS[2])]).astype(float)
    if w.center.nunique() > 1:
        complete = np.column_stack([complete, w.center.eq('HUASHAN').astype(float)])
    means, scale = complete.mean(0), complete.std(0)
    assert np.all(scale>0)
    complete = (complete-means)/scale
    return incomplete, complete


def impute_chain(original, fixed, arms, rng, cfg, chain, scope, method):
    """Type-1 predictive mean matching with a normal/inverse-chi-square draw."""
    observed = np.isfinite(original); x = original.copy()
    donors = {(j,a):np.flatnonzero(observed[:,j] & (arms==a)) for j in range(x.shape[1]) for a in ARMS}
    for j in range(x.shape[1]):
        for a in ARMS:
            ix=np.flatnonzero(~observed[:,j] & (arms==a)); pool=donors[j,a]
            assert len(pool)>=5, (scope,j,a,len(pool))
            x[ix,j]=original[rng.choice(pool,len(ix)),j]
    samples, trace, support = [], [], []
    diagnostics = dict(max_condition=0., min_residual_df=10000, donor_violations=0, max_observed_change=0.)
    burn, thin, draws = cfg['burn_in_cycles'], cfg['cycles_between_draws'], cfg['draws_per_chain']
    for cycle in range(1,burn+thin*draws+1):
        for j in rng.permutation(x.shape[1]):
            yy=original[observed[:,j],j]
            cov=np.column_stack([fixed,np.delete(x,j,axis=1)])
            mu, sd = cov.mean(0),cov.std(0)
            cov=(cov-mu)/sd
            design=np.column_stack([np.ones(len(x)),cov]); xo=design[observed[:,j]]
            beta,_,rank,sing=lstsq(xo,yy,lapack_driver='gelsd')
            assert rank==xo.shape[1], (scope,j,'rank deficiency')
            df=len(yy)-rank;assert df>10
            inverse=np.linalg.inv(np.einsum("ni,nj->ij",xo,xo,optimize=False))
            residual=yy-np.einsum("ij,j->i",xo,beta,optimize=False)
            sigma2=np.einsum("i,i->",residual,residual)/rng.chisquare(df)
            beta_draw=rng.multivariate_normal(beta,sigma2*inverse,check_valid='raise')
            donor_prediction=np.einsum("ij,j->i",design,beta,optimize=False); recipient_prediction=np.einsum("ij,j->i",design,beta_draw,optimize=False)
            for a in ARMS:
                miss=np.flatnonzero(~observed[:,j] & (arms==a)); pool=donors[j,a]
                if not len(miss):continue
                if method == 'normal_log' and j < 3:
                    x[miss,j] = recipient_prediction[miss] + rng.normal(0,np.sqrt(sigma2),len(miss))
                else:
                    distances=abs(recipient_prediction[miss,None]-donor_prediction[pool][None,:])
                    nearest=np.argsort(distances,axis=1,kind='stable')[:,:5]
                    picked=pool[nearest[np.arange(len(miss)),rng.integers(0,5,len(miss))]]
                    assert np.all(arms[picked]==a) and np.all(observed[picked,j])
                    x[miss,j]=original[picked,j]
                if cycle>burn and (cycle-burn)%thin==0 and j<3:
                    outside=(donor_prediction[miss]<donor_prediction[pool].min()) | (donor_prediction[miss]>donor_prediction[pool].max())
                    support.append(dict(scope=scope,method=method,chain=chain,cycle=cycle,week=WEEKS[j],treatment=a,
                                        fraction_predicted_outside_donor_range=float(outside.mean()),
                                        n_missing=len(miss),n_same_treatment_donors=len(pool)))
            diagnostics['max_condition']=max(diagnostics['max_condition'],float(sing[0]/sing[-1]))
            diagnostics['min_residual_df']=min(diagnostics['min_residual_df'],int(df))
        assert np.array_equal(x[observed],original[observed])
        for j,week in enumerate(WEEKS):
            for a in ARMS:
                miss=~observed[:,j] & (arms==a)
                if cycle % 10 == 0:
                    trace.append(dict(scope=scope,method=method,chain=chain,cycle=cycle,week=week,treatment=a,
                                      missing_log_mean=float(x[miss,j].mean()) if miss.any() else np.nan))
        if cycle>burn and (cycle-burn)%thin==0:samples.append(x.copy())
    return np.asarray(samples),trace,diagnostics,support


def chain_diagnostics(values):
    """Split-chain R-hat and conservative positive-sequence autocorrelation ESS."""
    a=np.asarray(values,float)
    n=a.shape[1]//2
    split=np.concatenate([a[:,:n],a[:,-n:]],axis=0)
    within=split.var(1,ddof=1).mean();between=n*split.mean(1).var(ddof=1)
    varhat=(n-1)/n*within+between/n
    rhat=np.sqrt(varhat/within) if within>0 else 1.
    rho=[]
    for lag in range(1,n):
        autocov=np.mean([np.mean((s[:-lag]-s.mean())*(s[lag:]-s.mean())) for s in split])
        rho.append(1-(within-autocov)/varhat if varhat>0 else 0.)
    total=0.
    for j in range(0,len(rho)-1,2):
        pair=rho[j]+rho[j+1]
        if pair<0:break
        total+=pair
    ess=min(a.size,a.size/max(1.,1+2*total))
    return float(rhat),float(ess)


def pool(q,u,df_complete,chains):
    q,u=np.asarray(q,float),np.asarray(u,float);m=len(q)
    Q=q.mean();U=u.mean();B=q.var(ddof=1);T=U+(1+1/m)*B
    lam=(1+1/m)*B/T if T>0 else 0.
    old=(m-1)/lam**2 if lam>1e-12 else 1e12
    obs=(df_complete+1)/(df_complete+3)*df_complete*(1-lam)
    df=1/(1/old+1/obs);se=np.sqrt(T);rad=stats.t.ppf(.975,df)*se
    rhat,ess=chain_diagnostics(q.reshape(chains,-1))
    mcse=np.sqrt(B/ess);r=(1+1/m)*B/U if U>0 else np.nan
    fmi=(r+2/(df+3))/(r+1)
    return dict(estimate=Q,ci_low=Q-rad,ci_high=Q+rad,se=se,df=df,p=2*stats.t.sf(abs(Q/se),df),
                within_imputation_variance=U,between_imputation_variance=B,
                missing_information=fmi,monte_carlo_se=mcse,mcse_over_se=mcse/se,
                split_rhat=rhat,effective_imputations=ess,m=m)


def analyze_mi(w, samples, scope, cfg):
    flat=samples.reshape(-1,len(w),samples.shape[-1]);obs=np.isfinite(np.log1p(w[['ige0','ige12','ige24']].to_numpy(float)))
    means, changes, contrasts=[] ,[],[]
    for factor in [.5,1.,2.]:
        transformed=flat[:,:,:3].copy()
        for j in [1,2]:transformed[:,~obs[:,j],j]=np.maximum(0,transformed[:,~obs[:,j],j]+np.log(factor))
        scenario={.5:'MNAR_half',1.:'MAR',2.:'MNAR_double'}[factor]
        for arm in ARMS:
            ix=np.flatnonzero(w.treatment.eq(arm));n=len(ix)
            for j,week in enumerate(WEEKS):
                z=transformed[:,ix,j];q=z.mean(1);u=z.var(1,ddof=1)/n
                result=pool(q,u,n-1,cfg['chains'])
                means.append(dict(scope=scope,treatment=arm,week=week,scenario=scenario,n_target=n,
                    n_observed=int(obs[ix,j].sum()),n_imputed=int((~obs[ix,j]).sum()),
                    shifted_geometric_mean=np.expm1(result['estimate']),level_ci_low=np.expm1(result['ci_low']),
                    level_ci_high=np.expm1(result['ci_high']),**result))
            for start,end in WINDOWS:
                j,k=WEEKS.index(start),WEEKS.index(end)
                z=transformed[:,ix,k]-transformed[:,ix,j];q=z.mean(1);u=z.var(1,ddof=1)/n
                result=pool(q,u,n-1,cfg['chains'])
                changes.append(dict(scope=scope,treatment=arm,window=f'W{start}_W{end}',scenario=scenario,
                    n_target=n,n_observed_pairs=int((obs[ix,j]&obs[ix,k]).sum()),
                    ratio=np.exp(result['estimate']),ratio_low=np.exp(result['ci_low']),ratio_high=np.exp(result['ci_high']),**result))
        contrasts.extend(adjusted_mi(w,transformed,scope,scenario,cfg['chains']))
    return means,changes,contrasts


def analysis_design(w, y):
    n=len(w);week=np.tile([12,24],n);a1=np.repeat(w.treatment.eq(ARMS[1]).to_numpy(float),2);a2=np.repeat(w.treatment.eq(ARMS[2]).to_numpy(float),2);t=(week==24).astype(float)
    # Baseline IgE is imputed separately within each complete data set.
    cols=[np.ones(2*n),a1,a2,t,a1*t,a2*t,np.repeat(y[:,0],2)]
    for col in [np.log1p(w.easi0),w.age,w.bmi,w.sex_male]:cols.append(np.repeat(np.asarray(col),2))
    if w.center.nunique()>1:cols.append(np.repeat(w.center.eq('HUASHAN').to_numpy(float),2))
    return np.column_stack(cols),y[:,1:].ravel()


def cluster_fit(x,y,w,n):
    a=x*np.sqrt(w[:,None]);b=y*np.sqrt(w);beta,_,rank,_=lstsq(a,b,lapack_driver='gelsd')
    assert rank==x.shape[1]
    bread=np.linalg.inv(np.einsum("ni,nj->ij",a,a,optimize=False));residual=y-np.einsum("ij,j->i",x,beta,optimize=False)
    scores=(x*(w*residual)[:,None]).reshape(n,2,-1).sum(1)
    meat=np.einsum("ni,nj->ij",scores,scores,optimize=False)
    cov=np.einsum("ij,jk,lk->il",bread,meat,bread,optimize=False)*n/(n-1)*(len(y)-1)/(len(y)-rank)
    assert np.isfinite(beta).all() and np.isfinite(cov).all()
    return beta,cov


def adjusted_mi(w, ys, scope, scenario, chains):
    n=len(w);ww=np.repeat(w.overlap_weight.to_numpy(),2);estimates={};variances={}
    for yy in ys:
        x,y=analysis_design(w,yy);beta,cov=cluster_fit(x,y,ww,n)
        for week in [12,24]:
            arm_vectors=[]
            for arm in ARMS:
                v=np.zeros(x.shape[1]);v[0]=1;v[3]=float(week==24)
                v[1]=float(arm==ARMS[1]);v[2]=float(arm==ARMS[2]);v[4]=v[1]*v[3];v[5]=v[2]*v[3]
                arm_vectors.append(v)
            for i,j in [(0,1),(0,2),(1,2)]:
                v=arm_vectors[i]-arm_vectors[j];key=(week,ARMS[i]+' - '+ARMS[j]);estimates.setdefault(key,[]).append(np.einsum('i,i->',v,beta));variances.setdefault(key,[]).append(np.einsum('i,ij,j->',v,cov,v,optimize=False))
    rows=[]
    for (week,pair),q in estimates.items():
        result=pool(q,variances[week,pair],n-1,chains)
        rows.append(dict(scope=scope,scenario=scenario,week=week,comparison=pair,n_patients=n,
                         ratio=np.exp(result['estimate']),ratio_low=np.exp(result['ci_low']),ratio_high=np.exp(result['ci_high']),**result))
    return rows


def available_mixed(w,scope):
    rows=[]
    for uid,r in w.iterrows():
        for week in WEEKS:
            if pd.notna(r[f'ige{week}']):rows.append(dict(patient_uid=uid,week=week,y=np.log1p(r[f'ige{week}']),
                treatment=r.treatment,center=r.center,age=r.age,bmi=r.bmi,sex=r.sex_male,baseline_EASI=np.log1p(r.easi0)))
    long=pd.DataFrame(rows);f='y ~ C(treatment)*C(week) + baseline_EASI + age + bmi + sex'
    if w.center.nunique()>1:f+=' + C(center)'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model=MixedLM.from_formula(f,groups='patient_uid',re_formula='1',data=long)
        fit=model.fit(reml=False,method=['lbfgs','powell'],maxiter=1000,disp=False)
    from patsy import build_design_matrices
    rows=[];reference=long.drop_duplicates('patient_uid')
    for arm in ARMS:
        for start,end in WINDOWS:
            a=np.asarray(build_design_matrices([model.data.design_info],reference.assign(treatment=arm,week=start))[0]).mean(0)
            b=np.asarray(build_design_matrices([model.data.design_info],reference.assign(treatment=arm,week=end))[0]).mean(0)
            v=b-a;estimate=np.einsum('i,i->',v,fit.fe_params.to_numpy());cov=fit.cov_params().iloc[:len(v),:len(v)].to_numpy();se=np.sqrt(np.einsum('i,ij,j->',v,cov,v,optimize=False))
            rows.append(dict(scope=scope,treatment=arm,window=f'W{start}_W{end}',method='Available-data random-intercept ML',
                 n_patients=long.patient_uid.nunique(),n_observations=len(long),estimate=estimate,ci_low=estimate-1.96*se,ci_high=estimate+1.96*se,
                 ratio=np.exp(estimate),ratio_low=np.exp(estimate-1.96*se),ratio_high=np.exp(estimate+1.96*se),
                 converged=bool(fit.converged),random_intercept_variance=float(fit.cov_re.iloc[0,0]),residual_variance=float(fit.scale)))
    return rows,[str(x.message) for x in caught]



def comparison_table(paired, mixed, mi):
    rows=[]
    for scope in ['CAMS','all_163']:
        for arm in ARMS:
            for start,end in WINDOWS:
                window=f'W{start}_W{end}'
                p=paired[paired.scope.eq(scope)&paired.treatment.eq(arm)&paired.window.eq(window)&paired.method.eq('all_available_pairs')].iloc[0]
                m=mixed[mixed.scope.eq(scope)&mixed.treatment.eq(arm)&mixed.window.eq(window)].iloc[0]
                selected=[('All available pairs',p,int(p.n)),('Available-data longitudinal model',m,int(m.n_patients))]
                for method,title in [('normal_log','Gaussian log-IgE multiple imputation'),('pmm','Predictive mean matching multiple imputation')]:
                    z=mi[mi.scope.eq(scope)&mi.treatment.eq(arm)&mi.window.eq(window)&mi.scenario.eq('MAR')&mi.method.eq(method)].iloc[0]
                    selected.append((title,z,int(z.n_target)))
                for title,z,n in selected:
                    rows.append(dict(scope=scope,treatment=arm,window=window,method_label=title,
                         n_model_or_group=n,n_observed_pairs=int(p.n),estimate=z.estimate,ci_low=z.ci_low,ci_high=z.ci_high,
                         ratio=z.ratio,ratio_low=z.ratio_low,ratio_high=z.ratio_high))
    return rows


def run(root=ROOT):
    root=Path(root);out=root/'results/tables/update_20261025/ige';report=root/'reports/update_20261025/ige'
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    cfg_path=root/'config/ige_missingness_20261025.json';cfg=json.loads(cfg_path.read_text());c,w,sources=wide_data(root)
    source_hashes={str(p.relative_to(root)):digest(p) for p in sources+[cfg_path]}
    coverage=descriptive(c,w,out)
    means,changes,contrasts,traces,impute_diagnostics,mixed,warnings_out,support,tails=[],[],[],[],[],[],{},[],[]
    for scope,ww in [('CAMS',w[w.center.eq('CAMS')].copy()),('all_163',w.copy())]:
        for method in ['pmm','normal_log']:
            original,fixed=prepare_imputation(ww,method);chains=[]
            for chain in range(cfg['imputation']['chains']):
                rng=np.random.default_rng(cfg['seed']+chain*1000+(0 if scope=='CAMS' else 10000)+(0 if method=='pmm' else 50000))
                draws,trace,diag,donor=impute_chain(original,fixed,ww.treatment.to_numpy(),rng,cfg['imputation'],chain,scope,method)
                chains.append(draws);traces.extend(trace);support.extend(donor);impute_diagnostics.append(dict(scope=scope,method=method,chain=chain,**diag))
            samples=np.asarray(chains)
            if method == 'normal_log':
                samples[:,:,:,:3]=np.logaddexp(0,samples[:,:,:,:3])
                for j in range(3):
                    measured=np.isfinite(original[:,j]);samples[:,:,measured,j]=np.log1p(ww[f'ige{WEEKS[j]}'].to_numpy())[measured]
            for j,week in enumerate(WEEKS):
                for arm in ARMS:
                    ix=ww.treatment.eq(arm).to_numpy() & ~np.isfinite(original[:,j])
                    val=np.expm1(samples[:,:,ix,j]).ravel()
                    tails.append(dict(scope=scope,method=method,treatment=arm,week=week,n_missing_patients=int(ix.sum()),
                         n_imputed_draws=len(val),p01=np.quantile(val,.01),median=np.median(val),p99=np.quantile(val,.99),
                         minimum=val.min(),maximum=val.max(),fraction_above_recorded5000=float((val>5000).mean())))
            a,b,d=analyze_mi(ww,samples,scope,cfg['imputation'])
            for rows in [a,b,d]:
                for row in rows:row['method']=method
            means.extend(a);changes.extend(b);contrasts.extend(d)
            print(scope,method,'complete',len(ww),'patients',samples.shape[0]*samples.shape[1],'imputations',flush=True)
        a,b=available_mixed(ww,scope);mixed.extend(a);warnings_out[scope]=b
    save(support,out/'donor_support_per_imputation.csv')
    support=pd.DataFrame(support).groupby(['scope','method','treatment','week'],as_index=False).agg(
        fraction_predicted_outside_donor_range=('fraction_predicted_outside_donor_range','mean'),
        n_missing=('n_missing','first'),n_same_treatment_donors=('n_same_treatment_donors','first'))
    save(support,out/'donor_support_summary.csv');save(tails,out/'imputation_tail_diagnostics.csv')
    save(pd.DataFrame(means).drop(columns='p'),out/'mi_log_levels.csv')
    change=save(changes,out/'mi_log_changes.csv');save(contrasts,out/'mi_adjusted_contrasts.csv')
    mixed=save(mixed,out/'available_longitudinal_changes.csv');save(traces,out/'imputation_chain_traces.csv')
    save(comparison_table(pd.read_csv(out/'paired_log_changes.csv'),
         pd.read_csv(out/'available_longitudinal_changes.csv'),pd.read_csv(out/'mi_log_changes.csv')),out/'method_comparison.csv')
    save(impute_diagnostics,out/'imputation_design_diagnostics.csv')
    assert not c[c.visit_week.eq(4)].ige_numeric.notna().any()
    assert all(digest(root/p)==sha for p,sha in source_hashes.items())
    assert change[change.scenario.eq('MAR')].split_rhat.max()<1.05
    assert change[change.scenario.eq('MAR')].mcse_over_se.max()<.1
    assert change.select_dtypes('number').notna().all().all()
    check=dict(status='COMPUTATION_VERIFIED_METHOD_SENSITIVITY_REQUIRED',n_clinical=163,n_CAMS=int(w.center.eq('CAMS').sum()),
               n_no_primary_visit_IgE=int(w[['ige0','ige12','ige24']].isna().all(1).sum()),
               n_complete_triplets=int(w[['ige0','ige12','ige24']].notna().all(1).sum()),
               observed_values_unchanged=True,original_files_unchanged=True,no_W4_IgE_values_created=True,
               independent_unit='patient',imputations_per_scope=cfg['imputation']['chains']*cfg['imputation']['draws_per_chain'],
               max_MAR_change_rhat=float(change[change.scenario.eq('MAR')].split_rhat.max()),
               max_MAR_change_mcse_over_se=float(change[change.scenario.eq('MAR')].mcse_over_se.max()),
               mixed_model_warnings=warnings_out,source_sha256=source_hashes)
    (report/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    files=[Path(__file__),cfg_path]+sorted(out.glob('*.csv'))
    (report/'manifest.json').write_text(json.dumps([dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in files],indent=2)+'\n')
    print(json.dumps(check,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.root)
