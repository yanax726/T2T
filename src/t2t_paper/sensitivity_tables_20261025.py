"""Numerical companions for matched input and IgE sensitivity analyses."""
import pandas as pd

COHORTS = [
 ('core_all','Complete core population','152/89/75'),
 ('itch_matched','Matched current-itch population','151/89/75'),
 ('adult_dlqi_matched','Matched adult itch and DLQI population','105/58/44'),
]
LABELS = {
 'ccl17':'CCL17', 'serum3':'Three serum markers', 'bsa':'BSA',
 'bsa_ccl17':'BSA + CCL17', 'bsa_serum3':'BSA + three markers',
 'itch_serum3':'PP-NRS + three markers', 'bsa_itch':'BSA + PP-NRS',
 'bsa_serum3_itch':'BSA + three markers + PP-NRS',
 'bsa_serum3_dlqi':'BSA + three markers + DLQI',
 'bsa_serum3_itch_dlqi':'BSA + three markers + PP-NRS + DLQI',
 'bsa_itch_dlqi':'BSA + PP-NRS + DLQI',
 'bsa_area':'BSA, fixed area structure', 'bsa_serum3_area':'BSA + three markers, fixed area structure',
 'without_CCL17':'Fixed area, without CCL17', 'without_CCL26':'Fixed area, without CCL26',
 'without_IL5':'Fixed area, without IL-5',
}


def start(caption,label,header,cols=r'p{53mm}rrr'):
    return ('{\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{'+cols+'}\n\\caption{'+caption+'}\\label{'+label+'}\\\\\n'
            '\\toprule '+header+r' \\ \midrule\endfirsthead'+'\n\\toprule '+header+r' \\ \midrule\endhead'+'\n')


def end():
    return '\\bottomrule\\end{longtable}\n}\n'


def tables(root,tex):
    out=root/'manuscript/tables'
    folder=root/'results/tables/update_20261025/ablation'
    d=pd.read_csv(folder/'overall_metrics.csv')
    s=start('Matched input-set performance.','tab:input_ablation','Inputs & W0 & W12 & W24')
    for metric,title in [('MAE','MAE in EASI points (95\\% CI)'),('correlations','Spearman / concordance correlation')]:
        s+=r'\multicolumn{4}{l}{\textbf{'+title+r'}} \\ \addlinespace'+'\n'
        for cohort,name,ns in COHORTS:
            q=d[d.cohort.eq(cohort)]
            s+=r'\multicolumn{4}{l}{'+tex(name+'; n='+ns)+r'} \\'+'\n'
            for model,g in q.groupby('model',sort=False):
                g=g.set_index('week').loc[[0,12,24]]
                if metric=='MAE':values=[f'{x.MAE:.2f} ({x.MAE_lo:.2f}, {x.MAE_hi:.2f})' for x in g.itertuples()]
                else:values=[f'{x.rho:.3f} / {x.CCC:.3f}' for x in g.itertuples()]
                s+=tex(LABELS[model])+' & '+' & '.join(values)+r' \\'+'\n'
            s+=r'\addlinespace'+'\n'
    s+=end()+r'''\noindent\footnotesize All models within a population use identical observations and patient folds. Three markers are CCL17, CCL26 and IL-5. BSA and questionnaires are from the current visit; DLQI requires age at least 18. W0/W12 training precedes previously inspected W24 temporal application. Core and adult populations must not be compared as though their participants were identical. Fixed-area rows hold the model structure constant; other BSA models select among linear, quadratic and area structures inside training folds. Intervals use 2,000 paired patient resamples conditional on saved predictions; all comparisons are exploratory. Full RMSE, bias, calibration and within-one/two-point metrics are retained in \texttt{overall\_metrics.csv}.\normalsize
'''
    (out/'input_ablation.tex').write_text(s)
    d=pd.read_csv(folder/'paired_comparisons.csv');d=d[d.metric.eq('MAE')]
    contrasts=[
        ('core_all','serum3','ccl17','Three markers versus CCL17'),
        ('core_all','bsa_serum3','bsa_ccl17','BSA + three markers versus BSA + CCL17'),
        ('core_all','bsa_serum3','bsa','BSA + three markers versus BSA'),
        ('core_all','bsa_serum3_area','bsa_area','Same area structure: add three markers'),
        ('itch_matched','bsa_serum3_itch','bsa_serum3','BSA + three markers: add PP-NRS'),
        ('itch_matched','bsa_serum3_itch','bsa_itch','BSA + PP-NRS: add three markers'),
        ('adult_dlqi_matched','bsa_serum3_itch_dlqi','bsa_serum3_itch','Adult BSA + three markers + PP-NRS: add DLQI'),
        ('adult_dlqi_matched','bsa_serum3_itch_dlqi','bsa_itch_dlqi','Adult BSA + PP-NRS + DLQI: add three markers'),
    ]
    s=start('Paired differences in mean absolute error.','tab:ablation_differences','Comparison & W0 & W12 & W24')
    for cohort,cand,ref,name in contrasts:
        q=d[d.cohort.eq(cohort)&d.candidate.eq(cand)&d.reference.eq(ref)].set_index('week').loc[[0,12,24]]
        values=[f'{x.difference:+.3f} ({x.ci_low:+.3f}, {x.ci_high:+.3f})' for x in q.itertuples()]
        s+=tex(name)+' & '+' & '.join(values)+r' \\ \addlinespace'+'\n'
    s+=end()+r'''\noindent\footnotesize Differences are the first or augmented model minus its stated reference; negative values indicate less error. Confidence intervals use the same patient resample for both models. These exploratory intervals are not adjusted for the many input comparisons, and do not establish clinical importance. The adult rows use 105/58/44 patients; itch rows use 151/89/75, and core rows use 152/89/75. In particular, a sub-point adult W24 error cannot be compared directly with a larger full-cohort error.\normalsize
'''
    (out/'ablation_differences.tex').write_text(s)
    ig=root/'results/tables/update_20261025/ige'
    s=start('IgE observation patterns and missing-data sensitivity.','tab:ige_missingness',
            'Initial group / analysis & W0--W12 & W0--W24 & W12--W24',cols=r'p{49mm}rrr')
    s+=r'\multicolumn{4}{l}{\textbf{Observed all-centre paired ratios of IgE+1 (95\% CI); paired n}} \\'+'\n'
    d=pd.read_csv(ig/'paired_log_changes.csv');d=d[d.scope.eq('all_163') & d.method.eq('all_available_pairs')]
    arms=['dupilumab','abrocitinib_100mg','abrocitinib_200mg'];names=['dupi','abro 100 mg','abro 200 mg'];windows=['W0_W12','W0_W24','W12_W24']
    for arm,name in zip(arms,names):
        q=d[d.treatment.eq(arm)].set_index('window').loc[windows]
        s+=name+' & '+' & '.join(f'{x.ratio:.2f} ({x.ratio_low:.2f}, {x.ratio_high:.2f}); {int(x.n)}' for x in q.itertuples())+r' \\'+'\n'
    d=pd.read_csv(ig/'mi_log_changes.csv')
    for method,mname in [('normal_log','Posterior log-normal imputation'),('pmm','Within-treatment donor imputation')]:
        s+=r'\addlinespace\multicolumn{4}{l}{\textbf{'+mname+r': CAMS, MAR}} \\'+'\n'
        for arm,name in zip(arms,names):
            q=d[d.scope.eq('CAMS')&d.scenario.eq('MAR')&d.method.eq(method)&d.treatment.eq(arm)].set_index('window').loc[windows]
            s+=name+' & '+' & '.join(f'{x.ratio:.2f} ({x.ratio_low:.2f}, {x.ratio_high:.2f})' for x in q.itertuples())+r' \\'+'\n'
    s+=end()
    s+=r'\textbf{CAMS W0--W24 sensitivity to missing-value assumptions}'+'\n'
    s+=r'{\small\begin{longtable}{p{49mm}rrr}\toprule Initial group / method & Half & MAR & Double \\ \midrule\endhead'+'\n'
    for method,mname in [('normal_log','Log-normal'),('pmm','Donor')]:
        for arm,name in zip(arms,names):
            q=d[d.scope.eq('CAMS')&d.method.eq(method)&d.treatment.eq(arm)&d.window.eq('W0_W24')].set_index('scenario').loc[['MNAR_half','MAR','MNAR_double']]
            s+=name+'; '+mname+' & '+' & '.join(f'{x.ratio:.2f} ({x.ratio_low:.2f}, {x.ratio_high:.2f})' for x in q.itertuples())+r' \\'+'\n'
    s+=end()+r'''\noindent\footnotesize Ratios exponentiate mean changes in log(IgE+1); they are not arithmetic mean ratios. All-centre observed pairs and CAMS imputation have different target populations. The second block varies missing-value assumptions for W0--W24. MAR assumes that included observed information explains missingness; half/double multiply only missing follow-up (IgE+1) by 0.5/2, retaining the physical lower bound, and illustrate departures from MAR. Imputation targets 130 CAMS patients, with 400 retained completed datasets; these are not 400 independent patient cohorts. Complete all-centre sensitivity, centre-specific coverage, observation-pattern summaries, donor support and numerical diagnostics remain in the source tables. No IgE was observed at W4 and none was created. The value 5000 is retained as recorded; its assay-censoring status is unverified.\normalsize
'''
    (out/'ige_missingness.tex').write_text(s)
