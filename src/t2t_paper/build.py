"""Build manuscript tables, figure legends and the LaTeX review documents."""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
ARM = dict(zip(ARMS, ['Dupilumab', 'Abrocitinib 100 mg', 'Abrocitinib 200 mg']))
TARGET = {'PP_NRS_le4':'PP-NRS at most 4','PP_NRS_01':'PP-NRS 0/1','PP_NRS_drop4':'PP-NRS decrease at least 4','EASI_le3':'EASI at most 3','EASI_le5':'EASI at most 5','IGA_01':'IGA 0/1','IGA_drop2':'IGA decrease at least 2','DLQI_le4':'Adult DLQI at most 4','DLQI_le10':'Adult DLQI at most 10','overall':'Both domains','physician':'Physician domain','patient':'Patient domain'}
TITLE = 'A common BSA--serum equation for EASI estimation in a Chinese atopic dermatitis cohort'

def supplement_number(source_number):
    """Stable internal figure number -> consecutive publication number."""
    source_number = int(source_number)
    order = json.loads((ROOT/'config/manuscript_revision_20261025.json').read_text())['supplement_source_order']
    if source_number not in order:
        raise ValueError('Retired figure must not be cited: source S' + str(source_number))
    return order.index(source_number) + 1

def section_text(path):
    return re.sub(r'\\suppfig\{(\d+)\}', lambda m: 'S' + str(supplement_number(m[1])), path.read_text())

def tex(value):
    s = str(value)
    replacements = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','≤':r'$\leq$','≥':r'$\geq$','₂':r'$_2$','−':'$-$','–':'--','—':'---','†':r'$\dagger$','α':r'$\alpha$','×':r'$\times$'}
    replacements.update({'ρ':r'$\rho$','²':r'$^2$','∞':r'$\infty$','→':r'$\rightarrow$','∩':r'$\cap$','±':r'$\pm$','∑':r'$\sum$'})
    return ''.join(replacements.get(c,c) for c in s)

def f(x, digits=2):
    return '--' if pd.isna(x) else f'{x:.{digits}f}'

def interval(x,lo,hi,digits=2):
    return f'${f(x,digits)}$ (${f(lo,digits)}$, ${f(hi,digits)}$)'

def preamble(title):
    return r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,graphicx,booktabs,longtable,array,amsmath,amssymb,float}
\usepackage[margin=18mm]{geometry}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hyphens]{url}
\usepackage[hidelinks]{hyperref}
\IfFileExists{microtype.sty}{\usepackage{microtype}}{}
\IfFileExists{caption.sty}{\usepackage[font=small,labelfont=bf]{caption}}{}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\setlength{\textfloatsep}{12pt}
\setlength{\intextsep}{12pt}
\setlength{\abovecaptionskip}{6pt}
\renewcommand{\topfraction}{.92}
\renewcommand{\bottomfraction}{.88}
\renewcommand{\textfraction}{.06}
\renewcommand{\floatpagefraction}{.82}
\widowpenalty=10000
\clubpenalty=10000
\makeatletter
\setlength{\@fptop}{0pt}
\makeatother
\newcommand{\figref}[1]{Fig.~\ref{fig:#1}}
\title{''' + title + r'''}
\author{}
\date{}
\begin{document}
'''

def table_begin(cols,caption,label):
    return f'\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\\toprule\n'

def safety_documentation_table(root):
    path=root/'manuscript/source_data/safety_documentation_windows.csv'
    data=pd.read_csv(path).set_index('treatment').loc[ARMS]
    required={'n_initial','n_with_note_through24','n_without_note_through24','n_with_note_all_followup','n_note_records_through24','adjudicated_event_incidence_available'}
    if not required.issubset(data.columns):raise ValueError('Safety documentation source lacks required window fields.')
    assert not data.index.duplicated().any()
    assert data.adjudicated_event_incidence_available.astype(str).str.lower().eq('false').all()
    assert (data.n_with_note_through24+data.n_without_note_through24).eq(data.n_initial).all()
    assert data.n_with_note_through24.le(data.n_with_note_all_followup).all()
    assert data.n_with_note_all_followup.le(data.n_initial).all()
    s=table_begin('p{31mm}rrrrr','Coverage of recorded safety documentation by follow-up window.','tab:safety_documentation')
    s+=r'Initial treatment & Initial N & \shortstack{With note through\\W24, n/N} & \shortstack{No note through\\W24, n} & \shortstack{With note at any\\follow-up, n/N} & \shortstack{Note records\\through W24, n} \\ \midrule'+'\n'
    for arm,r in data.iterrows():
        n=int(r.n_initial)
        s+=tex(ARM[arm])+f' & {n} & {int(r.n_with_note_through24)}/{n} & {int(r.n_without_note_through24)} & {int(r.n_with_note_all_followup)}/{n} & {int(r.n_note_records_through24)}'+r' \\'+'\n'
    s+=r'''\bottomrule\end{tabular}
\par\small\raggedright Patients are counted once within each documentation window. The through-W24 window uses recorded nominal visits up to W24; the all-follow-up column also includes later recorded visits. Note records can repeat within a patient and are not adverse-event counts. Absence of a note means unknown documentation, not absence of an event. Initial-group N is a cohort denominator, not a verified safety-observation denominator. These counts do not estimate AE or SAE incidence, severity or causality, and do not establish event-time treatment attribution. Source, \texttt{safety\_documentation\_windows.csv}.
\end{table}
'''
    return s

def reference_sensitivity_table(root):
    base=root/'results/tables/update_20261017/reference_sensitivity'
    metrics=pd.read_csv(base/'metrics.csv')
    paired=pd.read_csv(base/'paired_differences.csv')
    s=table_begin('p{56mm}rrr','Robustness to training-only exclusion of reference-inconsistent development visits.','tab:reference_sensitivity').replace('\\small','\\footnotesize')
    s+=r' & W0 & W12 & W24 \\ \midrule'+'\n'
    for evaluation,title in [('all_original_evaluable','All original held-out observations'),('strict_reference_consistent','Strict reference-consistent evaluation subset')]:
        s+=r'\multicolumn{4}{l}{\textbf{'+title+r'}} \\'+'\n'
        selected=metrics[metrics.evaluation.eq(evaluation)&metrics.model.eq('unified_selected')]
        p=selected[selected.analysis.eq('primary')].set_index('week').loc[[0,12,24]]
        q=selected[selected.analysis.eq('reference_exclusion')].set_index('week').loc[[0,12,24]]
        d=paired[paired.evaluation.eq(evaluation)&paired.contrast_type.eq('training_exclusion_minus_primary')&paired.candidate.eq('unified_selected')]
        assert p.n_patients.equals(q.n_patients)
        s+='Patients & '+' & '.join(str(int(v)) for v in p.n_patients)+r' \\'+'\n'
        for label,values in [('Primary MAE (95\\% CI)',[interval(r.MAE,r.MAE_lo,r.MAE_hi) for r in p.itertuples()]),
                             ('Training-exclusion MAE (95\\% CI)',[interval(r.MAE,r.MAE_lo,r.MAE_hi) for r in q.itertuples()])]:
            s+=label+' & '+' & '.join(values)+r' \\'+'\n'
        z=d[d.metric.eq('MAE')].set_index('week').loc[[0,12,24]]
        s+=r'Paired $\Delta$MAE (95\% CI) & '+' & '.join(interval(r.difference,r.ci_low,r.ci_high,3) for r in z.itertuples())+r' \\'+'\n'
        s+='Primary calibration slope & '+' & '.join(f(v,3) for v in p.calibration_slope)+r' \\'+'\n'
        s+='Training-exclusion calibration slope & '+' & '.join(f(v,3) for v in q.calibration_slope)+r' \\'+'\n'
        z=d[d.metric.eq('calibration_slope')].set_index('week').loc[[0,12,24]]
        s+=r'Paired $\Delta$slope (95\% CI) & '+' & '.join(interval(r.difference,r.ci_low,r.ci_high,3) for r in z.itertuples())+r' \\ \addlinespace'+'\n'
    s+=r'''\bottomrule\end{tabular}
\par\small\raggedright Eight W0 and two W12 visits with complete four-region scores whose sum differed from recorded total EASI by more than 0.15 points were excluded from development fitting only. Visits with missing regional references were retained. Original patient outer folds, evaluation patients and recorded outcomes were unchanged; preprocessing and inner structure/penalty selection were repeated. The strict evaluation subset requires all four regions and an absolute sum-to-total difference at most 0.01. Paired $\Delta$ is training-exclusion minus primary; negative $\Delta$MAE means less error. Calibration regresses observed on predicted EASI, with ideal slope 1. Metrics are scored within each repetition before averaging. Intervals use 2,000 paired patient resamples conditional on the saved predictions (seed 20261002); bootstrap resamples do not repeat fitting or selection. This is robustness to exclusion, not correction of clinical truth, independent validation or a replacement of the primary formula. Matched refitted BSA controls and complete metrics are in the \texttt{reference\_sensitivity} aggregate directory.
\end{table}
'''
    return s

def input_reference_specification_table():
    rows=[
        ('EASI reference','Recorded total EASI, 0--72; observed outcomes only, with no imputed validation truth.','Regional-sum discrepancies are audited; the authoritative field in discrepant records and blinded repeat-assessment reliability are not confirmed.'),
        ('BSA input','Affected BSA entered from 0 to 100 (10% is entered as 10). It supplies a clinical extent component of the hybrid estimate.','The source does not establish an observer-independent measurement protocol; this is not a blood-only score.'),
        ('Molecular inputs','CCL17, CCL26 and IL-5. Positive finite integration-export numerics recorded in pg/mL are log2 transformed once.','A positive numeric export is not proof of validated quantification. Transfer to another export or assay requires confirmed scale compatibility.'),
        ('Specimen and assay','Laboratory reports describe serum, Luminex LXSAHM-13/Luminex-200 and MSD K15067M-1/K15067M-2/SQ120 assays.','Matched specimen handling, technical-repeat precision and cross-platform bridging are not established by the analysis.'),
        ('Dilution and limits','Reports describe 50-microlitre serum and twofold laboratory dilution. No additional dilution factor is applied to the integration export.','Whether dilution is already corrected in that export, and validated lower/upper quantification limits, require source confirmation. Software calibration boundaries are not validated LLOQ/ULOQ.'),
        ('Clinical/serum timing','Patient identifier plus nominal W0, W12 or W24 links the clinical and molecular record.','Exact calendar-day separation and a permitted sampling window are unverified; nominal matching does not establish same-day measurement.'),
        ('Development scope','Age ≥12, observed EASI/BSA and all three positive molecular inputs. Fitting and selection use W0/W12 with patient-grouped inner and outer validation.','The marker panel arose from earlier exploration. W24 had previously been inspected and is same-cohort temporal application, not untouched external validation.'),
        ('Calculator output','The precise frozen coefficients return EASI units clipped to 0--72. Missing, nonfinite or nonpositive molecular inputs and invalid BSA are rejected.','BSA=0 returns zero by the area constraint; it does not prove absence of molecular inflammation. Independent validation of the single final coefficient vector is unavailable.'),
        ('Recorded IGA','Recorded grades 0--5 are retained. IGA 0/1 and decrease ≥2 use those recorded values without conversion to a 0--4 scale.','Exact instrument/version and correspondence to a validated IGA system require source confirmation.'),
        ('PP-NRS and DLQI','Recorded PP-NRS uses 0--10; adult DLQI uses 0--30 and age ≥18. A PP-NRS decrease ≥4 requires observed baseline ≥4 and observed follow-up.','PP-NRS recall window and exact questionnaire versions remain unconfirmed. Adult DLQI is unknown in adolescent composite rules.'),
        ('IgE and eosinophils','Analyses retain the recorded numeric values; adjusted models use log(1+x) and back-transformation.','Clinical laboratory units and the meaning of repeated IgE values of 5000 require source confirmation; no censoring threshold is inferred from repetition alone.'),
        ('Treatment targets','The September investigator framework supplies stage-specific physician and patient thresholds, with the stated three-valued missing-component logic.','The exact approved and prospectively implemented framework version and original voting records are not verified; target attainment is not validation of a new consensus or management effect.')]
    s=r'''\begin{longtable}{p{29mm}p{72mm}p{55mm}}
\caption{Input and clinical-reference specifications, with unresolved source requirements.}\label{tab:input_reference_specification}\\
\toprule Element & Current analysis specification & Evidence boundary \\ \midrule\endfirsthead
\toprule Element & Current analysis specification & Evidence boundary \\ \midrule\endhead
'''
    for row in rows:s+=' & '.join(tex(value) for value in row)+r' \\ \addlinespace'+'\n'
    return s+r'\bottomrule\end{longtable}'+'\n'

def tables(root):
    base = root/'results/tables/update_20261017/clinical'
    out = root/'manuscript/tables';out.mkdir(parents=True,exist_ok=True)
    b = pd.read_csv(base/'baseline_characteristics.csv')
    names = {'age':'Age, years','bmi':r'BMI, kg/m$^2$','sex_male':'Male sex, n/N (\\%)','easi_total':'EASI','pp_nrs':'PP-NRS','iga':'IGA, recorded grade','dlqi':'Adult DLQI','bsa':'BSA, \\%','scorad':'SCORAD','eo_count_numeric':'Eosinophils, recorded units','ige_numeric':'IgE, recorded units'}
    s = table_begin('p{45mm}rrr','Baseline characteristics of the clinical comparison population.','tab:baseline').replace('\\small','\\footnotesize')
    s += ' & Dupilumab & Abrocitinib 100 mg & Abrocitinib 200 mg \\\\ \n & (n=65) & (n=56) & (n=42) \\\\ \\midrule\n'
    for var,name in names.items():
        vals=[]
        for arm in ARMS:
            r=b[(b.variable==var)&(b.treatment==arm)].iloc[0]
            if var=='sex_male': val=f'{int(round(r["mean"]*r.n))}/{int(r.n)} ({100*r["mean"]:.1f})'
            else: val=f'{r["median"]:.1f} ({r.q25:.1f}--{r.q75:.1f})'
            vals.append(val)
        s += name+' & '+' & '.join(vals)+' \\\\ \n'
        if var!='sex_male' and any(int(b[(b.variable==var)&(b.treatment==arm)].iloc[0].n) != n for arm,n in zip(ARMS,[65,56,42])):
            s += '\\quad {\\scriptsize Available n} & '+' & '.join('{\\scriptsize '+str(int(b[(b.variable==var)&(b.treatment==arm)].iloc[0].n))+'}' for arm in ARMS)+' \\\\ \n'
    s += '\\bottomrule\\end{tabular}\n\\par\\small\\raggedright Values are median (interquartile range), unless stated otherwise. Quartiles use linear interpolation at rank $1+(n-1)p$ in the sorted observations (equivalent to Excel PERCENTILE.INC). Adult DLQI is restricted to age $\\geq18$. IGA is retained on the recorded scale pending instrument confirmation. The column-header n applies unless an available n is shown. Missing values are not filled.\\end{table}\n'
    (out/'table1.tex').write_text(s)
    all_metrics=pd.read_csv(root/'results/tables/update_20261017/model/overall_metrics.csv')
    m=all_metrics[all_metrics.model.eq('unified_selected')].sort_values('week')
    s=table_begin('lrrr','Primary performance of the common-model development procedure and matched controls.','tab:model')
    s+=' & W0 & W12 & W24 \\\\ \\midrule\n'
    rows=[('Patients',[str(int(x)) for x in m.n_patients]),('MAE (95\\% CI)',[interval(r.MAE,r.MAE_lo,r.MAE_hi) for r in m.itertuples()]),('RMSE',[f(x) for x in m.RMSE]),('Mean signed error',[f(x) for x in m.bias]),('Spearman correlation',[f(x,3) for x in m.rho]),('Concordance correlation',[f(x,3) for x in m.CCC]),(r'$R^2$',[f(x,3) for x in m.R2]),('Calibration intercept',[f(x) for x in m.calibration_intercept]),('Calibration slope',[f(x) for x in m.calibration_slope]),('Within one point, \\%',[f(100*x,1) for x in m.within1]),('Within two points, \\%',[f(100*x,1) for x in m.within2]),('Severity-band agreement, \\%',[f(100*x,1) for x in m.band_accuracy]),('Sensitivity for EASI >7, \\%',[f(100*x,1) for x in m.sensitivity_gt7]),('Specificity for EASI $\\leq$7, \\%',[f(100*x,1) for x in m.specificity_le7])]
    diagnostic=root/'results/tables/update_20261017/revision2_20261018'
    area=all_metrics[all_metrics.model.eq('bsa_area')].sort_values('week')
    serum=all_metrics[all_metrics.model.eq('serum_linear')].sort_values('week')
    paired=pd.read_csv(diagnostic/'primary_paired_control_MAE_uncertainty.csv')
    paired=paired[paired.primary.eq('unified_selected')&paired.comparator.eq('bsa_area')].sort_values('week')
    assert list(area.week)==list(serum.week)==list(paired.week)==list(m.week)==[0,12,24]
    assert area.n_patients.tolist()==serum.n_patients.tolist()==paired.n_patients.tolist()==m.n_patients.tolist()
    control_label=r'\shortstack[l]{Primary minus BSA-only area\\MAE difference (95\% CI)}'
    control_rows=[('BSA-only area MAE',[f(v) for v in area.MAE]),
                  ('Three-marker-only MAE',[f(v) for v in serum.MAE]),
                  (control_label,[interval(r.estimate,r.ci_low,r.ci_high) for r in paired.itertuples()])]
    rows=rows[:2]+control_rows+rows[2:]
    uncertainty=pd.read_csv(diagnostic/'primary_calibration_uncertainty.csv')
    for label,metric in [('Mean signed error','bias'),('Calibration intercept','calibration_intercept'),('Calibration slope','calibration_slope')]:
        u=uncertainty[uncertainty.metric.eq(metric)].sort_values('week')
        rows=[(name,[interval(r.estimate,r.ci_low,r.ci_high) for r in u.itertuples()] if name==label else values) for name,values in rows]
    ending='\\bottomrule\\end{tabular}\n\\par\\small\\raggedright W0/W12 use patient-grouped nested validation; W24 is same-cohort temporal application. Structure and penalty selection are internal to each training fold; earlier marker-panel selection is not repeated. Matched controls use the same patients and outer folds. BSA-only area uses the zero-intercept area structure; three-marker-only uses CCL17, CCL26 and IL-5 without BSA. Positive paired MAE differences mean greater primary-procedure error. Metrics are averaged after scoring each repetition. Intervals are 95\\% patient-bootstrap intervals conditional on saved predictions; selection and fitting are not repeated. MAE, mean absolute error; RMSE, root mean squared error. Positive signed error indicates overestimation. Severity-band agreement uses four EASI bands for observed and predicted values: $\\leq7$, $>7$ to $\\leq21$, $>21$ to $\\leq50$, and $>50$. Calibration regresses observed EASI on predicted EASI; ideal intercept/slope are 0/1. The single final formula is not independently validated.\\end{table}\n'
    full=s+''.join(name+' & '+' & '.join(vals)+' \\\\ \n' for name,vals in rows)+ending
    (out/'model_full_metrics.tex').write_text(full.replace('tab:model','tab:model_full'))
    core={'Patients','MAE (95\\% CI)','RMSE','Mean signed error','Spearman correlation','Concordance correlation','Within two points, \\%','Severity-band agreement, \\%'}
    core.update(name for name,_ in control_rows)
    s+=''.join(name+' & '+' & '.join(vals)+' \\\\ \n' for name,vals in rows if name in core)+ending.replace('Calibration regresses observed EASI on predicted EASI; ideal intercept/slope are 0/1. ', '')
    (out/'table2.tex').write_text(s)
    rates=pd.read_csv(base/'endpoint_rates.csv').sort_values(['week','target','treatment'])
    rates=rates[rates.week.isin([0,2,4,12,24])]
    s=r'\begin{longtable}{llrrr}'+'\n'+r'\caption{Exact target attainment, observed denominators and Wilson 95\% confidence intervals.}\label{tab:exact_target_rates}\\'+'\n'+r'\toprule Week and target & Initial treatment & Achieved/N & Rate, \% & 95\% CI \\ \midrule\endfirsthead'+'\n'+r'\toprule Week and target & Initial treatment & Achieved/N & Rate, \% & 95\% CI \\ \midrule\endhead'+'\n'
    for r in rates.itertuples():
        short={'dupilumab':'D','abrocitinib_100mg':'A100','abrocitinib_200mg':'A200'}[r.treatment]
        s+=f'W{r.week}: '+tex(TARGET.get(r.target,r.target))+f' & {short} & {r.k}/{r.n_evaluable} & {f(100*r.rate,1)} & {f(100*r.ci_low,1)}--{f(100*r.ci_high,1)} \\\\ \n'
    s+=r'\bottomrule\end{longtable}'+'\n'
    (out/'endpoint_rates.tex').write_text(s)
    bounds=pd.read_csv(diagnostic/'headline_missing_outcome_bounds.csv')
    support=pd.read_csv(diagnostic/'endpoint_available_ESS.csv')
    bounds=bounds.merge(support[['week','target','treatment','ESS']],on=['week','target','treatment'],validate='one_to_one')
    weighted=pd.read_csv(base/'endpoint_weighted_rates.csv')
    fields=['week','target','treatment','n_initial','n_evaluable','weighted_rate','ESS','lower_missing_bound','upper_missing_bound']
    weighted=weighted[fields].rename(columns={col:('weighted_available_rate' if col=='weighted_rate' else 'weighted_'+col) for col in fields if col not in ['week','target','treatment']})
    bounds=bounds.merge(weighted,on=['week','target','treatment'],validate='one_to_one')
    assert bounds.n_initial.eq(bounds.weighted_n_initial).all() and bounds.n_evaluable.eq(bounds.weighted_n_evaluable).all()
    assert (bounds.ESS-bounds.weighted_ESS).abs().lt(1e-8).all()
    s=r'\begin{center}\small\textbf{Unweighted observed outcomes and initial-cohort bounds}\par\smallskip\begin{tabular}{llrrrr}\toprule Target & Initial group & Success/evaluable & Initial N & \shortstack{Observed\\rate, \%} & \shortstack{Unweighted missing-\\outcome bounds, \%} \\ \midrule'+'\n'
    for r in bounds.itertuples():
        arm={'dupilumab':'D','abrocitinib_100mg':'A100','abrocitinib_200mg':'A200'}[r.treatment]
        target='W2 itch' if r.week==2 else 'W24 combined'
        s+=f'{target} & {arm} & {r.k}/{r.n_evaluable} & {r.n_initial} & {100*r.rate:.1f} & {100*r.lower_missing_bound:.1f}--{100*r.upper_missing_bound:.1f}'+r' \\'+'\n'
    s+=r'\bottomrule\end{tabular}\par\medskip\textbf{Matching baseline-overlap-weighted analysis}\par\smallskip\begin{tabular}{llrrr}\toprule Target & Initial group & \shortstack{Weighted available-\\case rate, \%} & \shortstack{Available-case\\weight ESS} & \shortstack{Weighted missing-\\outcome bounds, \%} \\ \midrule'+'\n'
    for r in bounds.itertuples():
        arm={'dupilumab':'D','abrocitinib_100mg':'A100','abrocitinib_200mg':'A200'}[r.treatment]
        target='W2 itch' if r.week==2 else 'W24 combined'
        s+=f'{target} & {arm} & {100*r.weighted_available_rate:.1f} & {r.ESS:.1f} & {100*r.weighted_lower_missing_bound:.1f}--{100*r.weighted_upper_missing_bound:.1f}'+r' \\'+'\n'
    s+=r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.97\linewidth}{\footnotesize Bounds assign all unknown outcomes failure or success in the initial cohort. Weighted bounds retain the same baseline overlap weights in observed and unknown records. Weighted available-case rates use only endpoint-observed patients. ESS is the effective sample size of those weighted available cases, not a patient count or an incidence denominator. Bounds are sensitivity limits, not confidence intervals.}
\end{center}
'''
    (out/'headline_target_support.tex').write_text(s)
    (out/'safety_documentation.tex').write_text(safety_documentation_table(root))
    (out/'reference_sensitivity.tex').write_text(reference_sensitivity_table(root))
    (out/'input_reference_specification.tex').write_text(input_reference_specification_table())
    # Machine-readable source index includes every aggregate table, not only plotted results.
    rows=[]
    for folder in ['clinical','model','model/auxiliary_targets','model/rounding','regional','redesign_20261018','revision2_20261018','reference_sensitivity']:
        d=root/'results/tables/update_20261017'/folder
        for p in sorted(d.glob('*.csv')):
            df=pd.read_csv(p)
            if 'private' in p.name or any(c in df for c in ['patient_uid','patient_id','source_record_id','record_id']):continue
            rows.append({'section':folder,'file':str(p.relative_to(root)),'rows':len(df),'columns':'; '.join(df.columns)})
    p=root/'manuscript/source_data/safety_documentation_windows.csv'
    data=pd.read_csv(p)
    rows.append({'section':'Safety documentation','file':str(p.relative_to(root)),'rows':len(data),'columns':'; '.join(data.columns)})
    pd.DataFrame(rows).to_csv(root/'manuscript/source_data/index.csv',index=False)
    s=r'\begin{longtable}{p{20mm}p{115mm}r}\toprule Section & Source file & Rows \\ \midrule\endhead'+'\n'
    for r in rows:s+=tex({'model/auxiliary_targets':'Other targets','model/rounding':'Rounding','redesign_20261018':'Diagnostics','revision2_20261018':'Revision audit','reference_sensitivity':'Reference sensitivity'}.get(r['section'],r['section']))+' & '+tex(Path(r['file']).name)+' & '+str(r['rows'])+' \\\\ \n'
    s+=r'\bottomrule\end{longtable}'
    (out/'source_index.tex').write_text(s)

def figure_page(row):
    height='.84' if row['id'].startswith('FigureS') else '.76'
    return '\\clearpage\n\\renewcommand{\\thefigure}{'+row['id'].replace('Figure','')+'}\n\\begin{figure}[p]\n\\centering\n\\includegraphics[width=\\linewidth,height='+height+'\\textheight,keepaspectratio]{figures/'+row['id']+'.pdf}\n\\caption{\\textbf{'+tex(row['title'])+'.} '+tex(row['caption'])+'}\\label{fig:'+row['id']+'}\n\\end{figure}\n'

def figure_inline(row):
    return '\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth,height=.72\\textheight,keepaspectratio]{figures/'+row['id']+'.pdf}\n\\caption{\\textbf{'+tex(row['title'])+'.} '+tex(row['caption'])+'}\\label{fig:'+row['id']+'}\n\\end{figure}\n'

def documents(root):
    m=root/'manuscript';catalog=json.loads((m/'figure_catalog.json').read_text())
    order=lambda r:(('S' in r['id']),int(re.search(r'\d+',r['id']).group()))
    catalog=sorted(catalog,key=order)
    main=preamble(TITLE)+r'\maketitle'+'\n'
    abstract=m/'sections/abstract.tex'
    if not abstract.exists():raise FileNotFoundError('The reviewed structured abstract is required.')
    main+='\\begin{abstract}\n'+section_text(abstract)+'\n\\end{abstract}\n'
    main+=r'\textbf{Keywords:} atopic dermatitis; biomarkers; EASI; longitudinal estimation; treat-to-target.'+'\n'
    figures={row['id']:row for row in catalog if 'S' not in row['id']}
    if set(figures)!={f'Figure{i}' for i in range(1,6)}:raise ValueError('The manuscript requires Figure1 through Figure5.')
    inserted=[]
    for sec in ['introduction','methods','results','discussion']:
        text=section_text(m/'sections'/f'{sec}.tex')
        def insert(match):
            key=match.group(1)
            if key in inserted:raise ValueError('Duplicate display: '+key)
            inserted.append(key)
            if key.startswith('Figure'):
                return figure_inline(figures[key])
            table_name={'Table1':'table1','Table2':'table2'}[key]
            return (m/'tables'/f'{table_name}.tex').read_text().replace('[htbp]','[!htbp]')
        text=re.sub(r'^% INSERT: (Figure[1-5]|Table[12])\s*$',insert,text,flags=re.MULTILINE)
        main+=text+'\n'
    required=set(figures)|{'Table1','Table2'}
    if set(inserted)!=required:
        raise ValueError('Missing inline display anchors: '+str(required-set(inserted)))
    main+=r'''\section*{Data and code availability}
Analysis code, versioned parameters, aggregate source tables and figure sources are available at \url{https://github.com/yanax726/T2T}. Figures and tables can be rebuilt from aggregates; model refitting requires the declared restricted input snapshots. Patient-level clinical records, serum measurements, dates and free-text notes are not included in the public repository. Access conditions and the institutional contact for these restricted records require author confirmation.
\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
'''
    (m/'main.tex').write_text(main)
    revision=json.loads((root/'config/manuscript_revision_20261025.json').read_text())
    groups = [(g['title'],g['source_ids'],g['description']) for g in revision['groups']]
    groups=[(title,[supplement_number(i) for i in ids],description) for title,ids,description in groups]
    supplement={r['id']:r for r in catalog if 'S' in r['id']}
    navigation_ids=[f'FigureS{i}' for _,ids,_ in groups for i in ids]
    if len(navigation_ids)!=len(supplement) or len(set(navigation_ids))!=len(supplement) or set(navigation_ids)!=set(supplement):
        raise ValueError('Supplement navigation must cover all current figures exactly once.')
    app=preamble('Supplementary information: '+TITLE)+r'''\maketitle
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theHtable}{S.\arabic{table}}
\section*{Supplementary guide}
The Supplementary Methods define the analysis populations, endpoints and validation. Tables S1--S5 provide exact target rates, complete model metrics, safety-documentation coverage, reference-exclusion sensitivity and input/reference specifications. Figures are grouped by their clinical or measurement question below. Complete aggregate results, including negative and non-estimable findings, remain in the linked CSV collection. Missing estimates are not zero effects.
'''
    for title,ids,description in groups:
        app+='\\hyperref[fig:FigureS'+str(ids[0])+']{\\textbf{'+tex(title)+'}}. '+tex(description)+'\\par\n'
    app+=r'''D denotes dupilumab; A100/A200 denote initial abrocitinib 100/200 mg. Treatment colours are fixed throughout. Molecular changes use log$_2$ concentration units; a change of $-1$ means a halving. Model errors use EASI points. Correlation describes ordering, whereas absolute errors and concordance address numerical agreement.
\clearpage
'''
    supplementary_methods=m/'sections/supplementary_methods.tex'
    if not supplementary_methods.exists():raise FileNotFoundError('The reviewed Supplementary Methods are required.')
    app+=section_text(supplementary_methods)+'\n'
    app+=r'''\clearpage\setcounter{table}{0}\section*{Supplementary tables}
N is the endpoint-specific evaluable denominator. Unknown outcomes are excluded from the observed rate and retained in missing-outcome bounds. Adult DLQI requires age at least 18. This table covers the specified visits through W24; exploratory W48 component rates remain in the CSV collection. Exact tests and correction families accompany \texttt{endpoint\_tests.csv} and \texttt{endpoint\_weighted\_contrasts.csv}.
{\setlength{\parskip}{0pt}\small\input{tables/endpoint_rates}}
\clearpage\input{tables/model_full_metrics}
\input{tables/safety_documentation}
\clearpage\input{tables/reference_sensitivity}
\clearpage{\small\input{tables/input_reference_specification}}
\clearpage
'''
    if (m/'sections/auxiliary_targets.tex').exists():app+=section_text(m/'sections/auxiliary_targets.tex')+'\n'
    app+=r'''\section*{Transport of published equations}
Published p-EASI coefficients were applied unchanged to identical eligible records, retaining both raw and 0--72-clipped outputs. The numerical source tables distinguish the primary nested procedure from fixed-structure, BSA-only and molecular-only comparisons. Published p-EASI uses serum alone, whereas the present procedure includes clinical BSA; local IL2RA is an unbridged proxy for sIL-2R. These local applications assess transport under unverified assay compatibility and do not establish model superiority or an ethnicity-specific difference.\citep{thijs2017,thijs2019,bakker2020}
'''
    for group_number,(title,ids,description) in enumerate(groups,1):
        for position,i in enumerate(ids):
            page=figure_page(supplement[f'FigureS{i}'])
            if position==0:
                page=page.replace('\\centering\n','\\centering\n\\pdfbookmark[1]{'+tex(title)+'}{group'+str(group_number)+'}\n{\\large\\bfseries '+tex(title)+'}\\par\\vspace{5pt}\n',1)
            app+=page
    app+=r'''\clearpage\section*{Aggregate source tables}
The \href{run:source_data/index.csv}{machine-readable source-table index} lists the complete aggregate collection, including unplotted and negative results. Figure-specific dependencies are in \texttt{figure\_catalog.json}. The \href{https://github.com/yanax726/T2T/tree/main/results/tables/update_20261017}{versioned aggregate directory} includes clinical, molecular, model, regional and reference-sensitivity tables; restricted patient-level files are not part of the public collection.
\bibliographystyle{unsrtnat}\bibliography{references}
'''
    app+='\\end{document}\n';(m/'appendix.tex').write_text(app)
    book=preamble('Complete figure book: clinical course, serum markers and EASI estimation')+r'\maketitle'+ '\n'
    book+=f'This review copy contains the five main figures followed by all {len(supplement)} supplementary figures. Captions specify populations, uncertainty and what each comparison can establish. Complete numerical results are supplied as CSV source tables.\n'
    for row in catalog:book+=figure_page(row)
    book+='\\end{document}\n';(m/'figure_book.tex').write_text(book)

def compile_pdf(root):
    env=os.environ.copy();env['SOURCE_DATE_EPOCH']='1792195200';env['FORCE_SOURCE_DATE']='1'
    logs=root/'reports/update_20261017/logs';logs.mkdir(parents=True,exist_ok=True)
    for name in ['main','appendix','figure_book']:
        commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]
        if name!='figure_book':commands += [['bibtex',name]]
        commands += [['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]*2
        with (logs/(name+'_latex.log')).open('w') as log:
            for cmd in commands:
                subprocess.run(cmd,cwd=root/'manuscript',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def run(root,compile=False):
    tables(root);documents(root)
    if compile:compile_pdf(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--compile',action='store_true');a=p.parse_args();run(a.root,a.compile)
