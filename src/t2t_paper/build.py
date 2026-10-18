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
    if source_number in {20, 36}:
        raise ValueError('Retired comparison figure must not be cited: S' + str(source_number))
    return source_number - (source_number > 20) - (source_number > 36)

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
    return f'{f(x,digits)} ({f(lo,digits)}--{f(hi,digits)})'

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
    s += '\\bottomrule\\end{tabular}\n\\par\\small Values are median (interquartile range), unless stated otherwise. Adult DLQI is restricted to age $\\geq18$. IGA is retained on the recorded scale pending instrument confirmation. The column-header n applies unless an available n is shown. Missing values are not filled.\\end{table}\n'
    (out/'table1.tex').write_text(s)
    m=pd.read_csv(root/'results/tables/update_20261017/model/overall_metrics.csv');m=m[m.model=='unified_selected'].sort_values('week')
    s=table_begin('lrrr','Primary performance of the common-model development procedure.','tab:model')
    s+=' & W0 & W12 & W24 \\\\ \\midrule\n'
    rows=[('Patients',[str(int(x)) for x in m.n_patients]),('MAE (95\\% CI)',[interval(r.MAE,r.MAE_lo,r.MAE_hi) for r in m.itertuples()]),('RMSE',[f(x) for x in m.RMSE]),('Mean signed error',[f(x) for x in m.bias]),('Spearman correlation',[f(x,3) for x in m.rho]),('Concordance correlation',[f(x,3) for x in m.CCC]),(r'$R^2$',[f(x,3) for x in m.R2]),('Calibration intercept',[f(x) for x in m.calibration_intercept]),('Calibration slope',[f(x) for x in m.calibration_slope]),('Within one point, \\%',[f(100*x,1) for x in m.within1]),('Within two points, \\%',[f(100*x,1) for x in m.within2]),('Severity-band agreement, \\%',[f(100*x,1) for x in m.band_accuracy]),('Sensitivity for EASI >7, \\%',[f(100*x,1) for x in m.sensitivity_gt7]),('Specificity for EASI $\\leq$7, \\%',[f(100*x,1) for x in m.specificity_le7])]
    diagnostic=root/'results/tables/update_20261017/revision2_20261018'
    uncertainty=pd.read_csv(diagnostic/'primary_calibration_uncertainty.csv')
    for label,metric in [('Mean signed error','bias'),('Calibration intercept','calibration_intercept'),('Calibration slope','calibration_slope')]:
        u=uncertainty[uncertainty.metric.eq(metric)].sort_values('week')
        rows=[(name,[interval(r.estimate,r.ci_low,r.ci_high) for r in u.itertuples()] if name==label else values) for name,values in rows]
    ending='\\bottomrule\\end{tabular}\n\\par\\small W0/W12 use patient-grouped nested validation; W24 is same-cohort temporal application. Structure and penalty selection are internal to each training fold; earlier marker-panel selection is not repeated. Metrics are averaged after scoring each repetition. Intervals are 95\\% patient-bootstrap intervals conditional on saved predictions; selection and fitting are not repeated. MAE, mean absolute error; RMSE, root mean squared error. Positive signed error indicates overestimation. Calibration regresses observed EASI on predicted EASI; ideal intercept/slope are 0/1. The single final formula is not independently validated.\\end{table}\n'
    full=s+''.join(name+' & '+' & '.join(vals)+' \\\\ \n' for name,vals in rows)+ending
    (out/'model_full_metrics.tex').write_text(full.replace('tab:model','tab:model_full'))
    core={'Patients','MAE (95\\% CI)','RMSE','Mean signed error','Spearman correlation','Concordance correlation','Within two points, \\%','Severity-band agreement, \\%'}
    s+=''.join(name+' & '+' & '.join(vals)+' \\\\ \n' for name,vals in rows if name in core)+ending.replace('Calibration regresses observed EASI on predicted EASI; ideal intercept/slope are 0/1. ', '')
    (out/'table2.tex').write_text(s)
    rates=pd.read_csv(base/'endpoint_rates.csv').sort_values(['week','target','treatment'])
    rates=rates[~(rates.week.eq(48)&rates.target.isin(['physician','patient','overall']))]
    s=r'\begin{longtable}{llrrr}'+'\n'+r'\caption{Exact target attainment, observed denominators and Wilson 95\% confidence intervals.}\\'+'\n'+r'\toprule Week and target & Initial treatment & Achieved/N & Rate, \% & 95\% CI \\ \midrule\endfirsthead'+'\n'+r'\toprule Week and target & Initial treatment & Achieved/N & Rate, \% & 95\% CI \\ \midrule\endhead'+'\n'
    for r in rates.itertuples():
        short={'dupilumab':'D','abrocitinib_100mg':'A100','abrocitinib_200mg':'A200'}[r.treatment]
        s+=f'W{r.week}: '+tex(TARGET.get(r.target,r.target))+f' & {short} & {r.k}/{r.n_evaluable} & {f(100*r.rate,1)} & {f(100*r.ci_low,1)}--{f(100*r.ci_high,1)} \\\\ \n'
    s+=r'\bottomrule\end{longtable}'+'\n'
    (out/'endpoint_rates.tex').write_text(s)
    bounds=pd.read_csv(diagnostic/'headline_missing_outcome_bounds.csv')
    support=pd.read_csv(diagnostic/'endpoint_available_ESS.csv')
    bounds=bounds.merge(support[['week','target','treatment','ESS']],on=['week','target','treatment'],validate='one_to_one')
    s=r'\begin{center}\small\begin{tabular}{llrrrr}\toprule Target & Initial group & Success/evaluable & Initial N & ESS & Missing-outcome bounds, \% \\ \midrule'+'\n'
    for r in bounds.itertuples():
        arm={'dupilumab':'D','abrocitinib_100mg':'A100','abrocitinib_200mg':'A200'}[r.treatment]
        target='W2 itch' if r.week==2 else 'W24 combined'
        s+=f'{target} & {arm} & {r.k}/{r.n_evaluable} & {r.n_initial} & {r.ESS:.1f} & {100*r.lower_missing_bound:.1f}--{100*r.upper_missing_bound:.1f}'+r' \\'+'\n'
    s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
    (out/'headline_target_support.tex').write_text(s)
    # Machine-readable source index includes every aggregate table, not only plotted results.
    rows=[]
    for folder in ['clinical','model','model/auxiliary_targets','model/rounding','regional','redesign_20261018','revision2_20261018']:
        d=root/'results/tables/update_20261017'/folder
        for p in sorted(d.glob('*.csv')):
            df=pd.read_csv(p)
            if 'private' in p.name or any(c in df for c in ['patient_uid','patient_id','source_record_id','record_id']):continue
            rows.append({'section':folder,'file':str(p.relative_to(root)),'rows':len(df),'columns':'; '.join(df.columns)})
    pd.DataFrame(rows).to_csv(root/'manuscript/source_data/index.csv',index=False)
    s=r'\begin{longtable}{p{20mm}p{115mm}r}\toprule Section & Source file & Rows \\ \midrule\endhead'+'\n'
    for r in rows:s+=tex({'model/auxiliary_targets':'Other targets','model/rounding':'Rounding','redesign_20261018':'Diagnostics','revision2_20261018':'Revision audit'}.get(r['section'],r['section']))+' & '+tex(Path(r['file']).name)+' & '+str(r['rows'])+' \\\\ \n'
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
        text=re.sub(r'^% INSERT: (Figure[1-4]|Table[12])\s*$',insert,text,flags=re.MULTILINE)
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
    app=preamble('Supplementary information: '+TITLE)+r'''\maketitle
\renewcommand{\thetable}{S\arabic{table}}
\section*{How to use this supplement}
This file follows the clinical, molecular and model sequence of the investigators' September figure outline. It includes every specified clinical visit, the complete 23-marker panel, all three paired change windows, actual treatment pathways, and model stress tests. The attached aggregate CSV collection retains estimates that are not displayed as figures, including negative findings, sparse cells, centre and assay-batch analyses and all multiplicity-adjusted tests. Missing cells denote unavailable or non-estimable results, never zero effects.

D denotes dupilumab; A100/A200 denote initial abrocitinib dose. Treatment colours are fixed throughout. In the three-visit molecular display, grey tones and distinct marker shapes distinguish W0, W12 and W24. A change of $-1$ on the log$_2$ scale is a halving of concentration; it is not a correlation of $-1$. In the mutually exclusive response-depth table, the $[90,\infty)$ EASI band means at least 90\% clinical improvement, and its molecular value is the mean paired log$_2$ change among those patients.
\section*{Interpretation of model measures}
MAE is the mean size of an error without its sign, in EASI points. RMSE gives larger errors more weight. Spearman correlation measures ordering; it can be high even when numerical predictions are wrong. Concordance additionally reflects numerical agreement. The percentages within one or two points count predictions meeting those error limits. Severity-band agreement, sensitivity and specificity depend on the declared categories and should not be compared across different cut-offs. None of these alone establishes replacement of clinical examination.
\section*{Published equations and matched numerical controls}
Published p-EASI equations were applied unchanged to the identical eligible records, with raw and 0--72-clipped predictions retained alongside BSA-only and molecular-only controls. The primary nested procedure and fixed-structure comparisons are labelled separately in the numerical source tables. The published equations use serum alone, and local IL2RA is an unbridged proxy for sIL-2R; these local-application checks do not establish superiority of the BSA--serum model.\citep{thijs2017,thijs2019,bakker2020}
\section*{Source and interpretation checks}
The earlier 92.1\% W2 100 mg rate counted one unresolved initial-dose record; the current 91.9\% is 34/37. Adult DLQI uses age at least 18 throughout the new clinical analysis. IGA includes recorded grade 5; the instrument version remains unresolved. Eosinophil W4 data are absent in the dupilumab records. Exact assay/examination-date matching, validated quantification limits, blinded repeat measurements and an independent external cohort are not available. The eDelphi summary is a source framework, not a newly verified consensus experiment. Safety tables describe documentation coverage rather than adjudicated adverse-event incidence. Outcomes proposed in the protocol but not observed cannot be reconstructed from the available files.

The analysis starts from versioned curated clinical, serum and exposure snapshots. Independent reruns verified the resulting tables and the primary model. This is not a claim that every manually curated field can be regenerated from the original Excel documents without adjudication. The package distinguishes aggregate rebuilding from restricted-data model refitting.
\clearpage\setcounter{table}{0}\renewcommand{\theHtable}{S.\arabic{table}}\section*{Supplementary Table S1: exact target rates}
D, dupilumab; A100/A200, initial abrocitinib 100/200 mg. N is the endpoint-specific evaluable denominator. Unknown outcomes are excluded from N and retained in the source table with missing-outcome bounds. W48 component thresholds remain exploratory; W48 composite targets are omitted from the active table because they were not specified in the manuscript's target framework. Historical CSV rows are retained. Exact P values, risk differences, confidence intervals and correction families accompany \texttt{endpoint\_tests.csv} and \texttt{endpoint\_weighted\_contrasts.csv}.
{\setlength{\parskip}{0pt}\small\input{tables/endpoint_rates}}
'''
    if (m/'sections/supplementary_methods.tex').exists():app+=section_text(m/'sections/supplementary_methods.tex')+'\n'
    app+='\\clearpage\\setcounter{table}{1}\\section*{Supplementary Table S2: complete model metrics}\n\\input{tables/model_full_metrics}\n'
    if (m/'sections/auxiliary_targets.tex').exists():app+=section_text(m/'sections/auxiliary_targets.tex')+'\n'
    app+=r'\bibliographystyle{unsrtnat}\bibliography{references}'+'\n'
    groups = [
        ('Cohort structure and treatment comparability', [39,40,41],
         'Parallel patient populations, sequential exclusions, baseline balance, common support and actual recorded exposure.'),
        ('Clinical outcomes and stage-specific targets', [1,2,3,4,15,16,17,21,23,24,25,26],
         'Complete specified visits, observed and adjusted trajectories, IGA composition, response targets, dose-reduction pathways and all treatment contrasts.'),
        ('The full molecular panel and regional disease', [5,6,7,8,9,10,11,12,13,14,18,19,27,28,29,30,31,32,33,34],
         'All 23 markers, three distinct change windows, baseline distributions, paired clinical associations, response-depth groups and regional analyses.'),
        ('Equation validation and other targets', [22,35,37,38,42],
         'Transport and severity checks, separately fitted PP-NRS/DLQI/IGA targets, and response classification. Matched numerical controls remain in source tables.')]
    groups=[(title,[supplement_number(i) for i in ids],description) for title,ids,description in groups]
    supplement={r['id']:r for r in catalog if 'S' in r['id']}
    expected={f'FigureS{i}' for _,ids,_ in groups for i in ids}
    if expected != set(supplement):raise ValueError('Supplement navigation omits or duplicates figure IDs.')
    contents='\\section*{Supplementary figure guide}\n'
    for title,ids,description in groups:
        contents+='\\textbf{'+tex(title)+'}. '+tex(description)+'\\par\n'
        contents+='; '.join('\\hyperref[fig:FigureS'+str(i)+']{Fig. S'+str(i)+'}' for i in ids)+'.\\par\n'
    app=app.replace('\\section*{How to use this supplement}',contents+'\\section*{How to use this supplement}',1)
    for group_number,(title,ids,description) in enumerate(groups,1):
        for position,i in enumerate(ids):
            page=figure_page(supplement[f'FigureS{i}'])
            if position==0:
                page=page.replace('\\centering\n','\\centering\n\\pdfbookmark[1]{'+tex(title)+'}{group'+str(group_number)+'}\n{\\large\\bfseries '+tex(title)+'}\\par\\vspace{5pt}\n',1)
            app+=page
    app+='\\clearpage\\section*{Complete source-table catalogue}\nEvery aggregate table is included, including unplotted and negative results. Figure-specific dependencies are recorded in \\texttt{figure\\_catalog.json}.\n'
    app+='\\renewcommand{\\theHtable}{index.\\arabic{table}}\n{\\setlength{\\parskip}{0pt}\\renewcommand{\\arraystretch}{0.95}\\small\\input{tables/source_index}}\n'
    app+='\\end{document}\n';(m/'appendix.tex').write_text(app)
    book=preamble('Complete figure book: clinical course, serum markers and EASI estimation')+r'\maketitle'+ '\n'
    book+='This review copy contains the four main figures followed by all supplementary figures. Captions specify populations, uncertainty and what each comparison can establish. Complete numerical results are supplied as CSV source tables.\n'
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
