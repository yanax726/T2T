"""Supplementary tables generated directly from the November analyses."""
from pathlib import Path
import pandas as pd


def start(caption,label,columns,header):
    return ('{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
            '\\begin{longtable}{'+columns+'}\n\\caption{'+caption+'}\\label{'+label+'}\\\\\n'
            '\\toprule '+header+' \\\\ \\midrule\\endfirsthead\n'
            '\\multicolumn{'+str(len(header.split(' & ')))+'}{l}{\\tablename~\\thetable: continued}\\\\\n'
            '\\toprule '+header+' \\\\ \\midrule\\endhead\n')


def end(note):
    return '\\bottomrule\\end{longtable}\n'+note+'\n}\n'


def run(root):
    from .build import tex
    root=Path(root);out=root/'manuscript/tables'
    folder=root/'results/tables/update_20261107'
    d=pd.read_csv(folder/'labs/baseline_marker_display.csv')
    s=start('Available pretreatment concentrations of all 23 serum markers.',
            'tab:baseline_markers','p{22mm}p{45mm}p{45mm}p{45mm}',
            'Marker & Dupilumab & Abrocitinib 100 mg & Abrocitinib 200 mg')
    for r in d.to_dict('records'):
        values=[]
        for arm in ['dupilumab','abrocitinib_100mg','abrocitinib_200mg']:
            values.append(tex(r[arm+'_median_q1_q3'])+'\\newline '+
                          f'n={r[arm+"_n"]}; outside range={r[arm+"_outside_quantification"]}')
        s+=tex(r['marker'])+' & '+' & '.join(values)+' \\\\[4pt]\n'
    s+=end('Values are median (Q1--Q3) in recorded pg/mL, calculated on individual original-scale concentrations. '
           'n counts retained positive numeric results; outside range counts reported below- or above-quantification results excluded '
           'from these summaries. Other missing or nonnumeric entries are not included in either count. The table therefore '
           'describes the retained numeric sample, not an imputed full-group distribution. Positive numeric export values do not establish analytical validity or verified clinical quantification limits. Quartiles use linear interpolation. '
           'Complete detection-status counts are in \\texttt{baseline\\_marker\\_summary.csv}.')
    (out/'baseline_markers.tex').write_text(s)

    d=pd.read_csv(folder/'framework/framework_comparison.csv')
    citations={'International T2T (2021)':'debruin2021t2t','Chinese systemic-treatment guidance (2022)':'zhao2026t2t',
               'Canadian T2T (2023)':'yeung2023t2t','AHEAD / MDA (2024)':'silverberg2024ahead',
               'STRIVE AD (2026)':'strive2026','IEC disease-control states (2026)':'iec2026control'}
    s=start('Published treatment-target frameworks and the local operational schedule.',
            'tab:framework_comparison','p{27mm}p{43mm}p{51mm}p{38mm}',
            'Framework / timing & Decision rule & Clinical and patient targets & Use in this cohort')
    for r in d.to_dict('records'):
        name=tex(r['Framework'])+'\\newline '+tex(r['Timing'])
        # Sources are linked directly, avoiding dependence on changing bibliography keys.
        source=str(r['Source']).split(';')[0]
        if source.startswith('http'):name+='\\newline \\href{'+source+'}{Source}'
        s+=name+' & '+tex(r['Decision_logic'])+'\\newline '+tex(r['Additional_requirements'])+' & '+tex(r['Clinical_signs'])+'\\newline '+tex(r['Patient_reported_targets'])+' & '+tex(r['Applicability_to_this_cohort'])+' \\\\[6pt]\n'
    s+=end('MDA, minimal disease activity; LDA/vLDA, low/very low disease activity; PtGA, patient global assessment; '
           'POEM, Patient-Oriented Eczema Measure. These frameworks are not interchangeable. AHEAD MDA depends on patient-selected '
           'priorities. IEC complete control and remission require duration criteria in addition to a low score. The 2022 Chinese '
           'guidance thresholds were checked against the 2026 publication by the guidance authors because the original Chinese '
           'full text was not available. The local source framework also plans months 9/12: EASI-90, IGA 0/1 or EASI $\\leq$3 '
           'in the clinical domain, and PP-NRS 0/1 or DLQI $\\leq$4 in the patient domain. Those later targets were not assessed '
           'in the 24-week cohort. Source-specific verification details are in \\texttt{framework\\_comparison.csv}.')
    (out/'framework_comparison.tex').write_text(s)

    d=pd.read_csv(folder/'model/overall_metrics.csv')
    d=d[d.cohort.eq('core')]
    models=[('three_ridge','Three markers: ridge'),('three_selected','Three markers: nested candidate selection'),
            ('full19_ridge','19 markers: ridge'),('full19_paired_rbf','19 markers: paired nonlinear'),
            ('full19_ordinal_rrr','19 markers: IGA auxiliary'),('full19_selected','19 markers: inner selection')]
    s=start('Molecular-only model extensions evaluated in the same patients and folds.',
            'tab:molecular_extensions','p{46mm}rrrrr',
            'Model & Week & n & MAE (95\\% CI) & Spearman & CCC')
    for model,label in models:
        for r in d[d.model.eq(model)].sort_values('week').itertuples():
            s+=f'{tex(label)} & {r.week} & {r.n_patients} & {r.MAE:.2f} ({r.MAE_lo:.2f}--{r.MAE_hi:.2f}) & {r.rho:.3f} & {r.CCC:.3f} \\\\ \n'
        s+='\\addlinespace\n'
    s+=end('All rows use the same legacy core evaluation set, complete for BSA and the three-marker panel, with 152/89/75 patients at W0/W12/W24. Other serum values are imputed within training folds. '
           'Only molecular concentrations are inputs at prediction. The three-marker set is CCL17, CCL26 and IL-5; the '
           '19-marker set follows the prespecified assay-quality screen. Training uses W0/W12, with the existing two repeats '
           'of five patient-level outer folds and three inner folds. Imputation, scaling and tuning are training-fold operations. '
           'The candidate set permits ridge, paired nonlinear constraints (including a zero-constraint setting) and IGA-assisted reduced-rank estimation; the inner '
           'criterion selects the candidate without using held-out scores. IGA is an auxiliary training outcome, not a prediction '
           'input. W24 is previously inspected same-cohort temporal application. Patient-bootstrap intervals condition on the '
           'fitted predictions and do not include all model-selection uncertainty. The extensions are exploratory after prior '
           'inspection of this cohort; they do not replace the frozen E-EASI equation. Complete RMSE, bias, calibration, severity '
           'strata, paired-change results, expanded-training populations and paired comparisons are in the aggregate CSV tables.')
    (out/'molecular_extensions.tex').write_text(s)

    ext=pd.read_csv(root/'results/tables/external_validation_20261101/molecular_metrics.csv')
    ext=ext[ext.stratum.eq('all')&ext.prediction_scale.eq('clip')&ext.scope.isin(['W0','W16'])]
    local=pd.read_csv(root/'results/tables/external_validation_20261101/local_matched_metrics.csv')
    s=start('Unchanged serum equations applied to matched local and OMIX002850 samples.',
            'tab:serum_transport','p{25mm}p{37mm}rrrr',
            'Dataset / visit & Equation & n & MAE (95\\% CI) & Spearman & CCC')
    for scope in ['W0','W16']:
        for model,label in [('core_all__serum3','Local three-marker serum'),('peasi_2017_precise','Published p-EASI')]:
            z=ext[ext.scope.eq(scope)&ext.model.eq(model)].set_index('metric')
            m=z.loc['MAE']
            s+=f'OMIX {scope} & {label} & {int(m.n_patients)} & {m.estimate:.2f} ({m.ci_low:.2f}--{m.ci_high:.2f}) & {z.loc["Spearman","estimate"]:.3f} & {z.loc["CCC","estimate"]:.3f} \\\\ \n'
    s+='\\addlinespace\n'
    for week in [0,12,24]:
        for model,label in [('serum3','Local three-marker serum'),('peasi_precise_clip','Published p-EASI')]:
            r=local[local.week.eq(week)&local.model.eq(model)].iloc[0]
            s+=f'Local W{week} & {label} & {int(r.n_patients)} & {r.MAE:.2f} ({r.MAE_lo:.2f}--{r.MAE_hi:.2f}) & {r.rho:.3f} & {r.CCC:.3f} \\\\ \n'
    s+=end('The serum-only model is a separately fitted CCL17/CCL26/IL-5 equation, not the hybrid E-EASI equation with BSA '
           'deleted. Local evaluations use held-out predictions and identical eligible records; published p-EASI is applied '
           'unchanged. OMIX estimates use the frozen serum equation without refitting; 125 pretreatment and 60 week-16 AD '
           'samples are available. Healthy controls are excluded. Both calculations are clipped to 0--72 here; raw outputs '
           'and the rounded 2020 p-EASI version are retained in source tables. OMIX lacks BSA, was previously inspected, and '
           'may share patients with the local source centre; participant nonoverlap and assay equivalence are unverified. '
           'This is a transport benchmark, not untouched independent validation of E-EASI. The local IL2RA assay is an '
           'unbridged proxy for published sIL-2R. The three-marker formula did not establish superiority over p-EASI.')
    (out/'serum_transport.tex').write_text(s)

    d=pd.concat([pd.read_csv(folder/'labs/joint_low_activity_rates.csv'),
                 pd.read_csv(folder/'labs/joint_low_activity_rates_overall.csv')],ignore_index=True)
    s=start('Single-visit low and very low disease activity using the IEC EASI-and-itch branch.',
            'tab:joint_low_activity','p{24mm}rp{20mm}p{50mm}p{50mm}',
            'Initial treatment & Week & Joint n / full N & LDA & vLDA')
    names={'dupilumab':'Dupilumab','abrocitinib_100mg':'Abro 100 mg',
           'abrocitinib_200mg':'Abro 200 mg','all_initial_treatments':'All treatments'}
    for arm,label in names.items():
        for week in [0,4,12,24]:
            z=d[d.treatment.eq(arm)&d.week.eq(week)].set_index('state')
            cells=[]
            for state in ['LDA','vLDA']:
                r=z.loc[state]
                cells.append(f'{int(r.n_success)}/{int(r.n_joint_observed)}: {100*r.proportion:.1f}\\% '
                             f'({100*r.ci_low:.1f}--{100*r.ci_high:.1f})\\newline '
                             f'Full-group bounds: {100*r.full_cohort_lower_bound:.1f}--{100*r.full_cohort_upper_bound:.1f}\\%')
            s+=f'{label} & {week} & {int(r.n_joint_observed)} / {int(r.N_initial)} & '+' & '.join(cells)+' \\\\[4pt]\n'
        s+='\\addlinespace\n'
    s+=end('LDA requires EASI $\\leq$7 AND PP-NRS $\\leq$4; vLDA requires EASI $\\leq$3 AND PP-NRS $\\leq$1. '
           'Percentages and Wilson 95\\% confidence intervals use patients with both observed scores at the same nominal visit. '
           'Joint n is that observed denominator; full N is the initial treatment population. Full-group bounds assign every '
           'unresolved AND state to failure or success, while any observed failing component remains a failure. These bounds '
           'are missing-outcome scenarios, not confidence intervals. They can differ from the observed-joint rate because '
           'a partially measured patient can already be known to fail. The analysis is post hoc, with no imputed clinical '
           'scores or new between-treatment significance tests. It evaluates a single-visit state, not individualized AHEAD '
           'MDA, six-month continuous control, remission or a causal treatment effect. All components and unknown counts '
           'are in \\texttt{joint\\_low\\_activity\\_rates.csv} and \\texttt{joint\\_low\\_activity\\_rates\\_overall.csv}.')
    (out/'joint_low_activity.tex').write_text(s)
