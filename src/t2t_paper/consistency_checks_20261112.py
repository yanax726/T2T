"""Check the current figure catalogue against its distributed aggregate data."""
from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import fitz
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / 'reports/update_20261112/consistency_update/audit'
OUT.mkdir(parents=True, exist_ok=True)
checks = []


def record(name, passed, detail):
    checks.append(dict(check=name, passed=bool(passed), detail=detail))


def csv(path):
    return pd.read_csv(ROOT / path)


def bh_check(path, keys):
    d = csv(path)
    errors = []
    for _, ix in d.groupby(keys, dropna=False).groups.items():
        x = d.loc[ix].dropna(subset=['p'])
        if len(x):
            errors.extend(abs(multipletests(x.p, method='fdr_bh')[1]-x.q))
    maximum = max(errors, default=0.)
    record(path + ' BH replay', maximum < 1e-12,
           dict(maximum_absolute_error=float(maximum), finite_tests=len(errors), groups=keys))


catalog = json.loads((ROOT/'manuscript/figure_catalog.json').read_text())
record('Live catalogue size', len(catalog) == 26,
       '5 main and 21 supplementary figures; checked current content, not archived numbering')
inventory = []
for figure in catalog:
    missing = [s for s in figure['sources'] if not (ROOT/s).is_file()]
    assets = [ext for ext in ['pdf', 'png', 'svg']
              if not (ROOT/'manuscript/figures'/f"{figure['id']}.{ext}").is_file()]
    record(figure['id'] + ' source and assets', not missing and not assets,
           dict(missing_sources=missing, missing_assets=assets))
    inventory.append(dict(figure=figure['id'], title=figure['title'],
                          sources='; '.join(figure['sources']),
                          source_files_exist=not missing,
                          current_export_triplet_exists=not assets))

source_files=sorted({s for r in catalog for s in r['sources']})
source_files += ['results/tables/update_20261112/itch_prediction/'+name+'.csv'
                 for name in ['metrics','paired_differences','repeat_metrics','null_reference']]
for source in source_files:
    if Path(source).suffix != '.csv':
        continue
    d=csv(source)
    for left,right in [('ci_low','ci_high'),('MAE_lo','MAE_hi'),
                       ('log_ci_low','log_ci_high'),('ratio_low','ratio_high')]:
        if left in d and right in d:
            finite=d[[left,right]].dropna()
            record(source+' ordered '+left+'/'+right,
                   finite[left].le(finite[right]+1e-12).all(),len(finite))
    for column in ['p','q','p_holm','q_exploratory']:
        if column in d:
            finite=d[column].dropna()
            record(source+' valid '+column,
                   finite.between(0,1).all(),len(finite))

clinical = 'results/tables/update_20261017/clinical/'
rates = csv(clinical+'endpoint_rates.csv')
record('Figure2 rates reproduce counts',
       np.allclose(rates.rate, rates.k/rates.n_evaluable, equal_nan=True), len(rates))
lo, hi = proportion_confint(rates.k, rates.n_evaluable, method='wilson')
record('Figure2 Wilson intervals reproduce',
       np.allclose(lo,rates.ci_low,equal_nan=True) and np.allclose(hi,rates.ci_high,equal_nan=True), len(rates))
record('Figure2 endpoint missingness arithmetic',
       ((rates.n_evaluable+rates.not_evaluable)==rates.n_initial).all()
       and (rates.k <= rates.n_evaluable).all(), len(rates))
iga = csv(clinical+'IGA_composition.csv')
record('Figure3 proportions reproduce counts', np.allclose(iga.proportion,iga.k/iga.n), len(iga))
record('Figure3 six grades add to denominator',
       iga.groupby(['treatment','week']).k.sum().equals(
           iga.groupby(['treatment','week']).n.first()), '12 treatment/visit bars')
for filename in ['endpoint_tests.csv','endpoint_weighted_contrasts.csv']:
    d=csv(clinical+filename)
    differences=[]
    for _,h in d[d.family.eq('three_pairwise')].groupby(['week','target']):
        if len(h):
            replay=multipletests(h.p,method='holm')[1]
            assert np.array_equal(np.isnan(replay),h.p_holm.isna())
            differences.extend(abs(replay[np.isfinite(replay)]-h.loc[h.p_holm.notna(),'p_holm']))
    maximum=max(differences,default=0.)
    record('S7-S10 '+filename+' Holm replay',maximum<1e-12,
           dict(maximum_absolute_error=float(maximum),tests=len(differences)))

balance=csv('results/tables/update_20261017/redesign_20261018/baseline_balance_full.csv')
record('S2 absolute SMD preserves signed source',
       np.allclose(balance.absolute_SMD,balance.SMD.abs(),equal_nan=True),len(balance))
bins=csv('results/tables/update_20261017/redesign_20261018/propensity_overlap_bins.csv')
record('S2 propensity histogram fractions reproduce counts',
       np.allclose(bins.fraction,bins.n/bins.n_group,equal_nan=True),len(bins))
record('S2 each propensity distribution sums to one',
       np.allclose(bins.groupby(['probability_arm','treatment']).fraction.sum(),1),
       'Three propensity axes by three actual initial arms')
ess=csv('results/tables/update_20261017/redesign_20261018/analysis_effective_sample_sizes.csv')
finite_ess=ess[ess.ESS.notna()]
record('S2 effective sample sizes bounded by observed patients',
       finite_ess.ESS.ge(0).all() and (finite_ess.ESS<=finite_ess.n_patients+1e-10).all(),
       dict(estimable=len(finite_ess),not_estimable=int(ess.ESS.isna().sum())))

tr = csv('results/tables/update_20261107/labs/molecular_three_visit_changes.csv')
ix = tr.week.ne(0)
expected = multipletests(tr.loc[ix,'p'].fillna(1),method='fdr_bh')[1]
record('Figure4 BH replay', np.allclose(expected[tr.loc[ix,'p'].notna()],
       tr.loc[ix & tr.p.notna(),'q']), dict(planned=int(ix.sum()),estimable=int(tr.p.notna().sum())))
record('Figure4 fixed counts and zero baseline',
       tr.groupby(['marker','treatment']).n.nunique().eq(1).all()
       and tr[tr.week.eq(0)].effect.dropna().eq(0).all(), '69 marker-treatment sets')
record('Figure4 small groups have no test or CI',
       tr[tr.n.lt(5)][['p','ci_low','ci_high']].isna().all().all(), 'n<5')
record('Figure4 displayed significance agrees with q',
       (tr.significant == tr.q.lt(.05)).all(), 'All saved star flags')

bh_check(clinical+'molecular_paired_changes.csv',['family'])
bh_check(clinical+'molecular_adjusted_contrasts.csv',['scope','analysis'])
bh_check(clinical+'molecular_clinical_change_correlations.csv',['family'])
bh_check(clinical+'molecular_regional_baseline_correlations.csv',['family'])
bh_check('results/tables/update_20261017/regional/regional_treatment_contrasts.csv',['analysis','outcome'])
pairs = csv(clinical+'molecular_paired_changes.csv')
record('S11-S13 small groups have no inference',
       pairs[pairs.n.lt(5)][['p','q','ci_low','ci_high']].isna().all().all(), 'All three windows')
corr = csv(clinical+'molecular_clinical_change_correlations.csv')
positive = corr[corr.q.lt(.05)]
record('S17 includes exactly six qualifying associations', len(positive)==6, len(positive))
record('S17 no W12-W24 corrected association',
       not ((positive.start==12)&(positive.end==24)).any(), 'Full association family, not selected-row correction')
reg = csv('results/tables/update_20261017/regional/regional_treatment_contrasts.csv')
record('S18 no adjusted regional contrast passes FDR',
       not reg[reg.analysis.eq('adjusted_HC3')].q.lt(.05).any(), 'All adjusted regional contrasts')
burden=csv('results/tables/update_20261017/regional/regional_normalized_burden.csv')
record('S18 regional normalized burden is within theoretical bounds',
       burden['mean'].dropna().between(0,1).all(),len(burden))

levels = csv('results/tables/update_20261112/labs/longitudinal_levels.csv')
coverage = csv('results/tables/update_20261112/labs/coverage_by_centre.csv')
record('S4-S5 interpolated cells are exactly unmeasured cells',
       levels[levels.kind.eq('assumption_based_interpolation')].n_observed_cell.eq(0).all()
       and levels[levels.kind.ne('assumption_based_interpolation')].n_observed_cell.gt(0).all(),
       'W4 IgE for all arms, W4 dupi eosinophils only')
record('S4-S5 model counts match declared coverage',
       levels[(levels.outcome=='IgE')&(levels.scope=='all_163')].n_observations.eq(258).all()
       and levels[(levels.outcome=='IgE')&(levels.scope=='all_163')].n_patients.eq(147).all()
       and levels[(levels.outcome=='EOS')&(levels.scope=='all_163')].n_observations.eq(320).all()
       and levels[(levels.outcome=='EOS')&(levels.scope=='all_163')].n_patients.eq(138).all(),
       'IgE 258 values/147 patients; EOS 320 values/138 patients')
for family, keys in [('between_treatment_changes.csv',['outcome','scope','model'])]:
    d=csv('results/tables/update_20261112/labs/'+family)
    errors=[]
    for _,h in d.groupby(keys):
        errors.extend(abs(multipletests(h.p,method='holm')[1]-h.p_holm))
    record('S5 Holm replay', max(errors)<1e-12, dict(maximum_absolute_error=float(max(errors))))

modelcells=csv('manuscript/source_data/model_count_cells.csv')
record('S19 cell counts retain all patients',
       modelcells.groupby('visit_week')['count'].sum().to_dict()=={0:152,12:89,24:75}, '152/89/75')
changecells=csv('manuscript/source_data/monitoring_count_cells.csv')
record('S20 cell counts retain all pairs',
       changecells.groupby('window')['count'].sum().to_dict()=={'W0_W12':85,'W0_W24':67,'W12_W24':55}, '85/67/55')
conf=csv('results/tables/update_20261017/redesign_20261018/monitoring_confusion_repeat0.csv')
record('S21 row fractions reproduce counts',np.allclose(conf.row_proportion,conf['count']/conf.n_observed_class),len(conf))
record('S21 all displayed repeats have valid paired baselines',conf.n_excluded.eq(0).all(),'All six panels')
record('S21 pair counts consistent across displayed cells',
       conf.groupby(['window','endpoint'])['count'].sum().to_dict()==
       {(w,e):n for w,n in [('W0_W12',85),('W0_W24',67)] for e in ['EASI50','EASI75','EASI90']},'Six confusion matrices')
metrics=csv('results/tables/update_20261017/model/overall_metrics.csv')
metrics=metrics[metrics.model.eq('unified_selected')].sort_values('week')
record('Main model MAE numbers round to primary source',
       list(np.round(metrics.MAE,2))==[3.19,1.30,1.19], list(metrics.MAE))
record('Main model Spearman numbers round to primary source',
       list(np.round(metrics.rho,3))==[.918,.927,.904],list(metrics.rho))
ecdf=csv('manuscript/source_data/error_ecdf.csv')
record('Figure5 d error curves agree with within-two percentages',
       list(np.round(100*ecdf[ecdf.absolute_error.eq(2)].sort_values('week').proportion,1))==[41.8,84.3,82.0],
       'Exact error threshold 2 evaluated separately by visit')

ablation=csv('results/tables/update_20261025/ablation/main_input_comparison.csv')
record('Figure5 a-c use identical comparison cohort sizes',
       ablation.groupby('week').n_patients.nunique().eq(1).all()
       and ablation.groupby('week').n_patients.first().to_dict()=={0:151,12:89,24:75},
       'Ablation 151/89/75 differs from primary 152/89/75 at baseline only; Figure5 caption states both')
record('Primary and monitoring population distinction',
       metrics.n_patients.to_list()==[152,89,75]
       and changecells.groupby('window')['count'].sum().to_dict()=={'W0_W12':85,'W0_W24':67,'W12_W24':55},
       'Individual prediction n differs from paired baseline/follow-up monitoring n')
triplets=csv(clinical+'clinical_fixed_triplets.csv')
record('Old fixed IgE subset contains 30 patients',
       triplets[triplets.target.eq('ige_numeric')].groupby('treatment').n.first().sum()==30,
       'Main text contrasts 30 complete-triplet patients with all-available model 147; not interchangeable')
eos=csv('results/tables/update_20261107/labs/eos_w4_estimates.csv')
record('S6 EOS sensitivity populations are 15 and seven patients',
       eos.groupby('population').n.first().to_dict()=={'complete_W0_W12_W24':7,'paired_W0_W12':15},
       'S6 median interpolation differs from S4 all-available adjusted mean interpolation')

tex=(ROOT/'manuscript/main.tex').read_text()
appendix=(ROOT/'manuscript/appendix.tex').read_text()
itch_folder='results/tables/update_20261112/itch_prediction/'
itch=csv(itch_folder+'metrics.csv')
itch_repeats=csv(itch_folder+'repeat_metrics.csv')
itch_table=(ROOT/'manuscript/tables/itch_prediction.tex').read_text()
record('S18 uses three fixed matched input sets',
       set(itch.model)=={'three_markers','clinical','clinical_three_markers'}
       and itch.groupby(['population','evaluation']).model.nunique().eq(3).all(),
       'Three serum markers, clinical reference and combined model')
record('S18 populations and events match source counts',
       itch[itch.population.eq('complete_three_markers')].n.eq(102).all()
       and itch[itch.population.eq('complete_three_markers')].events.eq(88).all()
       and itch[itch.population.ne('complete_three_markers')].n.eq(105).all()
       and itch[itch.population.ne('complete_three_markers')].events.eq(89).all(),
       '102/88/14 complete versus105/89/16 predictor-imputation sensitivity')
itch_mean=itch[itch.evaluation.eq('mean_repetition')]
itch_err=[]
for metric in ['AUROC','AUPRC','Brier','log_loss','mean_prediction']:
    a=itch_mean[itch_mean.metric.eq(metric)].set_index(['population','model']).estimate
    b=itch_repeats.groupby(['population','model'])[metric].mean().reindex(a.index)
    itch_err.extend(abs(a-b))
record('S18 mean-repetition metrics reproduce stored repetitions',
       max(itch_err)<1e-12,dict(maximum_absolute_error=float(max(itch_err))))
record('S18 all AUROC values and intervals match table text',
       all(f'{r.estimate:.3f} ({r.ci_low:.3f}--{r.ci_high:.3f})' in itch_table
           for r in itch_mean[itch_mean.metric.eq('AUROC')].itertuples()),'Both populations; mean-repetition estimates displayed')
record('S18 appears in current supplementary guide and methods',
       'Tables S1--S20' in appendix and r'\label{tab:itch_prediction}' in itch_table
       and r'\input{tables/itch_prediction}' in appendix
       and 'Exploratory baseline prediction of W12 itch improvement' in appendix,
       'Newly fitted prognostic models are separate from the frozen E-EASI equation')
itch_delta=csv(itch_folder+'paired_differences.csv')
matched_delta=itch_delta[itch_delta.population.eq('complete_three_markers')
                       &itch_delta.evaluation.eq('mean_repetition')
                       &itch_delta.model.eq('clinical_three_markers')]
record('S18 paired AUROC difference matches primary estimates',
       np.isclose(matched_delta[matched_delta.metric.eq('AUROC')].difference.iloc[0],
           itch_mean[itch_mean.population.eq('complete_three_markers')
                     &itch_mean.model.eq('clinical_three_markers')&itch_mean.metric.eq('AUROC')].estimate.iloc[0]
           -itch_mean[itch_mean.population.eq('complete_three_markers')
                      &itch_mean.model.eq('clinical')&itch_mean.metric.eq('AUROC')].estimate.iloc[0]),
       'Combined minus matched clinical reference, not historical111-patient comparator')
record('S18 retains uncertain molecular-increment conclusion',
       (matched_delta[matched_delta.metric.isin(['AUROC','Brier'])].ci_low.lt(0)
        &matched_delta[matched_delta.metric.isin(['AUROC','Brier'])].ci_high.gt(0)).all()
       and 'did not establish molecular added predictive value' in itch_table,
       'Both paired difference intervals include zero')
record('S18 labels conditional intervals and predictor-only imputation',
       'without refitting' in itch_table and 'response labels are never imputed' in itch_table,
       'No imputed response truth or external-validation claim')
record('Every current figure has a corresponding caption',
       all(('\\label{fig:'+r['id']+'}') in
           (tex+appendix) for r in catalog),
       'Five main captions and 21 supplementary captions')
catalog_by_id={r['id']:r for r in catalog}
record('Main text and Figure4 caption state complete-three-visit population',
       'all three visits' in tex and 'all three visits' in catalog_by_id['Figure4']['caption']
       and 'subset' in catalog_by_id['Figure4']['caption']
       and all(str(n) in tex for n in [138,124,193,207]),
       'Three-visit population and all-available-pair families distinguished')
for name in ['FigureS11','FigureS12','FigureS13']:
    record(name+' caption says third visit not required',
           'third visit is not required' in catalog_by_id[name]['caption'],
           'Interval-specific paired data, not main complete-triplet subset')
record('Appendix says third visit not required for pair analyses',
       'No measurement at the third visit was required' in appendix
       and appendix.count('third visit is not required')==3,
       'Supplementary methods and all three interval captions')
for name in ['FigureS14','FigureS15','FigureS16']:
    record(name+' caption states complete adjusted-comparison correction family',
           all(term in catalog_by_id[name]['caption'] for term in
               ['240 estimable','276 planned','three windows','four contrasts','separate families']),
           'All-centre primary: 240 estimable of 276 planned; centre/batch sensitivity separate')
record('Appendix captions retain 240/276 adjusted family',
       appendix.count('240 estimable tests among 276 planned')==3,
       'All three between-treatment interval captions')
support=(ROOT/'manuscript/tables/laboratory_support.tex').read_text()
record('Table S9 observed counts distinguish hollow W4 illustrations',
       'No W4 IgE or dupi eosinophil point was imputed' not in support
       and 'conditional temporal interpolations' in support
       and 'not included in these counts or tests' in support,
       'No interpolation is recast as an observed assay or treatment-test input')
for name,note in [('Figure4','Complete W0/W12/W24 records'),
                  ('FigureS11','All available W0/W12 pairs'),
                  ('FigureS12','All available W12/W24 pairs'),
                  ('FigureS13','All available W0/W24 pairs')]:
    with fitz.open(ROOT/'manuscript/figures'/f'{name}.pdf') as pdf:
        words=' '.join(page.get_text() for page in pdf)
    record(name+' exported PDF shows patient-population note',
           note in words and 'stars: adjusted q < 0.05' in words,
           'Text extracted from exported vector PDF, not inferred from source code')
record('Incompatible old script and export receipt no longer in current manuscript root',
       not (ROOT/'manuscript/script_zh.pdf').exists()
       and not (ROOT/'manuscript/figure_deck.export.json').exists(),
       'Historical files belong in identified archive, not current manuscript directory')
record('Current captions do not refer to a removed footer',
       not any(re.search(r'\bfooter\b',r['caption'],re.I) for r in catalog),
       'Figure-level prose was removed by the common title-free style; sample sizes and ESS must point to source tables')
style_review=[]
for figure in catalog:
    path=ROOT/'results/audit/update_20261112/figures'/f"{figure['id']}.style.json"
    style=json.loads(path.read_text()) if path.exists() else {}
    removed=style.get('figure_prose_removed_from_graphic',[])
    footer_claim=bool(re.search(r'\bfooter\b',figure['caption'],re.I))
    style_review.append(dict(figure=figure['id'],
                             removed_figure_prose=' | '.join(removed),
                             removed_axes_titles=' | '.join(style.get('axes_titles_removed',[])),
                             caption_refers_to_removed_footer=footer_claim,
                             review='Revise footer location claim' if footer_claim else
                             'No caption claim found that removed figure-level prose remains on the graphic'))
pd.DataFrame(style_review).to_csv(OUT/'removed_prose_caption_review.csv',index=False)

hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in source_files}
pd.DataFrame(inventory).to_csv(OUT/'figure_source_inventory.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'numerical_checks.csv',index=False)
(OUT/'aggregate_source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
stale=[str(p.relative_to(ROOT)) for p in (ROOT/'manuscript').glob('*')
       if p.name in ['script_zh.pdf','figure_deck.export.json']]
summary=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
             numerical_checks=len(checks),passed=sum(x['passed'] for x in checks),
             failed=[x for x in checks if not x['passed']],figures=len(catalog),
             source_files=len(source_files),stale_current_directory_candidates=stale,
             boundary='Read-only aggregate and reporting consistency audit; no new model fitting or drug-effect identification.')
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if summary['failed']:
    raise SystemExit(1)
