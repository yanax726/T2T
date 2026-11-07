"""Framework, fixed-patient molecular courses and laboratory sensitivity displays."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import palette
from .clinical_flow_20261025 import (
    _text, _panel, _box, _arrow, _line, INK, MUTED, RULE,
    CLINICAL_FILL, PATIENT_FILL, RESEARCH_FILL,
)


def framework(root, base):
    source = Path(root) / 'config/framework_figure_20261107.json'
    content = json.loads(source.read_text())
    fig = plt.figure(figsize=(183 / 25.4, 178 / 25.4))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.5, 1.0],
                          left=.06, right=.98, bottom=.045, top=.945, hspace=.28)
    a, b, c = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    _panel(a, 'a', 'Framework development')
    _panel(b, 'b', 'Targets evaluated through week 24')
    _panel(c, 'c', 'Clinical review and exploratory measurement')

    # Two source streams converge before the sequential review stages.
    for y, title in [(.60, 'Evidence review\nMeasures and frameworks'),
                     (.10, 'Stakeholder input\nClinicians, nurses, patients')]:
        _box(a, .0, y, .235, .32, face=CLINICAL_FILL)
        _text(a, .1175, y + .16, title, size=7.0)
    _line(a, [.235, .265, .265], [.76, .76, .51])
    _line(a, [.235, .265, .265], [.26, .26, .51])
    _arrow(a, (.265, .51), (.30, .51))
    stages = [(.30, 'Round 1', 'Select measures'),
              (.485, 'Feedback', 'Review and discuss'),
              (.67, 'Round 2', 'Targets and timing'),
              (.855, 'Final review', 'Local framework')]
    for i, (x, title, detail) in enumerate(stages):
        _box(a, x, .29, .145, .44, face=CLINICAL_FILL)
        _text(a, x + .0725, .60, title, size=7.0, weight='bold')
        _text(a, x + .0725, .43, detail.replace(' and ', '\nand '), size=6.5)
        if i < len(stages) - 1:
            _arrow(a, (x + .145, .51), (stages[i+1][0], .51))

    x0, width, gap = .18, .188, .022
    columns = [x0 + i * (width + gap) for i in range(4)]
    for row, ypos in zip(content['assessment']['rows'], [.60, .18]):
        _text(b, .01, ypos, row['label'].replace(' ', '\n', 1),
              ha='left', size=7.2, weight='bold')
    for j, (x, week) in enumerate(zip(columns, ['W2', 'W4', 'W12', 'W24'])):
        _text(b, x + width/2, .96, week, size=8, weight='bold')
        _box(b, x, .43, width, .34, face=CLINICAL_FILL)
        _box(b, x, .02, width, .30, face=PATIENT_FILL)
        clinical = content['assessment']['rows'][0]['cells'][j]
        patient = content['assessment']['rows'][1]['cells'][j].replace('DLQI', 'adult DLQI')
        _text(b, x+width/2, .60, clinical, size=7.0,
              color=MUTED if j == 0 else INK)
        _text(b, x+width/2, .17, patient, size=7.0)
        if j:
            _text(b, x+width/2, .375, 'AND', size=6.5, weight='bold')
    b.plot([x0, .998], [.835, .835], color=RULE, lw=.75)

    # Decision review is distinct from the post hoc model evaluation.
    for x, w, title in [(.0, .27, 'Review target attainment'),
                        (.36, .28, 'Safety, tolerability\nand patient priorities'),
                        (.73, .27, 'Continue or adjust care\nand reassess')]:
        _box(c, x, .56, w, .38, face=CLINICAL_FILL)
        _text(c, x+w/2, .75, title, size=7)
    _arrow(c, (.27, .75), (.36, .75))
    _arrow(c, (.64, .75), (.73, .75))
    _box(c, .0, .02, 1, .35, face=RESEARCH_FILL)
    _text(c, .02, .28, 'Research assessment', ha='left', size=7, weight='bold')
    _text(c, .235, .125, 'BSA + CCL17, CCL26, IL-5', size=7)
    _arrow(c, (.47, .125), (.54, .125))
    _text(c, .77, .125, 'E-EASI: estimated EASI', size=7, weight='bold')
    base.save(fig, 'Figure1', 'Development, clinical targets and the E-EASI research assessment',
              'a, The project materials describe evidence review, clinician/nurse/patient input, two questionnaire rounds, '
              'feedback and steering-group review. The planned agreement threshold was 75%; 49 returned questionnaires were reported. '
              'Original ballots and round-specific participation and agreement are unavailable, so the diagram reconstructs the reported '
              'process rather than verifying consensus. b, Operational targets for W2/W4/W12/W24. At W2 only PP-NRS ≤4 is required. '
              'From W4 onward at least one criterion in each domain is required; “or” is within a domain and “AND” joins domains. '
              'EASI-50/75 means at least 50%/75% improvement from positive pretreatment EASI; IGA decrease ≥2 is relative to baseline. '
              'IGA 0/1 and PP-NRS 0/1 mean recorded values 0 or 1. DLQI is analysed in adults aged ≥18. Missing components remain unknown '
              'unless another observed component resolves the domain. The recorded IGA instrument and implemented framework version '
              'require investigator confirmation. Longer-term planned targets and other published frameworks are compared in Table S13. '
              'c, Target review accompanies safety and patient priorities. E-EASI is a separate, post hoc hybrid clinical–molecular '
              'research estimate; it did not determine treatment changes. This observational cohort does not establish that the framework '
              'or the research estimate improves management.', [source, Path(root)/'config/clinical_framework.json'], root)


def clinical_composition(root, base):
    source = Path(root)/'results/tables/update_20261017/clinical/IGA_composition.csv'
    data = pd.read_csv(source)
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 84/25.4))
    fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.78, wspace=.25)
    colors = palette.sequence('recorded_iga_0_to_5')
    for j, (ax, arm, title) in enumerate(zip(axes, base.ARMS, base.LABELS)):
        d = data[data.treatment.eq(arm) & data.week.isin([0, 4, 12, 24])]
        bottom = np.zeros(4)
        for grade, color in enumerate(colors):
            z = d[d.category.eq(grade)].set_index('week').reindex([0, 4, 12, 24])
            bars = ax.bar(np.arange(4), 100*z.proportion, bottom=bottom,
                          color=color, width=.68, edgecolor='white', linewidth=.35)
            palette.bind_artist(fig, 'recorded_iga_0_to_5.'+str(grade), bars)
            bottom += 100*z.proportion.to_numpy()
        ns = d.groupby('week').n.first().reindex([0, 4, 12, 24]).astype(int)
        ax.set(title=title, xticks=range(4), xticklabels=[f'W{w}\nn={n}' for w,n in ns.items()],
               ylim=(0,100), yticks=[0,25,50,75,100])
        ax.annotate('abc'[j], (0,1), xycoords='axes fraction', xytext=(-12,12),
                    textcoords='offset points', fontsize=8, weight='bold')
    axes[0].set_ylabel('Patients in each recorded IGA grade (%)')
    fig.legend(handles=[Patch(facecolor=c, label=f'IGA {i}') for i,c in enumerate(colors)],
               loc='upper center', bbox_to_anchor=(.55,.99), ncol=6,
               frameon=False, handlelength=1.1, columnspacing=1.4)
    base.save(fig, 'Figure3', 'Recorded IGA distributions during follow-up',
              'a–c, Distribution of recorded IGA grades within each initial treatment group at W0, W4, W12 and W24. '
              'Each bar totals 100% of the patients with an observed grade; n below the bar is that visit’s denominator. '
              'Colour identifies grade, not treatment. Patient sets may differ between visits. Grade-specific counts and proportions '
              'are provided in the source table. The local 0–5 coding is retained; correspondence to a validated IGA instrument '
              'has not been confirmed. This figure describes disease activity, not dose-reduction success.', [source], root)


def molecular_response(root, base):
    source = Path(root)/'results/tables/update_20261107/labs/molecular_three_visit_changes.csv'
    data = pd.read_csv(source)
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 210/25.4))
    fig.subplots_adjust(left=.16, right=.945, top=.91, bottom=.075, wspace=.40)
    valid = data[data.n.ge(5) & data.week.gt(0)]
    lower = np.floor(min(0, valid[['effect','ci_low']].min().min()) * 2)/2
    upper = np.ceil(max(0, valid[['effect','ci_high']].max().max()) * 2)/2
    # Every finite descriptive point is retained even when no CI is warranted.
    lower = min(lower, np.floor(data.effect.min()*2)/2)
    upper = max(upper, np.ceil(data.effect.max()*2)/2)
    span = upper-lower
    offsets = [-.25, 0, .25]
    handles=[]
    for w in [0,12,24]:
        color = palette.register(fig, f'visit.W{w}')
        handles.append(Line2D([0],[0], marker='o', lw=0, color=color, ms=4, label=f'W{w}'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55,.995),
               ncol=3, frameon=False, columnspacing=2.4)
    for j,(ax,arm,title) in enumerate(zip(axes,base.ARMS,base.LABELS)):
        for offset,w in zip(offsets,[0,12,24]):
            z = data[data.treatment.eq(arm) & data.week.eq(w)].set_index('marker').reindex(base.MARKERS)
            color=palette.color(f'visit.W{w}')
            y=np.arange(len(base.MARKERS))+offset
            estimated=z.n.ge(5)&z.effect.notna()
            if w:
                ax.errorbar(z.loc[estimated,'effect'], y[estimated],
                            xerr=np.maximum(0, np.array([z.loc[estimated,'effect']-z.loc[estimated,'ci_low'],
                                                        z.loc[estimated,'ci_high']-z.loc[estimated,'effect']])),
                            fmt='o', color=color, ms=3, lw=.9, capsize=1.2, zorder=3)
            else:
                ax.plot(z.loc[z.effect.notna(),'effect'],y[z.effect.notna()],
                        'o',color=color,ms=2.7,zorder=3)
            sparse=z.n.between(1,4)&z.effect.notna()
            if w:
                ax.plot(z.loc[sparse,'effect'],y[sparse], 'o',color=color,ms=3,
                        markerfacecolor='white',markeredgewidth=.8,zorder=3)
                for i,r in enumerate(z.itertuples()):
                    if np.isfinite(r.q) and r.q < .05 and r.n >=5:
                        ax.text(1.055 if w==12 else 1.19,i,'*',color=color,fontsize=10,
                                va='center',ha='center',transform=ax.get_yaxis_transform())
        ax.axvline(0,color='#BBC6CB',lw=.65,zorder=0)
        ax.grid(axis='y',color='#E8EEF0',lw=.5,zorder=0)
        ax.set(yticks=np.arange(23),yticklabels=base.MARKERS if j==0 else ['']*23,
               ylim=(22.55,-.6),xlim=(lower-.04*span,upper+.04*span),
               xlabel='Log₂ change from W0',title=title)
        ax.tick_params(axis='y',length=0)
        ax.annotate('abc'[j],(0,1),xycoords='axes fraction',xytext=(-12,11),
                    textcoords='offset points',fontsize=8,weight='bold')
        for xpos,w in [(1.055,12),(1.19,24)]:
            ax.text(xpos,1.01,f'W{w}',transform=ax.transAxes,ha='center',va='bottom',
                    fontsize=6,color=INK)
    base.save(fig,'Figure4','Molecular changes across three visits in the same patients',
              'a–c, All 23 serum markers by initial treatment. For each marker and treatment, the same patients contribute at '
              'W0, W12 and W24; patient sets can differ between markers and treatments. W0 is the reference value zero, not zero '
              'concentration. Coloured points show mean within-person log₂ changes from W0; horizontal bars are 95% t confidence '
              'intervals for n≥5. A change of −1 means a geometric paired concentration ratio of 0.5. Hollow later-visit points '
              'denote n<5 without inference; blank rows have no complete three-visit set. The two star columns labelled W12 and W24 '
              'refer to the corresponding visit versus W0: *BH q<0.05 across all 138 planned marker-by-treatment-by-contrast tests, '
              'using paired sign-flip tests and retaining non-estimable hypotheses in the family. These stars are calculated in the '
              'displayed fixed patient sets, not copied from analyses of all available pairs. No baseline covariate adjustment is '
              'applied to this descriptive figure. Between-treatment adjusted comparisons and all available-pair windows are '
              'reported separately in the supplement. Baseline concentrations are given in Table S12; exact n, changes, intervals '
              'and q values are in the source table.',[source],root)


def laboratories(root, base, finish):
    root=Path(root)
    folder=root/'results/tables/update_20261017/clinical'
    obs=pd.read_csv(folder/'clinical_observed.csv')
    adj=pd.read_csv(folder/'clinical_trajectory_adjusted.csv')
    source=root/'results/tables/update_20261107/labs/eos_endpoint_complete_trajectory.csv'
    eos=pd.read_csv(source)
    from . import supplement_redesign_20261018 as p
    fig,axes=p.grid(2,2,155,left=.13,top=.88,bottom=.12,hspace=.70,wspace=.40)
    p.legend(fig)
    for var,oa,aa in [('ige_numeric',axes[0],axes[1]),('eo_count_numeric',None,axes[3])]:
        weeks=[0,12,24] if var=='ige_numeric' else [0,4,12,24]
        for j,arm in enumerate(base.ARMS):
            color=palette.register(fig,'treatment.'+arm)
            if oa is not None:
                z=obs[obs.target.eq(var)&obs.treatment.eq(arm)].set_index('week').reindex(weeks)
                p.line(oa,weeks,z['median'],z.q25,z.q75,color)
            z=adj[adj.target.eq(var)&adj.treatment.eq(arm)].set_index('week').reindex(weeks[1:])
            v=np.expm1(z[['adjusted_mean_scale','ci_low','ci_high']].to_numpy())
            p.line(aa,np.array(weeks[1:])+(j-1)*.2,v[:,0],v[:,1],v[:,2],color)
        for ax,x in [(oa,weeks),(aa,weeks[1:])]:
            if ax is None:continue
            ax.set(xticks=x,xlabel='Week',ylabel=p.DISPLAY[var],xlim=(-1,25))
            ax.grid(axis='y',color='#E8EEF0',lw=.4)
            if var=='ige_numeric':
                ax.set(yscale='log',ylim=(25,6500),yticks=[50,100,500,1000,5000],
                       yticklabels=['50','100','500','1000','5000'])
                ax.minorticks_off()
    axes[0].set_title('Observed median and IQR')
    axes[1].set_title('Baseline-adjusted estimate and 95% CI')
    ax=axes[2]
    z=eos[eos.population.eq('paired_W0_W12')].sort_values('week')
    color=palette.color('treatment.dupilumab')
    ax.plot(z.week,z['median'],color=color,lw=1.3)
    for r in z.itertuples():
        ax.errorbar(r.week,r.median,yerr=[[r.median-r.median_ci_low],[r.median_ci_high-r.median]],
                    fmt='o',color=color,ms=4,markerfacecolor='white' if r.week==4 else color,
                    lw=1.0,capsize=2)
    ax.annotate('W4 estimate',xy=(4,float(z.loc[z.week.eq(4),'median'].iloc[0])),
                xytext=(7,.53),fontsize=7,arrowprops={'arrowstyle':'-','color':color,'lw':.7})
    ax.set(title='dupi: paired endpoints (n=15)\nW4 estimated',xlabel='Week',
           ylabel='Eosinophils (recorded units)',xticks=[0,4,12],xlim=(-1,13),ylim=(0,.60))
    ax.grid(axis='y',color='#E8EEF0',lw=.4)
    axes[3].set_title('Baseline-adjusted estimate and 95% CI')
    finish(fig,4,'IgE follow-up and the unobserved W4 eosinophil value',
           'a, Available-case IgE medians and interquartile ranges; patient sets vary by visit. The vertical axis is logarithmic '
           'with labels in recorded units. b,d, Generalized-overlap-weighted estimates from log(1+x) models adjusted for baseline '
           'outcome, age, sex, BMI and centre, with patient-cluster 95% confidence intervals conditional on estimated weights. '
           'Back-transformed estimates are not arithmetic means. There are no observed dupi W4 eosinophil measurements, so d '
           'contains no estimate for that treatment/visit cell. c, A separate fixed set of 15 dupi patients with W0 and W12 '
           'eosinophil measurements: filled points are observed endpoint medians; the hollow W4 point is an assumption-based '
           'estimate. Each patient’s log(1+x) eosinophil value is linearly interpolated one-third of the way from W0 to W12, '
           'back-transformed, then summarized. Bars are patient-bootstrap 95% intervals for the median, conditional on this '
           'temporal assumption. The W4 estimate is 0.281 (0.183–0.448); it is not an observed result, validation outcome or '
           'treatment-comparison input. Original-scale interpolation, alternative time shapes and a seven-patient complete '
           'three-visit analysis are supplied separately. Initial treatment groups retain subsequent regimen changes.',
           [folder/'clinical_observed.csv',folder/'clinical_trajectory_adjusted.csv',source],root,base)


def eos_sensitivity(root,base,finish):
    root=Path(root)
    folder=root/'results/tables/update_20261107/labs'
    obs=pd.read_csv(root/'results/tables/update_20261017/clinical/clinical_observed.csv')
    estimates=pd.read_csv(folder/'eos_w4_estimates.csv')
    from . import supplement_redesign_20261018 as p
    fig,axes=p.grid(1,2,103,left=.12,top=.82,bottom=.21,wspace=.65)
    p.legend(fig)
    ax=axes[0]
    for j,arm in enumerate(base.ARMS):
        z=obs[obs.target.eq('eo_count_numeric')&obs.treatment.eq(arm)&obs.week.isin([0,4,12,24])]
        z=z[z.n.gt(0)].sort_values('week')
        p.line(ax,z.week,z['median'],z.q25,z.q75,p.COLORS[j])
    ax.set(title='Observed eosinophils',xlabel='Week',ylabel='Eosinophils (recorded units)',
           xticks=[0,4,12,24],xlim=(-1,25))
    choices=[('paired_W0_W12','log1p_linear_time','Linear log(1+x) · n=15'),
             ('paired_W0_W12','linear_original_scale','Linear original scale · n=15'),
             ('paired_W0_W12','early_log_time_log1p','Earlier change · n=15'),
             ('paired_W0_W12','late_quadratic_time_log1p','Later change · n=15'),
             ('complete_W0_W12_W24','log1p_linear_time','Three-visit subset · n=7')]
    ax=axes[1];color=palette.register(fig,'treatment.dupilumab')
    for i,(population,method,label) in enumerate(choices):
        r=estimates[estimates.population.eq(population)&estimates.method.eq(method)].iloc[0]
        ax.errorbar(r['median'],i,xerr=[[r['median']-r.median_ci_low],[r.median_ci_high-r['median']]],
                    fmt='o',color=color,ms=4,lw=1.1,capsize=2)
    ax.set(title='dupi: W4 estimation sensitivity',xlabel='Estimated median eosinophils (recorded units)',
           yticks=range(5),yticklabels=[r[2] for r in choices],ylim=(4.5,-.5),xlim=(0,.6))
    ax.tick_params(axis='y',labelsize=6.5)
    finish(fig,46,'Observed eosinophil data and sensitivity of the W4 estimate',
           'a, Available-case medians and interquartile ranges at observed visits. The dupi line connects W0 to W12 because '
           'no W4 measurement exists; crossing the W4 position does not create an observation. No imputed value enters this '
           'panel. b, Estimated dupi W4 medians and patient-bootstrap 95% intervals under different temporal assumptions. '
           'The first four estimates use the same 15 W0/W12 pairs. Earlier/later change replaces the linear time fraction '
           'with log(1+t)/log(13) or (t/12)² at t=4. The last row restricts to seven patients also observed at W24, illustrating '
           'selection sensitivity, not a better estimate. Intervals condition on the stated assumptions and do not cover '
           'all uncertainty about the unmeasured W4 distribution. These are post hoc descriptive estimates, not multiple '
           'imputation under an identified treatment-specific W4 model. All original observations remain unchanged.',
           [root/'results/tables/update_20261017/clinical/clinical_observed.csv',folder/'eos_w4_estimates.csv',
            folder/'eos_interpolation_backchecks.csv'],root,base)
