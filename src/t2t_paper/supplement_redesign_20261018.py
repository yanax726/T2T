"""Supplementary figures grouped by clinical and measurement questions."""
from pathlib import Path
import json
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

ARMS = ['dupilumab', 'abrocitinib_100mg', 'abrocitinib_200mg']
LABELS = ['Dupilumab', 'Abrocitinib 100 mg', 'Abrocitinib 200 mg']
COLORS = ['#0072B2', '#009E73', '#D55E00']
TIME_COLORS = ['#999999', '#555555', '#111111']
TIME_MARKERS = ['^', 'o', 's']
TREATMENT_MARKERS = ['o', 's', '^']
TIMES = [0, 12, 24]
CURRENT, CONTROL = '#333333', '#BDBDBD'
CMAP = LinearSegmentedColormap.from_list('signed_effect', ['#7B3294', '#FAFAF7', '#C29124'])
CMAP.set_bad('#E6E8EA')
MARKERS = ['IL-4','IL-5','IL-13','IL-31','TSLP','IL-33','CCL17','CCL18','IFN-gamma','IL-12p70','IL-18','TNF-alpha','IL-21','IL-6','IL-17A','IL-22','IL-10','IL2RA','CCL26','Periostin','CCL11','IL-36beta','IFN-beta']
TARGETS = ['PP_NRS_le4','EASI50','IGA_drop2','DLQI_le10','EASI75','EASI90','IGA_01','PP_NRS_01','DLQI_le4','EASI_le5']
TARGET_LABELS = ['PP-NRS ≤4','EASI50','IGA decrease ≥2','Adult DLQI ≤10','EASI75','EASI90','IGA 0/1','PP-NRS 0/1','Adult DLQI ≤4','EASI ≤5']
DISPLAY = {'easi_total':'EASI','pp_nrs':'PP-NRS','dlqi':'Adult DLQI','bsa':'BSA (%)','scorad':'SCORAD','ige_numeric':'IgE (recorded units)','eo_count_numeric':'Eosinophils\n(recorded units)'}
MODEL = 'unified_selected'
DESIGNS = []


def grid(rows, cols, height=165, left=.11, top=.9, bottom=.12, wspace=.42, hspace=.65):
    fig, axes = plt.subplots(rows, cols, figsize=(183/25.4, height/25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=.975, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    for i, ax in enumerate(axes.flat):
        ax.annotate(string.ascii_lowercase[i], (0,1), xycoords='axes fraction', xytext=(-17,11), textcoords='offset points', weight='bold', fontsize=8, annotation_clip=False)
        ax.tick_params(length=2.5, width=.6)
    return fig, list(axes.flat)


def legend(fig, labels=LABELS, colors=None, styles=None, y=.985, markers=None):
    colors = COLORS if colors is None else colors
    styles = styles or ['-']*len(labels)
    markers = markers or (TREATMENT_MARKERS if labels == LABELS else ['o']*len(labels))
    handles = [Line2D([0],[0], color=c, marker=m, ms=3, lw=.9, ls=s, label=l) for l,c,s,m in zip(labels,colors,styles,markers)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,y), ncol=min(3,len(labels)), frameon=False)


def line(ax, x, y, lo, hi, color, label=None, style='-', marker=None):
    marker = marker or (TREATMENT_MARKERS[COLORS.index(color)] if color in COLORS else 'o')
    x,y,lo,hi = [np.asarray(v,float) for v in (x,y,lo,hi)]
    ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt=marker,linestyle=style,color=color,label=label,ms=3,lw=.8,capsize=1.7)


def forest(ax, data, y, color, key='effect', marker='o', filled=True):
    ok = data[key].notna() & data.ci_low.notna() & data.ci_high.notna()
    ax.errorbar(data.loc[ok,key], np.asarray(y)[ok], xerr=[np.maximum(0,data.loc[ok,key]-data.loc[ok,'ci_low']),np.maximum(0,data.loc[ok,'ci_high']-data.loc[ok,key])],fmt=marker,color=color,ms=2.7,capsize=1.5,lw=.7,markerfacecolor=color if filled else 'white')
    missing = data[key].notna() & ~ok
    ax.scatter(data.loc[missing,key],np.asarray(y)[missing],s=10,facecolors='none',edgecolors=color,lw=.7,marker=marker)


def heat(ax, values, rowlabels, collabels, title, lim=1, text=None, counts=None):
    values=np.asarray(values,float)
    im=ax.imshow(values,cmap=CMAP,vmin=-lim,vmax=lim,aspect='auto',interpolation='none')
    ax.set(yticks=range(len(rowlabels)),yticklabels=rowlabels,xticks=range(len(collabels)),xticklabels=collabels,title=title)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value=values[i,j]
            label=(text[i][j] if text is not None else '')
            if not np.isfinite(value):
                label='×' + (' '+label if label else '')
            if counts is not None and 0 < np.asarray(counts)[i,j] < 5:
                ax.add_patch(Rectangle((j-.49,i-.49),.98,.98,fill=False,edgecolor='#24323C',lw=.8,linestyle=':'))
            if label:
                rgb=np.array(CMAP((value+lim)/(2*lim))[:3]) if np.isfinite(value) else np.ones(3)
                linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                luminance=float(linear @ np.array([.2126,.7152,.0722]))
                ax.text(j,i,label,ha='center',va='center',fontsize=6.3,color='white' if luminance<.18 else '#18232B')
    return im


def count_strip(ax, weeks, data, key='n', txkey='treatment', y=-.30, rowgap=.12):
    """Aligned per-visit denominators, never pooled across outcomes or groups."""
    for k,w in enumerate(weeks):
        ax.text((k+.5)/len(weeks),y+rowgap,f'W{w}',transform=ax.transAxes,ha='center',va='center',fontsize=6.3)
    for j,tx in enumerate(ARMS):
        z=data[data[txkey].eq(tx)].set_index('week')
        ax.text(-.02,y-j*rowgap,['D','A100','A200'][j],transform=ax.transAxes,ha='right',va='center',fontsize=6.3,color='#24323C')
        for k,w in enumerate(weeks):
            value=z.loc[w,key] if w in z.index else np.nan
            ax.text((k+.5)/len(weeks),y-j*rowgap,str(int(value)) if pd.notna(value) else '—',transform=ax.transAxes,ha='center',va='center',fontsize=6.3,color='#24323C')


def window_header(fig, label, y=.98):
    fig.text(.55,y,label,ha='center',va='top',fontsize=7.2,weight='bold')


def q_column(ax, data, y):
    ax.text(1.02,.99,'q / NE',transform=ax.transAxes,ha='left',va='bottom',fontsize=6.3)
    for yi,row in zip(y,data.itertuples()):
        if np.isfinite(row.q) and row.q<.05:
            ax.text(1.02,yi,'*',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=7)
        if not np.isfinite(row.effect):
            ax.text(1.02,yi,'NE',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=6.3,color='#59646A')


def colorbar(fig, im, label, lim=1):
    cax=fig.axes[0].inset_axes([.30,.075,.46,.016], transform=fig.transFigure);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-lim,0,lim]);cb.set_label(label,fontsize=6.5);cax.tick_params(labelsize=6.3,length=2)


def finish(fig, number, title, caption, sources, root, base, role, x, y, sample):
    from matplotlib.text import Text
    for txt in fig.findobj(match=Text):
        if txt.get_fontsize()<6.3: txt.set_fontsize(6.3)
    for ax in fig.axes:
        if ax.get_title(): ax.title.set_fontsize(7.2)
    name=f'FigureS{number}'
    base.save(fig,name,title,caption,[str(s) for s in sources],root)
    DESIGNS.append(dict(id=name,question=title,redesign=role,x=x,y=y,sample=sample,sources=';'.join(str(Path(s).relative_to(root)) for s in sources),height_mm=round(fig.get_figheight()*25.4,1),palette='Fixed treatment, time and comparator semantics; purple-white-gold signed effects'))


def clinical_courses(root, base):
    folder=root/'results/tables/update_20261017/clinical';source=folder/'endpoint_rates.csv';d=pd.read_csv(source)
    fig,axes=grid(5,2,220,left=.12,top=.945,bottom=.14,hspace=1.22,wspace=.42)
    legend(fig,y=.995)
    for ax,key,title in zip(axes,TARGETS,TARGET_LABELS):
        weeks=[2,4,12,24] if key=='PP_NRS_le4' else [4,12,24]
        subset=d[d.target.eq(key)&d.week.isin(weeks)]
        for j,tx in enumerate(ARMS):
            z=subset[subset.treatment.eq(tx)].sort_values('week')
            line(ax,z.week+(j-1)*.22,z.rate*100,z.ci_low*100,z.ci_high*100,COLORS[j])
        ax.set(title=title,ylabel='Attainment (%)',xticks=weeks,ylim=(-3,103),yticks=[0,50,100]);ax.grid(axis='y',color='#E7EBED',lw=.4)
        count_strip(ax,weeks,subset,key='n_evaluable',y=-.48,rowgap=.13)
    fig.text(.55,.012,'Horizontal axis: week · Below each axis: evaluable n by D / A100 / A200 · Points: rate and Wilson 95% CI',ha='center',fontsize=6.3)
    finish(fig,1,'Ten clinical targets over their specified visits','Observed proportions with Wilson 95% confidence intervals, by initial treatment; the count strips give the available-outcome denominator at every point, in D/A100/A200 order. Only PP-NRS ≤4 is shown at W2. Adult DLQI requires age ≥18; EASI90 requires positive observed baseline EASI. Missing outcomes are unknown, not failures. Connecting lines describe observed visit summaries and do not estimate fixed-dose causal effects.',[source],root,base,'Ten targets including EASI90; shared scales and explicit per-point evaluable counts','Specified week','Observed proportion; Wilson 95% CI','Endpoint-specific available patients')
    obs=pd.read_csv(folder/'clinical_observed.csv');adj=pd.read_csv(folder/'clinical_trajectory_adjusted.csv')
    overall_source=root/'manuscript/source_data/clinical_overall.csv'
    overall=pd.read_csv(overall_source)
    for number,variables in [(2,['easi_total','pp_nrs','dlqi']),(3,['bsa','scorad']),(4,['ige_numeric','eo_count_numeric'])]:
        fig,axes=grid(len(variables),3,210 if len(variables)==3 else 165,top=.90,bottom=.19,hspace=1.25)
        legend(fig)
        for row,var in enumerate(variables):
            z=overall[overall.target.eq(var)].sort_values('week');line(axes[row*3],z.week,z['mean'],z.ci_low,z.ci_high,CONTROL)
            for j,tx in enumerate(ARMS):
                z=obs[obs.target.eq(var)&obs.treatment.eq(tx)].sort_values('week');line(axes[row*3+1],z.week,z['median'],z.q25,z.q75,COLORS[j])
                z=adj[adj.target.eq(var)&adj.treatment.eq(tx)].sort_values('week');v=z[['adjusted_mean_scale','ci_low','ci_high']].to_numpy()
                if var in ['ige_numeric','eo_count_numeric']:v=np.expm1(v)
                line(axes[row*3+2],z.week+(j-1)*.2,v[:,0],v[:,1],v[:,2],COLORS[j])
            weeks=[0,12,24] if var=='ige_numeric' else ([0,2,4,12,24] if var=='pp_nrs' else [0,4,12,24])
            for col,title in enumerate(['Overall: mean + 95% CI','By treatment: median + IQR','Adjusted estimate + 95% CI']):
                ax=axes[row*3+col];ax.set(title=title if row==0 else None,ylabel=DISPLAY[var],xticks=weeks if col<2 else weeks[1:]);ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(axis='y',lw=.35,color='#E7EBED')
                if col==0:
                    zz=overall[overall.target.eq(var)].set_index('week')
                    ax.text(-.02,-.41,'n',transform=ax.transAxes,ha='right',fontsize=6.3)
                    for k,w in enumerate(weeks):
                        ax.text((k+.5)/len(weeks),-.28,f'W{w}',transform=ax.transAxes,ha='center',fontsize=6.3)
                        if w in zz.index: ax.text((k+.5)/len(weeks),-.41,str(int(zz.loc[w,'n'])),transform=ax.transAxes,ha='center',fontsize=6.3)
                else:
                    count_strip(ax,weeks if col==1 else weeks[1:],obs[obs.target.eq(var)] if col==1 else adj[adj.target.eq(var)],key='n' if col==1 else 'n_observed',y=-.41,rowgap=.13)
        fig.text(.55,.02,'Below axes: available n (D / A100 / A200 for treatment columns); adjusted counts are observed outcomes',ha='center',fontsize=6.3)
        finish(fig,number,'Observed and baseline-adjusted '+', '.join(DISPLAY[x].replace('\n',' ') for x in variables),'Each row is one outcome. Left, overall available-case mean and 95% t confidence interval, exactly combined from group moments. Middle, initial-treatment-specific median and interquartile range. Right, overlap-weighted adjusted trajectories and patient-cluster robust 95% confidence intervals, controlling baseline outcome, age, sex, BMI and centre. IgE/eosinophils use log(1+x) models and back-transformation; these adjusted estimates are not arithmetic means. Visit composition may change. Only investigator-specified visits are shown; no causal current-dose effect is claimed. Exact sample sizes and contrasts accompany the source tables.',[overall_source,folder/'clinical_observed.csv',folder/'clinical_trajectory_adjusted.csv'],root,base,'Rows are outcomes; columns separate three different estimands','Specified week','Outcome in original recorded scale','Available observations per visit; adult DLQI')


def molecular_panels(root,base):
    folder=root/'results/tables/update_20261017/clinical';d=pd.read_csv(folder/'molecular_paired_changes.csv');windows=[(0,12),(12,24),(0,24)]
    for k,(start,end) in enumerate(windows):
        fig,axes=grid(1,3,165,left=.17,top=.91,bottom=.13,wspace=.62)
        fig.subplots_adjust(right=.92)
        window_header(fig,f'W{start} → W{end}: paired molecular change')
        subset=d[d.start.eq(start)&d.end.eq(end)]
        limits=[min(-.3,subset[['effect','ci_low']].min().min()-.2),max(.3,subset[['effect','ci_high']].max().max()+.2)]
        for j,(ax,tx,title) in enumerate(zip(axes,ARMS,LABELS)):
            z=subset[subset.treatment.eq(tx)].set_index('marker').reindex(MARKERS);forest(ax,z,np.arange(23),COLORS[j]);ax.axvline(0,color='#B2BBC1',lw=.6)
            labs=[f'{m}  [{int(n)}]' for m,n in zip(MARKERS,z.n)] if j==0 else [f'[{int(n)}]' for n in z.n]
            ax.set(yticks=range(23),yticklabels=labs,ylim=(22.6,-.6),xlim=limits,xlabel='Later − earlier log₂ concentration',title=title);ax.grid(axis='y',color='#EDF0F2',lw=.35)
            q_column(ax,z,np.arange(23))
        fig.text(.52,.025,f'W{start} → W{end} paired interval · [n] = complete pairs · * FDR q<.05 · NE: not estimable',ha='center',fontsize=6.5)
        finish(fig,5+k,f'All molecular changes within W{start} → W{end}','Mean within-patient log₂ concentration changes and 95% t confidence intervals. Negative values mean a decline: −1 equals a geometric paired ratio of 0.5. Each marker/treatment uses its available pairs, shown as [n]; sets may differ between windows. Hollow points have fewer than five pairs and no inferential interval; rows without an estimable mean are marked NE (not estimable) in the outside status column. Stars indicate BH q<0.05 across the full 207 marker-by-treatment-by-window family. All 23 measured markers are included.',[folder/'molecular_paired_changes.csv'],root,base,'All markers in shared-scale treatment forests, n and FDR visible','Mean paired log₂ change; t 95% CI','All 23 markers','Marker-specific complete pairs')
    contrasts=pd.read_csv(folder/'molecular_adjusted_contrasts.csv')
    comps=['D - A100','D - A200','A200 - A100','D - pooled A']
    for k,(start,end) in enumerate(windows):
        fig,axes=grid(1,4,165,left=.16,top=.90,bottom=.13,wspace=.36)
        fig.subplots_adjust(right=.92)
        window_header(fig,f'W{start} → W{end}: baseline-adjusted change contrasts')
        subset=contrasts[contrasts.start.eq(start)&contrasts.end.eq(end)&contrasts.scope.eq('all_centers')&contrasts.analysis.eq('primary')]
        bound=np.ceil(np.nanmax(np.abs(subset[['ci_low','ci_high']].to_numpy()))*2)/2
        for j,(ax,comp) in enumerate(zip(axes,comps)):
            z=subset[subset.comparison.eq(comp)].set_index('marker').reindex(MARKERS);forest(ax,z,np.arange(23),CURRENT);ax.axvline(0,color='#AAB2B8',lw=.7)
            ax.set(yticks=range(23),yticklabels=MARKERS if j==0 else ['']*23,ylim=(22.6,-.6),xlim=(-bound,bound),xlabel='First − second\n(log₂ change)',title=comp.replace(' - ',' minus\n'));ax.grid(axis='y',color='#EDF0F2',lw=.35)
            q_column(ax,z,np.arange(23))
        fig.text(.53,.025,f'W{start} → W{end} paired interval · points and HC3 95% CI · * FDR q<.05 · NE: not estimable',ha='center',fontsize=6.5)
        finish(fig,8+k,f'Adjusted treatment contrasts in molecular change: W{start} → W{end}','Baseline-adjusted contrasts with HC3 95% confidence intervals. All 23 markers and four contrasts are shown. D denotes dupilumab and A100/A200 initial abrocitinib doses. Negative means the first treatment has a smaller, more negative change; it is not a negative concentration. Models control starting log₂ concentration, baseline total EASI, age, sex, BMI and centre; source tables provide exact n, exclusions, BH q values, centre and batch sensitivities. Pooled abrocitinib retains the observed initial-dose mixture. These are observational initial-treatment contrasts, not randomized current-dose effects.',[folder/'molecular_adjusted_contrasts.csv'],root,base,'Four directly labelled contrasts; common horizontal scale; FDR marker','Adjusted difference in paired log₂ change','All 23 markers','Complete covariate and marker pairs; exact n in source')
    correlations=pd.read_csv(folder/'molecular_clinical_change_correlations.csv');targets=['easi_total','pp_nrs','iga','dlqi']
    for k,(start,end) in enumerate(windows):
        fig,axes=grid(1,3,165,left=.15,top=.91,bottom=.19,wspace=.16)
        window_header(fig,f'W{start} → W{end}: clinical–molecular change correlations')
        for j,(ax,tx,title) in enumerate(zip(axes,ARMS,LABELS)):
            z=correlations[correlations.start.eq(start)&correlations.end.eq(end)&correlations.treatment.eq(tx)]
            v=z.pivot(index='marker',columns='outcome',values='rho').reindex(index=MARKERS,columns=targets)
            n=z.pivot(index='marker',columns='outcome',values='n').reindex(index=MARKERS,columns=targets)
            q=z.pivot(index='marker',columns='outcome',values='q').reindex(index=MARKERS,columns=targets)
            texts=[['n='+str(int(n.iloc[i,j]))+('*' if q.iloc[i,j]<.05 else '') if pd.notna(n.iloc[i,j]) else '' for j in range(4)] for i in range(23)]
            im=heat(ax,v,MARKERS if j==0 else ['']*23,['EASI','PP-NRS','IGA','DLQI'],title,1,texts,counts=n)
        colorbar(fig,im,'Colour: Spearman ρ · n: pairs · ×: not estimable · dotted border: n<5 · * FDR q<.05')
        finish(fig,11+k,f'Clinical–molecular change associations: W{start} → W{end}','Colour is Spearman correlation between paired clinical change and paired log₂ molecular change, both later minus earlier. Cell numbers show complete-pair patient counts and stars BH q<0.05. Positive correlations indicate changes in the same direction; negative correlations indicate opposite changes. Cells marked × have no estimable correlation; printed n distinguishes no pairs from an unestimable coefficient among available pairs. Dotted borders flag n<5. Adult DLQI requires age ≥18. Correlation does not establish numerical interchangeability or causal effects. All coefficients, bootstrap confidence intervals and multiplicity-adjusted tests remain in source data.',[folder/'molecular_clinical_change_correlations.csv'],root,base,'Correlation magnitude and evaluable n are visually distinct','Four clinical outcomes','23 markers; colour = signed rho','Cell-specific complete patient pairs')
    source=folder/'molecular_regional_baseline_correlations.csv';d=pd.read_csv(source);regions=['head_neck','upper_limb','trunk','lower_limb']
    v=d.pivot(index='marker',columns='region',values='rho').reindex(index=MARKERS,columns=regions);q=d.pivot(index='marker',columns='region',values='q').reindex(index=MARKERS,columns=regions)
    fig,axes=grid(1,1,165,left=.18,top=.92,bottom=.18);texts=[[f'ρ={v.iloc[i,j]:.2f}'+('*' if q.iloc[i,j]<.05 else '') for j in range(4)] for i in range(23)]
    im=heat(axes[0],v,MARKERS,['Head/neck','Upper limbs','Trunk','Lower limbs'],'Before treatment: regional EASI and serum markers',1,texts);colorbar(fig,im,'Colour and cell value: Spearman ρ · ×: not estimable · * FDR q<.05')
    finish(fig,14,'Baseline molecular associations with four skin regions','Observed baseline Spearman correlations pooled before systemic treatment, using source-consistent regional EASI and matched serum measurements. Values are printed in cells; stars indicate BH q<0.05. Regional records must be complete, within theoretical bounds and sum to total EASI within 0.01. Counts and bootstrap confidence intervals are in the source table. Serum association alone does not establish site-specific pathogenesis.',[source],root,base,'Exact signed correlations on one common matrix','Four regional EASI contributions','All 23 markers','Matched source-consistent baseline records')


def clinical_paths(root,base):
    folder=root/'results/tables/update_20261017/clinical'
    iga=pd.read_csv(folder/'IGA_composition.csv')
    fig,axes=grid(1,3,115,top=.76,bottom=.18,wspace=.35)
    grades=['#FFFFFF','#DDDDDD','#BBBBBB','#999999','#666666','#333333']
    for ax,tx,title in zip(axes,ARMS,LABELS):
        z=iga[iga.treatment.eq(tx)&iga.week.isin([0,4,12,24])];weeks=sorted(z.week.unique());bottom=np.zeros(len(weeks))
        for grade,col in enumerate(grades):
            records=[z[z.week.eq(w)&z.category.eq(grade)].iloc[0] for w in weeks]
            vals=np.array([r.proportion*100 for r in records])
            ax.bar(range(len(weeks)),vals,bottom=bottom,color=col,width=.68,label=f'IGA {grade}',linewidth=.45,edgecolor='#566771')
            for i,(value,r) in enumerate(zip(vals,records)):
                if value >= 10:
                    ax.text(i,bottom[i]+value/2,f'{grade}: {int(r.k)}',ha='center',va='center',fontsize=6.3,color='white' if grade>=4 else '#24323C')
            bottom+=vals
        ns=[int(z[z.week.eq(w)].n.iloc[0]) for w in weeks]
        ax.set(title=title,xticks=range(len(weeks)),xticklabels=[f'W{w}\nn={n}' for w,n in zip(weeks,ns)],ylim=(0,100),ylabel='Recorded grades (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.92),ncol=6,frameon=False)
    window_header(fig,'Recorded IGA: instrument identity unconfirmed',y=.99)
    fig.text(.54,.04,'Labels inside large segments: grade: patient count · Small segments retain grade colour; all counts in source',ha='center',fontsize=6.3)
    finish(fig,15,'Recorded IGA composition at four visits','Each bar is the recorded IGA composition among evaluable patients, whose total n is printed below the bar. Neutral intensity encodes grade, not treatment; borders distinguish grade 0 from the background. Labels in segments occupying at least 10% show grade and exact patient count; all grade counts remain in source data. Grade 5 remains as recorded because instrument identity is unresolved; no conversion to a standard 0–4 scale was made. EASI90 is shown with the other clinical targets in Figure S1.',[folder/'IGA_composition.csv'],root,base,'Three matched IGA-only facets; explicit unverified instrument and six identifiable grades','Specified visit','Proportion of available patients','IGA n printed per bar and counts in large segments')
    paths=pd.read_csv(folder/'clinical_paths_fixed_triplets.csv')
    keys=['W12_not_EASI75','responder_no_recorded_reduction','responder_reduced_after_W12']
    names=['W12 EASI75 not reached','W12 EASI75; no recorded reduction','W12 EASI75; later reduction']
    fig,axes=grid(1,3,105,top=.73,bottom=.16,wspace=.38)
    legend(fig,names,['#485861']*3,[':','--','-'],markers=['^','s','o'])
    for ax,tx,title,col in zip(axes,ARMS,LABELS,COLORS):
        counts=[]
        for key,label,style,marker in zip(keys,names,[':','--','-'],['^','s','o']):
            z=paths[paths.treatment.eq(tx)&paths.target.eq('easi_total')&paths.path.eq(key)&paths.n.gt(0)].sort_values('week')
            if len(z):line(ax,z.week,z['median'],z.ci_low,z.ci_high,col,style=style,marker=marker);counts.append(int(z.n.iloc[0]))
            else:counts.append(0)
        ax.set(title=title,xlabel='Week',ylabel='EASI',xticks=TIMES,ylim=(0,np.ceil(paths[paths.target.eq('easi_total')&paths.path.isin(keys)].ci_high.max()/5)*5))
        ax.set_title(title+'\nPathway n: '+', '.join(map(str,counts)),fontsize=7)
    finish(fig,16,'EASI trajectories before and after the W12 treatment decision','Each trajectory keeps the same patients observed at W0/W12/W24. Treatment is encoded by colour; line style distinguishes W12 EASI75 not reached, reached without recorded reduction, and reached followed by reduction. Median and patient-bootstrap 95% confidence intervals are shown; n is listed in legend order within each panel. Paths depend on later information and are descriptive, not randomized treatment strategies. Other paths, including unknown exposure, stopping and switching, remain in the source tables.',[folder/'clinical_paths_fixed_triplets.csv'],root,base,'Treatment colour preserved; pathway distinguished by line style','Actual week 0/12/24','Median EASI; bootstrap 95% CI','Fixed triplets within future-defined pathway')
    source=folder/'maintenance_rates.csv';d=pd.read_csv(source);d=d[d.classification.eq('group')]
    pathnames={'W12_missing':'W12 unknown','W12_not_EASI75':'W12 target not reached','responder_no_recorded_reduction':'Reached; no reduction','responder_reduced_after_W12':'Reached; later reduction','responder_stopped_or_switched':'Reached; stopped/switched','responder_exposure_unknown':'Reached; exposure unknown','responder_exposure_conflict':'Reached; conflicting exposure','reduced_before_or_at_W12':'Reduced before/by W12'}
    categories=list(dict.fromkeys(d.group));fig,axes=grid(3,1,205,left=.40,top=.92,bottom=.09,hspace=.68)
    window_header(fig,'Future-defined pathways: descriptive W24 EASI75 status')
    for ax,tx,title,col in zip(axes,ARMS,LABELS,COLORS):
        z=d[d.treatment.eq(tx)].set_index('group').reindex(categories)
        z[['n_group','n_evaluable','k','n_missing']]=z[['n_group','n_evaluable','k','n_missing']].fillna(0)
        for y,row in enumerate(z.itertuples()):
            if row.n_evaluable>0:
                ax.errorbar(row.rate*100,y,xerr=[[100*(row.rate-row.ci_low)],[100*(row.ci_high-row.rate)]],fmt='o',color=col,ms=3,capsize=2,lw=.8)
            else:
                ax.text(50,y,'Not estimable',ha='center',va='center',fontsize=6.3,color='#59646A')
            rate_count=f'{int(row.k)}/{int(row.n_evaluable)}' if row.n_evaluable>0 else '—'
            ax.text(1.05,y,f'{int(row.n_group):>2}   {rate_count:>4}   {int(row.n_missing)}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=6.3,fontfamily='DejaVu Sans Mono')
        ax.text(1.05,1.03,'N   k/eval  Miss',transform=ax.transAxes,fontsize=6.3,ha='left')
        ax.set(yticks=range(len(categories)),yticklabels=[pathnames.get(t,t.replace('_',' ')) for t in categories],ylim=(len(categories)-.5,-.5),xlim=(-3,103),xticks=[0,50,100],xlabel='W24 EASI75 among evaluable patients (%)',title=title)
        ax.grid(False)
        pos=ax.get_position();ax.set_position([pos.x0,pos.y0,pos.width*.76,pos.height])
    fig.text(.55,.025,'Points and Wilson 95% CI · N: all pathway patients · k/eval: attained/evaluable · Miss: W24 score missing',ha='center',fontsize=6.3)
    finish(fig,17,'W24 EASI75 status in future-defined recorded pathways','Each row shows the observed W24 EASI75 proportion and Wilson 95% confidence interval among evaluable patients. The table states complete pathway size N, attained/evaluable k/eval and the count missing a W24 score. All recorded paths are retained, including zero-size rows, which explicitly show N=0 and Not estimable. An evaluable denominator of zero is not a zero response rate. W12 target status and later recorded treatment define these pathways; they are descriptive future-defined groups, not randomized strategies or causal tapering effects. Missing outcomes are unknown, not failures. Missing-outcome bounds remain in source data.',[source],root,base,'Direct counts and conditional Wilson forests replace two-denominator stacked bars','Observed proportion among evaluable patients with Wilson 95% CI','Every recorded future-defined pathway','Complete N, evaluable n, k and missing printed')


def distributions_and_triplets(root,base):
    folder=root/'results/tables/update_20261017/clinical';source=folder/'molecular_distributions.csv';d=pd.read_csv(source)
    fig,axes=grid(1,3,165,left=.17,top=.91,bottom=.12,wspace=.38)
    for j,(ax,tx,title,col) in enumerate(zip(axes,ARMS,LABELS,COLORS)):
        z=d[d.week.eq(0)&d.treatment.eq(tx)].set_index('marker').reindex(MARKERS)
        ax.errorbar(z['median'],range(23),xerr=[z['median']-z.q25,z.q75-z['median']],fmt='o',ms=2.7,capsize=1.5,lw=.7,color=col)
        ax.set(yticks=range(23),yticklabels=[f'{m} [{int(n)}]' for m,n in zip(MARKERS,z.n)] if j==0 else [f'[{int(n)}]' for n in z.n],ylim=(22.6,-.6),xlabel='Log₂ recorded concentration',title=title,xlim=(d[d.week.eq(0)].q25.min()-1,d[d.week.eq(0)].q75.max()+1));ax.grid(axis='y',lw=.35,color='#EDF0F2')
    finish(fig,18,'Baseline concentration distributions for the full measured panel','Baseline medians and interquartile ranges of log₂ recorded concentrations, with available patient counts [n]. Treatment colours are fixed across the manuscript. Compare treatment groups within each marker; absolute values across different analytes do not represent comparable biological activity. All 23 markers are retained, including sparse or censored assays. Censoring and repeated values are separately audited.',[source],root,base,'All baseline markers, visible n, median/IQR rather than inferred normality','Log₂ concentration; median and IQR','23 markers','Available baseline assays per marker/treatment')
    source=folder/'molecular_three_visit_paired.csv';d=pd.read_csv(source);fig,axes=grid(1,3,165,left=.17,top=.89,bottom=.13,wspace=.28);legend(fig,[f'W{w}' for w in TIMES],TIME_COLORS,styles=[':','--','-'],markers=TIME_MARKERS)
    bound=np.ceil(np.nanmax(np.abs(d[['ci_low','ci_high']].to_numpy())))
    for j,(ax,tx,title) in enumerate(zip(axes,ARMS,LABELS)):
        for k,w in enumerate(TIMES):
            z=d[d.treatment.eq(tx)&d.week.eq(w)].set_index('marker').reindex(MARKERS);forest(ax,z,np.arange(23)+(k-1)*.21,TIME_COLORS[k],marker=TIME_MARKERS[k])
        z=d[d.treatment.eq(tx)&d.week.eq(0)].set_index('marker').reindex(MARKERS)
        ax.axvline(0,color='#AFB8BD',lw=.6);ax.set(yticks=range(23),yticklabels=[f'{m} [{int(n)}]' for m,n in zip(MARKERS,z.n)] if j==0 else [f'[{int(n)}]' for n in z.n],ylim=(22.6,-.6),xlim=(-bound,bound),xlabel='Log₂ change from own W0',title=title);ax.grid(axis='y',color='#EDF0F2',lw=.35)
    finish(fig,19,'Molecular W0/W12/W24 changes in identical patient sets','Within each marker and initial treatment, all three colours use the same patients observed at W0/W12/W24. Fixed triplet counts are shown as [n]. W0 equals zero by definition; W12/W24 are mean within-patient log₂ changes from W0 with 95% t confidence intervals when n≥5. Hollow points denote smaller descriptive sets. Negative values indicate lower concentration, not absence of the analyte. This includes every measured marker and its uncertainty, complementing the overview heatmap.',[source],root,base,'Three true visit colours; same triplet n for each marker','Mean log₂ change relative to own W0','23 markers; three visit symbols','Fixed complete triplets per marker/treatment')
    source=folder/'clinical_fixed_triplets.csv';d=pd.read_csv(source);fig,axes=grid(1,3,100,top=.88,bottom=.18)
    for ax,tx,title,col in zip(axes,ARMS,LABELS,COLORS):
        z=d[d.target.eq('ige_numeric')&d.treatment.eq(tx)].sort_values('week');line(ax,z.week,z['median'],z.ci_low,z.ci_high,col)
        ax.set(title=f'{title}\nSame {int(z.n.iloc[0])} patients',xlabel='Week',ylabel='IgE (recorded units)',xticks=TIMES);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    finish(fig,21,'IgE with the same patients retained at all three visits','Medians and patient-bootstrap 95% confidence intervals in fixed complete W0/W12/W24 sets. Each treatment panel retains identical patients at all three visits. Vertical scales differ across treatment panels so that within-group trajectories remain visible; compare absolute values using the tick labels. These denominators differ from cross-sectional and W0–W24-only paired sets. This figure tests whether changing sample composition may explain an apparent visit-level rise; it does not establish a causal drug effect.',[source],root,base,'Fixed-patient IgE diagnostic with plain numerical ticks','Week 0/12/24','IgE median and bootstrap 95% CI','Same complete triplets')


def treatment_contrasts(root,base):
    folder=root/'results/tables/update_20261017/clinical';raw=pd.read_csv(folder/'endpoint_tests.csv');adj=pd.read_csv(folder/'endpoint_weighted_contrasts.csv')
    comparisons=['dupilumab - abrocitinib_100mg','dupilumab - abrocitinib_200mg','abrocitinib_100mg - abrocitinib_200mg','dupilumab - pooled_abrocitinib']
    labels=['D minus A100','D minus A200','A100 minus A200','D minus pooled A']
    targets=['EASI50','EASI75','EASI90','EASI_le5','PP_NRS_drop4','PP_NRS_le4','PP_NRS_01','IGA_drop2','IGA_01','DLQI_le10','DLQI_le4','physician','patient','overall']
    names=['EASI50','EASI75','EASI90','EASI ≤5','PP-NRS decrease ≥4','PP-NRS ≤4','PP-NRS 0/1','IGA decrease ≥2','IGA 0/1','Adult DLQI ≤10','Adult DLQI ≤4','Physician target','Patient target','Both targets']
    for k,week in enumerate([2,4,12,24]):
        if week==2:
            fig,axes=grid(1,1,85,left=.27,top=.80,bottom=.22);ax=axes[0]
            for j,(data,col) in enumerate([(raw,CONTROL),(adj,CURRENT)]):
                z=data[data.week.eq(week)&data.target.eq('PP_NRS_le4')].set_index('comparison').reindex(comparisons);forest(ax,z,np.arange(4)+(j-.5)*.2,col)
            ax.axvline(0,color='#AAB2B8',lw=.7);ax.set(yticks=range(4),yticklabels=labels,ylim=(3.6,-.6),xlim=(-1.03,1.03),xlabel='Difference in PP-NRS ≤4 attainment',title='Week 2: the only prespecified target')
        else:
            fig,axes=grid(1,4,145,left=.25,top=.88,bottom=.15,wspace=.18)
            for ax,comp,title in zip(axes,comparisons,labels):
                for j,(data,col) in enumerate([(raw,CONTROL),(adj,CURRENT)]):
                    z=data[data.week.eq(week)&data.comparison.eq(comp)].set_index('target').reindex(targets);forest(ax,z,np.arange(len(targets))+(j-.5)*.23,col)
                ax.axvline(0,color='#AAB2B8',lw=.7);ax.set(yticks=range(len(targets)),yticklabels=names if ax==axes[0] else ['']*len(names),ylim=(len(targets)-.4,-.6),xlim=(-1.03,1.03),xticks=[-1,0,1],xlabel='Risk difference',title=title);ax.grid(axis='y',lw=.35,color='#EDF0F2')
        window_header(fig,f'Week {week}: initial-treatment target contrasts',y=.995)
        legend(fig,['Observed','Baseline adjusted'],[CONTROL,CURRENT],y=.965);fig.text(.56,.035,'Negative favours second group     |     Positive favours first group',ha='center',fontsize=6.3)
        finish(fig,23+k,f'Week {week}: all initial-treatment target contrasts','Observed and baseline-adjusted risk differences with nominal 95% confidence intervals. Positive values favour the first named treatment for target attainment. Adjustment uses generalized overlap weights from baseline EASI, age, sex, BMI and centre; it does not estimate a randomized current-dose effect. D denotes dupilumab; A100/A200 the initial abrocitinib doses. Pooled abrocitinib is supplementary. W2 includes PP-NRS ≤4 only. Missing estimates are left absent; all exact denominators, Fisher/weighted tests, Holm-adjusted P values and exploratory FDR values are in the source tables.',[folder/'endpoint_tests.csv',folder/'endpoint_weighted_contrasts.csv'],root,base,'W2 compressed into one purposeful comparison forest; later visits all endpoints','Attainment risk difference; nominal 95% CI','Four contrasts or complete target list','Endpoint-specific evaluable patients; weighted ESS in source')


def model_comparisons(root,base):
    folder=root/'results/tables/update_20261017/model'
    source=folder/'held_out_group_metrics.csv';d=pd.read_csv(source);fig,axes=grid(2,3,145,left=.15,top=.91,bottom=.13,hspace=.75)
    mapping={'dupilumab':'D','abrocitinib_100mg':'A100','abrocitinib_200mg':'A200','HUASHAN':'Huashan','CAMS':'CAMS'}
    bound=np.ceil(d.MAE_hi.max())
    for row,dimension in enumerate(['center','treatment']):
        for j,w in enumerate(TIMES):
            z=d[d.held_out_dimension.eq(dimension)&d.week.eq(w)];ax=axes[row*3+j];forest(ax,z.rename(columns={'MAE':'effect','MAE_lo':'ci_low','MAE_hi':'ci_high'}),np.arange(len(z)),CURRENT)
            ax.set(yticks=range(len(z)),yticklabels=[f'{mapping.get(r.held_out_group,r.held_out_group)}\nn={int(r.n_patients)}' for r in z.itertuples()],ylim=(len(z)-.4,-.6),xlim=(0,bound),xlabel='MAE (EASI points)',title=f'W{w}: unseen {dimension}')
    finish(fig,22,'Estimation when a centre or treatment group is left out','Every displayed centre or initial-treatment group was entirely absent from fitting and model-form selection. Points are held-out MAE with patient-bootstrap 95% confidence intervals; patient counts appear next to each group. The same error scale is used in all panels. These internal transport stress tests are not independent external validation. Sparse groups and different severity distributions affect precision.',[source],root,base,'Horizontal interval forests with all sample sizes and shared error scale','Held-out MAE; patient-bootstrap 95% CI','Held-out centre or initial treatment','Each named group excluded from all fitting')
    source=folder/'auxiliary_targets/overall_metrics.csv';d=pd.read_csv(source)
    fig,axes=grid(1,3,110,left=.12,top=.80,bottom=.20,wspace=.46)
    window_header(fig,'Separate target-specific models: W0/W12 development; W24 temporal')
    for ax,target,title in zip(axes,['pp_nrs','dlqi','iga'],['PP-NRS','Adult DLQI','Recorded IGA']):
        metric='normalized_RPS' if target=='iga' else 'MAE'
        z=d[d.target.eq(target)&d.model.eq('BSA_three')].sort_values('week')
        for k,row in enumerate(z.itertuples()):
            estimate=getattr(row,metric);lo=getattr(row,metric+'_lo');hi=getattr(row,metric+'_hi')
            ax.errorbar(row.week,estimate,yerr=[[estimate-lo],[hi-estimate]],fmt=TIME_MARKERS[k],color=TIME_COLORS[k],ms=4,capsize=2,lw=.8)
        ax.set(xticks=TIMES,xticklabels=[f'W{int(r.week)}\nn={int(r.n_patients)}' for r in z.itertuples()],ylabel='Normalized ranked probability score' if target=='iga' else 'MAE (score points)',title=title)
        ax.set_ylim(bottom=0)
    fig.text(.55,.04,'Lower is better · Points and patient-bootstrap 95% CI · IGA instrument unconfirmed',ha='center',fontsize=6.3)
    finish(fig,35,'Independently fitted PP-NRS, DLQI and ordinal IGA models','Each target has separate coefficients and patient-grouped nested selection using current BSA and the fixed three-marker panel. PP-NRS and adult DLQI use continuous-outcome models; recorded IGA uses ordinal proportional-odds regression and normalized ranked probability score. Points and patient-bootstrap 95% confidence intervals show the independently fitted target-specific models; lower is better. Visits use neutral tones and distinct symbols. W0/W12 are development visits evaluated in patient-held-out outer folds; W24 is previously inspected same-cohort temporal transfer, not independent external validation. BSA-only controls remain in the numerical source table and are not plotted. No treatment, time label or EASI is used as an input. These results do not establish clinically adequate replacement, and the shared panel is not claimed optimal for every target. Recorded IGA includes grade 5 pending instrument verification.',[source],root,base,'Three independently fitted targets with appropriate metrics; no cross-method ranking','Evaluation visit, unconnected','Target-specific MAE or ordinal RPS','Available patients per target and visit')
    source=folder/'model_metrics.csv';d=pd.read_csv(source);d=d[d.model.eq(MODEL)&d.group_type.eq('severity')];fig,axes=grid(2,3,150,left=.11,top=.91,bottom=.13,hspace=.75,wspace=.42)
    cats=['EASI_le7','EASI_7_to21','EASI_gt21'];labs=['≤7','>7–21','>21'];bound=max(7,np.ceil(d.MAE_hi.max()))
    for j,w in enumerate(TIMES):
        z=d[d.week.eq(w)].set_index('group').reindex(cats);ns=z.n_patients.fillna(0).astype(int);ok=z.MAE.notna();x=np.arange(3)
        axes[j].errorbar(x[ok],z.MAE[ok],yerr=[z.MAE[ok]-z.MAE_lo[ok],z.MAE_hi[ok]-z.MAE[ok]],fmt=TIME_MARKERS[j],color=TIME_COLORS[j],capsize=2);axes[j].set(ylim=(0,bound),ylabel='MAE (EASI points)',title=f'Week {w}')
        axes[j+3].scatter(x[ok],100*z.within2[ok],color=TIME_COLORS[j],marker=TIME_MARKERS[j],s=20)
        for ix in x[ok]:
            axes[j+3].annotate(f'{100*z.within2.iloc[ix]:.1f}%',(ix,100*z.within2.iloc[ix]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6.3)
        axes[j+3].set(ylim=(-5,118),yticks=[0,50,100],ylabel='Within ±2 points (%)',title=f'Week {w}: descriptive rate')
        for ax in [axes[j],axes[j+3]]:
            ax.set(xticks=x,xticklabels=[f'{lab}\nn={n}' for lab,n in zip(labs,ns)],xlabel='Observed EASI stratum',xlim=(-.5,2.5))
            for ix in x[~ok]:ax.text(ix,.05,'No data',ha='center',transform=ax.get_xaxis_transform(),fontsize=6.3)
    window_header(fig,'Severity coverage: both outer repetitions averaged')
    fig.text(.55,.025,'Rate intervals are unavailable in this aggregate source · n=1 is descriptive · No data means n=0',ha='center',fontsize=6.3)
    finish(fig,37,'Accuracy across observed EASI severity levels','Primary nested estimator results averaged over both outer repetitions. Top, Mean absolute error and patient-bootstrap 95% confidence intervals. Bottom, Percentage within two EASI points, with direct values including zero; aggregate-source rate intervals are unavailable and are not fabricated. The n=1 result is descriptive. This is a descriptive accuracy threshold, not established clinical equivalence. Strata use observed EASI for evaluation only, not routing or predictor input. W12 has one severe patient and W24 none, so post-treatment accuracy cannot establish reliable severe-disease performance. Axes retain all interval extents.',[source],root,base,'Severity coverage and severe-case uncertainty made explicit','Observed EASI evaluation stratum','MAE or within-two-point rate','Patient counts printed per stratum')


def response_depth(root,base):
    source=root/'results/tables/update_20261017/clinical/doctor_stratified_response_depth.csv';d=pd.read_csv(source);number=27
    for start,end in [(0,12),(12,24),(0,24)]:
        for target in ['easi_total','iga']:
            bands=['[-inf, 50.0)','[50.0, 75.0)','[75.0, 90.0)','[90.0, inf)'] if target=='easi_total' else ['[-inf, 0.0)','[0.0, 1.0)','[1.0, 2.0)','[2.0, inf)']
            labs=['<50%','50–<75%','75–<90%','≥90%'] if target=='easi_total' else ['<0','0','1','≥2']
            fig,axes=grid(1,3,165,left=.15,top=.91,bottom=.20,wspace=.16)
            window_header(fig,f'Exploratory W{start} → W{end}: molecular change by {"EASI" if target=="easi_total" else "recorded IGA"} response')
            for j,(ax,tx,title) in enumerate(zip(axes,ARMS,LABELS)):
                z=d[d.start.eq(start)&d.end.eq(end)&d.outcome.eq(target)&d.treatment.eq(tx)]
                values=z.pivot(index='marker',columns='band',values='mean_change').reindex(index=MARKERS,columns=bands);ns=z.pivot(index='marker',columns='band',values='n').reindex(index=MARKERS,columns=bands)
                texts=[['n='+str(int(ns.iloc[i,j])) if pd.notna(ns.iloc[i,j]) else 'n=0' for j in range(4)] for i in range(23)]
                im=heat(ax,values,MARKERS if j==0 else ['']*23,labs,title,6,texts,counts=ns);ax.set_xlabel('EASI improvement (%)\nFrom interval start' if target=='easi_total' else 'Recorded IGA decrease (grades)')
            colorbar(fig,im,'Mean log₂ change · n: pairs · ×: no observations · dotted border: n<5 (descriptive)',6)
            finish(fig,number,f'Molecular changes by {"EASI" if target=="easi_total" else "IGA"} response: W{start} → W{end}','Columns are mutually exclusive clinical-response categories within the named paired interval. Colour is mean within-patient log₂ concentration change; numbers are complete-pair patient counts, with a dotted cell border for n<5. Negative means lower later concentration. These are neither correlation coefficients nor change per clinical point. EASI improvement is relative to the start of the named interval; IGA uses recorded ordinal decrease. Cells marked × and n=0 have no observations; absence is not a zero molecular change. All 23 markers, exact means and intervals are retained in the source table; the small cells are descriptive and do not validate a prediction model.',[source],root,base,'Mean molecular change distinguished from cell n; same signed scale across all six figures','Mutually exclusive response categories','23 markers; signed log₂ change colour','Complete patient pairs within category');number+=1


def regional_panels(root,base):
    folder=root/'results/tables/update_20261017/regional';d=pd.read_csv(folder/'regional_paired_improvement.csv');itch=pd.read_csv(folder/'whole_body_PP_NRS_improvement.csv');regions=['head_neck','upper_limb','trunk','lower_limb'];labels=['Head/neck','Upper limbs','Trunk','Lower limbs']
    fig,axes=grid(3,2,170,left=.17,top=.90,bottom=.11,hspace=.85,wspace=.58);legend(fig,['Dupilumab','Pooled abrocitinib'],[COLORS[0],CONTROL])
    for row,(start,end) in enumerate([(0,12),(0,24),(12,24)]):
        for k,(scope,col) in enumerate([('dupilumab',COLORS[0]),('pooled_abrocitinib',CONTROL)]):
            outcome='normalized_improvement' if start==12 else 'pct_improvement'
            z=d[d.scope.eq(scope)&d.start.eq(start)&d.end.eq(end)&d.outcome.eq(outcome)].set_index('region').reindex(regions).copy()
            if start==12:z[['mean','ci_low','ci_high']]*=100
            forest(axes[row*2],z,np.arange(4)+(k-.5)*.2,col,key='mean')
            z=itch[itch.scope.eq(scope)&itch.start.eq(start)&itch.end.eq(end)];forest(axes[row*2+1],z,[k],col,key='mean')
        ax=axes[row*2];ax.set(yticks=range(4),yticklabels=labels,ylim=(3.6,-.6),xlabel='Decrease (% of regional maximum)' if start==12 else 'Regional EASI improvement (%)',title=f'W{start} → W{end}: '+('regional maximum normalization' if start==12 else 'relative to starting regional EASI'));ax.axvline(0,color='#AAB2B8',lw=.6)
        ax=axes[row*2+1];ax.set(yticks=[0,1],yticklabels=['Dupilumab','Pooled A'],ylim=(1.5,-.5),xlabel='PP-NRS decrease (points)',title=f'W{start} → W{end}: whole-body itch');ax.axvline(0,color='#AAB2B8',lw=.6)
    for ax in axes[4:]: ax.set_facecolor('#F3F5F6')
    finish(fig,33,'Separate paired regional and whole-body itch intervals','Rows are separate paired intervals that can share patients; they are not a continuous cumulative sequence. First two rows show mean regional percentage improvement relative to positive W0 regional EASI. Last row shows W12→W24 decrease as a percentage of the regional theoretical maximum, avoiding division by near-zero W12 values. Right, Whole-body PP-NRS decrease in original score points, not regional itch. Positive values indicate improvement, negative worsening. Bars are 95% t confidence intervals. Source-consistent regional records are required; the separate-dose and adjusted comparisons remain in the tables. No adjusted regional contrast passed the FDR threshold. The pooled abrocitinib group retains its observed dose mixture.',[folder/'regional_paired_improvement.csv',folder/'whole_body_PP_NRS_improvement.csv',folder/'regional_treatment_contrasts.csv'],root,base,'Horizontal effects avoid crammed region labels; each row a separately named interval','Paired improvement with 95% CI; distinct units explicit','Region or treatment','Source-consistent region pairs; whole-body itch independent available pairs')
    source=folder/'regional_normalized_burden.csv';d=pd.read_csv(source);fig,axes=grid(2,2,170,top=.87,bottom=.22,hspace=1.10);legend(fig)
    for ax,region,title in zip(axes,regions,labels):
        for j,tx in enumerate(ARMS):
            z=d[d.scope.eq(tx)&d.region.eq(region)].sort_values('week');line(ax,z.week+(j-1)*.2,z['mean']*100,z.ci_low*100,z.ci_high*100,COLORS[j])
        ax.set(title=title,xticks=TIMES,ylabel='Regional maximum (%)',ylim=(0,65));ax.grid(axis='y',lw=.35,color='#EDF0F2')
        count_strip(ax,TIMES,d[d.region.eq(region)],txkey='scope',y=-.40,rowgap=.13)
    fig.text(.55,.025,'Below each axis: available n in D / A100 / A200 order',ha='center',fontsize=6.3)
    finish(fig,34,'Regional burden on a common interpretable scale','Recorded regional EASI contributions divided by theoretical regional maxima (head/neck 7.2, upper limbs 14.4, trunk 21.6, lower limbs 28.8) and multiplied by 100. Points show observed means and 95% t confidence intervals, by initial treatment. This rescales skin-region burden; it does not measure molecular inflammation or demonstrate intrinsic anatomical susceptibility. Available patients may differ between visits. Exact n and sensitivity populations accompany the source table.',[source],root,base,'Common percent scale for all four regions, actual 0/12/24 visits','Week','Percentage of theoretical regional maximum','Source-consistent available records')


def monitoring_classification(root,base):
    source=root/'results/tables/update_20261017/redesign_20261018/monitoring_confusion_repeat0.csv';d=pd.read_csv(source)
    fig,axes=grid(2,3,155,left=.12,top=.90,bottom=.13,wspace=.47,hspace=.68)
    cmap=LinearSegmentedColormap.from_list('current_proportion',['#F6F8F8',CURRENT])
    for row,window in enumerate(['W0_W12','W0_W24']):
        for col,endpoint in enumerate(['EASI50','EASI75','EASI90']):
            ax=axes[row*3+col];z=d[d.window.eq(window)&d.endpoint.eq(endpoint)]
            counts=z.pivot(index='observed_response',columns='predicted_response',values='count').reindex(index=[False,True],columns=[False,True]);proportions=z.pivot(index='observed_response',columns='predicted_response',values='row_proportion').reindex(index=[False,True],columns=[False,True])
            ax.imshow(proportions.to_numpy(float),cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='none')
            for i in range(2):
                for j in range(2):
                    p=proportions.iloc[i,j];ax.text(j,i,f'n = {int(counts.iloc[i,j])}\n{100*p:.1f}% of row',ha='center',va='center',fontsize=6.5,color='white' if p>.65 else '#24323C')
            n=int(z.n_evaluable.iloc[0]);ax.set(xticks=[0,1],xticklabels=['Not reached','Reached'],yticks=[0,1],yticklabels=['Not reached','Reached'],xlabel='Model classification',ylabel='Observed response' if col==0 else '',title=f'{window.replace("_"," → ")} · {endpoint}\nn={n} paired patients')
            ax.tick_params(length=0)
            for spine in ax.spines.values():spine.set_visible(False)
    window_header(fig,'First outer repetition: illustration of response classification',y=.99)
    finish(fig,38,'Who is correctly classified as reaching EASI50, EASI75 and EASI90?','Each panel shows actual patient counts from the first outer repetition, separately for W0→W12 and W0→W24. Rows are observed response and columns model-estimated response. Diagonal cells are correct; upper-right cells falsely indicate response and lower-left cells miss an observed responder. Percentages are within each observed-response row, so the lower-right percentage is sensitivity and the upper-left percentage specificity. Observed response uses observed baseline/follow-up scores, whereas estimated response uses predicted baseline/follow-up scores. Counts are integers from one repetition, not the primary two-repetition average; the full metrics table retains the latter. The paired denominators are 85 and 67, with no excluded pair for invalid baseline in this displayed repetition. The two intervals can share patients. This assesses response classification, not alternative-treatment outcomes or continuous-score equivalence.',[source,root/'results/tables/update_20261017/model/improvement_classification.csv'],root,base,'Replace isolated metric dots with six directly readable confusion matrices','Predicted response: no/yes','Observed response: no/yes','First repeat same-patient pairs n85/n67; actual integer counts')


def actual_exposure(root,base):
    source=root/'results/tables/update_20261017/clinical/actual_exposure_counts.csv';d=pd.read_csv(source)
    fig,axes=grid(1,3,160,left=.18,top=.88,bottom=.20,wspace=.75)
    cmap=LinearSegmentedColormap.from_list('recorded_proportion',['#F5F5F5','#666666'])
    labels={'unknown':'Unknown','unknown_stop_timing':'Stop timing\nunknown','stopped_or_switched':'Stopped /\nswitched'}
    for ax,tx,title in zip(axes,ARMS,LABELS):
        z=d[d.initial_treatment.eq(tx)];regimens=sorted(set(z.previsit_regimen),key=lambda s:('unknown' in s or 'stopped' in s,float(s.split('mg')[0]) if 'mg' in s and s.split('mg')[0].isdigit() else float('inf'),float(re.search(r'q([0-9.]+)',s).group(1)) if re.search(r'q([0-9.]+)',s) else float('inf'),s))
        counts=z.pivot(index='previsit_regimen',columns='visit_week',values='n').reindex(index=regimens,columns=TIMES).fillna(0)
        bounds=z.pivot(index='previsit_regimen',columns='visit_week',values='boundary_unknown_n').reindex(index=regimens,columns=TIMES).fillna(0);totals=counts.sum(axis=0)
        values=counts/totals
        im=ax.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='none')
        for i in range(len(regimens)):
            for j in range(3):
                n=int(counts.iloc[i,j]);b=int(bounds.iloc[i,j]);conflict=(j==0 and n>0 and ((tx=='abrocitinib_100mg' and regimens[i].startswith('200mg/')) or (tx=='abrocitinib_200mg' and regimens[i].startswith('100mg/'))))
                s=f'n={n}'+('†' if conflict else '')+(f'\nu={b}' if b else '')
                ax.text(j,i,s,ha='center',va='center',fontsize=6.3,color='white' if values.iloc[i,j]>.65 else '#24323C')
        names=[labels.get(s,s.replace('mg/',' mg\n').replace('q','every ').replace('days',' d')) for s in regimens]
        ax.set(yticks=range(len(regimens)),yticklabels=names,xticks=range(3),xticklabels=[f'W{w}\nN={int(totals[w])}' for w in TIMES],xlabel='Visit',title=title)
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    cax=axes[0].inset_axes([.35,.085,.40,.018],transform=fig.transFigure)
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1]);cb.ax.set_xticklabels(['0%','50%','100%']);cb.ax.set_title('Proportion of full initial-treatment group',fontsize=6.3,pad=4)
    fig.text(.55,.03,'n: patients · u: subset with uncertain visit-boundary timing · All recorded regimen labels retained',ha='center',fontsize=6.3)
    fig.text(.55,.009,'† W0 source conflicts: two A100 patients recorded at 200 mg; one A200 patient at 100 mg. Classification unresolved.',ha='center',fontsize=6.3)
    finish(fig,41,'Recorded regimens evolve after the initial treatment assignment','Within each initial-treatment group, the table-like heatmap shows all recorded regimens at W0/W12/W24. Cell n values are patient counts; u denotes the subset with uncertain timing relative to the visit boundary. Grey intensity is the proportion of the full initial-treatment group, with a fixed 0–100% scale. Zero cells are genuine zero counts. Unknown regimens, uncertain stop timing and stopping/switching are retained; all 163 patients remain represented. W0 is the recorded starting regimen and must not be read as prior systemic exposure. Daggers flag three unresolved source conflicts at W0: two patients assigned to the initial A100 source group have a recorded 200 mg regimen and one assigned to A200 has a recorded 100 mg regimen. The source group and recorded regimen are both preserved; these are not presumed switches. Recorded 75 mg and 150 mg labels are preserved without assuming that they mean alternate-day 100 mg or 200 mg. This aggregate display is not an individual transition flow and does not resolve whether a treatment decision preceded or followed the outcome assessment.',[source],root,base,'All recorded dose/interval categories and uncertainty visible; no invented transitions','W0/W12/W24','Recorded dose/interval or unresolved state','All initial-group patients, including unknown exposure')


def absolute_agreement(root,base):
    folder=root/'manuscript/source_data';bins=pd.read_csv(folder/'model_agreement_bins.csv');summary=pd.read_csv(folder/'model_agreement_summary.csv').set_index('week')
    fig,axes=grid(1,3,112,left=.12,top=.80,bottom=.22,wspace=.43)
    lower=np.floor(min(bins.difference_bin.min(),summary.LoA_low.min())/5)*5-1
    upper=np.ceil(max(bins.difference_bin.max(),summary.LoA_high.max())/5)*5+1
    for j,(ax,week) in enumerate(zip(axes,TIMES)):
        z=bins[bins.visit_week.eq(week)];r=summary.loc[week]
        ax.axhspan(-2,2,color='#E4E8EA',alpha=.75,zorder=0);ax.axhline(0,color='#B7C0C5',lw=.6)
        ax.axhline(r.bias,color=TIME_COLORS[j],lw=.9)
        for limit in [r.LoA_low,r.LoA_high]:ax.axhline(limit,color=TIME_COLORS[j],lw=.8,ls='--')
        ax.scatter(z.mean_bin,z.difference_bin,s=8+8*np.sqrt(z['count']),alpha=.7,color=TIME_COLORS[j],marker=TIME_MARKERS[j],linewidths=0)
        ax.set(xlim=(-1,max(10,z.mean_bin.max()+2)),ylim=(lower,upper),xlabel='Mean of observed and estimated EASI',ylabel='Estimated − observed EASI',title=f'Week {week} (n={int(r.n)})\nBias {r.bias:+.2f}; limits {r.LoA_low:.1f} to {r.LoA_high:.1f}')
    handles=[Line2D([0],[0],color='#55646D',lw=.9,label='Mean bias'),Line2D([0],[0],color='#55646D',lw=.8,ls='--',label='95% limits of agreement'),Patch(facecolor='#E4E8EA',label='±2-point reading band')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.94),ncol=3,frameon=False)
    handles=[plt.scatter([],[],s=8+8*np.sqrt(n),color=CONTROL,label=f'{n} '+('patient' if n==1 else 'patients')) for n in [1,5,10]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.08),ncol=3,frameon=False)
    window_header(fig,'First outer repetition · Positions binned (mean: 1 EASI; difference: 0.5 EASI)',y=.995)
    finish(fig,42,'Individual agreement across the full range of EASI','First outer-repetition Bland–Altman displays, with prediction minus observed EASI on the vertical axis and their mean horizontally. Symbol area reflects binned patient counts; display coordinates are rounded to one point horizontally and half a point vertically. Solid coloured lines show mean bias; dashed lines are bias ±1.96 SD (limits of agreement, not confidence intervals). The common vertical scale retains every observed error and both limits. Shading marks ±2 points for reading the plot, not a prespecified equivalence boundary. Negative errors are underestimates. W0/W12 predictions are held out by patient; W24 is within-cohort temporal application after previous inspection. The primary performance table averages both outer repetitions and should not be confused with these single-repetition display statistics.',[folder/'model_agreement_bins.csv',folder/'model_agreement_summary.csv'],root,base,'Restore complementary size-dependent error evidence; all errors and agreement limits retained','Mean observed/estimated EASI','Estimated minus observed EASI','First-repetition held-out patient predictions, n152/89/75')


def run(root,base=None):
    if base is None:
        from . import figures as base
    root=Path(root).resolve();DESIGNS.clear()
    global COLORS,TIME_COLORS,CURRENT,CONTROL,CMAP
    style=base.STYLE if hasattr(base,'STYLE') else json.loads((root/'config/paper_style.json').read_text())
    COLORS=[style['treatment_colors'][x] for x in ARMS];TIME_COLORS=[style['visit_colors'][str(w)] for w in TIMES]
    CURRENT=style['method_colors']['current'];CONTROL=style['method_colors']['reference']
    CMAP=LinearSegmentedColormap.from_list('signed_effect',style['diverging_colors']);CMAP.set_bad(style.get('unknown_color','#D9DEE2'))
    clinical_courses(root,base)
    molecular_panels(root,base)
    clinical_paths(root,base)
    distributions_and_triplets(root,base)
    treatment_contrasts(root,base)
    model_comparisons(root,base)
    response_depth(root,base)
    regional_panels(root,base)
    monitoring_classification(root,base)
    actual_exposure(root,base)
    absolute_agreement(root,base)
    report=root/'reports/update_20261018/supplement_redesign';report.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(DESIGNS).sort_values('id',key=lambda x:x.str.extract(r'(\d+)')[0].astype(int)).to_csv(report/'design_and_source_map.csv',index=False)
    actual={x['id'] for x in DESIGNS};expected={f'FigureS{i}' for i in range(1,39) if i not in (20,36)}|{'FigureS41','FigureS42'}
    if actual!=expected:raise AssertionError(f'Supplement coverage differs: {actual^expected}')
    return DESIGNS


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args();run(args.root)
