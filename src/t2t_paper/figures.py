"""Render the manuscript figures from versioned aggregate source tables."""
from pathlib import Path
import argparse, json, string, re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from cytot2t.nature_qa.audit_panel_alignment_redevelopment import require_matplotlib_panel_alignment
from . import palette

ROOT=Path(__file__).resolve().parents[2]
ARMS=['dupilumab','abrocitinib_100mg','abrocitinib_200mg']
LABELS=['Dupilumab','Abrocitinib 100 mg','Abrocitinib 200 mg']
STYLE=json.loads((ROOT/'config/paper_style.json').read_text())
STYLE.update(treatment_colors={a:palette.color('treatment.'+a) for a in ARMS},
             treatment_markers={a:palette.marker('treatment.'+a) for a in ARMS},
             visit_colors={str(w):palette.color('visit.W'+str(w)) for w in [0,12,24]},
             visit_markers={str(w):palette.marker('visit.W'+str(w)) for w in [0,12,24]},
             method_colors={k:palette.color('method.'+k) for k in ['reference','current']})
COLORS=[palette.color('treatment.'+a) for a in ARMS]
TIMES=[0,12,24]
TIME_COLORS=[STYLE['visit_colors'][str(w)] for w in TIMES]
TIME_MARKERS=[STYLE['visit_markers'][str(w)] for w in TIMES]
TIME_STYLES=[STYLE['visit_linestyles'][str(w)] for w in TIMES]
TREATMENT_MARKERS=[STYLE['treatment_markers'][a] for a in ARMS]
MARKERS=['IL-4','IL-5','IL-13','IL-31','TSLP','IL-33','CCL17','CCL18','IFN-gamma','IL-12p70','IL-18','TNF-alpha','IL-21','IL-6','IL-17A','IL-22','IL-10','IL2RA','CCL26','Periostin','CCL11','IL-36beta','IFN-beta']
DISPLAY={'pp_nrs':'PP-NRS','easi_total':'EASI','bsa':'BSA (%)','dlqi':'Adult DLQI','scorad':'SCORAD','ige_numeric':'IgE (recorded units)','eo_count_numeric':'Eosinophils (recorded units)'}
CATALOG=[]
REVISION=json.loads((ROOT/'config/manuscript_revision_20261025.json').read_text())
SUPPLEMENT_ORDER=REVISION['supplement_source_order']
plt.rcParams.update({'font.family':STYLE['font'],'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'t2t-palette-v4','savefig.dpi':600})

def publication_id(source_id):
 if not source_id.startswith('FigureS'):return source_id
 n=int(source_id.removeprefix('FigureS'))
 if n not in SUPPLEMENT_ORDER:return None
 return 'FigureS'+str(SUPPLEMENT_ORDER.index(n)+1)

def remap_figure_references(text):
 def replace(match):
  n=int(match.group(2));mapped=publication_id('FigureS'+str(n))
  return match.group(1)+mapped.removeprefix('FigureS') if mapped else 'the corresponding source table'
 return re.sub(r'(Supplementary Fig\. S|Fig\. S|FigureS)(\d+)',replace,text)

def layout(rows,cols,height=155):
 fig,axes=plt.subplots(rows,cols,figsize=(183/25.4,height/25.4),squeeze=False)
 fig.subplots_adjust(left=.105,right=.975,bottom=.10,top=.88,wspace=.43,hspace=.58)
 for i,ax in enumerate(axes.flat):
  ax.annotate(string.ascii_lowercase[i],(0,1),xycoords='axes fraction',xytext=(-18,12),textcoords='offset points',weight='bold',fontsize=8,annotation_clip=False)
  ax.tick_params(length=2.5,width=.6)
 return fig,axes.flat

def shared_legend(fig,kind='treatment'):
 labs,colors=(LABELS,COLORS) if kind=='treatment' else ([f'W{x}' for x in TIMES],TIME_COLORS)
 markers=TREATMENT_MARKERS if kind=='treatment' else TIME_MARKERS
 handles=[Line2D([0],[0],color=c,marker=m,ms=4,lw=1.35,label=l) for l,c,m in zip(labs,colors,markers)]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.985),ncol=3,frameon=False)

def errorline(ax,x,y,lo,hi,color,label=None):
 x,y,lo,hi=map(lambda a:np.asarray(a,float),(x,y,lo,hi))
 marker=TREATMENT_MARKERS[COLORS.index(color)] if color in COLORS else 'o'
 outline=palette.role_style('treatment.'+ARMS[COLORS.index(color)])['outline'] if color in COLORS else color
 ax.errorbar(x,y,yerr=np.vstack([np.maximum(y-lo,0),np.maximum(hi-y,0)]),fmt=marker+'-',ms=4,lw=1.35,capsize=2,color=color,markeredgecolor=outline,markeredgewidth=.55,label=label)

def save(fig,name,title,caption,sources,root=ROOT):
 source_id=name;name=publication_id(source_id)
 if name is None:
  plt.close(fig)
  return
 out=root/'manuscript/figures';qa=root/'results/audit/update_20261017/figures';out.mkdir(parents=True,exist_ok=True);qa.mkdir(parents=True,exist_ok=True)
 fig.canvas.draw()
 (qa/(name+'.colour.json')).write_text(json.dumps(palette.audit_figure(fig,name),indent=2)+'\n')
 require_matplotlib_panel_alignment(fig,json_out=str(qa/(name+'.alignment.json')),strict=True)
 for ext in ['pdf','svg','png']:
  fig.savefig(out/(name+'.'+ext),dpi=300 if ext=='png' else 600,metadata=({'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)))
 plt.close(fig)
 CATALOG.append(dict(id=name,source_id=source_id,title=title,caption=remap_figure_references(caption),sources=[str(Path(s).relative_to(root)) if Path(s).is_absolute() else s for s in sources]))

def run(root):
 from .plot_source_aggregates_20261018 import run as prepare_plot_sources
 from .main_revision_20261025 import run as main_figures
 from .supplement_revision_20261025 import run as supplementary_figures
 from .revision_integration_20261025 import supporting_figures
 root=Path(root).resolve()
 prepare_plot_sources(root)
 CATALOG.clear()
 main_figures(root)
 supplementary_figures(root)
 supporting_figures(root)
 rows={r['id']:r for r in CATALOG}
 expected=[f'Figure{i}' for i in range(1,6)]+[f'FigureS{i}' for i in range(1,len(SUPPLEMENT_ORDER)+1)]
 if set(rows)!=set(expected) or len(CATALOG)!=len(expected):
  raise RuntimeError('Figure catalogue coverage or duplicate error.')
 (root/'manuscript/figure_catalog.json').write_text(json.dumps([rows[name] for name in expected],ensure_ascii=False,indent=2))
 (root/'config/figure_identity_map.json').write_text(json.dumps({f'FigureS{i}':publication_id(f'FigureS{i}') for i in range(1,46)},indent=2)+'\n')
 (root/'config/paper_figure_plan.json').write_text(json.dumps({'version':'2026-10-25-reviewed-manuscript','status':f'5 main and {len(SUPPLEMENT_ORDER)} supplementary manuscript figures. Prior presentation remains unchanged.','figures':[rows[name] for name in expected]},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
 from t2t_paper.figures import run as entry
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);entry(p.parse_args().root)
