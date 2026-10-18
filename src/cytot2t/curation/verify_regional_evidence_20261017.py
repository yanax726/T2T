"""Read-only replay check for the regional supplement; writes only its audit report."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from cytot2t.curation.regional_evidence_20261017 import run


def main():
    root=Path(os.environ.get('T2T_ROOT',Path(__file__).resolve().parents[3])).resolve()
    clinical=root/'results/tables/update_20261017/clinical'; output=root/'results/tables/update_20261017/regional'
    prior={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in clinical.glob('*.csv')}
    source=Path('results/tables/update_20260925/clinical.csv')
    with tempfile.TemporaryDirectory(prefix='regional_replay_') as td:
        other=Path(td);(other/source).parent.mkdir(parents=True);shutil.copy2(root/source,other/source)
        manifest=run(other)
        identical={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==manifest['outputs'][p.name] for p in output.glob('*.csv')}
    unchanged=all(hashlib.sha256((clinical/p).read_bytes()).hexdigest()==h for p,h in prior.items())
    raw=pd.read_csv(root/source);raw=raw[raw.age.ge(12)&~raw.dose_ambiguous&raw.regional_reliable]
    pairs=raw[raw.treatment.eq('dupilumab')].pivot(index='patient_uid',columns='visit_week',values='easi_head_neck')[[0,12]].dropna()
    x=100*(pairs[0]-pairs[12])/pairs[0].where(pairs[0].gt(0)); x=x.dropna()
    summary=pd.read_csv(output/'regional_paired_improvement.csv');row=summary[(summary.scope=='dupilumab')&(summary.region=='head_neck')&(summary.start==0)&(summary.end==12)&(summary.outcome=='pct_improvement')].iloc[0]
    independent=bool(int(row.n)==len(x) and np.isclose(row['mean'],x.mean(),rtol=0,atol=1e-12))
    report={'pass':all(identical.values()) and unchanged and independent,'regional_outputs_identical':identical,'original_clinical_tables_unchanged':unchanged,'clinical_tables_checked':len(prior),'independent_head_neck_percentage_check':independent,'n_pairs':len(x),'mean_percentage_improvement':x.mean()}
    destination=root/'reports/update_20261017/clinical/regional_replay_verification.json';destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));assert report['pass']


if __name__=='__main__': main()
