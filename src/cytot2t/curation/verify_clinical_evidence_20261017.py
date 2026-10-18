"""Verify a clinical replay in a temporary data root without changing inputs."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('T2T_ROOT',Path(__file__).resolve().parents[3])).resolve()
TABLES=ROOT/'results/tables/update_20261017/clinical'
REPORT=ROOT/'reports/update_20261017/clinical'
INPUTS=[
    'results/tables/update_20260925/clinical.csv',
    'results/tables/redevelopment/molecular.csv',
    'results/tables/redevelopment/assay_long.csv',
    'config/marker_registry.csv',
    'results/tables/update_20261002/maintenance_patient_ledger.csv',
    'results/tables/update_20261002/exposure_at_visits.csv',
    'results/tables/update_20261002/reduction_maintenance_classification.csv',
    'data/restricted/treatment_episode_v0.csv',
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_hashes={name:sha(ROOT/name) for name in INPUTS}
    names=sorted(p.name for p in TABLES.glob('*.csv'))
    first={name:sha(TABLES/name) for name in names}
    with tempfile.TemporaryDirectory(prefix='t2t-clinical-replay-') as temporary:
        replay=Path(temporary)
        for name in INPUTS:
            destination=replay/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        env=os.environ.copy();env['T2T_ROOT']=str(replay)
        source_root=Path(__file__).resolve().parents[2]
        env['PYTHONPATH']=str(source_root)+os.pathsep+env.get('PYTHONPATH','')
        process=subprocess.run([sys.executable,'-m','cytot2t.curation.clinical_evidence_20261017','--root',str(replay),'--bootstrap','1000'],env=env,capture_output=True,text=True)
        (REPORT/'independent_replay.log').write_text(process.stdout+process.stderr)
        if process.returncode:raise RuntimeError('Isolated replay failed; see independent_replay.log')
        second_tables=replay/'results/tables/update_20261017/clinical'
        second={name:sha(second_tables/name) for name in names}
        mismatches=[name for name in names if first[name]!=second[name]]
        assert not mismatches,mismatches
    assert source_hashes=={name:sha(ROOT/name) for name in INPUTS}
    c=pd.read_csv(ROOT/INPUTS[0]);c=c[c.age.ge(12)&~c.dose_ambiguous]
    m=pd.read_csv(ROOT/INPUTS[1]);merged=c.merge(m,on=['patient_uid','visit_week'],validate='one_to_one')
    triples=pd.read_csv(TABLES/'molecular_three_visit_paired.csv')
    for r in triples.itertuples():
        z=merged[merged.treatment.eq(r.treatment)].pivot(index='patient_uid',columns='visit_week',values='log2_'+r.marker).reindex(columns=[0,12,24]).dropna()
        assert len(z)==r.n
        if len(z):assert np.isclose((z[r.week]-z[0]).mean(),r.effect,atol=1e-12)
    rates=pd.read_csv(TABLES/'endpoint_rates.csv')
    row=rates[rates.week.eq(2)&rates.target.eq('PP_NRS_le4')&rates.treatment.eq('abrocitinib_100mg')].iloc[0]
    assert row.k==34 and row.n_evaluable==37
    assert rates[rates.week.eq(2)].target.unique().tolist()==['PP_NRS_le4']
    pd.DataFrame([dict(path=str(Path('results/tables/update_20261017/clinical')/name),sha256=first[name],replay_sha256=second[name],identical=True) for name in names]).to_csv(REPORT/'replay_hashes.csv',index=False)
    result=dict(status='PASS',aggregate_csv_files_compared=len(names),byte_identical=True,inputs_unchanged=True,
                W2_independently_checked=True,all_207_molecular_three_visit_rows_independently_checked=True,
                isolated_root_supported=True,bootstrap_replicates=1000,
                boundary='Replay starts from the curated clinical and processed assay snapshots and audited exposure ledger; it does not claim a raw-XLSX rebuild.')
    (REPORT/'replay_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
