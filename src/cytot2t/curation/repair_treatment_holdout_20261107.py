"""Rebuild treatment stress tests while preserving frozen estimates and centre rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from cytot2t.curation import unified_model_20261017 as model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dimension_lines(text, value):
    lines=text.splitlines(keepends=True)
    header=next(csv.reader([lines[0]]))
    position=header.index('held_out_dimension')
    selected=[]
    for line in lines[1:]:
        row=next(csv.reader([line]))
        assert len(row)==len(header)
        if row[position]==value:
            selected.append(line)
    return lines[0],selected


def run(root):
    report=root/'reports/update_20261107/model'
    manifest=json.loads((report/'holdout_before_hashes.json').read_text())
    archive=report/'holdout_before.zip'
    assert sha(archive)==manifest['archive_sha256']
    for path,value in manifest['protected'].items():
        assert sha(root/path)==value,path
    cfg=json.loads((root/'config/unified_model_20261017.json').read_text())
    data=pd.read_csv(root/cfg['input_path'])
    ambiguous=set(data.loc[data.dose_ambiguous.astype(bool),'patient_uid'])
    original_select=model.select_model
    fitted_inputs=[]

    def checked_select(train,candidates,settings):
        assert ambiguous.isdisjoint(train.patient_uid)
        assert train.visit_week.isin([0,12]).all()
        assert not train.dose_ambiguous.astype(bool).any()
        fitted_inputs.append(dict(n_training_patients=train.patient_uid.nunique(),
                                  n_training_visits=len(train),
                                  train_ids='|'.join(sorted(train.patient_uid.unique())),
                                  ambiguous_patients_in_training=0))
        return original_select(train,candidates,settings)

    model.select_model=checked_select
    try:
        predictions,audit=model.held_out_groups(data,cfg,dimensions=('treatment',))
    finally:
        model.select_model=original_select
    assert len(fitted_inputs)==3 and len(audit)==3
    assert ambiguous.isdisjoint(predictions.patient_uid)
    assert not predictions.dose_ambiguous.astype(bool).any()
    assert set(predictions.held_out_dimension)=={'treatment'}
    training=pd.DataFrame(fitted_inputs)
    training['held_out_group']=audit.held_out_group.to_numpy()
    for row in training.itertuples():
        testids=set(predictions.loc[predictions.held_out_group.eq(row.held_out_group),'patient_uid'])
        assert testids.isdisjoint(row.train_ids.split('|'))
    training.to_csv(report/'holdout_training_audit_PRIVATE.csv',index=False)
    rows=[]
    for (dimension,group,week),sub in predictions.groupby(['held_out_dimension','held_out_group','visit_week']):
        rows.append(dict(held_out_dimension=dimension,held_out_group=group,week=week,**model.metric(sub,cfg)))
    metrics=pd.DataFrame(rows)
    frames={'predictions_private':predictions,'fit_audit':audit,'metrics':metrics}
    unchanged_centres={}
    with ZipFile(archive) as z:
        old_metrics=None
        for suffix,frame in frames.items():
            path=f'results/tables/update_20261017/model/held_out_group_{suffix}.csv'
            before=z.read(path).decode()
            assert hashlib.sha256(before.encode()).hexdigest()==manifest['before'][path]
            header,centre=dimension_lines(before,'center')
            columns=next(csv.reader([header]))
            assert set(frame.columns)==set(columns)
            new_text=frame[columns].to_csv(index=False)
            _,treatment=dimension_lines(new_text,'treatment')
            after=header+''.join(centre+treatment)
            (root/path).write_text(after)
            _,centre_after=dimension_lines(after,'center')
            assert centre_after==centre
            unchanged_centres[suffix]=len(centre)
            if suffix=='metrics':
                old_metrics=pd.read_csv(io.StringIO(before))
    comparison=old_metrics[old_metrics.held_out_dimension.eq('treatment')].merge(
        metrics,on=['held_out_dimension','held_out_group','week'],validate='one_to_one',suffixes=('_before','_after'))
    assert np.array_equal(comparison.n_patients_before,comparison.n_patients_after)
    comparison['MAE_change']=comparison.MAE_after-comparison.MAE_before
    comparison.to_csv(report/'holdout_metric_changes.csv',index=False)
    for path,value in manifest['protected'].items():
        assert sha(root/path)==value,path
    outcome=dict(status='PASS',ambiguous_patients_removed_from_treatment_training=len(ambiguous),
                 treatment_fit_count=len(training),patient_overlap=0,ambiguous_test_patients=0,
                 frozen_outputs_unchanged=True,centre_csv_rows_byte_identical=unchanged_centres,
                 treatment_evaluation_counts_unchanged=True,
                 max_abs_treatment_MAE_change=float(abs(comparison.MAE_change).max()),
                 before_hashes=manifest,
                 after_sha256={path:sha(root/path) for path in manifest['before']},
                 repair_source_sha256=sha(Path(__file__)))
    (report/'holdout_repair_verification.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print(comparison[['held_out_group','week','n_patients_after','MAE_before','MAE_after','MAE_change']].to_string(index=False))
    print('PASS: only treatment stress-test rows rebuilt; centre rows and frozen outputs unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    run(p.parse_args().root.resolve())
