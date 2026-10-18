"""Matched sensitivity applying identical 0..72 output bounds to published p-EASI."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from cytot2t.curation.unified_model_20261017 import metric


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('T2T_ROOT',Path(__file__).resolve().parents[3])))
    args=parser.parse_args();root=args.root.resolve();out=root/'results/tables/update_20261017/model'
    cfg=json.loads((root/'config/unified_model_20261017.json').read_text())
    data=pd.read_csv(out/'published_matched_predictions_private.csv');data=data[data.model.eq('unified_selected')].copy()
    rows=[]
    for week,frame in data.groupby('visit_week'):
        unique=frame.drop_duplicates(['patient_uid','visit_week'])
        assert not frame.duplicated(['patient_uid','visit_week','repeat']).any()
        for model,values in [('unified_selected',frame.prediction),('published_original_raw',frame.published_prediction),('published_clipped_0_72',frame.published_prediction.clip(0,72))]:
            result=metric(frame.assign(prediction=values),cfg)
            rows.append(dict(model=model,week=week,published_n_below0=int(unique.published_prediction.lt(0).sum()),published_n_above72=int(unique.published_prediction.gt(72).sum()),comparison='Same exact participants and visits; no refitting; matched to nested unified selection',**result))
    pd.DataFrame(rows).to_csv(out/'published_clipping_sensitivity.csv',index=False)
    existing=pd.read_csv(out/'published_matched_metrics.csv');existing=existing[existing.matched_candidate.eq('unified_selected')]
    for row in rows:
        if row['model']=='published_clipped_0_72':continue
        reference=existing[existing.model.eq(row['model'])&existing.week.eq(row['week'])].iloc[0]
        for name in ['MAE','rho','CCC','band_accuracy']:
            assert np.isclose(row[name],reference[name],atol=1e-12)
    print(pd.DataFrame(rows)[['model','week','n_patients','MAE','rho','CCC','band_accuracy','published_n_below0','published_n_above72']].to_string(index=False))


if __name__=='__main__':main()
