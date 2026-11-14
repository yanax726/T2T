"""Check exploratory prognosis tables against their distributed aggregates."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


def run(root):
    root = Path(root)
    report = root/'reports/update_20261114/prognosis_integration'
    report.mkdir(parents=True, exist_ok=True)
    checks, sources = [], {}

    def record(check, passed, detail=''):
        checks.append(dict(check=check, passed=bool(passed), detail=detail))

    def read(path):
        p = root/path
        sources[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p)

    appendix = (root/'manuscript/appendix.tex').read_text()
    main = (root/'manuscript/main.tex').read_text()
    t19 = (root/'manuscript/tables/itch_panel_size.tex').read_text()
    t20 = (root/'manuscript/tables/landmark_three.tex').read_text()
    record('Supplementary guide reaches S20', 'Tables S1--S20' in appendix)
    record('New tables appear in the appendix', all('tables/'+name in appendix
           for name in ['itch_panel_size', 'landmark_three']))
    record('Main text separates future response from E-EASI',
           'Future-response prediction remained exploratory' in main
           and 'rather than contemporaneous EASI estimation' in main
           and 'Supplementary Tables~S18--S20' in main)

    panel = read('results/tables/update_20261112/itch_panel_size/metrics.csv')
    stress = read('results/tables/update_20261112/itch_panel_size/batch_stress/aggregate_metrics.csv')
    land = read('results/tables/update_20261114/landmark_three/metrics.csv')
    delta = read('results/tables/update_20261114/landmark_three/paired_differences.csv')
    reps = read('results/tables/update_20261114/landmark_three/repeat_metrics.csv')
    for table, data, text in [('S19 patient held out', panel[panel.evaluation.eq('mean_repetition')], t19),
                              ('S20 landmark', land[land.evaluation.eq('mean_repetition')], t20)]:
        for population, z in data.groupby('population'):
            for model, rows in z.groupby('model'):
                a = rows[rows.metric.eq('AUROC')].iloc[0]
                literal = f'{a.estimate:.3f} ({a.ci_low:.3f}--{a.ci_high:.3f})'
                record(table+'/'+population+'/'+model+' AUROC matches text', literal in text)
                record(table+'/'+population+'/'+model+' CI is ordered',
                       0 <= a.ci_low <= a.estimate <= a.ci_high <= 1)
    for (population, model), z in stress.groupby(['population', 'model']):
        a = z[z.metric.eq('AUROC')].iloc[0]
        literal = f'{a.estimate:.3f} ({a.ci_low:.3f}--{a.ci_high:.3f})'
        record('S19 file holdout/'+population+'/'+model+' AUROC matches text', literal in t19)

    for population, n, events in [('complete_three_markers', 68, 59),
                                  ('available_serum_imputation_sensitivity', 72, 63)]:
        z = land[land.population.eq(population)]
        record('Landmark '+population+' population',
               z.n.eq(n).all() and z.events.eq(events).all() and z.nonevents.eq(n-events).all())
        s = reps[reps.population.eq(population)]
        record('Landmark '+population+' five scored repetitions per model',
               s.groupby('model').size().eq(5).all())
        z = z[z.evaluation.eq('mean_repetition')]
        for model in z.model.unique():
            a = z[z.model.eq(model)].set_index('metric')
            r = s[s.model.eq(model)]
            record('Landmark '+population+'/'+model+' repetition replay',
                   all(abs(r[m].mean()-a.loc[m, 'estimate']) < 1e-12
                       for m in ['AUROC', 'AUPRC', 'Brier', 'log_loss', 'mean_prediction']))
        d = delta[delta.population.eq(population) & delta.evaluation.eq('mean_repetition')
                  & delta.model.eq('clinical_three_markers') & delta.metric.eq('AUROC')].iloc[0]
        expected = (z[z.model.eq('clinical_three_markers') & z.metric.eq('AUROC')].estimate.iloc[0]
                    - z[z.model.eq('clinical') & z.metric.eq('AUROC')].estimate.iloc[0])
        record('Landmark '+population+' paired difference', abs(d.difference-expected) < 1e-12)
        record('Landmark '+population+' uncertain increment retained', d.ci_low < 0 < d.ci_high)
    record('S19 identifies selection procedure and source uncertainty',
           all(s in t19 for s in ['not one fixed marker list', 'not a noninferiority test',
                                 'not a verified physical assay batch', 'not external validation']))
    record('S20 defines outcome and time boundary',
           all(s in t20.lower() for s in ['does not require easi75 at w12', 'rather than w0--w12 differences',
                                         'no w24 outcome is imputed', 'conditional']))
    index = read('manuscript/source_data/index.csv')
    record('Source index includes both new analyses',
           set(sources).difference({'manuscript/source_data/index.csv'}).issubset(set(index.file)))
    pd.DataFrame(checks).to_csv(report/'publication_checks.csv', index=False)
    summary = dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
                   checks=len(checks), failed=[x for x in checks if not x['passed']],
                   source_sha256=sources, version=(root/'VERSION').read_text().strip(),
                   scope='Aggregate-to-publication consistency; independent raw-data audits are separate.')
    (report/'publication_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ['status', 'checks', 'failed']}, indent=2))
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    run(Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])))
