"""Reconcile original clinical and assay workbooks against frozen identity decisions.

Adapted from existing phase1 semantic extraction, v2 center/name/code serum
linkage, and v4 clinical range QC. No legacy results are used as raw data.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

SECTIONS={'基线':0,'第一周':1,'第二周':2,'第四周':4,'第八周':8,
          '第十二周':12,'第十六周':16,'第二十周':20,'第二十四周':24,'第四十八周':48}
BASE={'患者编号':'local_patient_id_raw','患者门诊号':'outpatient_id_raw',
      '患者姓名':'patient_name_raw','分中心名称':'center_name','入组时间':'enrollment_date_raw',
      '性别':'sex','年龄':'age','BMI':'bmi','患者分组':'treatment_group_raw'}
VARS={'PtGA':'ptga','IGA':'iga','PP-NRS':'pp_nrs','头颈EASI':'easi_head_neck',
      '上肢EASI':'easi_upper_limb','躯干EASI':'easi_trunk','下肢EASI':'easi_lower_limb',
      '总EASI':'easi_total','EASI':'easi_total','DLQI':'dlqi','SCORAD':'scorad','BSA':'bsa',
      'IgE':'ige','IGE':'ige','EO#':'eo_count','血清':'serum_flag','PBMC':'pbmc_flag',
      '不良事件':'adverse_event_text','复发情况':'relapse_text','治疗总费用':'total_cost_this_period',
      '皮损形态图像':'image_flag','皮肤表面菌群样本':'skin_microbiome_flag'}


def norm(x):
    if pd.isna(x):return ''
    return re.sub(r'\.0$','',str(x).strip())


def name(x):
    return re.split(r'[（(]',norm(x))[0].strip()


def local_id(x):
    return norm(x).lstrip('0') or '0'


def treatment(x):
    s=norm(x).replace(' ','')
    if s=='达必妥':return 'dupilumab'
    for dose in (100,200):
        if s==f'阿布昔替尼{dose}mgQD':return f'abrocitinib_{dose}mg'
    raise ValueError('Unmapped raw treatment encoding')


def clinical_from_raw(root,cfg):
    meta=pd.read_csv(root/cfg['identity_records'])
    episodes=pd.read_csv(root/cfg['episode_decisions'])
    records=[]; cells=[]; headers=[]
    for key,filename in cfg['clinical_workbooks'].items():
        path=root/cfg['raw_root']/filename
        d=pd.read_excel(path,header=None)
        section=None; baseidx={}; visitidx={}
        for j in range(d.shape[1]):
            if pd.notna(d.iat[0,j]):section=str(d.iat[0,j]).strip()
            var=d.iat[1,j]
            if pd.isna(var):continue
            var=str(var).strip()
            headers.append(dict(source=filename,column=j+1,section=section,raw_header=var,
                                week=SECTIONS.get(section),canonical=BASE.get(var,VARS.get(var,''))))
            if section=='基本信息' and var in BASE:baseidx[BASE[var]]=j
            if section in SECTIONS and var in VARS:
                k=(SECTIONS[section],VARS[var])
                if k in visitidx:raise ValueError(f'Duplicate semantic column {filename}/{k}')
                visitidx[k]=j
        for j in range(2,len(d)):
            r=d.iloc[j]
            if all(pd.isna(r[v]) for v in baseidx.values()):continue
            rid=f'{key}__row{j-2:03d}'
            record=dict(record_id=rid,source_workbook=filename,excel_row=j+1,
                        **{k:r[v] for k,v in baseidx.items()})
            records.append(record)
            for (week,var),col in visitidx.items():
                cells.append(dict(record_id=rid,visit_week=week,variable=var,value_raw=r[col],
                                  source_workbook=filename,excel_row=j+1,excel_column=col+1))
    records=pd.DataFrame(records); cells=pd.DataFrame(cells)
    if set(records.record_id)!=set(meta.record_id):raise ValueError('Raw treatment record universe differs from identity decisions')
    check=records.merge(meta,on='record_id',suffixes=('_rawread','_decision'),validate='one_to_one')
    for col in ['patient_name_raw','local_patient_id_raw','center_name','sex','age','bmi','treatment_group_raw','enrollment_date_raw']:
        a=check[col+'_rawread'].map(norm);b=check[col+'_decision'].map(norm)
        if not a.equals(b):
            bad=check.loc[a!=b,'record_id'].tolist()
            raise ValueError(f'Identity input changed: {col}, records {bad}')
    records=records.merge(meta[['record_id','patient_uid']],on='record_id',validate='one_to_one')
    records=records.merge(episodes[['source_record_id','treatment_episode_id','episode_order','episode_treatment',
                                  'n_episodes_this_patient','same_day_dual_dose_flag']],
                          left_on='record_id',right_on='source_record_id',validate='one_to_one')
    records['treatment']=records.treatment_group_raw.map(treatment)
    assert (records.treatment==records.episode_treatment).all(), 'Raw treatment versus episode discrepancy'
    first=records[records.episode_order==cfg['primary_episode']].copy()
    assert not first.patient_uid.duplicated().any()
    assert first.patient_uid.nunique()==records.patient_uid.nunique()
    long=cells.pivot(index=['record_id','visit_week'],columns='variable',values='value_raw').reset_index()
    long=long.merge(first[['record_id','patient_uid','treatment_episode_id']],on='record_id',validate='many_to_one')
    numeric={**cfg['clinical_numeric_ranges'],'eo_count':[0,20],'total_cost_this_period':[0,np.inf]}
    edits=[]
    for var,bounds in numeric.items():
        if var not in long:continue
        col=var+'_numeric';long[col]=pd.to_numeric(long[var],errors='coerce')
        for idx in long.index[long[var].notna() & long[col].isna()]:
            edits.append(dict(patient_uid=long.at[idx,'patient_uid'],visit_week=int(long.at[idx,'visit_week']),
                              variable=var,raw_value=str(long.at[idx,var]),action='set_missing',reason='nonnumeric_raw_cell'))
        bad=long[col].notna() & ~long[col].between(*bounds)
        for idx in long.index[bad]:
            edits.append(dict(patient_uid=long.at[idx,'patient_uid'],visit_week=int(long.at[idx,'visit_week']),
                              variable=var,raw_value=str(long.at[idx,var]),action='set_missing',reason=f'outside_range_{bounds}'))
        long.loc[bad,col]=np.nan
    # Preserve every scheduled slot; endpoints use actual measurement masks.
    long['any_raw_visit_data']=long[sorted(c for c in set(VARS.values()) if c in long)].notna().any(axis=1)
    regions=[v+'_numeric' for v in cfg['clinical_numeric_ranges'] if v.startswith('easi_') and v!='easi_total']
    long['regional_sum']=long[regions].sum(axis=1,min_count=4)
    long['regional_total_diff']=(long.regional_sum-long.easi_total_numeric).abs()
    long['regional_reliable']=long.regional_total_diff.le(0.01)&long[regions].notna().all(axis=1)
    first['center']=first.center_name.map({'中国医学科学院皮肤病医院':'CAMS','上海华山医院':'HUASHAN'})
    assert first.center.notna().all(), 'Unknown center'
    first['sex_male']=first.sex.map({'男':1,'女':0})
    assert first.sex_male.notna().all(), 'Unrecognized sex value; do not silently label female'
    first['abro_dose']=first.treatment.str.extract(r'_(100|200)mg')[0].astype(float)
    first['dose_ambiguous']=first.same_day_dual_dose_flag.astype(bool)
    # Only deidentified fields leave the in-memory raw extraction.
    cohort=first[['patient_uid','treatment_episode_id','record_id','source_workbook','excel_row',
                  'treatment','center','sex_male','age','bmi','abro_dose','dose_ambiguous','n_episodes_this_patient']].copy()
    for c,bounds in [('age',(0,100)),('bmi',(10,60))]:
        cohort[c]=pd.to_numeric(cohort[c],errors='coerce');cohort.loc[~cohort[c].between(*bounds),c]=np.nan
    return cohort,long,records,pd.DataFrame(headers),pd.DataFrame(edits)


def samples_from_raw(root,cfg,records):
    raw=root/cfg['raw_root']/cfg['outsource_workbook']
    cw=pd.read_csv(root/cfg['identity_crosswalk'])
    cw['nm']=cw.patient_name_clean.map(name);cw['cd']=cw.local_patient_id_raw.map(local_id)
    rows=[]
    for sheet in pd.ExcelFile(raw).sheet_names:
        d=pd.read_excel(raw,sheet_name=sheet,header=None)
        header=next(i for i in range(min(5,len(d))) if (d.iloc[i].astype(str).str.contains('基线名称')).any())
        cols={norm(v):j for j,v in enumerate(d.iloc[header]) if pd.notna(v)}
        for i in range(header+1,len(d)):
            r=d.iloc[i]
            if sheet=='达标':
                center='中国医学科学院皮肤病医院';code=r[cols['达标患者编号']]
            else:
                center='上海华山医院';code=r[cols['编号']]
            nm=name(r[cols['姓名']]);cd=local_id(code)
            for column,week in cfg['serum_week_by_column'].items():
                if column not in cols or pd.isna(r[cols[column]]):continue
                sid=norm(r[cols[column]])
                # Explicit worksheet total, not a patient or sample; never use
                # a fixed nrows cutoff to discard unknown rows.
                if re.fullmatch(r'共\d+',sid) and nm=='' and pd.isna(code):
                    continue
                if not re.fullmatch(r'P\w+BT[1-4]',sid):
                    raise ValueError(f'Unrecognized serum cell: sheet={sheet}, row={i+1}, column={column}')
                match=cw[(cw.center_name==center)&(cw.nm==nm)&(cw.cd==cd)]
                method='center_name_local_code'
                if len(match)==0:
                    match=cw[(cw.center_name==center)&(cw.nm==nm)]
                    method='center_unique_name_alternate_code'
                if len(match)!=1:raise ValueError(f'Nonunique serum patient linkage: sheet={sheet}, row={i+1}')
                uid=match.iloc[0].patient_uid
                expected={0:'BT1',12:'BT2',24:'BT3',48:'BT4'}[week]
                assert sid.endswith(expected), 'Serum suffix and raw visit column disagree'
                ambiguous=bool(records.loc[records.patient_uid==uid,'same_day_dual_dose_flag'].any())
                rows.append(dict(sample_id=sid,patient_uid=uid,visit_week=week,outsource_sheet=sheet,
                                 source_excel_row=i+1,source_excel_column=cols[column]+1,
                                 link_method=method,dose_ambiguous=ambiguous))
    link=pd.DataFrame(rows)
    assert not link.sample_id.duplicated().any(), 'Duplicate outsource samples'
    assert not link[['patient_uid','visit_week']].duplicated().any(), 'Multiple samples per patient/visit need adjudication'
    return link


def assay_from_raw(root,cfg,link):
    labs=list((root/cfg['raw_root']).glob('*LABEX*'))
    assert len(labs)==1, 'Raw assay directory ambiguous'
    frames=[]; inputs=[]
    for platform in ['Luminex','MSD']:
        p=labs[0]/f'Processed_{platform}'/'Summary/4.Integration/Summary.xlsx'
        inputs.append(p)
        d=pd.read_excel(p).rename(columns={'Sample':'sample_id','Concentration':'raw_value','Indicator':'raw_marker',
                                          'MFI':'mfi','Unit':'unit','File Name':'assay_file'})
        d['sample_id']=d.sample_id.map(norm);d['platform']=platform
        d['analyte']=d.raw_marker.replace(cfg['marker_aliases'])
        assert d.analyte.notna().all()
        assert not d[['sample_id','analyte']].duplicated().any(), 'Unadjudicated assay replicates'
        frames.append(d)
    allraw=pd.concat(frames,ignore_index=True)
    measured=allraw.merge(link,on='sample_id',validate='many_to_one')
    s=measured.raw_value.astype(str).str.strip()
    lower=s.str.contains('[<＜↓]',regex=True);upper=s.str.contains('[>＞↑]',regex=True)
    numbers=pd.to_numeric(s.str.replace('[<>＜＞↑↓]','',regex=True),errors='coerce')
    measured['state']=np.select([lower,upper,numbers.isna(),numbers.le(0)],
                               ['below_lloq','above_uloq','not_measured','assay_failure'],default='observed')
    measured['censor_bound']=numbers.where(lower|upper)
    measured['concentration']=numbers.where(measured.state=='observed')
    measured['log2_concentration']=np.log2(measured.concentration.where(measured.concentration.gt(0)))
    # Retain all actual assay rows, even if no marker is quantifiable.
    wide=measured.pivot(index=['patient_uid','visit_week'],columns='analyte',values='log2_concentration')
    wide=wide.add_prefix('log2_').reset_index()
    wide['n_analytes_observed']=wide.filter(regex='^log2_').notna().sum(axis=1)
    universe=link.merge(measured[['sample_id']].drop_duplicates().assign(has_assay=True),on='sample_id',how='left')
    universe['has_assay']=universe.has_assay.eq(True)
    return measured,wide,universe,inputs


def compare_numeric(new,old,keys,columns):
    assert not new[keys].duplicated().any()
    assert not old[keys].duplicated().any()
    m=new.merge(old,on=keys,how='outer',suffixes=('_raw_rebuilt','_legacy'),indicator=True)
    rows=[]
    for col in columns:
        if col not in new or col not in old:continue
        a=pd.to_numeric(m[col+'_raw_rebuilt'],errors='coerce')
        b=pd.to_numeric(m[col+'_legacy'],errors='coerce')
        good=np.isclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True)
        for idx in m.index[~good]:
            rows.append({**{k:m.at[idx,k] for k in keys},'variable':col,'rebuilt':a.at[idx],'legacy':b.at[idx],
                         'key_match':m.at[idx,'_merge']})
    return pd.DataFrame(rows,columns=keys+['variable','rebuilt','legacy','key_match'])


def run(root):
    root=Path(root);cfg=json.loads((root/'config/input_sources.json').read_text())
    out=root/'data/processed';out.mkdir(parents=True,exist_ok=True)
    reports=root/'reports/takeover';reports.mkdir(parents=True,exist_ok=True)
    cohort,clinical,records,headers,edits=clinical_from_raw(root,cfg)
    link=samples_from_raw(root,cfg,records)
    measured,wide,universe,assayinputs=assay_from_raw(root,cfg,link)
    oldclinical=pd.read_csv(root/'data/reference/clinical_long.csv')
    oldwide=pd.read_csv(root/'data/reference/molecular_wide_log2.csv')
    oldlink=pd.read_csv(root/'data/reference/cytokine_sample_linkage.csv')
    assert set(cohort.patient_uid)==set(pd.read_csv(root/'data/reference/analysis_cohort.csv').patient_uid)
    keys=['patient_uid','visit_week']
    a=link[['sample_id']+keys].sort_values('sample_id').reset_index(drop=True)
    b=oldlink[['sample_id']+keys].sort_values('sample_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    clinical_diff=compare_numeric(clinical,oldclinical,keys,[v+'_numeric' for v in cfg['clinical_numeric_ranges']])
    molecular_diff=compare_numeric(wide,oldwide,keys,list(wide.filter(regex='^log2_').columns))
    clinical_diff.to_csv(reports/'clinical_reconciliation_differences.csv',index=False)
    molecular_diff.to_csv(reports/'molecular_reconciliation_differences.csv',index=False)
    edits.to_csv(reports/'raw_cell_qc_actions.csv',index=False)
    headers.to_csv(reports/'raw_clinical_header_mapping.csv',index=False)
    cohort.to_csv(out/'analysis_cohort_raw_rebuilt.csv',index=False)
    clinical.to_csv(out/'clinical_long_raw_rebuilt.csv',index=False)
    link.to_csv(out/'sample_linkage_raw_rebuilt.csv',index=False)
    measured.to_csv(out/'assay_long_raw_rebuilt.csv',index=False)
    wide.to_csv(out/'molecular_wide_raw_rebuilt.csv',index=False)
    universe.to_csv(reports/'serum_sample_reconciliation.csv',index=False)
    summary={'raw_records':len(records),'patients':len(cohort),'treatments':cohort.treatment.value_counts().to_dict(),
             'clinical_scheduled_slots':len(clinical),'clinical_data_bearing_slots':int(clinical.any_raw_visit_data.sum()),
             'clinical_evaluable_by_week':clinical.groupby('visit_week')[['easi_total_numeric','pp_nrs_numeric']].count().to_dict(),
             'outsource_samples':len(link),'assay_samples':measured.sample_id.nunique(),
             'assay_visits':sorted(measured.visit_week.unique().tolist()),
             'assayed_markers':measured.analyte.nunique(),'serum_by_week':wide.groupby('visit_week').size().to_dict(),
             'clinical_value_differences':len(clinical_diff),'molecular_value_differences':len(molecular_diff),
             'identity_policy':cfg['identity_policy'],
             'input_time':'Serum only at raw outsource-column-defined visits; no invented W2/W4 molecular measurements.'}
    (reports/'raw_reconciliation_summary.json').write_text(json.dumps(summary,indent=2))
    paths=[root/'config/input_sources.json']+[root/cfg['raw_root']/v for v in cfg['clinical_workbooks'].values()]
    paths += [root/cfg['raw_root']/cfg['outsource_workbook']]+assayinputs
    paths += [root/cfg[k] for k in ['identity_records','episode_decisions','identity_crosswalk','historical_score_equations']]
    input_lock=[{'path':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    lock=root/'config/raw_input_manifest.json'
    if lock.exists():assert json.loads(lock.read_text())==input_lock, 'Locked raw input changed'
    else:lock.write_text(json.dumps(input_lock,ensure_ascii=False,indent=2))
    print(json.dumps(summary,indent=2))
    if len(clinical_diff) or len(molecular_diff):
        raise ValueError('Raw-versus-legacy discrepancies require cell-level adjudication before downstream stages; see reconciliation CSVs.')
    return {'evidence':['reports/takeover/raw_reconciliation_summary.json','config/raw_input_manifest.json'],
            'raw_recomputed_patients':len(cohort)}
