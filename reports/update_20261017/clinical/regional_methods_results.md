# 区域分析补齐：F40/F41

本补充按历史代码 `src/cytot2t/redevelopment/molecular.py:regions` 的可核定义重算，未读取旧汇总数作为新结果。输入为当前 `results/tables/update_20260925/clinical.csv` 临床快照；不再要求患者同时有血液样本。因此这是临床区域改善分析，不是血液亚组分析。原始 Excel 到该快照的逐单元格血缘并非本模块所重建；输入 SHA256 见区域 `manifest.json`。

分析统一纳入年龄≥12岁且初始治疗清楚的163人（D=65、A100=56、A200=42）。四区域必须均有值、在各区域允许范围内，且区域和与记录EASI总分相差≤0.01分。重新计算的可靠标记与快照完全一致。W0/W12/W24合格区域记录分别151/120/88；各组与缺失数见 `regional_availability.csv`。没有以总EASI反推、修补区域分数。

## 图和表如何解释

- `regional_normalized_burden.csv`：区域EASI贡献分别除以头颈7.2、上肢14.4、躯干21.6、下肢28.8。数值0表示没有该区域负担，1为该区域理论满分。每时点为实际可评估患者的均值和95% t置信区间，患者集合可以变化。若纵轴用百分数，整个效应和区间一起×100。
- `regional_paired_improvement.csv`：同一患者、同一区域两端均可靠。`raw_improvement`=起点−终点（区域EASI分）；`normalized_improvement`=该差值/区域理论满分；`pct_improvement`=100×该差值/起点，仅起点>0可计算。正值表示改善，负值表示恶化。起点为0的人不被写为0%改善，其人数单列。每组n≥5才给推断区间。
- F40所需D对合并A及完整三组结果都保留。合并A是实际患者的混合，未把两个剂量组均值简单相加或等权平均。
- `regional_treatment_contrasts.csv`：四区域、三个窗口、D−A100/D−A200/A200−A100/D−合并A，均给原始分和百分比改善的组间差及区间。观察差用Welch检验；调整模型以初始三治疗组、起点区域分、W0总EASI、年龄、性别、BMI和中心解释改善，HC3异方差稳健协方差，残余自由度t区间。合并A调整对比依该窗口可评估人群的剂量比例标准化。最大杠杆值0.620，全部模型对比可估。每种分析、每种结果各48项检验分别BH校正，区间为名义95%，并非多重比较同时区间。
- `whole_body_PP_NRS_improvement.csv`：另外给全身瘙痒起点−终点。CRF没有可核的四区域瘙痒记录，不能把PP-NRS人为拆成四区域。

W0→W12四区域平均改善在D组约63.5%–75.3%，合并A组约71.9%–79.7%；W0→W24为74.4%–82.4%和73.0%–91.4%。这些百分比使用各区域起点>0的配对患者，分母随区域而异。W24归一化均值的D/合并A分别为头颈0.0608/0.0435，上肢0.0729/0.0308，躯干0.0553/0.0275，下肢0.0465/0.0207。所有区域组间改善检验在相应48项家族校正后均无q<0.05。不能将本结果解释成相同疗效，也不能由观察均值宣布某药更好。

W12→W24的起点经常已接近0；同样小的反弹除以很小起点可得到很大负百分比，最小个体值−950%。这是定义的分母效应，源值已核，不应为得到好看的图而删去。建议该窗口主图用原始区域分或归一化分的变化，同时把所有百分比结果保留在完整数据表。W0→W12和W0→W24仍可采用老师要求的改善百分比。

初始组不因之后减药而改组。本补充不识别固定剂量的纯药物效应，未对达到目标才减药造成的时间变化混杂进行因果识别；其调整仅控制列出的基线/起点差异。

## 可复现入口与验证

```sh
PYTHONPATH=src .venv/bin/python -m cytot2t.curation.regional_evidence_20261017 --root /path/to/project
T2T_ROOT=/path/to/project PYTHONPATH=src .venv/bin/python -m cytot2t.curation.verify_regional_evidence_20261017
```

仅需要上面一个临床CSV输入，运行代码两文件和Python环境。计算依赖NumPy/Pandas/SciPy/Patsy/Statsmodels（多重校正），不依赖其他历史研究脚本。输出在 `results/tables/update_20261017/regional/`，患者配对明细在其 `restricted/` 内，不应随公开包上传。重跑验证仅把输入复制到临时根目录再算，5张公开CSV与正式输出字节一致，另独立核算D组头颈W0→W12的39对及75.298891%均值，并确认原临床46张表SHA256完全不变。日志与JSON在本目录。

## Methods text

Regional EASI contributions were analysed among patients aged at least 12 years with a known initial regimen. Records required four observed regional contributions within their allowable ranges and agreement between their sum and recorded total EASI within 0.01 points. Regional burden was divided by the theoretical maximum of 7.2, 14.4, 21.6 and 28.8 for head/neck, upper limbs, trunk and lower limbs, respectively. Paired improvement was defined as the earlier minus later contribution. Percentage improvement additionally required a positive earlier contribution; zero-start regions were reported separately rather than assigned an improvement value. We retained weeks 0–12, 12–24 and 0–24. Means and nominal 95% t confidence intervals were reported when at least five paired observations were available. Treatment contrasts were estimated with Welch intervals and, as an adjusted exploratory analysis, linear regression including initial treatment, starting regional contribution, baseline total EASI, age, sex, BMI and centre, with HC3 covariance and residual-degree-of-freedom t inference. The supplementary pooled-abrocitinib contrast used the observed evaluable dose mixture. Benjamini–Hochberg correction was applied across 48 regional-window-contrast tests separately for each outcome and analytical approach. Initial treatment labels were retained after treatment changes. These estimates describe associations under observed treatment pathways and do not identify causal fixed-dose effects. PP-NRS was analysed as a whole-body outcome because regional itch measurements were unavailable.
