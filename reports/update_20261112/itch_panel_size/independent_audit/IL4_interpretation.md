# IL-4 selection and assay-file grouping

Stable feature selection does not independently validate IL-4 as a prognostic biomarker. IL-4 was repeatedly selected within the current training folds, but the candidate data also show assay-file and case-mix structure.

In the 102-patient complete-three-marker population:

| Anonymous report-file group | Centre | n | Responders | Nonresponders | IL-4 median log2 concentration |
|---|---|---:|---:|---:|---:|
| assay_file_1 | CAMS | 33 | 24 | 9 | 6.380 |
| assay_file_2 | CAMS | 40 | 37 | 3 | 6.516 |
| assay_file_3 | Huashan | 14 | 14 | 0 | 6.502 |
| assay_file_4 | Huashan | 15 | 13 | 2 | 6.576 |

The first group contains nine of the 14 nonresponders and has lower IL-4 values overall. This structure exists within CAMS, so including centre alone cannot automatically account for it. File groups also have different treatment mixtures: the first contains mostly dupi, the second mostly abrocitinib, the third only dupi, and the fourth mostly abrocitinib 100 mg. Initial treatment is already a clinical predictor, but this does not demonstrate complete control of file-specific or centre-by-treatment differences.

Within-file directions are not uniform. IL-4 medians for nonresponders versus responders are 6.305 versus6.380 in file1, 6.577 versus6.516 in file2 and6.257 versus6.606 in file4; the latter two have only three and two nonresponders. File3 has no nonresponder. Thus the descriptive data cannot establish that the predictive contribution is entirely technical, or that it is entirely biological.

These groups are based on the observed `assay_file` field. A file is not a verified plate/run identifier, and this audit does not establish an assay calibration shift. The observation raises a technical/case-mix sensitivity question. Random patient folds mix these file groups between training and testing; they do not independently test transport to a new assay run. A claim of biological prognostic information requires more support than the frequency with which IL-4 was selected.

No new predictive model, adjusted effect or significance test was fitted for this description. The numerical replay remains valid, while the biological interpretation and transferability remain exploratory.
