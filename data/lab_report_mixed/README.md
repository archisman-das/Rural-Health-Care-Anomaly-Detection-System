# Mixed lab report dataset

This folder contains upload-testable lab report files with both anomaly-free and anomaly cases.

Fields included in every report:

- Fasting glucose
- Postprandial glucose
- HbA1c
- Hemoglobin
- WBC count
- Platelet count
- LDL
- HDL
- Triglycerides
- AST
- ALT
- Bilirubin
- Albumin
- Creatinine
- Urea
- eGFR
- Sodium
- Potassium
- Chloride
- Bicarbonate

Files:

- `mixed-normal-01.csv` and `mixed-normal-01.pdf`: Routine Panel - Normal (anomaly_free)
- `mixed-normal-02.csv` and `mixed-normal-02.pdf`: Routine Panel - Normal (anomaly_free)
- `mixed-prediabetes.csv` and `mixed-prediabetes.pdf`: Prediabetes and Sugar Drift (anomaly)
- `mixed-diabetes.csv` and `mixed-diabetes.pdf`: Diabetes Control Panel (anomaly)
- `mixed-anemia.csv` and `mixed-anemia.pdf`: Anemia and CBC Review (anomaly)
- `mixed-infection.csv` and `mixed-infection.pdf`: Inflammation / Infection Pattern (anomaly)
- `mixed-kidney.csv` and `mixed-kidney.pdf`: Kidney Function Focus (anomaly)
- `mixed-liver.csv` and `mixed-liver.pdf`: Liver Panel Review (anomaly)
- `mixed-electrolyte.csv` and `mixed-electrolyte.pdf`: Electrolyte Imbalance Panel (anomaly)
- `mixed-combined.csv` and `mixed-combined.pdf`: Combined Multi-System Abnormality (anomaly)
- `mixed-normal-03.csv` and `mixed-normal-03.pdf`: Routine Panel - Normal (anomaly_free)
- `mixed-normal-04.csv` and `mixed-normal-04.pdf`: Routine Panel - Normal (anomaly_free)
- `mixed-borderline-cbc.csv` and `mixed-borderline-cbc.pdf`: Borderline CBC Variation (anomaly)
- `mixed-borderline-lipids.csv` and `mixed-borderline-lipids.pdf`: Borderline Lipid Shift (anomaly)
- `mixed-dehydration.csv` and `mixed-dehydration.pdf`: Dehydration and Salt Stress (anomaly)
- `mixed-hepatic-severe.csv` and `mixed-hepatic-severe.pdf`: Severe Liver Stress (anomaly)
- `mixed-hyperglycemia-mild.csv` and `mixed-hyperglycemia-mild.pdf`: Mild Hyperglycemia (anomaly)
- `mixed-renal-mild.csv` and `mixed-renal-mild.pdf`: Mild Kidney Strain (anomaly)
- `mixed-electrolyte-mild.csv` and `mixed-electrolyte-mild.pdf`: Mild Electrolyte Shift (anomaly)
- `mixed-combined-02.csv` and `mixed-combined-02.pdf`: Combined Multi-System Abnormality (anomaly)
- `mixed-cbc-normal-variant.csv` and `mixed-cbc-normal-variant.pdf`: Routine Panel - Normal (anomaly_free)
- `mixed-metabolic-shift.csv` and `mixed-metabolic-shift.pdf`: Metabolic Risk Shift (anomaly)
- `mixed-hospital-followup.csv` and `mixed-hospital-followup.pdf`: Hospital Follow-up Abnormality (anomaly)

Also included:

- `mixed-dataset.csv`: combined tabular dataset with all samples. Use this when you want one upload-ready CSV that mixes normal and anomalous cases in a single file.
- `mixed-manifest.csv`: quick index of every generated file with the sample name, format, and clinical pattern it represents.

### `mixed-dataset.csv`

This file is the merged dataset for the mixed lab report folder. It contains the same field set as the individual reports, but each row represents one sample record instead of one report file.

Use it when you want to:

- Train or test the model on a larger tabular file
- Run split generation for train, validation, and test folders
- Check how the pipeline behaves on a mixed normal/anomaly dataset
- Load a single CSV instead of uploading separate report files one by one

The combined file is meant for model experimentation and dashboard testing, not as a clinical source of truth.
