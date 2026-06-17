# Lab report samples

Upload these directly in the Lab Investigation step.

- `normal`: Routine Panel - Normal (anomaly_free)
- `diabetes`: Diabetes Control Panel (anomaly)
- `anemia`: Anemia and CBC Review (anomaly)
- `kidney`: Kidney Function Focus (anomaly)
- `liver`: Liver Panel Review (anomaly)
- `electrolyte`: Electrolyte Imbalance Panel (anomaly)

Each file uses the same field names the uploader parses: fasting glucose, postprandial glucose, HbA1c, hemoglobin, WBC count, platelet count, LDL, HDL, triglycerides, AST, ALT, bilirubin, albumin, creatinine, urea, eGFR, sodium, potassium, chloride, and bicarbonate.

### Dataset notes

These sample files are meant to test the upload parser and the lab-investigation form.

- The normal file shows the baseline pattern the dashboard should accept without raising anomaly flags.
- The disease-pattern files are intentionally varied so you can see how the system responds to different abnormal lab combinations.
- The CSV and PDF versions contain the same clinical content in different formats.

If you want a single merged dataset instead of individual samples, use `data/lab_report_mixed/mixed-dataset.csv`.
