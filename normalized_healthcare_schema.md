# Normalized Healthcare Database Schema

This schema normalizes the previously defined clinical, lab, demographic, environmental, and operational fields into separate related tables.

## Design Goals

- Keep patient identity separate from encounter events
- Store measurements in domain-specific tables
- Avoid repeating shared context across every row
- Support one-to-many and longitudinal data cleanly
- Preserve raw values and source metadata

## Core Tables

### 1. `patients`

Stores relatively stable patient profile data.

Fields:

- `patient_id` string, primary key
- `age_years` integer, nullable
- `gender` string, nullable
- `created_at` datetime, not null
- `updated_at` datetime, not null

Suggested constraints:

- `age_years >= 0`
- `gender` should use a controlled vocabulary

### 2. `encounters`

Stores one clinical encounter or measurement event.

Fields:

- `encounter_id` string, primary key
- `patient_id` string, foreign key to `patients.patient_id`
- `recorded_at` datetime, not null
- `location_type` string, not null
- `source_type` string, not null
- `operator_id` string, nullable
- `device_id` string, nullable
- `measurement_posture` string, nullable
- `measurement_context` text, nullable
- `notes` text, nullable
- `data_quality_flag` string, not null
- `created_at` datetime, not null
- `updated_at` datetime, not null

Suggested constraints:

- `location_type` should use a controlled vocabulary
- `source_type` should use `manual`, `device`, or `imported`
- `data_quality_flag` should use `ok`, `missing`, `suspect`, or `repeat_required`

## Clinical Measurement Tables

### 3. `vital_sign_observations`

Stores bedside and point-of-care measurements for a single encounter.

Fields:

- `vital_sign_id` string, primary key
- `encounter_id` string, foreign key to `encounters.encounter_id`
- `heart_rate_bpm` integer, nullable
- `systolic_bp_mmhg` integer, nullable
- `diastolic_bp_mmhg` integer, nullable
- `spo2_percent` decimal(5,2), nullable
- `body_temperature_c` decimal(5,2), nullable
- `respiratory_rate_bpm` integer, nullable
- `weight_kg` decimal(6,2), nullable
- `height_cm` decimal(6,2), nullable
- `bmi_kg_m2` decimal(6,2), nullable
- `glucose_fasting_mg_dl` decimal(6,2), nullable
- `glucose_postprandial_mg_dl` decimal(6,2), nullable
- `created_at` datetime, not null
- `updated_at` datetime, not null

Suggested constraints:

- `heart_rate_bpm > 0`
- `systolic_bp_mmhg >= diastolic_bp_mmhg`
- `spo2_percent between 0 and 100`
- `body_temperature_c > 0`
- `respiratory_rate_bpm > 0`
- `weight_kg > 0` when present
- `height_cm > 0` when present
- `bmi_kg_m2 > 0` when present

### 4. `lab_results`

Stores all laboratory results in a normalized, extensible form.

Fields:

- `lab_result_id` string, primary key
- `encounter_id` string, foreign key to `encounters.encounter_id`
- `lab_panel` string, not null
- `test_name` string, not null
- `value_numeric` decimal(12,4), nullable
- `value_text` text, nullable
- `unit` string, nullable
- `result_context` string, nullable
- `specimen_time` datetime, nullable
- `source_lab` string, nullable
- `created_at` datetime, not null
- `updated_at` datetime, not null

Suggested panel values:

- `cbc`
- `hba1c`
- `lipid_panel`
- `liver_function`
- `kidney_function`
- `electrolytes`

Suggested test names:

- CBC: `hb`, `wbc`, `platelets`
- HbA1c: `hba1c`
- Lipids: `ldl`, `hdl`, `triglycerides`
- Liver: `alt`, `ast`, `bilirubin`
- Kidney: `creatinine`, `bun`, `egfr`
- Electrolytes: `sodium`, `potassium`, `calcium`

Recommended uniqueness rule:

- `encounter_id + lab_panel + test_name` should be unique when a single result per test is expected

## Context Tables

### 5. `patient_comorbidities`

Stores one or more comorbidities per patient.

Fields:

- `patient_comorbidity_id` string, primary key
- `patient_id` string, foreign key to `patients.patient_id`
- `condition_name` string, not null
- `status` string, nullable
- `created_at` datetime, not null

Suggested condition values:

- `diabetes`
- `hypertension`
- `tb`

### 6. `patient_medications`

Stores current medications per patient or encounter.

Fields:

- `patient_medication_id` string, primary key
- `patient_id` string, foreign key to `patients.patient_id`
- `encounter_id` string, nullable foreign key to `encounters.encounter_id`
- `medication_name` string, not null
- `dose` string, nullable
- `frequency` string, nullable
- `route` string, nullable
- `active` boolean, not null
- `created_at` datetime, not null

### 7. `environmental_context`

Stores area-level social and environmental context.

Fields:

- `environmental_context_id` string, primary key
- `encounter_id` string, foreign key to `encounters.encounter_id`
- `malaria_prevalence_level` string, nullable
- `dengue_prevalence_level` string, nullable
- `sanitation_index` decimal(5,2), nullable
- `nutritional_score` decimal(5,2), nullable
- `distance_to_nearest_facility_km` decimal(8,2), nullable
- `context_scope` string, nullable
- `created_at` datetime, not null

Suggested constraints:

- prevalence levels should use `low`, `moderate`, `high`, or `outbreak`
- `sanitation_index` should use a documented scale, commonly 0 to 1
- `nutritional_score` should use a documented scale, commonly 0 to 100
- `distance_to_nearest_facility_km >= 0`

### 8. `operational_metrics`

Stores longitudinal follow-up and care-process signals.

Fields:

- `operational_metric_id` string, primary key
- `patient_id` string, foreign key to `patients.patient_id`
- `encounter_id` string, nullable foreign key to `encounters.encounter_id`
- `treatment_response_score` decimal(5,2), nullable
- `readmission_frequency` integer, nullable
- `drug_adherence_rate` decimal(5,2), nullable
- `days_between_visits_trend` json/text, nullable
- `window_start_date` date, nullable
- `window_end_date` date, nullable
- `calculation_method` string, nullable
- `created_at` datetime, not null
- `updated_at` datetime, not null

Suggested constraints:

- `treatment_response_score` should use a documented scale, commonly 0 to 1
- `readmission_frequency >= 0`
- `drug_adherence_rate` should use a documented scale, commonly 0 to 1

## Relationship Summary

- One `patient` can have many `encounters`
- One `encounter` can have one `vital_sign_observations` row
- One `encounter` can have many `lab_results`
- One `patient` can have many `patient_comorbidities`
- One `patient` can have many `patient_medications`
- One `encounter` can have one `environmental_context` row
- One `patient` can have many `operational_metrics`

## Index Recommendations

- Index `encounters(patient_id, recorded_at)`
- Index `lab_results(encounter_id, lab_panel, test_name)`
- Index `patient_comorbidities(patient_id, condition_name)`
- Index `patient_medications(patient_id, active)`
- Index `environmental_context(encounter_id)`
- Index `operational_metrics(patient_id, window_end_date)`

## Example Normalized Record Flow

1. Create a row in `patients`
2. Create a row in `encounters`
3. Insert one row in `vital_sign_observations`
4. Insert multiple rows in `lab_results`
5. Insert zero or more rows in `patient_comorbidities`
6. Insert zero or more rows in `patient_medications`
7. Insert one row in `environmental_context` when area context is available
8. Insert one row in `operational_metrics` for follow-up summaries

## Notes

- If your platform prefers a document store, this schema still maps cleanly into nested collections.
- If you want strict normalization, split `days_between_visits_trend` into a separate `visit_intervals` table.
- If you want to support coding systems, add reference tables for diagnoses, medications, and lab tests.

## Preprocessing Before Modeling

Before any model consumes the data, apply these preprocessing steps in a fixed order.

### 1. Missing Value Handling

- Use KNN imputation for numerical features.
- Use mode imputation for categorical features.
- Keep an explicit missingness indicator where possible so the model can learn patterns of absence.
- In rural data, missingness is common, so imputation quality directly affects anomaly scores.
- Fit imputation parameters on the training set only to avoid leakage.

Recommended examples:

- Numerical: heart rate, blood pressure, SpO2, temperature, lab values, adherence rate, sanitation index
- Categorical: gender, location type, source type, comorbidity labels, prevalence levels

Implementation notes:

- Standardize numerical features before KNN imputation if distance-based imputation is used.
- Impute within clinically similar groups when feasible, such as age bands or facility catchment areas.
- Do not impute target labels or outcome fields used for evaluation.

### 2. Feature Scaling

- Apply `StandardScaler` or `MinMaxScaler` to numeric features before model training.
- Use one scaler consistently for a given model family and deployment pipeline.
- Isolation Forest is sensitive to scale, so normalized inputs improve comparability across features.
- Autoencoders also converge faster and more stably on normalized input.
- Fit scaling parameters on the training set only, then reuse them for validation, test, and production inference.

Recommended guidance:

- Use `StandardScaler` when features are roughly Gaussian or when z-score style normalization is preferred.
- Use `MinMaxScaler` when the model benefits from bounded inputs, especially for neural networks.
- Keep the original unscaled values available for clinical reporting and audit trails.

### 3. Feature Engineering and Dimensionality Reduction

- Engineer rolling 7-day and 30-day statistics such as mean and standard deviation.
- Add lag features from previous visits to capture change over time.
- Add interaction terms where clinically useful, such as `age x glucose`.
- Consider encounter-level deltas, trend slopes, and rate-of-change features when longitudinal data is available.
- Apply PCA for dimensionality reduction if the feature count exceeds approximately 50.

Recommended examples:

- Rolling statistics:
  - `heart_rate_mean_7d`
  - `heart_rate_std_30d`
  - `glucose_mean_7d`
  - `glucose_std_30d`
- Lag features:
  - `heart_rate_lag1`
  - `systolic_bp_lag1`
  - `hba1c_lag1`
- Interaction terms:
  - `age_x_glucose`
  - `bmi_x_systolic_bp`
  - `adherence_x_visits_last_90_days`

Implementation notes:

- Build rolling and lagged features from encounter history ordered by `recorded_at`.
- Avoid leakage by computing features only from prior visits, not future data.
- If PCA is used, retain enough components to preserve clinically meaningful variance and document the explained-variance threshold.
- Keep a mapping from transformed features back to the raw clinical variables for interpretability.
