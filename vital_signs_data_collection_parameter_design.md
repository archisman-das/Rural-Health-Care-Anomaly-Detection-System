# Vital Signs Data Collection & Parameter Design

## 1. Purpose

This document defines the minimum data model and capture rules for collecting core patient vital signs in a rural health-care anomaly detection workflow.

Target measurements:

- Heart rate
- Blood pressure, systolic and diastolic
- SpO2
- Body temperature
- Respiratory rate
- BMI
- Blood glucose, fasting and postprandial
- CBC: Hb, WBC, platelets
- HbA1c
- Lipid panel: LDL, HDL, triglycerides
- Liver function: ALT, AST, bilirubin
- Kidney function: creatinine, BUN, eGFR
- Electrolytes: sodium, potassium, calcium
- Demographic and history: age, gender, comorbidities, medications, recent visit frequency, symptom duration
- Environmental and social: seasonal disease prevalence, sanitation, nutrition, distance to care
- Temporal and operational: treatment response, readmissions, adherence, visit interval trends

## 2. Collection Strategy

### 2.1 Collection Points

- Registration or triage
- Pre-consultation assessment
- Post-consultation follow-up
- Home visit or remote screening, if available

### 2.2 Data Sources

- Manual entry by nurse, ASHA worker, or clinician
- Connected digital devices such as pulse oximeter, BP monitor, thermometer, glucometer, weighing scale, and stadiometer
- EMR import, if available

### 2.3 Capture Principles

- Record one observation set per encounter timestamp
- Store raw values exactly as measured
- Store units explicitly for every metric
- Store device/source metadata for traceability
- Allow missing fields when a measurement is not available

## 3. Data Schema

### 3.1 Core Entity: VitalSignObservation

Recommended fields:

- `patient_id`
- `encounter_id`
- `age_years`
- `gender`
- `comorbidities`
- `current_medications`
- `visits_last_90_days`
- `symptom_duration_days`
- `malaria_prevalence_level`
- `dengue_prevalence_level`
- `sanitation_index`
- `nutritional_score`
- `distance_to_nearest_facility_km`
- `treatment_response_score`
- `readmission_frequency`
- `drug_adherence_rate`
- `days_between_visits_trend`
- `recorded_at`
- `location_type`
- `source_type`
- `operator_id`
- `device_id`
- `heart_rate_bpm`
- `systolic_bp_mmhg`
- `diastolic_bp_mmhg`
- `spo2_percent`
- `body_temperature_c`
- `respiratory_rate_bpm`
- `weight_kg`
- `height_cm`
- `bmi_kg_m2`
- `glucose_fasting_mg_dl`
- `glucose_postprandial_mg_dl`
- `hb_g_dl`
- `wbc_count_10e9_l`
- `platelets_10e9_l`
- `hba1c_percent`
- `ldl_mg_dl`
- `hdl_mg_dl`
- `triglycerides_mg_dl`
- `alt_u_l`
- `ast_u_l`
- `bilirubin_mg_dl`
- `creatinine_mg_dl`
- `bun_mg_dl`
- `egfr_ml_min_1_73m2`
- `sodium_mmol_l`
- `potassium_mmol_l`
- `calcium_mg_dl`
- `measurement_posture`
- `measurement_context`
- `notes`
- `data_quality_flag`

### 3.2 Suggested Data Types

| Field | Type | Example | Required |
| --- | --- | --- | --- |
| `age_years` | integer | `54` | No |
| `gender` | enum | `female`, `male`, `other`, `prefer_not_to_say` | No |
| `comorbidities` | array[string] | `["diabetes", "hypertension"]` | No |
| `current_medications` | array[string] | `["metformin", "amlodipine"]` | No |
| `visits_last_90_days` | integer | `3` | No |
| `symptom_duration_days` | integer | `12` | No |
| `malaria_prevalence_level` | enum | `low`, `moderate`, `high`, `outbreak` | No |
| `dengue_prevalence_level` | enum | `low`, `moderate`, `high`, `outbreak` | No |
| `sanitation_index` | number | `0.72` | No |
| `nutritional_score` | number | `68` | No |
| `distance_to_nearest_facility_km` | number | `4.6` | No |
| `treatment_response_score` | number | `0.8` | No |
| `readmission_frequency` | integer | `2` | No |
| `drug_adherence_rate` | number | `0.92` | No |
| `days_between_visits_trend` | array[number] | `[14, 21, 30]` | No |
| `patient_id` | string | `P-000124` | Yes |
| `encounter_id` | string | `ENC-20260611-001` | Yes |
| `recorded_at` | datetime | `2026-06-11T09:15:00+05:30` | Yes |
| `location_type` | enum | `triage`, `home_visit`, `clinic`, `telehealth` | Yes |
| `source_type` | enum | `manual`, `device`, `imported` | Yes |
| `operator_id` | string | `NURSE-17` | No |
| `device_id` | string | `BP-0442` | No |
| `heart_rate_bpm` | integer | `78` | No |
| `systolic_bp_mmhg` | integer | `118` | No |
| `diastolic_bp_mmhg` | integer | `76` | No |
| `spo2_percent` | number | `97.0` | No |
| `body_temperature_c` | number | `36.8` | No |
| `respiratory_rate_bpm` | integer | `16` | No |
| `weight_kg` | number | `64.2` | No |
| `height_cm` | number | `168.0` | No |
| `bmi_kg_m2` | number | `22.7` | No |
| `glucose_fasting_mg_dl` | number | `92` | No |
| `glucose_postprandial_mg_dl` | number | `128` | No |
| `hb_g_dl` | number | `13.4` | No |
| `wbc_count_10e9_l` | number | `6.2` | No |
| `platelets_10e9_l` | number | `240` | No |
| `hba1c_percent` | number | `6.1` | No |
| `ldl_mg_dl` | number | `102` | No |
| `hdl_mg_dl` | number | `48` | No |
| `triglycerides_mg_dl` | number | `156` | No |
| `alt_u_l` | number | `28` | No |
| `ast_u_l` | number | `24` | No |
| `bilirubin_mg_dl` | number | `0.8` | No |
| `creatinine_mg_dl` | number | `1.0` | No |
| `bun_mg_dl` | number | `14` | No |
| `egfr_ml_min_1_73m2` | number | `92` | No |
| `sodium_mmol_l` | number | `138` | No |
| `potassium_mmol_l` | number | `4.2` | No |
| `calcium_mg_dl` | number | `9.4` | No |
| `measurement_posture` | enum | `sitting`, `standing`, `lying` | No |
| `measurement_context` | string | `Resting, before medication` | No |
| `notes` | string | `Patient reported dizziness` | No |
| `data_quality_flag` | enum | `ok`, `missing`, `suspect`, `repeat_required` | Yes |

## 4. Parameter Design

### 4.1 Heart Rate

- Field name: `heart_rate_bpm`
- Unit: beats per minute
- Type: integer
- Validation:
  - Must be positive
  - Flag extremely low or high values for review
- Capture source:
  - Manual count
  - Pulse oximeter
  - ECG device, if available

### 4.2 Blood Pressure

- Field names: `systolic_bp_mmhg`, `diastolic_bp_mmhg`
- Unit: mmHg
- Type: integer
- Validation:
  - Systolic must be greater than or equal to diastolic
  - Both values must be positive
  - Record cuff size and posture where possible in notes or metadata

### 4.3 SpO2

- Field name: `spo2_percent`
- Unit: percent
- Type: number
- Validation:
  - Store one decimal place if the device supports it
  - Must be between 0 and 100
  - Use pulse-oximeter source metadata when available

### 4.4 Body Temperature

- Field name: `body_temperature_c`
- Unit: degree Celsius
- Type: number
- Validation:
  - Store at one decimal place
  - Keep device and site of measurement consistent where possible

### 4.5 Respiratory Rate

- Field name: `respiratory_rate_bpm`
- Unit: breaths per minute
- Type: integer
- Validation:
  - Must be positive
  - Prefer observed count over estimated value

### 4.6 BMI

- Field name: `bmi_kg_m2`
- Unit: kg/m2
- Type: number
- Derivation:
  - If `weight_kg` and `height_cm` are available, calculate BMI automatically
  - Formula: `weight_kg / (height_m * height_m)`
- Validation:
  - If stored, keep both raw inputs and derived output when possible

### 4.7 Blood Glucose

- Field names:
  - `glucose_fasting_mg_dl`
  - `glucose_postprandial_mg_dl`
- Unit: mg/dL
- Type: number
- Validation:
  - Store the fasting or postprandial context explicitly
  - Do not mix values without context
  - If both are present, keep them as separate fields

### 4.8 Lab Results

#### CBC

- Field names:
  - `hb_g_dl`
  - `wbc_count_10e9_l`
  - `platelets_10e9_l`
- Units:
  - Hb: g/dL
  - WBC: 10^9/L
  - Platelets: 10^9/L
- Validation:
  - Store each result separately
  - Keep reference ranges configurable by facility
  - Mark hemolysis, clotted sample, or invalid specimen in notes when known

#### HbA1c

- Field name: `hba1c_percent`
- Unit: percent
- Type: number
- Validation:
  - Store one decimal place if available
  - Use the lab-reported value, not an estimated glucose average

#### Lipid Panel

- Field names:
  - `ldl_mg_dl`
  - `hdl_mg_dl`
  - `triglycerides_mg_dl`
- Unit: mg/dL
- Type: number
- Validation:
  - Store results independently
  - Preserve fasting status in `measurement_context` if the lab requires fasting

#### Liver Function

- Field names:
  - `alt_u_l`
  - `ast_u_l`
  - `bilirubin_mg_dl`
- Units:
  - ALT: U/L
  - AST: U/L
  - Bilirubin: mg/dL
- Validation:
  - Keep direct and indirect bilirubin in notes if the lab provides them separately
  - Allow both numeric values and test comments

#### Kidney Function

- Field names:
  - `creatinine_mg_dl`
  - `bun_mg_dl`
  - `egfr_ml_min_1_73m2`
- Units:
  - Creatinine: mg/dL
  - BUN: mg/dL
  - eGFR: mL/min/1.73m2
- Validation:
  - Store eGFR as reported by the lab or calculated from the lab formula, but label the source consistently
  - Retain age/sex context if eGFR is calculated externally

#### Electrolytes

- Field names:
  - `sodium_mmol_l`
  - `potassium_mmol_l`
  - `calcium_mg_dl`
- Units:
  - Sodium: mmol/L
  - Potassium: mmol/L
  - Calcium: mg/dL
- Validation:
  - Flag extreme values for clinical review
  - Keep albumin-corrected calcium separate if used

### 4.9 Demographic and History

#### Age

- Field name: `age_years`
- Unit: years
- Type: integer
- Validation:
  - Must be zero or greater
  - If date of birth is available, calculate age consistently from the same reference date

#### Gender

- Field name: `gender`
- Type: enum/string
- Suggested values:
  - `female`
  - `male`
  - `other`
  - `prefer_not_to_say`
- Validation:
  - Keep the vocabulary consistent across facilities

#### Comorbidities

- Field name: `comorbidities`
- Type: array of strings
- Suggested values:
  - `diabetes`
  - `hypertension`
  - `tb`
- Validation:
  - Store multiple active conditions when present
  - Preserve the original coded value if the source system uses one

#### Current Medications

- Field name: `current_medications`
- Type: array of strings
- Validation:
  - Store medication names as reported
  - If possible, capture dose and frequency in a separate medication-detail field later

#### Number of Visits in Last 90 Days

- Field name: `visits_last_90_days`
- Unit: count
- Type: integer
- Validation:
  - Must be zero or greater
  - Count only completed visits unless your workflow defines otherwise

#### Symptom Duration

- Field name: `symptom_duration_days`
- Unit: days
- Type: integer
- Validation:
  - Must be zero or greater
  - Store the duration from symptom onset to the encounter date

### 4.10 Environmental and Social

#### Seasonal Disease Prevalence

- Field names:
  - `malaria_prevalence_level`
  - `dengue_prevalence_level`
- Type: enum
- Suggested values:
  - `low`
  - `moderate`
  - `high`
  - `outbreak`
- Validation:
  - Use area-level prevalence, not individual diagnosis
  - Keep the source date and geography consistent with the facility catchment area

#### Sanitation Index

- Field name: `sanitation_index`
- Type: number
- Suggested scale:
  - 0 to 1, where higher indicates better sanitation
- Validation:
  - Store the scale definition in project metadata if a different scale is used
  - Keep the value derived from a consistent household or village assessment method

#### Nutritional Score

- Field name: `nutritional_score`
- Type: number
- Suggested scale:
  - 0 to 100, where higher indicates better nutritional status
- Validation:
  - Define the scoring rubric centrally
  - Use the same scoring method across all sites

#### Distance to Nearest Facility

- Field name: `distance_to_nearest_facility_km`
- Unit: km
- Type: number
- Validation:
  - Must be zero or greater
  - Prefer road distance or travel distance if consistently available

### 4.11 Temporal and Operational

#### Treatment Response Score

- Field name: `treatment_response_score`
- Type: number
- Suggested scale:
  - 0 to 1, where higher indicates better response
- Validation:
  - Keep the scoring rubric consistent across programs
  - Store the source of the score if it is clinician-rated or model-generated

#### Readmission Frequency

- Field name: `readmission_frequency`
- Unit: count
- Type: integer
- Validation:
  - Must be zero or greater
  - Count readmissions within the project-defined window

#### Drug Adherence Rate

- Field name: `drug_adherence_rate`
- Type: number
- Suggested scale:
  - 0 to 1, where 1 means fully adherent
- Validation:
  - Store the computation method, such as pill count, refill history, or self-report
  - Keep the time window explicit in metadata when possible

#### Time Between Visits Trends

- Field name: `days_between_visits_trend`
- Type: array of numbers
- Unit: days
- Validation:
  - Store visit intervals in chronological order when available
  - Keep raw visit dates in the encounter history if the trend is derived

## 5. Business Rules

- Use a single encounter timestamp for a measurement batch
- Do not overwrite a previous raw value unless the data entry was incorrect and the audit trail is preserved
- If a measurement is not taken, store `null` and set `data_quality_flag` accordingly
- If a value is out of expected range, do not auto-correct it; mark it as `suspect`
- Use standardized units across all facilities

## 6. Quality Checks

- Heart rate, blood pressure, and respiratory rate should all be numeric and positive
- Blood pressure should satisfy systolic greater than or equal to diastolic
- SpO2 should remain within 0 to 100
- BMI should be recalculated when height or weight changes
- Glucose values should include whether the sample is fasting or postprandial
- CBC, lipid, liver, kidney, and electrolyte values should carry their test units explicitly
- Demographic and history values should use consistent coding and list formats across all records
- Environmental and social values should use one defined scale per field across the deployment
- Temporal and operational values should specify the scoring window or calculation method in project metadata
- Flag duplicate measurements from the same device and timestamp

## 7. Recommended Storage Format

Use a row-based table or JSON record with one observation set per row. Example JSON:

```json
{
  "patient_id": "P-000124",
  "encounter_id": "ENC-20260611-001",
  "age_years": 54,
  "gender": "female",
  "comorbidities": ["diabetes", "hypertension"],
  "current_medications": ["metformin", "amlodipine"],
  "visits_last_90_days": 3,
  "symptom_duration_days": 12,
  "malaria_prevalence_level": "moderate",
  "dengue_prevalence_level": "high",
  "sanitation_index": 0.72,
  "nutritional_score": 68,
  "distance_to_nearest_facility_km": 4.6,
  "treatment_response_score": 0.8,
  "readmission_frequency": 2,
  "drug_adherence_rate": 0.92,
  "days_between_visits_trend": [14, 21, 30],
  "recorded_at": "2026-06-11T09:15:00+05:30",
  "location_type": "clinic",
  "source_type": "device",
  "operator_id": "NURSE-17",
  "device_id": "BP-0442",
  "heart_rate_bpm": 78,
  "systolic_bp_mmhg": 118,
  "diastolic_bp_mmhg": 76,
  "spo2_percent": 97.0,
  "body_temperature_c": 36.8,
  "respiratory_rate_bpm": 16,
  "weight_kg": 64.2,
  "height_cm": 168.0,
  "bmi_kg_m2": 22.7,
  "glucose_fasting_mg_dl": 92,
  "glucose_postprandial_mg_dl": null,
  "hb_g_dl": 13.4,
  "wbc_count_10e9_l": 6.2,
  "platelets_10e9_l": 240,
  "hba1c_percent": 6.1,
  "ldl_mg_dl": 102,
  "hdl_mg_dl": 48,
  "triglycerides_mg_dl": 156,
  "alt_u_l": 28,
  "ast_u_l": 24,
  "bilirubin_mg_dl": 0.8,
  "creatinine_mg_dl": 1.0,
  "bun_mg_dl": 14,
  "egfr_ml_min_1_73m2": 92,
  "sodium_mmol_l": 138,
  "potassium_mmol_l": 4.2,
  "calcium_mg_dl": 9.4,
  "measurement_posture": "sitting",
  "measurement_context": "Resting before consultation",
  "notes": "No visible distress",
  "data_quality_flag": "ok"
}
```

## 8. Implementation Notes

- Keep validation rules in configuration so clinical staff can tune thresholds later
- Make all fields optional except patient identity, encounter identity, timestamp, source type, and quality flag
- Support both manual and device-based entry from the same schema
- Store derived fields like BMI separately from raw source inputs
- Keep lab values in the same observation record when they are collected at the same encounter, or split into a linked lab-result record if your database prefers normalization
- Keep environmental and social context either in the encounter record or a linked area-assessment record if it is shared across many patients
- Keep temporal and operational metrics either as encounter-level derived features or as longitudinal summaries linked to the patient history

## 9. Minimal Form Layout

1. Patient details
2. Encounter metadata
3. Vital signs
4. Body measurements
5. Glucose section
6. Notes and quality flag

## 10. Suggested Next Step

If this is going into an application, the next layer should be:

- a database schema
- a form UI
- validation logic
- anomaly-scoring rules based on these fields
