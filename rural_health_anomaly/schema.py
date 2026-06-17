"""Schema-backed feature lists for the rural health anomaly project."""

SCHEMA_NUMERIC_FEATURES: list[str] = [
    "age_years",
    "visits_last_90_days",
    "symptom_duration_days",
    "sanitation_index",
    "nutritional_score",
    "distance_to_nearest_facility_km",
    "treatment_response_score",
    "readmission_frequency",
    "drug_adherence_rate",
    "heart_rate_bpm",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "spo2_percent",
    "body_temperature_c",
    "respiratory_rate_bpm",
    "weight_kg",
    "height_cm",
    "bmi_kg_m2",
    "glucose_fasting_mg_dl",
    "glucose_postprandial_mg_dl",
    "hb_g_dl",
    "wbc_count_10e9_l",
    "platelets_10e9_l",
    "hba1c_percent",
    "ldl_mg_dl",
    "hdl_mg_dl",
    "triglycerides_mg_dl",
    "alt_u_l",
    "ast_u_l",
    "bilirubin_mg_dl",
    "creatinine_mg_dl",
    "bun_mg_dl",
    "egfr_ml_min_1_73m2",
    "sodium_mmol_l",
    "potassium_mmol_l",
    "calcium_mg_dl",
]

SCHEMA_CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "location_type",
    "source_type",
    "operator_id",
    "device_id",
    "measurement_posture",
    "data_quality_flag",
    "malaria_prevalence_level",
    "dengue_prevalence_level",
]

SCHEMA_MULTI_VALUE_FEATURES: list[str] = [
    "comorbidities",
    "current_medications",
]

SCHEMA_LIST_NUMERIC_FEATURES: list[str] = [
    "days_between_visits_trend",
]

SCHEMA_EXCLUDED_TEXT_FEATURES: list[str] = [
    "measurement_context",
    "notes",
]
