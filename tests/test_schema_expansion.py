import unittest

import numpy as np
import pandas as pd

from rural_health_anomaly import PreprocessingConfig, HealthcarePreprocessor


class SchemaExpansionTests(unittest.TestCase):
    def setUp(self):
        self.training_df = pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2"],
                "recorded_at": [
                    "2026-06-01T09:00:00+05:30",
                    "2026-06-08T09:00:00+05:30",
                    "2026-06-03T10:15:00+05:30",
                ],
                "age_years": [54, 54, 61],
                "gender": ["female", "female", "male"],
                "location_type": ["clinic", "clinic", "home_visit"],
                "source_type": ["device", "manual", "device"],
                "operator_id": ["N1", "N1", "N2"],
                "device_id": ["D1", "D1", "D2"],
                "measurement_posture": ["sitting", "sitting", "standing"],
                "data_quality_flag": ["ok", "ok", "suspect"],
                "malaria_prevalence_level": ["moderate", "moderate", "high"],
                "dengue_prevalence_level": ["high", "high", "moderate"],
                "comorbidities": [
                    ["diabetes", "hypertension"],
                    ["diabetes"],
                    ["tb"],
                ],
                "current_medications": [
                    ["metformin", "amlodipine"],
                    ["metformin"],
                    ["isoniazid"],
                ],
                "days_between_visits_trend": [[14, 21, 30], [7, 14], [30, 45]],
                "visits_last_90_days": [3, 4, 2],
                "symptom_duration_days": [12, 11, 8],
                "sanitation_index": [0.72, 0.71, 0.55],
                "nutritional_score": [68, 67, 59],
                "distance_to_nearest_facility_km": [4.6, 4.6, 8.2],
                "treatment_response_score": [0.80, 0.82, 0.61],
                "readmission_frequency": [2, 2, 1],
                "drug_adherence_rate": [0.92, 0.94, 0.71],
                "heart_rate_bpm": [78, 81, 92],
                "systolic_bp_mmhg": [118, 120, 136],
                "diastolic_bp_mmhg": [76, 78, 88],
                "spo2_percent": [97.0, 96.0, 94.0],
                "body_temperature_c": [36.8, 36.7, 37.4],
                "respiratory_rate_bpm": [16, 16, 18],
                "weight_kg": [64.2, 64.0, 70.1],
                "height_cm": [168.0, 168.0, 172.0],
                "bmi_kg_m2": [22.7, 22.7, 23.7],
                "glucose_fasting_mg_dl": [92, 110, 140],
                "glucose_postprandial_mg_dl": [128, 142, 180],
                "hb_g_dl": [13.4, 13.3, 12.2],
                "wbc_count_10e9_l": [6.2, 6.4, 8.1],
                "platelets_10e9_l": [240, 238, 180],
                "hba1c_percent": [6.1, 6.2, 7.2],
                "ldl_mg_dl": [102, 100, 128],
                "hdl_mg_dl": [48, 49, 42],
                "triglycerides_mg_dl": [156, 158, 210],
                "alt_u_l": [28, 29, 36],
                "ast_u_l": [24, 25, 33],
                "bilirubin_mg_dl": [0.8, 0.8, 1.1],
                "creatinine_mg_dl": [1.0, 1.0, 1.2],
                "bun_mg_dl": [14, 15, 18],
                "egfr_ml_min_1_73m2": [92, 92, 74],
                "sodium_mmol_l": [138, 139, 136],
                "potassium_mmol_l": [4.2, 4.1, 4.7],
                "calcium_mg_dl": [9.4, 9.5, 9.0],
                "measurement_context": ["resting", "resting", "follow-up"],
                "notes": ["", "", ""],
            }
        )

        self.config = PreprocessingConfig(apply_pca=False)

    def test_schema_fields_expand_into_feature_map(self):
        preprocessor = HealthcarePreprocessor(self.config)
        preprocessor.fit(self.training_df)

        feature_map = preprocessor.export_feature_map()
        final_features = set(feature_map["final_feature"].tolist())

        self.assertIn("comorbidities__diabetes", final_features)
        self.assertIn("comorbidities__hypertension", final_features)
        self.assertIn("current_medications__metformin", final_features)
        self.assertIn("current_medications__amlodipine", final_features)
        self.assertIn("days_between_visits_trend_mean", final_features)
        self.assertIn("days_between_visits_trend_std", final_features)
        self.assertIn("days_between_visits_trend_last", final_features)
        self.assertIn("days_between_visits_trend_count", final_features)

        feature_types = dict(zip(feature_map["final_feature"], feature_map["feature_type"]))
        self.assertEqual(feature_types["comorbidities__diabetes"], "expanded_multi_value")
        self.assertEqual(feature_types["days_between_visits_trend_mean"], "direct")

    def test_transform_preserves_shape_with_unseen_list_values(self):
        preprocessor = HealthcarePreprocessor(self.config)
        preprocessor.fit(self.training_df)

        inference_df = pd.DataFrame(
            {
                "patient_id": ["P3"],
                "recorded_at": ["2026-06-14T09:10:00+05:30"],
                "age_years": [57],
                "gender": ["female"],
                "location_type": ["clinic"],
                "source_type": ["device"],
                "operator_id": ["N3"],
                "device_id": ["D3"],
                "measurement_posture": ["sitting"],
                "data_quality_flag": ["ok"],
                "malaria_prevalence_level": ["moderate"],
                "dengue_prevalence_level": ["high"],
                "comorbidities": [["diabetes", "asthma"]],
                "current_medications": [["metformin", "insulin"]],
                "days_between_visits_trend": [[10, 20, 35]],
                "visits_last_90_days": [2],
                "symptom_duration_days": [6],
                "sanitation_index": [0.69],
                "nutritional_score": [65],
                "distance_to_nearest_facility_km": [5.1],
                "treatment_response_score": [0.74],
                "readmission_frequency": [1],
                "drug_adherence_rate": [0.88],
                "heart_rate_bpm": [84],
                "systolic_bp_mmhg": [126],
                "diastolic_bp_mmhg": [80],
                "spo2_percent": [95.0],
                "body_temperature_c": [36.9],
                "respiratory_rate_bpm": [17],
                "weight_kg": [66.0],
                "height_cm": [165.0],
                "bmi_kg_m2": [24.2],
                "glucose_fasting_mg_dl": [124],
                "glucose_postprandial_mg_dl": [166],
                "hb_g_dl": [12.8],
                "wbc_count_10e9_l": [7.1],
                "platelets_10e9_l": [210],
                "hba1c_percent": [6.8],
                "ldl_mg_dl": [114],
                "hdl_mg_dl": [46],
                "triglycerides_mg_dl": [172],
                "alt_u_l": [30],
                "ast_u_l": [27],
                "bilirubin_mg_dl": [0.7],
                "creatinine_mg_dl": [0.9],
                "bun_mg_dl": [13],
                "egfr_ml_min_1_73m2": [88],
                "sodium_mmol_l": [137],
                "potassium_mmol_l": [4.3],
                "calcium_mg_dl": [9.2],
                "measurement_context": ["follow-up"],
                "notes": [""],
            }
        )

        transformed = preprocessor.transform(inference_df)
        self.assertEqual(transformed.shape[1], preprocessor.fit_transform(self.training_df).shape[1])
        self.assertTrue(np.isfinite(transformed).all())

    def test_transform_handles_missing_scalars_and_unseen_multi_value_tokens(self):
        preprocessor = HealthcarePreprocessor(self.config)
        preprocessor.fit(self.training_df)

        inference_df = pd.DataFrame(
            {
                "patient_id": ["P4"],
                "recorded_at": ["2026-06-20T09:00:00+05:30"],
                "age_years": [None],
                "gender": [" none "],
                "location_type": ["clinic"],
                "source_type": ["device"],
                "operator_id": ["N4"],
                "device_id": ["D4"],
                "measurement_posture": ["sitting"],
                "data_quality_flag": ["ok"],
                "malaria_prevalence_level": ["moderate"],
                "dengue_prevalence_level": ["high"],
                "comorbidities": [["diabetes", "asthma"]],
                "current_medications": [["metformin", "insulin"]],
                "days_between_visits_trend": [[None, 20, 35]],
                "visits_last_90_days": [None],
                "symptom_duration_days": [None],
                "sanitation_index": [None],
                "nutritional_score": [None],
                "distance_to_nearest_facility_km": [None],
                "treatment_response_score": [None],
                "readmission_frequency": [None],
                "drug_adherence_rate": [None],
                "heart_rate_bpm": [None],
                "systolic_bp_mmhg": [None],
                "diastolic_bp_mmhg": [None],
                "spo2_percent": [None],
                "body_temperature_c": [None],
                "respiratory_rate_bpm": [None],
                "weight_kg": [None],
                "height_cm": [None],
                "bmi_kg_m2": [None],
                "glucose_fasting_mg_dl": [None],
                "glucose_postprandial_mg_dl": [None],
                "hb_g_dl": [None],
                "wbc_count_10e9_l": [None],
                "platelets_10e9_l": [None],
                "hba1c_percent": [None],
                "ldl_mg_dl": [None],
                "hdl_mg_dl": [None],
                "triglycerides_mg_dl": [None],
                "alt_u_l": [None],
                "ast_u_l": [None],
                "bilirubin_mg_dl": [None],
                "creatinine_mg_dl": [None],
                "bun_mg_dl": [None],
                "egfr_ml_min_1_73m2": [None],
                "sodium_mmol_l": [None],
                "potassium_mmol_l": [None],
                "calcium_mg_dl": [None],
                "measurement_context": ["follow-up"],
                "notes": [""],
            }
        )

        transformed = preprocessor.transform(inference_df)
        self.assertEqual(transformed.shape[1], preprocessor.fit_transform(self.training_df).shape[1])
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
