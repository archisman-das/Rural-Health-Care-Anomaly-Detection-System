import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rural_health_anomaly import HealthcarePreprocessor, PreprocessingConfig


class FeatureMapExportTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
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

    def test_feature_map_csv_and_json_exports(self):
        preprocessor = HealthcarePreprocessor(PreprocessingConfig(apply_pca=False))
        preprocessor.fit(self.df)

        feature_map = preprocessor.export_feature_map()
        comorbidity_row = feature_map.loc[feature_map["final_feature"] == "comorbidities__diabetes"].iloc[0]
        self.assertEqual(comorbidity_row["source_columns"], ["comorbidities"])
        self.assertEqual(comorbidity_row["transformation_path"], ["raw", "multi_value_expand"])
        self.assertEqual(comorbidity_row["provenance_depth"], 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "feature_map.csv"
            json_path = Path(tmpdir) / "feature_map.json"

            csv_text = preprocessor.export_feature_map_csv(csv_path)
            json_text = preprocessor.export_feature_map_json(json_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("final_feature", csv_text)

            loaded_json = json.loads(json_text)
            self.assertIsInstance(loaded_json, list)
            self.assertGreater(len(loaded_json), 0)
            self.assertIn("final_feature", loaded_json[0])

            csv_frame = pd.read_csv(csv_path)
            self.assertIn("feature_type", csv_frame.columns)
            self.assertIn("transformation_path", csv_frame.columns)
            self.assertGreater(len(csv_frame), 0)


if __name__ == "__main__":
    unittest.main()
