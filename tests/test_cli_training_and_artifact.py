import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rural_health_anomaly.cli import run_predict, run_retrain_feedback, run_split_data, run_train
from rural_health_anomaly.example import build_large_training_data
from rural_health_anomaly.training import (
    _clinical_risk_assessment,
    _clinical_risk_component,
    _generate_risk_score,
    _risk_category_from_score,
    load_pipeline,
    score_records,
)


class CliTrainingAndArtifactTests(unittest.TestCase):
    def _write_csv(self, path: Path, frame: pd.DataFrame) -> None:
        frame.to_csv(path, index=False)

    def _build_training_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
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
                "heart_rate_bpm": [78, 81, 92],
                "systolic_bp_mmhg": [118, 120, 136],
                "diastolic_bp_mmhg": [76, 78, 88],
                "glucose_fasting_mg_dl": [92, 110, 140],
                "measurement_context": ["resting", "resting", "follow-up"],
                "notes": ["", "", ""],
            }
        )

    def _build_inference_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "patient_id": ["P3", "P3"],
                "recorded_at": [
                    "2026-06-10T09:00:00+05:30",
                    "2026-06-17T09:00:00+05:30",
                ],
                "age_years": [57, 57],
                "gender": ["female", "female"],
                "location_type": ["clinic", "clinic"],
                "source_type": ["device", "manual"],
                "operator_id": ["N3", "N3"],
                "device_id": ["D3", "D3"],
                "measurement_posture": ["sitting", "sitting"],
                "data_quality_flag": ["ok", "ok"],
                "comorbidities": [["diabetes"], ["diabetes"]],
                "current_medications": [["metformin"], ["metformin"]],
                "days_between_visits_trend": [[10, 20], [7, 14]],
                "visits_last_90_days": [2, 3],
                "symptom_duration_days": [6, 5],
                "heart_rate_bpm": [84, 88],
                "systolic_bp_mmhg": [126, 132],
                "diastolic_bp_mmhg": [80, 84],
                "glucose_fasting_mg_dl": [124, 130],
                "measurement_context": ["follow-up", "follow-up"],
                "notes": ["", ""],
            }
        )

    def test_train_cli_saves_model_and_feature_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            feature_map_path = tmpdir_path / "feature_map.csv"
            config_path = tmpdir_path / "config.json"

            self._write_csv(train_path, self._build_training_frame())
            config_path.write_text(
                json.dumps({"apply_pca": False, "knn_neighbors": 2, "scaler": "standard"}),
                encoding="utf-8",
            )

            args = argparse.Namespace(
                input=str(train_path),
                output=str(model_path),
                feature_map=str(feature_map_path),
                config_json=str(config_path),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_train(args)

            self.assertTrue(model_path.exists())
            self.assertTrue(feature_map_path.exists())

            pipeline = load_pipeline(model_path)
            self.assertIn("preprocessor", pipeline.named_steps)
            self.assertIn("model", pipeline.named_steps)

            feature_map = pipeline.named_steps["preprocessor"].export_feature_map()
            self.assertIn("source_columns", feature_map.columns)
            self.assertIn("transformation_path", feature_map.columns)
            self.assertGreater(len(feature_map), 0)

    def test_split_data_cli_writes_train_validation_and_test_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "dataset.csv"
            output_dir = tmpdir_path / "split_dataset"
            frame = build_large_training_data(target_rows=30)
            frame["label"] = [1 if idx % 7 == 0 else 0 for idx in range(len(frame))]
            self._write_csv(input_path, frame)

            args = argparse.Namespace(
                input=str(input_path),
                output_dir=str(output_dir),
                train_fraction=0.7,
                validation_fraction=0.15,
                test_fraction=0.15,
                group_column="patient_id",
                random_state=42,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_split_data(args)

            self.assertTrue((output_dir / "train" / "data.csv").exists())
            self.assertTrue((output_dir / "validation" / "data.csv").exists())
            self.assertTrue((output_dir / "test" / "data.csv").exists())

            train_frame = pd.read_csv(output_dir / "train" / "data.csv")
            validation_frame = pd.read_csv(output_dir / "validation" / "data.csv")
            test_frame = pd.read_csv(output_dir / "test" / "data.csv")
            self.assertGreater(len(train_frame), 0)
            self.assertGreater(len(validation_frame), 0)
            self.assertGreater(len(test_frame), 0)
            self.assertTrue(set(train_frame["patient_id"]).isdisjoint(validation_frame["patient_id"]))
            self.assertTrue(set(train_frame["patient_id"]).isdisjoint(test_frame["patient_id"]))

    def test_train_cli_can_use_split_directory_and_validation_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "dataset.csv"
            split_dir = tmpdir_path / "splits"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            frame = build_large_training_data(target_rows=30)
            frame["label"] = [1 if idx % 6 == 0 else 0 for idx in range(len(frame))]
            self._write_csv(input_path, frame)
            config_path.write_text(
                json.dumps(
                    {
                        "apply_pca": False,
                        "knn_neighbors": 2,
                        "scaler": "standard",
                        "calibration_min_samples": 1,
                    }
                ),
                encoding="utf-8",
            )

            split_args = argparse.Namespace(
                input=str(input_path),
                output_dir=str(split_dir),
                train_fraction=0.7,
                validation_fraction=0.15,
                test_fraction=0.15,
                group_column="patient_id",
                random_state=42,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_split_data(split_args)

            train_args = argparse.Namespace(
                input=None,
                split_dir=str(split_dir),
                output=str(model_path),
                feature_map=None,
                config_json=str(config_path),
                label_column="label",
                labels_file=None,
                labels_column=None,
                synthetic_demo_data=False,
                synthetic_demo_rows=9600,
                synthetic_demo_seed=42,
                calibrate_threshold=True,
                calibration_min_samples=1,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_train(train_args)

            self.assertTrue(model_path.exists())
            pipeline = load_pipeline(model_path)
            self.assertIn("model", pipeline.named_steps)
            self.assertEqual(getattr(pipeline.named_steps["model"], "calibration_source_", None), "validation")

    def test_predict_cli_writes_scored_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            output_path = tmpdir_path / "predictions.csv"
            config_path = tmpdir_path / "config.json"

            self._write_csv(train_path, self._build_training_frame())
            self._write_csv(infer_path, self._build_inference_frame())
            config_path.write_text(
                json.dumps({"apply_pca": False, "knn_neighbors": 2, "scaler": "standard"}),
                encoding="utf-8",
            )

            train_args = argparse.Namespace(
                input=str(train_path),
                output=str(model_path),
                feature_map=None,
                config_json=str(config_path),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_train(train_args)

            predict_args = argparse.Namespace(
                model=str(model_path),
                input=str(infer_path),
                output=str(output_path),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_predict(predict_args)

            self.assertTrue(output_path.exists())

            scored = pd.read_csv(output_path)
            self.assertIn("isolation_forest_anomaly_score", scored.columns)
            self.assertIn("one_class_svm_anomaly_score", scored.columns)
            self.assertIn("local_outlier_factor_anomaly_score", scored.columns)
            self.assertIn("autoencoder_anomaly_score", scored.columns)
            self.assertIn("autoencoder_reconstruction_error", scored.columns)
            self.assertIn("autoencoder_reconstruction_mae", scored.columns)
            self.assertIn("anomaly_transformer_anomaly_score", scored.columns)
            self.assertIn("anomaly_transformer_reconstruction_error", scored.columns)
            self.assertIn("anomaly_transformer_attention_discrepancy", scored.columns)
            self.assertIn("variational_autoencoder_anomaly_score", scored.columns)
            self.assertIn("variational_autoencoder_reconstruction_error", scored.columns)
            self.assertIn("variational_autoencoder_reconstruction_mae", scored.columns)
            self.assertIn("ganomaly_anomaly_score", scored.columns)
            self.assertIn("ganomaly_reconstruction_error", scored.columns)
            self.assertIn("ganomaly_latent_consistency_error", scored.columns)
            self.assertIn("deep_svdd_distance", scored.columns)
            self.assertIn("raw_anomaly_score", scored.columns)
            self.assertIn("anomaly_score", scored.columns)
            self.assertIn("risk_level", scored.columns)
            self.assertIn("risk_score", scored.columns)
            self.assertIn("alert_triggered", scored.columns)
            self.assertIn("anomaly_flag", scored.columns)
            self.assertIn("is_anomaly", scored.columns)
            self.assertIn("training_time_seconds", scored.columns)
            self.assertIn("training_time_ms", scored.columns)
            self.assertIn("model_size_bytes", scored.columns)
            self.assertIn("estimated_ram_usage_bytes", scored.columns)
            self.assertIn("inference_batch_latency_ms", scored.columns)
            self.assertIn("inference_latency_ms_per_patient", scored.columns)
            self.assertIn("inference_throughput_rows_per_second", scored.columns)
            self.assertEqual(len(scored), 2)

    def test_predict_cli_assigns_risk_categories_from_score_thresholds(self):
        self.assertEqual(_risk_category_from_score(0.0), "Low")
        self.assertEqual(_risk_category_from_score(0.39), "Low")
        self.assertEqual(_risk_category_from_score(0.4), "Medium")
        self.assertEqual(_risk_category_from_score(0.64), "Medium")
        self.assertEqual(_risk_category_from_score(0.65), "High")
        self.assertEqual(_risk_category_from_score(0.99), "High")

    def test_clinical_risk_assessment_applies_critical_override(self):
        row = {
            "anomaly_score": 0.1,
            "glucose_fasting_mg_dl": 425,
            "medicalHistory.comorbidities": "Type 2 Diabetes",
        }
        assessment = _clinical_risk_assessment(row, anomaly_score=0.1)
        self.assertGreaterEqual(assessment["risk_score_normalized"], 0.75)
        self.assertEqual(assessment["risk_category"], "High")
        self.assertTrue(assessment["critical_override_triggered"])
        self.assertTrue(any("Blood glucose" in warning for warning in assessment["risk_warnings"]))

    def test_clinical_risk_assessment_applies_comorbidity_floor(self):
        row = {
            "anomaly_score": 0.05,
            "medicalHistory.comorbidities": "Hypertension, Type 2 Diabetes, COPD, Obesity, Asthma",
        }
        assessment = _clinical_risk_assessment(row, anomaly_score=0.05)
        self.assertGreaterEqual(assessment["risk_score_normalized"], 0.55)
        self.assertEqual(assessment["risk_category"], "Medium")
        self.assertEqual(assessment["comorbidity_count"], 5)

    def test_generate_risk_score_scales_to_percentage(self):
        self.assertEqual(_generate_risk_score(0.0), 0.0)
        self.assertEqual(_generate_risk_score(0.435), 43.5)
        self.assertEqual(_generate_risk_score(1.0), 100.0)
        self.assertEqual(_generate_risk_score(1.7), 100.0)

    def test_risk_scoring_weights_from_config_json_are_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._write_csv(train_path, self._build_training_frame())
            self._write_csv(infer_path, self._build_inference_frame())
            config_path.write_text(
                json.dumps(
                    {
                        "apply_pca": False,
                        "risk_scoring_weights": {
                            "anomaly": 1.0,
                            "vitals": 0.0,
                            "labs": 0.0,
                            "access": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            train_args = argparse.Namespace(
                input=str(train_path),
                output=str(model_path),
                feature_map=None,
                config_json=str(config_path),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_train(train_args)

            pipeline = load_pipeline(model_path)
            inference_frame = self._build_inference_frame()
            scored = score_records(pipeline, inference_frame)
            self.assertEqual(pipeline.risk_scoring_weights_["anomaly"], 1.0)
            self.assertEqual(pipeline.risk_scoring_weights_["vitals"], 0.0)
            self.assertEqual(pipeline.risk_scoring_weights_["labs"], 0.0)
            self.assertEqual(pipeline.risk_scoring_weights_["access"], 0.0)
            expected_risk_score = scored["anomaly_score"].astype(float).map(_generate_risk_score)
            self.assertTrue((scored["risk_score"].astype(float) == expected_risk_score).all())

    def test_clinical_risk_component_increases_with_worse_context(self):
        baseline = {
            "anomaly_score": 0.15,
            "heart_rate_bpm": 78,
            "systolic_bp_mmhg": 120,
            "diastolic_bp_mmhg": 78,
            "spo2_percent": 98,
            "body_temperature_c": 36.8,
            "respiratory_rate_bpm": 16,
            "bmi_kg_m2": 22,
            "glucose_fasting_mg_dl": 96,
            "glucose_postprandial_mg_dl": 132,
            "hba1c_percent": 5.4,
            "hemoglobin_g_dl": 13.8,
            "wbc_count_10e9_l": 7.0,
            "platelets_10e9_l": 240,
            "ldl_mg_dl": 92,
            "hdl_mg_dl": 54,
            "triglycerides_mg_dl": 130,
            "creatinine_mg_dl": 0.9,
            "egfr_ml_min_1_73m2": 94,
            "visits_last_90_days": 0,
            "symptom_duration_days": 1,
            "distance_to_nearest_facility_km": 1.5,
            "readmission_frequency": 0,
            "days_between_visits_trend": [30, 28, 31],
            "sanitation_index": 0.92,
            "drug_adherence_rate": 0.95,
            "treatment_response_score": 0.96,
        }
        elevated = dict(baseline)
        elevated.update(
            {
                "heart_rate_bpm": 112,
                "systolic_bp_mmhg": 168,
                "diastolic_bp_mmhg": 104,
                "spo2_percent": 88,
                "body_temperature_c": 39.1,
                "respiratory_rate_bpm": 28,
                "bmi_kg_m2": 31.5,
                "glucose_fasting_mg_dl": 182,
                "glucose_postprandial_mg_dl": 286,
                "hba1c_percent": 9.4,
                "hemoglobin_g_dl": 9.2,
                "wbc_count_10e9_l": 13.2,
                "platelets_10e9_l": 145,
                "ldl_mg_dl": 168,
                "hdl_mg_dl": 32,
                "triglycerides_mg_dl": 294,
                "creatinine_mg_dl": 1.7,
                "egfr_ml_min_1_73m2": 48,
                "visits_last_90_days": 6,
                "symptom_duration_days": 18,
                "distance_to_nearest_facility_km": 14.0,
                "readmission_frequency": 3,
                "days_between_visits_trend": [9, 8, 7],
                "sanitation_index": 0.42,
                "drug_adherence_rate": 0.48,
                "treatment_response_score": 0.41,
            }
        )

        self.assertLess(
            _clinical_risk_component(baseline, anomaly_score=0.15),
            _clinical_risk_component(elevated, anomaly_score=0.15),
        )

    def test_predict_cli_includes_autoencoder_score_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            output_path = tmpdir_path / "predictions.csv"
            config_path = tmpdir_path / "config.json"

            self._write_csv(train_path, self._build_training_frame())
            self._write_csv(infer_path, self._build_inference_frame())
            config_path.write_text(
                json.dumps(
                    {
                        "apply_pca": False,
                        "autoencoder_latent_dim": 8,
                        "autoencoder_threshold_percentile": 97.5,
                    }
                ),
                encoding="utf-8",
            )

            train_args = argparse.Namespace(
                input=str(train_path),
                output=str(model_path),
                feature_map=None,
                config_json=str(config_path),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_train(train_args)

            predict_args = argparse.Namespace(
                model=str(model_path),
                input=str(infer_path),
                output=str(output_path),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_predict(predict_args)

            scored = pd.read_csv(output_path)
            self.assertIn("autoencoder_anomaly_score", scored.columns)
            self.assertIn("autoencoder_reconstruction_error", scored.columns)
            self.assertIn("autoencoder_reconstruction_mae", scored.columns)
            self.assertIn("anomaly_transformer_anomaly_score", scored.columns)
            self.assertIn("variational_autoencoder_anomaly_score", scored.columns)
            self.assertIn("ganomaly_anomaly_score", scored.columns)

    def test_retrain_feedback_cli_rebuilds_model_from_clinician_ledger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            feedback_path = tmpdir_path / "feedback.jsonl"
            output_path = tmpdir_path / "retrained.joblib"
            config_path = tmpdir_path / "config.json"

            self._write_csv(train_path, self._build_training_frame())
            config_path.write_text(
                json.dumps({"apply_pca": False, "knn_neighbors": 2, "scaler": "standard"}),
                encoding="utf-8",
            )

            feedback_record = {
                "patient": self._build_inference_frame().iloc[0].to_dict(),
                "prediction": {"anomaly_score": 0.93, "risk_level": "High"},
                "is_true_positive": True,
                "reviewer": "clinician-a",
                "notes": "Confirmed anomaly during follow-up.",
            }
            feedback_path.write_text(json.dumps(feedback_record) + "\n", encoding="utf-8")

            args = argparse.Namespace(
                input=str(train_path),
                feedback_file=str(feedback_path),
                output=str(output_path),
                config_json=str(config_path),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_retrain_feedback(args)

            self.assertTrue(output_path.exists())
            retrained = load_pipeline(output_path)
            self.assertIn("preprocessor", retrained.named_steps)
            self.assertIn("model", retrained.named_steps)


if __name__ == "__main__":
    unittest.main()
