import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from rural_health_anomaly.backend import create_app
from rural_health_anomaly.config import PreprocessingConfig
from rural_health_anomaly.example import build_inference_data, build_training_data
from rural_health_anomaly.training import save_pipeline, train_anomaly_pipeline


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(cls._tmpdir.name)
        cls.model_path = tmpdir_path / "model.joblib"
        cls.metadata_path = cls.model_path.with_suffix(".metadata.json")
        cls.feedback_path = tmpdir_path / "feedback_ledger.jsonl"

        config = PreprocessingConfig(
            apply_pca=False,
            scaler="standard",
            autoencoder_max_epochs=2,
            autoencoder_patience=1,
            autoencoder_validation_fraction=0.4,
            deep_svdd_max_epochs=2,
            deep_svdd_pretrain_epochs=1,
            deep_svdd_validation_fraction=0.4,
            deep_svdd_pretrain_autoencoder=False,
        )
        pipeline = train_anomaly_pipeline(build_training_data(), config=config)
        save_pipeline(pipeline, cls.model_path)
        cls.auth_token = "test-secret-token"
        cls.app = create_app(cls.model_path, auth_token=cls.auth_token, feedback_store=cls.feedback_path)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_health_endpoint_returns_model_metadata(self):
        response = self.client.get("/health", headers={"X-API-Token": self.auth_token})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(Path(payload["model_path"]).name, "model.joblib")
        self.assertTrue(self.metadata_path.exists())
        self.assertIn("model_type", payload)
        self.assertIn("feature_count", payload)
        self.assertIn("model_name", payload)
        self.assertIn("model_version", payload)
        self.assertNotEqual(payload["model_version"], "unknown")

    def test_models_endpoint_returns_loaded_model_metadata(self):
        response = self.client.get("/models", headers={"X-API-Token": self.auth_token})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["count"], 1)
        self.assertEqual(len(payload["models"]), 1)
        self.assertEqual(Path(payload["models"][0]["model_path"]).name, "model.joblib")
        self.assertIn("model_type", payload["models"][0])
        self.assertIn("feature_count", payload["models"][0])

    def test_feedback_endpoint_appends_clinician_review(self):
        patient = build_inference_data().iloc[0].to_dict()

        response = self.client.post(
            "/feedback",
            json={
                "patient": patient,
                "prediction": {"anomaly_score": 0.91, "risk_level": "High"},
                "is_true_positive": True,
                "reviewer": "clinician-a",
                "notes": "Confirmed case during review.",
            },
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["count"], 1)
        self.assertTrue(self.feedback_path.exists())
        ledger_text = self.feedback_path.read_text(encoding="utf-8")
        self.assertIn("true_positive", ledger_text)
        self.assertIn("clinician-a", ledger_text)

        overview = self.client.get("/feedback", headers={"X-API-Token": self.auth_token})
        self.assertEqual(overview.status_code, 200)
        overview_payload = overview.json()
        self.assertTrue(overview_payload["exists"])
        self.assertEqual(overview_payload["count"], 1)

    def test_predict_endpoint_returns_single_patient_prediction(self):
        patient = build_inference_data().iloc[0].to_dict()

        response = self.client.post("/predict", json=patient, headers={"X-API-Token": self.auth_token})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertIn("input", payload)
        self.assertIn("prediction", payload)
        self.assertIn("anomaly_score", payload)
        self.assertIn("risk_score", payload)
        self.assertIn("risk_category", payload)
        self.assertIn("risk_level", payload)
        self.assertIn("alert_triggered", payload)
        self.assertIn("is_anomaly", payload)
        self.assertIn("explanation", payload)
        self.assertIn("latent_manifold", payload)
        self.assertIn("reconstruction_residual_heatmap", payload)
        self.assertEqual(payload["input"]["patient_id"], patient["patient_id"])
        self.assertIn("model_info", payload)
        self.assertIn("model_name", payload["model_info"])
        self.assertIn("model_version", payload["model_info"])
        self.assertIn("anomaly_score", payload["prediction"])
        self.assertIn("risk_score", payload["prediction"])
        self.assertIn("risk_category", payload["prediction"])
        self.assertIn("risk_level", payload["prediction"])

    def test_batch_predict_endpoint_returns_multiple_predictions(self):
        patients = build_inference_data().to_dict(orient="records")

        response = self.client.post("/batch-predict", json=patients, headers={"X-API-Token": self.auth_token})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["count"], len(patients))
        self.assertEqual(len(payload["predictions"]), len(patients))
        self.assertIn("anomaly_score", payload["predictions"][0])
        self.assertIn("risk_score", payload["predictions"][0])
        self.assertIn("risk_category", payload["predictions"][0])
        self.assertIn("risk_level", payload["predictions"][0])

    def test_predict_file_endpoint_scores_csv_uploads(self):
        csv_text = build_inference_data().to_csv(index=False)

        response = self.client.post(
            "/predict_file",
            files={"file": ("patients.csv", csv_text.encode("utf-8"), "text/csv")},
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["filename"], "patients.csv")
        self.assertEqual(payload["count"], len(build_inference_data()))
        self.assertEqual(len(payload["predictions"]), len(build_inference_data()))
        self.assertIn("anomaly_score", payload["predictions"][0])
        self.assertIn("risk_score", payload["predictions"][0])
        self.assertIn("risk_category", payload["predictions"][0])
        self.assertIn("risk_level", payload["predictions"][0])

    def test_batch_endpoint_scores_csv_uploads(self):
        csv_text = build_inference_data().to_csv(index=False)

        response = self.client.post(
            "/batch",
            files={"file": ("patients.csv", csv_text.encode("utf-8"), "text/csv")},
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["filename"], "patients.csv")
        self.assertEqual(payload["count"], len(build_inference_data()))
        self.assertEqual(len(payload["predictions"]), len(build_inference_data()))
        self.assertIn("anomaly_score", payload["predictions"][0])
        self.assertIn("risk_score", payload["predictions"][0])
        self.assertIn("risk_category", payload["predictions"][0])
        self.assertIn("risk_level", payload["predictions"][0])

    def test_protected_endpoints_reject_missing_or_invalid_token(self):
        patient = build_inference_data().iloc[0].to_dict()

        unauthorized_health = self.client.get("/health")
        unauthorized_predict = self.client.post("/predict", json=patient)
        invalid_predict = self.client.post("/predict", json=patient, headers={"X-API-Token": "wrong"})

        self.assertEqual(unauthorized_health.status_code, 401)
        self.assertEqual(unauthorized_predict.status_code, 401)
        self.assertEqual(invalid_predict.status_code, 401)
        self.assertIn("X-API-Token", unauthorized_health.json()["detail"])

    def test_explain_endpoint_returns_feature_importance(self):
        patient = build_inference_data().iloc[0].to_dict()

        response = self.client.post(
            "/explain?top_k=5",
            json=patient,
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertIn("prediction", payload)
        self.assertIn("model_info", payload)
        self.assertIn("explanation", payload)
        self.assertIn("feature_engineering", payload)
        self.assertIn("data_scaling", payload)
        self.assertIn("data_encoding", payload)
        self.assertIn("method", payload["explanation"])
        self.assertIn("feature_explanations", payload["explanation"])
        self.assertIn("interaction_heatmap", payload["explanation"])
        self.assertEqual(len(payload["explanation"]["feature_explanations"]), 5)
        self.assertIn("feature", payload["explanation"]["feature_explanations"][0])
        self.assertIn("shap_value", payload["explanation"]["feature_explanations"][0])
        self.assertIn("absolute_shap_value", payload["explanation"]["feature_explanations"][0])
        self.assertIn("feature_names", payload["explanation"]["interaction_heatmap"])
        self.assertIn("matrix", payload["explanation"]["interaction_heatmap"])
        self.assertIn("points", payload["latent_manifold"])
        self.assertIn("current_point", payload["latent_manifold"])
        self.assertIn("deep_svdd", payload["latent_manifold"])
        self.assertIn("reconstruction_residual_heatmap", payload)
        self.assertIn("feature_names", payload["reconstruction_residual_heatmap"])
        self.assertIn("models", payload["reconstruction_residual_heatmap"])
        self.assertIn("matrix", payload["reconstruction_residual_heatmap"])
        self.assertIn("engineered_features", payload["feature_engineering"])
        self.assertGreater(payload["feature_engineering"]["feature_count"], 0)
        self.assertIn("scaled_features", payload["data_scaling"])
        self.assertGreater(payload["data_scaling"]["feature_count"], 0)
        self.assertIn("encoded_features", payload["data_encoding"])
        self.assertGreaterEqual(payload["data_encoding"]["feature_count"], 0)
        if payload["data_encoding"]["encoded_features"]:
            self.assertIn("source_feature", payload["data_encoding"]["encoded_features"][0])
            self.assertIn("category_value", payload["data_encoding"]["encoded_features"][0])
            self.assertIn("explanation", payload["data_encoding"]["encoded_features"][0])

    def test_feature_engineering_endpoint_returns_engineered_features(self):
        patient = build_inference_data().iloc[0].to_dict()

        response = self.client.post(
            "/feature-engineering?top_k=8",
            json=patient,
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertIn("input", payload)
        self.assertIn("model_info", payload)
        self.assertIn("feature_engineering", payload)
        self.assertIn("data_scaling", payload)
        self.assertIn("data_encoding", payload)
        feature_engineering = payload["feature_engineering"]
        data_scaling = payload["data_scaling"]
        data_encoding = payload["data_encoding"]
        self.assertEqual(feature_engineering["top_k"], 8)
        self.assertGreater(feature_engineering["feature_count"], 0)
        self.assertEqual(len(feature_engineering["engineered_features"]), 8)
        self.assertIn("engineered_value", feature_engineering["engineered_features"][0])
        self.assertGreater(data_scaling["feature_count"], 0)
        self.assertIn("scaled_features", data_scaling)
        self.assertIn("encoded_features", data_encoding)
        if data_encoding["encoded_features"]:
            self.assertIn("source_feature", data_encoding["encoded_features"][0])
            self.assertIn("category_value", data_encoding["encoded_features"][0])
            self.assertIn("explanation", data_encoding["encoded_features"][0])

    def test_explain_file_endpoint_returns_feature_importance_for_most_anomalous_row(self):
        csv_text = build_inference_data().to_csv(index=False)

        response = self.client.post(
            "/explain_file?top_k=5",
            files={"file": ("patients.csv", csv_text.encode("utf-8"), "text/csv")},
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["filename"], "patients.csv")
        self.assertIn("selected_row_index", payload)
        self.assertIn("prediction", payload)
        self.assertIn("explanation", payload)
        self.assertIn("feature_explanations", payload["explanation"])
        self.assertEqual(len(payload["explanation"]["feature_explanations"]), 5)
        self.assertIn("anomaly_score", payload["prediction"])
        self.assertIn("risk_score", payload["prediction"])
        self.assertIn("risk_category", payload["prediction"])
        self.assertIn("risk_level", payload["prediction"])

    def test_explain_batch_endpoint_returns_feature_importance_for_multiple_patients(self):
        patients = build_inference_data().to_dict(orient="records")

        response = self.client.post(
            "/explain_batch?top_k=5",
            json=patients,
            headers={"X-API-Token": self.auth_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["count"], len(patients))
        self.assertEqual(len(payload["results"]), len(patients))
        self.assertIn("patient_index", payload["results"][0])
        self.assertIn("prediction", payload["results"][0])
        self.assertIn("explanation", payload["results"][0])
        self.assertIn("feature_explanations", payload["results"][0]["explanation"])
        self.assertEqual(len(payload["results"][0]["explanation"]["feature_explanations"]), 5)


if __name__ == "__main__":
    unittest.main()
