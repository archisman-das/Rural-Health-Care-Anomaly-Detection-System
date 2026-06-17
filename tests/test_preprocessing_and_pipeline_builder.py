import unittest

import numpy as np
import pandas as pd

from rural_health_anomaly import (
    AnomalyTransformer,
    DeepAutoencoder,
    CNNAutoencoder,
    DeepSVDD,
    HealthcarePreprocessor,
    GANomaly,
    IsolationForestAnomalyModel,
    PreprocessingConfig,
    LocalOutlierFactorAnomalyModel,
    OneClassSVMAnomalyModel,
    VariationalAutoencoder,
    build_anomaly_pipeline,
)
from rural_health_anomaly.ensemble import ParallelAnomalyEnsemble, _calibrate_threshold_from_scores


class PreprocessingAndPipelineBuilderTests(unittest.TestCase):
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

        self.config = PreprocessingConfig(apply_pca=False)

    def test_preprocessor_fits_and_exports_feature_map(self):
        preprocessor = HealthcarePreprocessor(self.config)
        transformed = preprocessor.fit_transform(self.df)

        self.assertEqual(transformed.shape[0], len(self.df))
        self.assertGreater(transformed.shape[1], 0)

        feature_map = preprocessor.export_feature_map()
        self.assertFalse(feature_map.empty)
        self.assertIn("final_feature", feature_map.columns)
        self.assertIn("feature_type", feature_map.columns)
        self.assertTrue(any(name.startswith("comorbidities__") for name in feature_map["final_feature"]))

    def test_pipeline_builder_returns_working_pipeline(self):
        pipeline = build_anomaly_pipeline(self.config)
        pipeline.fit(self.df)

        predictions = pipeline.predict(self.df)
        scores = pipeline.decision_function(self.df)

        self.assertEqual(len(predictions), len(self.df))
        self.assertEqual(len(scores), len(self.df))
        self.assertTrue(set(predictions).issubset({1, -1}))

        preprocessor = pipeline.named_steps["preprocessor"]
        self.assertIsInstance(preprocessor, HealthcarePreprocessor)
        self.assertGreater(len(preprocessor.get_feature_names_out()), 0)
        self.assertIsInstance(pipeline.named_steps["model"], ParallelAnomalyEnsemble)
        self.assertEqual(
            set(pipeline.named_steps["model"].estimators_.keys()),
            {
                "isolation_forest",
                "one_class_svm",
                "local_outlier_factor",
                "autoencoder",
                "anomaly_transformer",
                "variational_autoencoder",
                "ganomaly",
                "cnn_autoencoder",
                "deep_svdd",
            },
        )

    def test_pipeline_builder_configs_isolation_forest_hyperparameters(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_n_jobs=1,
            isolation_forest_n_estimators=400,
            isolation_forest_contamination=0.07,
            isolation_forest_max_samples="auto",
            isolation_forest_max_features=0.75,
            isolation_forest_bootstrap=True,
            isolation_forest_random_state=7,
            isolation_forest_n_jobs=1,
        )

        pipeline = build_anomaly_pipeline(config)
        model = pipeline.named_steps["model"]

        self.assertEqual(model.isolation_forest_n_estimators, 400)
        self.assertEqual(model.contamination, 0.07)
        self.assertEqual(model.isolation_forest_max_samples, "auto")
        self.assertEqual(model.isolation_forest_max_features, 0.75)
        self.assertTrue(model.isolation_forest_bootstrap)
        self.assertEqual(model.isolation_forest_random_state, 7)
        self.assertEqual(model.isolation_forest_n_jobs, 1)
        self.assertEqual(model.n_jobs, 1)

    def test_autoencoder_threshold_is_derived_from_validation_errors(self):
        autoencoder = DeepAutoencoder(
            threshold_percentile=95.0,
            validation_fraction=0.25,
            max_epochs=5,
            patience=2,
            random_state=7,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        autoencoder.fit(data)

        self.assertTrue(hasattr(autoencoder, "threshold_"))
        self.assertGreaterEqual(autoencoder.threshold_, 0.0)
        self.assertEqual(autoencoder.predict(data).shape[0], data.shape[0])

    def test_cnn_autoencoder_produces_anomaly_scores(self):
        cnn_autoencoder = CNNAutoencoder(
            filters=4,
            kernel_size=3,
            latent_dim=4,
            threshold_percentile=95.0,
            validation_fraction=0.25,
            max_epochs=5,
            patience=2,
            random_state=7,
            verbose=False,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        cnn_autoencoder.fit(data)

        self.assertTrue(hasattr(cnn_autoencoder, "threshold_"))
        self.assertGreaterEqual(cnn_autoencoder.threshold_, 0.0)
        self.assertEqual(cnn_autoencoder.reconstruction_error(data).shape[0], data.shape[0])
        self.assertEqual(cnn_autoencoder.predict(data).shape[0], data.shape[0])

    def test_anomaly_transformer_produces_anomaly_scores(self):
        anomaly_transformer = AnomalyTransformer(
            hidden_dim=16,
            latent_dim=4,
            attention_weight=0.5,
            attention_temperature=1.0,
            threshold_percentile=95.0,
            validation_fraction=0.25,
            max_epochs=5,
            patience=2,
            random_state=7,
            verbose=False,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        anomaly_transformer.fit(data)

        self.assertTrue(hasattr(anomaly_transformer, "threshold_"))
        self.assertGreaterEqual(anomaly_transformer.threshold_, 0.0)
        self.assertEqual(anomaly_transformer.reconstruction_error(data).shape[0], data.shape[0])
        self.assertEqual(anomaly_transformer.attention_discrepancy(data).shape[0], data.shape[0])
        self.assertEqual(anomaly_transformer.predict(data).shape[0], data.shape[0])

    def test_variational_autoencoder_produces_anomaly_scores(self):
        variational_autoencoder = VariationalAutoencoder(
            hidden_dim=16,
            latent_dim=4,
            beta=0.5,
            threshold_percentile=95.0,
            validation_fraction=0.25,
            max_epochs=5,
            patience=2,
            random_state=7,
            verbose=False,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        variational_autoencoder.fit(data)

        self.assertTrue(hasattr(variational_autoencoder, "threshold_"))
        self.assertGreaterEqual(variational_autoencoder.threshold_, 0.0)
        self.assertEqual(variational_autoencoder.reconstruction_error(data).shape[0], data.shape[0])
        self.assertEqual(variational_autoencoder.predict(data).shape[0], data.shape[0])

    def test_ganomaly_produces_anomaly_scores(self):
        ganomaly = GANomaly(
            hidden_dim=16,
            latent_dim=4,
            consistency_weight=0.5,
            threshold_percentile=95.0,
            validation_fraction=0.25,
            max_epochs=5,
            patience=2,
            random_state=7,
            verbose=False,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        ganomaly.fit(data)

        self.assertTrue(hasattr(ganomaly, "threshold_"))
        self.assertGreaterEqual(ganomaly.threshold_, 0.0)
        self.assertEqual(ganomaly.reconstruction_error(data).shape[0], data.shape[0])
        self.assertEqual(ganomaly.latent_consistency_error(data).shape[0], data.shape[0])
        self.assertEqual(ganomaly.predict(data).shape[0], data.shape[0])

    def test_deep_svdd_threshold_is_derived_from_validation_distances(self):
        deep_svdd = DeepSVDD(
            nu=0.05,
            center_fixed=True,
            latent_dim=8,
            max_epochs=5,
            pretrain_autoencoder=False,
            random_state=7,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        deep_svdd.fit(data)

        self.assertTrue(hasattr(deep_svdd, "radius_"))
        self.assertGreaterEqual(deep_svdd.radius_, 0.0)
        self.assertEqual(deep_svdd.predict(data).shape[0], data.shape[0])

    def test_deep_svdd_1d_cnn_architecture_fits_and_predicts(self):
        deep_svdd = DeepSVDD(
            nu=0.05,
            center_fixed=True,
            architecture="1d_cnn",
            latent_dim=8,
            max_epochs=5,
            pretrain_autoencoder=False,
            random_state=7,
        )
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])
        deep_svdd.fit(data)

        self.assertTrue(hasattr(deep_svdd, "conv1_weights_"))
        self.assertTrue(hasattr(deep_svdd, "conv2_weights_"))
        self.assertTrue(hasattr(deep_svdd, "dense_weights_"))
        self.assertTrue(hasattr(deep_svdd, "radius_"))
        self.assertEqual(deep_svdd.predict(data).shape[0], data.shape[0])

    def test_individual_models_expose_common_fit_and_score_interface(self):
        numeric = self.df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        data = np.vstack([numeric, numeric, numeric])

        models = [
            IsolationForestAnomalyModel(n_estimators=50, random_state=7),
            OneClassSVMAnomalyModel(nu=0.05),
            LocalOutlierFactorAnomalyModel(n_neighbors=5, contamination=0.05),
            DeepAutoencoder(latent_dim=4, max_epochs=5, patience=2, random_state=7),
            AnomalyTransformer(hidden_dim=16, latent_dim=4, max_epochs=5, patience=2, random_state=7),
            GANomaly(hidden_dim=16, latent_dim=4, max_epochs=5, patience=2, random_state=7),
            VariationalAutoencoder(hidden_dim=16, latent_dim=4, max_epochs=5, patience=2, random_state=7),
            DeepSVDD(latent_dim=4, max_epochs=5, pretrain_autoencoder=False, random_state=7),
        ]

        for model in models:
            fitted = model.fit(data)
            scores = fitted.score(data)
            self.assertEqual(scores.shape[0], data.shape[0])
            self.assertTrue(np.all(np.isfinite(scores)))

    def test_parallel_ensemble_uses_minmax_weighted_fusion(self):
        config = PreprocessingConfig(
            apply_pca=False,
            cnn_autoencoder_weight=0.2,
            anomaly_transformer_weight=0.1,
            ganomaly_weight=0.15,
            vae_weight=0.1,
            ensemble_fusion_weights={
                "isolation_forest": 0.3,
                "one_class_svm": 0.0,
                "local_outlier_factor": 0.0,
                "autoencoder": 0.4,
                "anomaly_transformer": 0.1,
                "deep_svdd": 0.3,
            },
        )
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df)

        transformed = pipeline.named_steps["preprocessor"].transform(self.df)
        model = pipeline.named_steps["model"]
        component_scores = model.score_components(transformed)
        fused_scores = model.raw_anomaly_score(transformed)

        self.assertIn("cnn_autoencoder_anomaly_score", component_scores.columns)
        self.assertIn("anomaly_transformer_anomaly_score", component_scores.columns)
        self.assertIn("ganomaly_anomaly_score", component_scores.columns)
        self.assertIn("variational_autoencoder_anomaly_score", component_scores.columns)
        self.assertTrue(((component_scores >= 0.0) & (component_scores <= 1.0)).all().all())
        self.assertTrue(np.all(fused_scores >= 0.0))
        self.assertTrue(np.all(fused_scores <= 1.0))

        expected = (
            component_scores["isolation_forest_anomaly_score"] * 0.3
            + component_scores["one_class_svm_anomaly_score"] * 0.0
            + component_scores["local_outlier_factor_anomaly_score"] * 0.0
            + component_scores["autoencoder_anomaly_score"] * 0.4
            + component_scores["anomaly_transformer_anomaly_score"] * 0.1
            + component_scores["variational_autoencoder_anomaly_score"] * 0.1
            + component_scores["ganomaly_anomaly_score"] * 0.15
            + component_scores["cnn_autoencoder_anomaly_score"] * 0.2
            + component_scores["deep_svdd_anomaly_score"] * 0.3
        ) / 1.55
        np.testing.assert_allclose(fused_scores, expected.to_numpy(dtype=float), rtol=1e-6, atol=1e-6)

        self.assertAlmostEqual(model.fusion_weights_["cnn_autoencoder"], 0.2 / 1.55)
        self.assertAlmostEqual(model.fusion_weights_["anomaly_transformer"], 0.1 / 1.55)
        self.assertAlmostEqual(model.fusion_weights_["variational_autoencoder"], 0.1 / 1.55)
        self.assertAlmostEqual(model.fusion_weights_["ganomaly"], 0.15 / 1.55)

    def test_parallel_ensemble_uses_max_score_voting(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_fusion_strategy="max_score_voting",
            ensemble_max_score_threshold=0.5,
        )
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df)

        transformed = pipeline.named_steps["preprocessor"].transform(self.df)
        model = pipeline.named_steps["model"]
        component_scores = model.score_components(transformed)
        fused_scores = model.raw_anomaly_score(transformed)
        expected = component_scores.max(axis=1).to_numpy(dtype=float)
        expected_flags = np.where(expected >= 0.5, -1, 1)

        self.assertEqual(model.fusion_strategy_, "max_score_voting")
        np.testing.assert_allclose(fused_scores, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(model.predict(transformed), expected_flags)
        self.assertTrue(np.all((fused_scores >= 0.0) & (fused_scores <= 1.0)))

    def test_parallel_ensemble_uses_stacking_meta_classifier(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_fusion_strategy="stacking",
            stacking_meta_model_type="mlp",
            stacking_hidden_layer_sizes=(16,),
            stacking_max_iter=200,
        )
        labels = np.array([0, 0, 1])
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df, labels)

        transformed = pipeline.named_steps["preprocessor"].transform(self.df)
        model = pipeline.named_steps["model"]
        fused_scores = model.raw_anomaly_score(transformed)
        scores = model.score(transformed)

        self.assertEqual(model.fusion_strategy_, "stacking")
        self.assertTrue(hasattr(model, "stacking_meta_model_"))
        self.assertTrue(hasattr(model.stacking_meta_model_, "predict_proba"))
        self.assertEqual(getattr(model, "stacking_meta_model_type_", None), type(model.stacking_meta_model_).__name__)
        self.assertTrue(np.all((fused_scores >= 0.0) & (fused_scores <= 1.0)))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(model.predict(transformed).shape[0], len(self.df))

    def test_parallel_ensemble_uses_moe_gate_with_labels(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_fusion_strategy="moe",
            moe_gate_hidden_dim=16,
            moe_gate_max_epochs=25,
            moe_gate_patience=5,
        )
        labels = np.array([0, 0, 1], dtype=int)
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df, labels)

        transformed = pipeline.named_steps["preprocessor"].transform(self.df)
        model = pipeline.named_steps["model"]
        fused_scores = model.raw_anomaly_score(transformed)
        gate_weights = model.gate_weights(transformed)

        self.assertEqual(model.fusion_strategy_, "moe")
        self.assertTrue(hasattr(model, "moe_gate_"))
        self.assertEqual(getattr(model, "moe_gate_training_signal_", None), "label_alignment")
        self.assertEqual(gate_weights.shape[1], len(model.component_names_))
        np.testing.assert_allclose(gate_weights.sum(axis=1).to_numpy(dtype=float), np.ones(len(self.df)), rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all((fused_scores >= 0.0) & (fused_scores <= 1.0)))
        self.assertEqual(model.predict(transformed).shape[0], len(self.df))

    def test_parallel_ensemble_uses_moe_gate_without_labels(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_fusion_strategy="moe",
            moe_gate_hidden_dim=16,
            moe_gate_max_epochs=25,
            moe_gate_patience=5,
        )
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df)

        transformed = pipeline.named_steps["preprocessor"].transform(self.df)
        model = pipeline.named_steps["model"]
        gate_weights = model.gate_weights(transformed)

        self.assertEqual(model.fusion_strategy_, "moe")
        self.assertTrue(hasattr(model, "moe_gate_"))
        self.assertEqual(getattr(model, "moe_gate_training_signal_", None), "disagreement_routing")
        self.assertEqual(gate_weights.shape[0], len(self.df))
        np.testing.assert_allclose(gate_weights.sum(axis=1).to_numpy(dtype=float), np.ones(len(self.df)), rtol=1e-6, atol=1e-6)

    def test_threshold_calibration_prefers_high_f1_cutoffs(self):
        scores = np.array([0.1, 0.2, 0.35, 0.8, 0.9], dtype=float)
        labels = np.array([0, 0, 0, 1, 1], dtype=int)

        threshold, metrics = _calibrate_threshold_from_scores(scores, labels, candidate_count=21)

        self.assertGreaterEqual(threshold, 0.75)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_parallel_ensemble_can_disable_threshold_calibration(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_fusion_strategy="weighted_average",
            calibrate_threshold=False,
        )
        labels = np.array([0, 0, 1], dtype=int)
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df, labels)

        model = pipeline.named_steps["model"]
        self.assertFalse(hasattr(model, "calibrated_threshold_"))
        self.assertFalse(hasattr(model, "calibration_metrics_"))
        self.assertEqual(model.calibrate_threshold, False)

    def test_parallel_ensemble_skips_calibration_with_too_few_labels(self):
        config = PreprocessingConfig(
            apply_pca=False,
            ensemble_fusion_strategy="weighted_average",
            calibrate_threshold=True,
            calibration_min_samples=10,
        )
        labels = np.array([0, 0, 1], dtype=int)
        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(self.df, labels)

        model = pipeline.named_steps["model"]
        self.assertFalse(hasattr(model, "calibrated_threshold_"))
        self.assertFalse(hasattr(model, "calibration_metrics_"))
        self.assertFalse(model.calibration_applied_)
        self.assertEqual(model.calibration_min_samples, 10)


if __name__ == "__main__":
    unittest.main()
