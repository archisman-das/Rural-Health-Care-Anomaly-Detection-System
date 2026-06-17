import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from example_training_inference import (
    build_inference_data,
    build_large_training_data,
    build_training_data,
    main as example_main,
)
from rural_health_anomaly import PreprocessingConfig, build_anomaly_pipeline
from rural_health_anomaly.common_interface_demo import main as common_interface_demo_main
from rural_health_anomaly.training import score_records, train_anomaly_pipeline


class ExamplePipelineTests(unittest.TestCase):
    def test_example_pipeline_trains_and_scores(self):
        training_df = build_training_data()
        inference_df = build_inference_data()

        config = PreprocessingConfig(
            interaction_terms=(
                ("age_years", "glucose_fasting_mg_dl"),
                ("bmi_kg_m2", "systolic_bp_mmhg"),
                ("drug_adherence_rate", "visits_last_90_days"),
            ),
            scaler="standard",
            apply_pca=True,
        )

        pipeline = build_anomaly_pipeline(config)
        pipeline.fit(training_df)

        feature_map = pipeline.named_steps["preprocessor"].export_feature_map()
        self.assertFalse(feature_map.empty)
        self.assertIn("final_feature", feature_map.columns)

        scores = pipeline.decision_function(inference_df)
        flags = pipeline.predict(inference_df)

        self.assertEqual(len(scores), len(inference_df))
        self.assertEqual(len(flags), len(inference_df))
        self.assertTrue(set(flags).issubset({1, -1}))

        scored_frame = pipeline.named_steps["model"].raw_anomaly_score(pipeline.named_steps["preprocessor"].transform(inference_df))
        self.assertEqual(len(scored_frame), len(inference_df))

    def test_point_anomaly_head_emits_zscore_columns(self):
        training_df = build_training_data()
        inference_df = build_inference_data()

        pipeline = train_anomaly_pipeline(training_df, config=PreprocessingConfig(scaler="standard", apply_pca=False))

        point_summary = getattr(pipeline, "point_anomaly_summary_", {})

        self.assertEqual(point_summary.get("status"), "trained")
        scored = score_records(pipeline, inference_df)

        self.assertIn("point_anomaly_score", scored.columns)
        self.assertIn("point_anomaly_top_feature", scored.columns)
        self.assertIn("point_anomaly_top_feature_zscore", scored.columns)
        self.assertIn("contextual_anomaly_score", scored.columns)
        self.assertIn("contextual_anomaly_top_feature", scored.columns)
        self.assertIn("contextual_anomaly_top_feature_zscore", scored.columns)
        self.assertIn("contextual_anomaly_history_length", scored.columns)
        self.assertIn("collective_anomaly_score", scored.columns)
        self.assertIn("collective_anomaly_top_group", scored.columns)
        self.assertIn("collective_anomaly_top_group_error", scored.columns)
        point_zscore_columns = [column for column in scored.columns if column.startswith("point_zscore__")]
        contextual_zscore_columns = [column for column in scored.columns if column.startswith("contextual_zscore__")]
        collective_group_columns = [column for column in scored.columns if column.startswith("collective_group_error__")]
        self.assertGreater(len(point_zscore_columns), 0)
        self.assertGreater(len(contextual_zscore_columns), 0)
        self.assertGreater(len(collective_group_columns), 0)
        self.assertTrue(np.isfinite(scored["point_anomaly_score"].to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(scored["contextual_anomaly_score"].to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(scored["collective_anomaly_score"].to_numpy(dtype=float)).all())

    def test_distribution_monitor_emits_staleness_alarm_on_shifted_batch(self):
        training_df = build_training_data()
        inference_df = build_inference_data().copy()
        numeric_columns = inference_df.select_dtypes(include=[np.number]).columns
        inference_df[numeric_columns] = inference_df[numeric_columns] * 8.0 + 50.0

        pipeline = train_anomaly_pipeline(training_df, config=PreprocessingConfig(scaler="standard", apply_pca=False))
        scored = score_records(pipeline, inference_df)

        self.assertIn("distribution_mmd_score", scored.columns)
        self.assertIn("distribution_mmd_threshold", scored.columns)
        self.assertIn("distribution_staleness_alarm", scored.columns)
        self.assertTrue(np.isfinite(scored["distribution_mmd_score"].to_numpy(dtype=float)).all())
        self.assertTrue(bool(scored["distribution_staleness_alarm"].iloc[0]))

    def test_example_main_runs_without_error(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            example_main(training_rows=240, inference_rows=24)

        output = buffer.getvalue()
        self.assertIn("Feature map:", output)
        self.assertIn("anomaly_score", output)

    def test_large_synthetic_training_data_scales_to_thousands_of_rows(self):
        synthetic_df = build_large_training_data(target_rows=9600)

        self.assertEqual(len(synthetic_df), 9600)
        self.assertEqual(set(synthetic_df.columns), set(build_training_data().columns))
        self.assertTrue(synthetic_df["patient_id"].str.startswith("TR").all())

    def test_common_interface_demo_runs_without_error(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            common_interface_demo_main()

        output = buffer.getvalue()
        self.assertIn("model", output)
        self.assertIn("mean_score", output)
        self.assertIn("deep_svdd", output)


if __name__ == "__main__":
    unittest.main()
