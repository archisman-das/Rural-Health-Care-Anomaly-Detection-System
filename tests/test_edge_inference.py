from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rural_health_anomaly import PreprocessingConfig
from rural_health_anomaly.edge_export import export_edge_bundle
from rural_health_anomaly.edge_inference import build_parser, load_edge_bundle, score_bundle_frame
from rural_health_anomaly.example import build_inference_data, build_training_data
from rural_health_anomaly.training import train_anomaly_pipeline


class EdgeInferenceTests(unittest.TestCase):
    def test_edge_inference_parser_includes_expected_flags(self):
        parser = build_parser()
        option_strings = {
            option_string
            for action in parser._actions
            for option_string in action.option_strings
            if option_string != "-h"
        }

        self.assertIn("--bundle-dir", option_strings)
        self.assertIn("--input", option_strings)
        self.assertIn("--output", option_strings)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("skl2onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None,
        "onnxruntime edge inference dependencies are not installed",
    )
    def test_edge_bundle_scores_dataframe(self):
        config = PreprocessingConfig(
            apply_pca=False,
            deep_svdd_pretrain_autoencoder=False,
        )
        pipeline = train_anomaly_pipeline(build_training_data(), config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            export_edge_bundle(pipeline, bundle_dir)
            bundle = load_edge_bundle(bundle_dir)
            scored = score_bundle_frame(bundle, build_inference_data())

            self.assertEqual(len(scored), len(build_inference_data()))
            self.assertIn("anomaly_score", scored.columns)
            self.assertIn("risk_level", scored.columns)
            self.assertIn("isolation_forest_anomaly_score", scored.columns)
            self.assertIn("autoencoder_anomaly_score", scored.columns)
            self.assertIn("anomaly_transformer_anomaly_score", scored.columns)
            self.assertIn("variational_autoencoder_anomaly_score", scored.columns)
            self.assertIn("ganomaly_anomaly_score", scored.columns)
            self.assertIn("deep_svdd_anomaly_score", scored.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("skl2onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None,
        "onnxruntime edge inference dependencies are not installed",
    )
    def test_edge_bundle_scores_dataframe_with_stacking_meta_model(self):
        config = PreprocessingConfig(
            apply_pca=False,
            deep_svdd_pretrain_autoencoder=False,
            ensemble_fusion_strategy="stacking",
            stacking_meta_model_type="mlp",
            stacking_hidden_layer_sizes=(16,),
            stacking_max_iter=200,
        )
        data = build_training_data()
        labels = [0] * len(data)
        if labels:
            labels[-1] = 1
        pipeline = train_anomaly_pipeline(data, y=labels, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            export_edge_bundle(pipeline, bundle_dir)
            bundle = load_edge_bundle(bundle_dir)
            scored = score_bundle_frame(bundle, build_inference_data())

            self.assertEqual(len(scored), len(build_inference_data()))
            self.assertIn("anomaly_score", scored.columns)
            self.assertIn("stacking_meta_model", bundle.manifest["artifacts"])
            self.assertIn("isolation_forest_anomaly_score", scored.columns)
            self.assertIn("autoencoder_anomaly_score", scored.columns)


if __name__ == "__main__":
    unittest.main()
