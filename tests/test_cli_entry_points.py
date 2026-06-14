import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import anomaly_cli
import dashboard_server
import train_pipeline
from rural_health_anomaly.cli import build_parser, evaluate_main, predict_main, train_main
from rural_health_anomaly.training import load_pipeline


class CliEntryPointTests(unittest.TestCase):
    def _write_standard_config(self, config_path: Path) -> None:
        config_path.write_text(
            json.dumps({"apply_pca": False, "knn_neighbors": 2, "scaler": "standard"}),
            encoding="utf-8",
        )

    def _run_command(self, argv: list[str], entrypoint) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("sys.argv", argv):
                entrypoint()

    def _assert_scored_predictions(self, predictions_path: Path, expected_rows: int) -> None:
        self.assertTrue(predictions_path.exists())
        scored = pd.read_csv(predictions_path)
        self.assertIn("anomaly_score", scored.columns)
        self.assertIn("anomaly_flag", scored.columns)
        self.assertIn("is_anomaly", scored.columns)
        self.assertEqual(len(scored), expected_rows)

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

    def test_top_level_cli_train_and_predict_subcommands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            predictions_path = tmpdir_path / "predictions.csv"
            feature_map_path = tmpdir_path / "feature_map.csv"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._build_inference_frame().to_csv(infer_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--feature-map",
                    str(feature_map_path),
                    "--config-json",
                    str(config_path),
                ],
                anomaly_cli.main,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(feature_map_path.exists())

            self._run_command(
                [
                    "anomaly-cli",
                    "predict",
                    "--model",
                    str(model_path),
                    "--input",
                    str(infer_path),
                    "--output",
                    str(predictions_path),
                ],
                anomaly_cli.main,
            )

            self._assert_scored_predictions(predictions_path, len(self._build_inference_frame()))

            labels_path = tmpdir_path / "labels.csv"
            metrics_path = tmpdir_path / "metrics.json"
            report_path = tmpdir_path / "report.md"
            html_report_path = tmpdir_path / "report.html"
            pd.DataFrame({"label": [0, 1]}).to_csv(labels_path, index=False)

            self._run_command(
                [
                    "anomaly-cli",
                    "evaluate",
                    "--input",
                    str(predictions_path),
                    "--labels-file",
                    str(labels_path),
                    "--labels-column",
                    "label",
                    "--output",
                    str(metrics_path),
                    "--report-md",
                    str(report_path),
                    "--report-html",
                    str(html_report_path),
                ],
                anomaly_cli.main,
            )

            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(metrics), 1)
            self.assertIn("precision", metrics[0])
            self.assertIn("recall", metrics[0])
            self.assertIn("f1", metrics[0])
            self.assertIn("roc_auc", metrics[0])
            self.assertIn("auprc", metrics[0])
            self.assertIn("score_mean", metrics[0])
            self.assertIn("score_p95", metrics[0])
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("# Anomaly Evaluation Report", report_text)
            self.assertIn("## Model Comparison", report_text)
            self.assertIn("Best model:", report_text)
            self.assertIn("Ensemble score: anomaly_score", report_text)
            self.assertIn("score_spread", report_text)
            self.assertNotIn("## Metrics", report_text)
            self.assertNotIn("## Runtime Comparison", report_text)
            self.assertTrue(html_report_path.exists())
            html_text = html_report_path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html_text.lower())
            self.assertIn("Anomaly Executive Summary", html_text)
            self.assertIn("evaluation-report-table", html_text)
            self.assertIn("Model Comparison", html_text)
            self.assertIn("Best model:", html_text)
            self.assertIn("Ensemble score:", html_text)
            self.assertNotIn("Runtime Comparison", html_text)
            self.assertNotIn("Score Distribution", html_text)

    def test_top_level_cli_evaluate_without_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            metrics_path = tmpdir_path / "metrics.json"
            report_path = tmpdir_path / "report.md"
            html_report_path = tmpdir_path / "report.html"

            pd.DataFrame(
                {
                    "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                    "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                    "autoencoder_anomaly_score": [0.2, 0.3, 0.9, 0.85],
                    "deep_svdd_anomaly_score": [0.1, 0.2, 0.88, 0.9],
                    "autoencoder_reconstruction_error": [0.05, 0.08, 0.56, 0.61],
                }
            ).to_csv(scores_path, index=False)

            self._run_command(
                [
                    "anomaly-cli",
                    "evaluate",
                    "--input",
                    str(scores_path),
                    "--output",
                    str(metrics_path),
                    "--report-md",
                    str(report_path),
                    "--report-html",
                    str(html_report_path),
                ],
                anomaly_cli.main,
            )

            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(metrics), 1)
            self.assertIn("score_mean", metrics[0])
            self.assertIn("score_spread", metrics[0])
            self.assertNotIn("precision", metrics[0])
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("# Anomaly Evaluation Report", report_text)
            self.assertIn("Labels available: no", report_text)
            self.assertIn("## Model Comparison", report_text)
            self.assertIn("Ensemble score: anomaly_score", report_text)
            self.assertNotIn("## Metrics", report_text)
            self.assertNotIn("## Score Distribution", report_text)
            self.assertNotIn("## Reconstruction Error Histograms", report_text)
            self.assertNotIn("## Model Agreement", report_text)
            self.assertNotIn("## Highest Disagreement Rows", report_text)
            self.assertTrue(html_report_path.exists())
            html_text = html_report_path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html_text.lower())
            self.assertIn("Anomaly Executive Summary", html_text)
            self.assertIn("Labels available: no", html_text)
            self.assertIn("Model Comparison", html_text)
            self.assertIn("Ensemble score:", html_text)
            self.assertNotIn("Metrics", html_text)
            self.assertNotIn("Score Distribution", html_text)
            self.assertNotIn("Model Agreement", html_text)
            self.assertNotIn("Highest Disagreement Rows", html_text)

    def test_top_level_cli_evaluate_executive_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            labels_path = tmpdir_path / "labels.csv"
            report_path = tmpdir_path / "report.md"
            html_report_path = tmpdir_path / "report.html"

            pd.DataFrame(
                {
                    "anomaly_score": [0.1, 0.95],
                    "isolation_forest_anomaly_score": [0.15, 0.9],
                    "autoencoder_anomaly_score": [0.2, 0.85],
                    "deep_svdd_anomaly_score": [0.1, 0.88],
                }
            ).to_csv(scores_path, index=False)
            pd.DataFrame({"label": [0, 1]}).to_csv(labels_path, index=False)

            self._run_command(
                [
                    "anomaly-cli",
                    "evaluate",
                    "--input",
                    str(scores_path),
                    "--labels-file",
                    str(labels_path),
                    "--labels-column",
                    "label",
                    "--report-md",
                    str(report_path),
                    "--report-html",
                    str(html_report_path),
                    "--executive-summary",
                ],
                anomaly_cli.main,
            )

            report_text = report_path.read_text(encoding="utf-8")
            html_text = html_report_path.read_text(encoding="utf-8")
            self.assertIn("## Model Comparison", report_text)
            self.assertIn("Best model:", report_text)
            self.assertIn("Ensemble score: anomaly_score", report_text)
            self.assertNotIn("## Metrics", report_text)
            self.assertNotIn("## Runtime Comparison", report_text)
            self.assertIn("Anomaly Executive Summary", html_text)
            self.assertIn("Model Comparison", html_text)
            self.assertNotIn("Runtime Comparison", html_text)
            self.assertNotIn("Score Distribution", html_text)

    def test_top_level_cli_evaluate_with_report_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            report_prefix = tmpdir_path / "evaluation" / "bundle"
            json_report_path = tmpdir_path / "evaluation" / "bundle.json"
            md_report_path = tmpdir_path / "evaluation" / "bundle.md"
            html_report_path = tmpdir_path / "evaluation" / "bundle.html"

            pd.DataFrame(
                {
                    "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                    "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                    "autoencoder_anomaly_score": [0.2, 0.3, 0.9, 0.85],
                    "deep_svdd_anomaly_score": [0.1, 0.2, 0.88, 0.9],
                    "autoencoder_reconstruction_error": [0.05, 0.08, 0.56, 0.61],
                }
            ).to_csv(scores_path, index=False)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with patch(
                    "sys.argv",
                    [
                        "anomaly-cli",
                        "evaluate",
                        "--input",
                        str(scores_path),
                        "--report-prefix",
                        str(report_prefix),
                    ],
                ):
                    anomaly_cli.main()

            self.assertTrue(json_report_path.exists())
            self.assertTrue(md_report_path.exists())
            self.assertTrue(html_report_path.exists())
            json_metrics = json.loads(json_report_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(json_metrics), 1)
            self.assertIn("score_mean", json_metrics[0])
            self.assertIn("score_spread", json_metrics[0])
            md_text = md_report_path.read_text(encoding="utf-8")
            html_text = html_report_path.read_text(encoding="utf-8")
            self.assertIn("## Model Comparison", md_text)
            self.assertIn("Anomaly Executive Summary", html_text)
            self.assertIn("Model Comparison", html_text)
            self.assertIn("Labels available: no", md_text)
            self.assertIn("Labels available: no", html_text)
            self.assertIn("Ensemble score: anomaly_score", md_text)
            self.assertIn("Ensemble score:", html_text)
            self.assertNotIn("## Metrics", md_text)
            self.assertNotIn("## Runtime Comparison", md_text)
            self.assertNotIn("## Score Distribution", md_text)
            self.assertNotIn("## Metrics", html_text)
            self.assertNotIn("## Runtime Comparison", html_text)
            self.assertNotIn("Score Distribution", html_text)
            self.assertIn("Executive summary mode enabled for report output.", stdout.getvalue())

    def test_top_level_cli_evaluate_with_report_prefix_full_report_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            report_prefix = tmpdir_path / "evaluation" / "bundle"
            md_report_path = tmpdir_path / "evaluation" / "bundle.md"
            html_report_path = tmpdir_path / "evaluation" / "bundle.html"

            pd.DataFrame(
                {
                    "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                    "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                    "autoencoder_anomaly_score": [0.2, 0.3, 0.9, 0.85],
                    "deep_svdd_anomaly_score": [0.1, 0.2, 0.88, 0.9],
                    "autoencoder_reconstruction_error": [0.05, 0.08, 0.56, 0.61],
                }
            ).to_csv(scores_path, index=False)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with patch(
                    "sys.argv",
                    [
                        "anomaly-cli",
                        "evaluate",
                        "--input",
                        str(scores_path),
                        "--report-prefix",
                        str(report_prefix),
                        "--no-executive-summary",
                    ],
                ):
                    anomaly_cli.main()

            md_text = md_report_path.read_text(encoding="utf-8")
            html_text = html_report_path.read_text(encoding="utf-8")
            self.assertIn("## Metrics", md_text)
            self.assertIn("## Score Distribution", md_text)
            self.assertIn("## Model Agreement", md_text)
            self.assertIn("## Highest Disagreement Rows", md_text)
            self.assertIn("Metrics", html_text)
            self.assertIn("Score Distribution", html_text)
            self.assertIn("Model Agreement", html_text)
            self.assertNotIn("Executive summary mode enabled for report output.", stdout.getvalue())

    def test_top_level_cli_evaluate_dashboard_html_full_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            labels_path = tmpdir_path / "labels.csv"
            dashboard_path = tmpdir_path / "dashboard.html"
            metrics_path = tmpdir_path / "metrics.json"

            pd.DataFrame(
                {
                    "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                    "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                    "autoencoder_anomaly_score": [0.2, 0.3, 0.9, 0.85],
                    "deep_svdd_anomaly_score": [0.1, 0.2, 0.88, 0.9],
                    "autoencoder_reconstruction_error": [0.05, 0.08, 0.56, 0.61],
                    "training_time_seconds": [4.2, 4.2, 4.2, 4.2],
                    "training_time_ms": [4200.0, 4200.0, 4200.0, 4200.0],
                    "model_size_bytes": [48_000_000, 48_000_000, 48_000_000, 48_000_000],
                    "estimated_ram_usage_bytes": [120_000_000, 120_000_000, 120_000_000, 120_000_000],
                    "inference_batch_latency_ms": [120.0, 120.0, 120.0, 120.0],
                    "inference_latency_ms_per_patient": [60.0, 60.0, 60.0, 60.0],
                    "inference_throughput_rows_per_second": [16.7, 16.7, 16.7, 16.7],
                }
            ).to_csv(scores_path, index=False)
            pd.DataFrame({"label": [0, 1, 1, 1]}).to_csv(labels_path, index=False)

            self._run_command(
                [
                    "anomaly-cli",
                    "evaluate",
                    "--input",
                    str(scores_path),
                    "--labels-file",
                    str(labels_path),
                    "--labels-column",
                    "label",
                    "--output",
                    str(metrics_path),
                    "--dashboard-html",
                    str(dashboard_path),
                ],
                anomaly_cli.main,
            )

            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(metrics), 1)
            self.assertIn("precision", metrics[0])
            self.assertIn("recall", metrics[0])
            self.assertIn("f1", metrics[0])
            self.assertIn("roc_auc", metrics[0])
            self.assertIn("auprc", metrics[0])

            self.assertTrue(dashboard_path.exists())
            dashboard_text = dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Anomaly Evaluation Report", dashboard_text)
            self.assertIn("<h2>Metrics</h2>", dashboard_text)
            self.assertIn("<h2>Score Distribution</h2>", dashboard_text)
            self.assertIn("<h2>Reconstruction Error Histograms</h2>", dashboard_text)
            self.assertIn("<h2>Model Agreement</h2>", dashboard_text)
            self.assertIn("<h2>Highest Disagreement Rows</h2>", dashboard_text)
            self.assertIn("<h2>Runtime Comparison</h2>", dashboard_text)
            self.assertIn("Edge readiness", dashboard_text)
            self.assertIn("precision", dashboard_text)
            self.assertIn("recall", dashboard_text)
            self.assertIn("f1", dashboard_text)
            self.assertIn("roc_auc", dashboard_text)
            self.assertIn("auprc", dashboard_text)

    def test_legacy_anomaly_cli_py_train_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            predictions_path = tmpdir_path / "predictions.csv"
            feature_map_path = tmpdir_path / "feature_map.csv"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._build_inference_frame().to_csv(infer_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly_cli.py",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--feature-map",
                    str(feature_map_path),
                    "--config-json",
                    str(config_path),
                ],
                anomaly_cli.main,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(feature_map_path.exists())

            self._run_command(
                [
                    "anomaly_cli.py",
                    "predict",
                    "--model",
                    str(model_path),
                    "--input",
                    str(infer_path),
                    "--output",
                    str(predictions_path),
                ],
                anomaly_cli.main,
            )

            self._assert_scored_predictions(predictions_path, len(self._build_inference_frame()))

    def test_dedicated_evaluate_command_entry_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            labels_path = tmpdir_path / "labels.csv"
            model_path = tmpdir_path / "model.joblib"
            predictions_path = tmpdir_path / "predictions.csv"
            metrics_path = tmpdir_path / "metrics.json"
            report_path = tmpdir_path / "report.md"
            html_report_path = tmpdir_path / "report.html"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._build_inference_frame().to_csv(infer_path, index=False)
            pd.DataFrame({"label": [0, 1]}).to_csv(labels_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                ],
                train_main,
            )

            self._run_command(
                [
                    "anomaly-predict",
                    "--model",
                    str(model_path),
                    "--input",
                    str(infer_path),
                    "--output",
                    str(predictions_path),
                ],
                predict_main,
            )

            self._run_command(
                [
                    "anomaly-evaluate",
                    "--input",
                    str(predictions_path),
                    "--labels-file",
                    str(labels_path),
                    "--labels-column",
                    "label",
                    "--output",
                    str(metrics_path),
                    "--report-md",
                    str(report_path),
                    "--report-html",
                    str(html_report_path),
                ],
                evaluate_main,
            )

            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(metrics), 1)
            self.assertIn("precision", metrics[0])
            self.assertIn("recall", metrics[0])
            self.assertIn("f1", metrics[0])
            self.assertIn("roc_auc", metrics[0])
            self.assertIn("auprc", metrics[0])
            self.assertIn("score_mean", metrics[0])
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Labels available: yes", report_text)
            self.assertIn("## Model Comparison", report_text)
            self.assertIn("Best model:", report_text)
            self.assertIn("Ensemble score: anomaly_score", report_text)
            self.assertNotIn("## Metrics", report_text)
            self.assertNotIn("## Runtime Comparison", report_text)
            self.assertTrue(html_report_path.exists())
            html_text = html_report_path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html_text.lower())
            self.assertIn("Anomaly Executive Summary", html_text)
            self.assertIn("Labels available: yes", html_text)
            self.assertIn("Model Comparison", html_text)
            self.assertIn("Best model:", html_text)
            self.assertIn("Ensemble score:", html_text)
            self.assertNotIn("Runtime Comparison", html_text)
            self.assertNotIn("Score Distribution", html_text)

    def test_legacy_anomaly_cli_py_predict_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            predictions_path = tmpdir_path / "predictions.csv"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._build_inference_frame().to_csv(infer_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly_cli.py",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                ],
                anomaly_cli.main,
            )

            self._run_command(
                [
                    "anomaly_cli.py",
                    "predict",
                    "--model",
                    str(model_path),
                    "--input",
                    str(infer_path),
                    "--output",
                    str(predictions_path),
                ],
                anomaly_cli.main,
            )

            self._assert_scored_predictions(predictions_path, len(self._build_inference_frame()))

    def test_legacy_anomaly_cli_py_help_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as exc:
                with patch("sys.argv", ["anomaly_cli.py", "--help"]):
                    anomaly_cli.main()

        self.assertEqual(exc.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("train", help_text)
        self.assertIn("predict", help_text)
        self.assertIn("evaluate", help_text)

    def test_legacy_train_pipeline_py_help_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as exc:
                with patch("sys.argv", ["train_pipeline.py", "--help"]):
                    train_pipeline.main()

        self.assertEqual(exc.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--input", help_text)
        self.assertIn("--output", help_text)
        self.assertIn("--feature-map", help_text)
        self.assertIn("--config-json", help_text)
        self.assertIn("--calibrate-threshold", help_text)
        self.assertIn("--no-calibrate-threshold", help_text)
        self.assertIn("--synthetic-demo-data", help_text)
        self.assertIn("--synthetic-demo-rows", help_text)
        self.assertIn("--synthetic-demo-seed", help_text)

    def test_legacy_train_pipeline_py_train_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            feature_map_path = tmpdir_path / "feature_map.csv"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "train_pipeline.py",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--feature-map",
                    str(feature_map_path),
                    "--config-json",
                    str(config_path),
                ],
                train_pipeline.main,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(feature_map_path.exists())

    def test_legacy_train_pipeline_py_synthetic_demo_training_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_path = tmpdir_path / "synthetic_model.joblib"
            feature_map_path = tmpdir_path / "synthetic_feature_map.csv"
            config_path = tmpdir_path / "config.json"

            self._write_standard_config(config_path)

            self._run_command(
                [
                    "train_pipeline.py",
                    "--synthetic-demo-data",
                    "--synthetic-demo-rows",
                    "120",
                    "--synthetic-demo-seed",
                    "7",
                    "--output",
                    str(model_path),
                    "--feature-map",
                    str(feature_map_path),
                    "--config-json",
                    str(config_path),
                ],
                train_pipeline.main,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(feature_map_path.exists())

    def test_legacy_train_pipeline_py_can_disable_threshold_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            training_frame = self._build_training_frame().copy()
            training_frame["label"] = [0, 0, 1]
            training_frame.to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "train_pipeline.py",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--no-calibrate-threshold",
                    "--label-column",
                    "label",
                ],
                train_pipeline.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertFalse(model.calibrate_threshold)
            self.assertFalse(hasattr(model, "calibrated_threshold_"))
            self.assertFalse(hasattr(model, "calibration_metrics_"))

    def test_legacy_train_pipeline_py_can_override_calibration_min_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            training_frame = self._build_training_frame().copy()
            training_frame["label"] = [0, 0, 1]
            training_frame.to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "train_pipeline.py",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--calibration-min-samples",
                    "7",
                    "--label-column",
                    "label",
                ],
                train_pipeline.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.calibration_min_samples, 7)

    def test_dedicated_train_command_entry_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            feature_map_path = tmpdir_path / "feature_map.csv"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--feature-map",
                    str(feature_map_path),
                    "--config-json",
                    str(config_path),
                ],
                train_main,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(feature_map_path.exists())

    def test_train_cli_accepts_deep_svdd_architecture_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--deep-svdd-nu",
                    "0.08",
                    "--deep-svdd-architecture",
                    "1d_cnn",
                    "--deep-svdd-latent-dim",
                    "4",
                    "--deep-svdd-max-epochs",
                    "3",
                    "--no-deep-svdd-pretrain-autoencoder",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.deep_svdd_architecture, "1d_cnn")
            self.assertEqual(model.deep_svdd_nu, 0.08)
            self.assertEqual(model.deep_svdd_latent_dim, 4)
            self.assertEqual(model.deep_svdd_max_epochs, 3)
            self.assertFalse(model.deep_svdd_pretrain_autoencoder)

    def test_train_cli_accepts_autoencoder_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--autoencoder-latent-dim",
                    "4",
                    "--autoencoder-threshold-percentile",
                    "95.0",
                    "--autoencoder-dropout",
                    "0.15",
                    "--autoencoder-learning-rate",
                    "0.0005",
                    "--autoencoder-batch-size",
                    "16",
                    "--autoencoder-validation-fraction",
                    "0.25",
                    "--autoencoder-max-epochs",
                    "5",
                    "--autoencoder-patience",
                    "2",
                    "--autoencoder-l2",
                    "1e-4",
                    "--autoencoder-random-state",
                    "11",
                    "--no-autoencoder-verbose",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.autoencoder_latent_dim, 4)
            self.assertEqual(model.autoencoder_threshold_percentile, 95.0)
            self.assertEqual(model.autoencoder_dropout, 0.15)
            self.assertEqual(model.autoencoder_learning_rate, 0.0005)
            self.assertEqual(model.autoencoder_batch_size, 16)
            self.assertEqual(model.autoencoder_validation_fraction, 0.25)
            self.assertEqual(model.autoencoder_max_epochs, 5)
            self.assertEqual(model.autoencoder_patience, 2)
            self.assertEqual(model.autoencoder_l2, 1e-4)
            self.assertEqual(model.autoencoder_random_state, 11)
            self.assertFalse(model.autoencoder_verbose)

    def test_train_cli_accepts_ganomaly_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--ganomaly-hidden-dim",
                    "16",
                    "--ganomaly-latent-dim",
                    "4",
                    "--ganomaly-dropout",
                    "0.1",
                    "--ganomaly-learning-rate",
                    "0.0005",
                    "--ganomaly-batch-size",
                    "16",
                    "--ganomaly-consistency-weight",
                    "0.6",
                    "--ganomaly-threshold-percentile",
                    "95.0",
                    "--ganomaly-validation-fraction",
                    "0.25",
                    "--ganomaly-max-epochs",
                    "5",
                    "--ganomaly-patience",
                    "2",
                    "--ganomaly-l2",
                    "1e-4",
                    "--ganomaly-random-state",
                    "11",
                    "--no-ganomaly-verbose",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.ganomaly_hidden_dim, 16)
            self.assertEqual(model.ganomaly_latent_dim, 4)
            self.assertEqual(model.ganomaly_dropout, 0.1)
            self.assertEqual(model.ganomaly_learning_rate, 0.0005)
            self.assertEqual(model.ganomaly_batch_size, 16)
            self.assertEqual(model.ganomaly_consistency_weight, 0.6)
            self.assertEqual(model.ganomaly_threshold_percentile, 95.0)
            self.assertEqual(model.ganomaly_validation_fraction, 0.25)
            self.assertEqual(model.ganomaly_max_epochs, 5)
            self.assertEqual(model.ganomaly_patience, 2)
            self.assertEqual(model.ganomaly_l2, 1e-4)
            self.assertEqual(model.ganomaly_random_state, 11)
            self.assertFalse(model.ganomaly_verbose)

    def test_train_cli_accepts_anomaly_transformer_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--anomaly-transformer-hidden-dim",
                    "16",
                    "--anomaly-transformer-latent-dim",
                    "4",
                    "--anomaly-transformer-dropout",
                    "0.1",
                    "--anomaly-transformer-learning-rate",
                    "0.0005",
                    "--anomaly-transformer-batch-size",
                    "16",
                    "--anomaly-transformer-attention-weight",
                    "0.7",
                    "--anomaly-transformer-attention-temperature",
                    "0.8",
                    "--anomaly-transformer-threshold-percentile",
                    "95.0",
                    "--anomaly-transformer-validation-fraction",
                    "0.25",
                    "--anomaly-transformer-max-epochs",
                    "5",
                    "--anomaly-transformer-patience",
                    "2",
                    "--anomaly-transformer-l2",
                    "1e-4",
                    "--anomaly-transformer-random-state",
                    "11",
                    "--no-anomaly-transformer-verbose",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.anomaly_transformer_hidden_dim, 16)
            self.assertEqual(model.anomaly_transformer_latent_dim, 4)
            self.assertEqual(model.anomaly_transformer_dropout, 0.1)
            self.assertEqual(model.anomaly_transformer_learning_rate, 0.0005)
            self.assertEqual(model.anomaly_transformer_batch_size, 16)
            self.assertEqual(model.anomaly_transformer_attention_weight, 0.7)
            self.assertEqual(model.anomaly_transformer_attention_temperature, 0.8)
            self.assertEqual(model.anomaly_transformer_threshold_percentile, 95.0)
            self.assertEqual(model.anomaly_transformer_validation_fraction, 0.25)
            self.assertEqual(model.anomaly_transformer_max_epochs, 5)
            self.assertEqual(model.anomaly_transformer_patience, 2)
            self.assertEqual(model.anomaly_transformer_l2, 1e-4)
            self.assertEqual(model.anomaly_transformer_random_state, 11)
            self.assertFalse(model.anomaly_transformer_verbose)

    def test_train_cli_accepts_ensemble_fusion_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--ensemble-fusion-strategy",
                    "max_score_voting",
                    "--ensemble-max-score-threshold",
                    "0.65",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.fusion_strategy, "max_score_voting")
            self.assertEqual(model.max_score_threshold, 0.65)
            self.assertEqual(model.fusion_strategy_, "max_score_voting")
            self.assertEqual(model.offset_, 0.65)

    def test_train_cli_supports_stacking_label_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            training_frame = self._build_training_frame().copy()
            training_frame["label"] = [0, 0, 1]
            training_frame.to_csv(train_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--ensemble-fusion-strategy",
                    "stacking",
                    "--label-column",
                    "label",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.fusion_strategy, "stacking")
            self.assertTrue(hasattr(model, "stacking_meta_model_"))
            self.assertEqual(model.fusion_strategy_, "stacking")

    def test_train_cli_supports_stacking_labels_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            labels_path = tmpdir_path / "labels.csv"
            model_path = tmpdir_path / "model.joblib"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            pd.DataFrame({"label": [0, 0, 1]}).to_csv(labels_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-cli",
                    "train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                    "--ensemble-fusion-strategy",
                    "stacking",
                    "--labels-file",
                    str(labels_path),
                    "--labels-column",
                    "label",
                ],
                anomaly_cli.main,
            )

            pipeline = load_pipeline(model_path)
            model = pipeline.named_steps["model"]
            self.assertEqual(model.fusion_strategy, "stacking")
            self.assertTrue(hasattr(model, "stacking_meta_model_"))
            self.assertEqual(model.fusion_strategy_, "stacking")

    def test_dedicated_predict_command_entry_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.csv"
            infer_path = tmpdir_path / "infer.csv"
            model_path = tmpdir_path / "model.joblib"
            predictions_path = tmpdir_path / "predictions.csv"
            config_path = tmpdir_path / "config.json"

            self._build_training_frame().to_csv(train_path, index=False)
            self._build_inference_frame().to_csv(infer_path, index=False)
            self._write_standard_config(config_path)

            self._run_command(
                [
                    "anomaly-train",
                    "--input",
                    str(train_path),
                    "--output",
                    str(model_path),
                    "--config-json",
                    str(config_path),
                ],
                train_main,
            )

            self._run_command(
                [
                    "anomaly-predict",
                    "--model",
                    str(model_path),
                    "--input",
                    str(infer_path),
                    "--output",
                    str(predictions_path),
                ],
                predict_main,
            )

            self._assert_scored_predictions(predictions_path, len(self._build_inference_frame()))

    def test_dedicated_dashboard_command_entry_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            dashboard_path = tmpdir_path / "dashboard.html"

            pd.DataFrame(
                {
                    "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                    "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                    "autoencoder_anomaly_score": [0.2, 0.3, 0.9, 0.85],
                    "deep_svdd_anomaly_score": [0.1, 0.2, 0.88, 0.9],
                    "autoencoder_reconstruction_error": [0.05, 0.08, 0.56, 0.61],
                    "training_time_seconds": [4.2, 4.2, 4.2, 4.2],
                    "training_time_ms": [4200.0, 4200.0, 4200.0, 4200.0],
                    "model_size_bytes": [48_000_000, 48_000_000, 48_000_000, 48_000_000],
                    "estimated_ram_usage_bytes": [120_000_000, 120_000_000, 120_000_000, 120_000_000],
                    "inference_batch_latency_ms": [120.0, 120.0, 120.0, 120.0],
                    "inference_latency_ms_per_patient": [60.0, 60.0, 60.0, 60.0],
                    "inference_throughput_rows_per_second": [16.7, 16.7, 16.7, 16.7],
                }
            ).to_csv(scores_path, index=False)

            with patch("dashboard_server.serve_dashboard") as serve_mock:
                self._run_command(
                    [
                        "anomaly-dashboard",
                        "--input",
                        str(scores_path),
                        "--output",
                        str(dashboard_path),
                        "--no-open-browser",
                    ],
                    dashboard_server.main,
                )
            serve_mock.assert_called_once()

            self.assertTrue(dashboard_path.exists())
            html_text = dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Anomaly Evaluation Report", html_text)
            self.assertIn("<h2>Metrics</h2>", html_text)
            self.assertIn("<h2>Runtime Comparison</h2>", html_text)

    def test_help_output_includes_commands(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as exc:
                with patch("sys.argv", ["anomaly-cli", "--help"]):
                    anomaly_cli.main()

        self.assertEqual(exc.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("train", help_text)
        self.assertIn("predict", help_text)

        with contextlib.redirect_stdout(io.StringIO()) as train_stdout:
            with self.assertRaises(SystemExit) as train_exc:
                with patch("sys.argv", ["anomaly-train", "--help"]):
                    train_main()

        self.assertEqual(train_exc.exception.code, 0)
        train_help = train_stdout.getvalue()
        self.assertIn("--input", train_help)
        self.assertIn("--output", train_help)
        self.assertIn("--feature-map", train_help)
        self.assertIn("--config-json", train_help)
        self.assertIn("--autoencoder-latent-dim", train_help)
        self.assertIn("--autoencoder-threshold-percentile", train_help)
        self.assertIn("--no-autoencoder-verbose", train_help)
        self.assertIn("--deep-svdd-architecture", train_help)
        self.assertIn("--deep-svdd-nu", train_help)
        self.assertIn("--deep-svdd-latent-dim", train_help)
        self.assertIn("--vae-hidden-dim", train_help)
        self.assertIn("--vae-latent-dim", train_help)
        self.assertIn("--vae-beta", train_help)
        self.assertIn("--vae-learning-rate", train_help)
        self.assertIn("--anomaly-transformer-hidden-dim", train_help)
        self.assertIn("--anomaly-transformer-latent-dim", train_help)
        self.assertIn("--anomaly-transformer-attention-weight", train_help)
        self.assertIn("--anomaly-transformer-weight", train_help)
        self.assertIn("--no-anomaly-transformer-verbose", train_help)
        self.assertIn("--ganomaly-hidden-dim", train_help)
        self.assertIn("--ganomaly-latent-dim", train_help)
        self.assertIn("--ganomaly-consistency-weight", train_help)
        self.assertIn("--ganomaly-weight", train_help)
        self.assertIn("--no-ganomaly-verbose", train_help)
        self.assertIn("--vae-weight", train_help)
        self.assertIn("--no-deep-svdd-pretrain-autoencoder", train_help)
        self.assertIn("--ensemble-fusion-strategy", train_help)
        self.assertIn("--ensemble-max-score-threshold", train_help)
        self.assertIn("--calibrate-threshold", train_help)
        self.assertIn("--no-calibrate-threshold", train_help)
        self.assertIn("--calibration-min-samples", train_help)
        self.assertIn("--label-column", train_help)
        self.assertIn("--labels-file", train_help)
        self.assertIn("--labels-column", train_help)
        self.assertIn("--synthetic-demo-data", train_help)
        self.assertIn("--synthetic-demo-rows", train_help)
        self.assertIn("--synthetic-demo-seed", train_help)
        self.assertIn("autoencoder_latent_dim", train_help)
        self.assertIn("autoencoder_threshold_percentile", train_help)
        self.assertIn("autoencoder_dropout", train_help)
        self.assertIn("deep_svdd_nu", train_help)
        self.assertIn("deep_svdd_architecture", train_help)

        with contextlib.redirect_stdout(io.StringIO()) as cli_train_stdout:
            with self.assertRaises(SystemExit) as cli_train_exc:
                with patch("sys.argv", ["anomaly-cli", "train", "--help"]):
                    anomaly_cli.main()

        self.assertEqual(cli_train_exc.exception.code, 0)
        cli_train_help = cli_train_stdout.getvalue()
        self.assertIn("autoencoder_latent_dim", cli_train_help)
        self.assertIn("autoencoder_threshold_percentile", cli_train_help)
        self.assertIn("autoencoder_dropout", cli_train_help)
        self.assertIn("--autoencoder-latent-dim", cli_train_help)
        self.assertIn("vae_hidden_dim", cli_train_help)
        self.assertIn("vae_latent_dim", cli_train_help)
        self.assertIn("vae_beta", cli_train_help)
        self.assertIn("anomaly_transformer_hidden_dim", cli_train_help)
        self.assertIn("anomaly_transformer_latent_dim", cli_train_help)
        self.assertIn("anomaly_transformer_attention_weight", cli_train_help)
        self.assertIn("anomaly_transformer_weight", cli_train_help)
        self.assertIn("ganomaly_hidden_dim", cli_train_help)
        self.assertIn("ganomaly_latent_dim", cli_train_help)
        self.assertIn("ganomaly_consistency_weight", cli_train_help)
        self.assertIn("ganomaly_weight", cli_train_help)
        self.assertIn("--vae-hidden-dim", cli_train_help)
        self.assertIn("--anomaly-transformer-hidden-dim", cli_train_help)
        self.assertIn("--anomaly-transformer-latent-dim", cli_train_help)
        self.assertIn("--anomaly-transformer-attention-weight", cli_train_help)
        self.assertIn("--anomaly-transformer-weight", cli_train_help)
        self.assertIn("--no-anomaly-transformer-verbose", cli_train_help)
        self.assertIn("deep_svdd_nu", cli_train_help)
        self.assertIn("deep_svdd_architecture", cli_train_help)
        self.assertIn("--deep-svdd-architecture", cli_train_help)
        self.assertIn("--deep-svdd-nu", cli_train_help)
        self.assertIn("--deep-svdd-latent-dim", cli_train_help)
        self.assertIn("--ganomaly-hidden-dim", cli_train_help)
        self.assertIn("--ganomaly-latent-dim", cli_train_help)
        self.assertIn("--ganomaly-consistency-weight", cli_train_help)
        self.assertIn("--ganomaly-weight", cli_train_help)
        self.assertIn("--no-ganomaly-verbose", cli_train_help)
        self.assertIn("--vae-weight", cli_train_help)
        self.assertIn("--ensemble-fusion-strategy", cli_train_help)
        self.assertIn("--ensemble-max-score-threshold", cli_train_help)
        self.assertIn("--calibrate-threshold", cli_train_help)
        self.assertIn("--no-calibrate-threshold", cli_train_help)
        self.assertIn("--calibration-min-samples", cli_train_help)
        self.assertIn("--label-column", cli_train_help)
        self.assertIn("--labels-file", cli_train_help)
        self.assertIn("--labels-column", cli_train_help)
        self.assertIn("--synthetic-demo-data", cli_train_help)
        self.assertIn("--synthetic-demo-rows", cli_train_help)
        self.assertIn("--synthetic-demo-seed", cli_train_help)

        with contextlib.redirect_stdout(io.StringIO()) as predict_stdout:
            with self.assertRaises(SystemExit) as predict_exc:
                with patch("sys.argv", ["anomaly-predict", "--help"]):
                    predict_main()

        self.assertEqual(predict_exc.exception.code, 0)
        predict_help = predict_stdout.getvalue()
        self.assertIn("--model", predict_help)
        self.assertIn("--input", predict_help)
        self.assertIn("--output", predict_help)

        with contextlib.redirect_stdout(io.StringIO()) as export_edge_stdout:
            with self.assertRaises(SystemExit) as export_edge_exc:
                with patch("sys.argv", ["anomaly-cli", "export-edge", "--help"]):
                    anomaly_cli.main()

        self.assertEqual(export_edge_exc.exception.code, 0)
        export_edge_help = export_edge_stdout.getvalue()
        self.assertIn("--model", export_edge_help)
        self.assertIn("--output-dir", export_edge_help)
        self.assertIn("--opset", export_edge_help)

        with contextlib.redirect_stdout(io.StringIO()) as retrain_feedback_stdout:
            with self.assertRaises(SystemExit) as retrain_feedback_exc:
                with patch("sys.argv", ["anomaly-cli", "retrain-feedback", "--help"]):
                    anomaly_cli.main()

        self.assertEqual(retrain_feedback_exc.exception.code, 0)
        retrain_feedback_help = retrain_feedback_stdout.getvalue()
        self.assertIn("--input", retrain_feedback_help)
        self.assertIn("--feedback-file", retrain_feedback_help)
        self.assertIn("--output", retrain_feedback_help)
        self.assertIn("--config-json", retrain_feedback_help)

        with contextlib.redirect_stdout(io.StringIO()) as evaluate_stdout:
            with self.assertRaises(SystemExit) as evaluate_exc:
                with patch("sys.argv", ["anomaly-evaluate", "--help"]):
                    evaluate_main()

        self.assertEqual(evaluate_exc.exception.code, 0)
        evaluate_help = evaluate_stdout.getvalue()
        self.assertIn("--input", evaluate_help)
        self.assertIn("--score-column", evaluate_help)
        self.assertIn("--score-columns", evaluate_help)
        self.assertIn("--threshold", evaluate_help)
        self.assertIn("--output", evaluate_help)
        self.assertIn("--report-prefix", evaluate_help)
        self.assertIn("--report-md", evaluate_help)
        self.assertIn("--report-html", evaluate_help)
        self.assertIn("--dashboard-html", evaluate_help)
        self.assertIn("short comparison-only report", evaluate_help.lower())
        self.assertIn("--top-fraction", evaluate_help)
        self.assertIn("--executive-summary", evaluate_help)
        self.assertIn("--no-executive-summary", evaluate_help)
        self.assertIn("--labels-file", evaluate_help)
        self.assertIn("--labels-column", evaluate_help)
        self.assertIn("--label-column", evaluate_help)

        with contextlib.redirect_stdout(io.StringIO()) as dashboard_stdout:
            with self.assertRaises(SystemExit) as dashboard_exc:
                with patch("sys.argv", ["anomaly-dashboard", "--help"]):
                    dashboard_server.main()

        self.assertEqual(dashboard_exc.exception.code, 0)
        dashboard_help = dashboard_stdout.getvalue()
        self.assertIn("--input", dashboard_help)
        self.assertIn("--score-column", dashboard_help)
        self.assertIn("--score-columns", dashboard_help)
        self.assertIn("--threshold", dashboard_help)
        self.assertIn("--top-fraction", dashboard_help)
        self.assertIn("--labels-file", dashboard_help)
        self.assertIn("--labels-column", dashboard_help)
        self.assertIn("--label-column", dashboard_help)
        self.assertIn("--host", dashboard_help)
        self.assertIn("--port", dashboard_help)
        self.assertIn("--output", dashboard_help)
        self.assertIn("--open-browser", dashboard_help)
        self.assertIn("--no-open-browser", dashboard_help)

    def test_readme_command_examples_match_cli_entry_points(self):
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")

        self.assertIn("anomaly-cli train", readme_text)
        self.assertIn("anomaly-cli predict", readme_text)
        self.assertIn("anomaly-cli evaluate", readme_text)
        self.assertIn("anomaly-cli export-edge", readme_text)
        self.assertIn("anomaly-cli retrain-feedback", readme_text)
        self.assertIn("anomaly-train", readme_text)
        self.assertIn("anomaly-predict", readme_text)
        self.assertIn("anomaly-evaluate", readme_text)
        self.assertIn("anomaly-edge-export", readme_text)
        self.assertIn("anomaly-edge-infer", readme_text)
        self.assertIn("anomaly-dashboard", readme_text)
        self.assertIn("--input", readme_text)
        self.assertIn("--output", readme_text)
        self.assertIn("--model", readme_text)
        self.assertIn("--feature-map", readme_text)

    def test_readme_examples_match_parser_structure(self):
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")
        parser = build_parser()
        subparsers_action = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )

        self.assertEqual(set(subparsers_action.choices), {"train", "split-data", "predict", "export-edge", "retrain-feedback", "evaluate"})
        self.assertIn("anomaly-cli train", readme_text)
        self.assertIn("anomaly-cli predict", readme_text)
        self.assertIn("anomaly-cli evaluate", readme_text)
        self.assertIn("anomaly-cli export-edge", readme_text)
        self.assertIn("anomaly-cli retrain-feedback", readme_text)
        self.assertIn("anomaly-train", readme_text)
        self.assertIn("anomaly-predict", readme_text)
        self.assertIn("anomaly-evaluate", readme_text)
        self.assertIn("anomaly-edge-export", readme_text)
        self.assertIn("anomaly-edge-infer", readme_text)
        self.assertIn("anomaly-dashboard", readme_text)

        expected_flags = {
            "train": {
                "--input",
                "--output",
                "--feature-map",
                "--config-json",
                "--calibrate-threshold",
                "--no-calibrate-threshold",
                "--calibration-min-samples",
                "--anomaly-transformer-hidden-dim",
                "--anomaly-transformer-latent-dim",
                "--anomaly-transformer-attention-weight",
                "--anomaly-transformer-weight",
                "--ganomaly-hidden-dim",
                "--ganomaly-latent-dim",
                "--ganomaly-consistency-weight",
                "--ganomaly-weight",
                "--vae-hidden-dim",
                "--vae-latent-dim",
                "--vae-beta",
                "--vae-learning-rate",
                "--ganomaly-hidden-dim",
                "--ganomaly-latent-dim",
                "--ganomaly-consistency-weight",
                "--ganomaly-weight",
                "--anomaly-transformer-hidden-dim",
                "--anomaly-transformer-latent-dim",
                "--anomaly-transformer-attention-weight",
                "--anomaly-transformer-weight",
                "--vae-weight",
                "--synthetic-demo-data",
                "--synthetic-demo-rows",
                "--synthetic-demo-seed",
            },
            "split-data": {"--input", "--output-dir", "--train-fraction", "--validation-fraction", "--test-fraction", "--group-column", "--random-state"},
            "predict": {"--model", "--input", "--output"},
            "export-edge": {"--model", "--output-dir", "--opset"},
            "retrain-feedback": {"--input", "--feedback-file", "--output", "--config-json"},
            "evaluate": {"--input", "--score-column", "--score-columns", "--threshold", "--output", "--report-prefix", "--report-md", "--report-html", "--dashboard-html", "--top-fraction", "--executive-summary", "--no-executive-summary"},
        }

        for command_name, parser_for_command in subparsers_action.choices.items():
            option_strings = {
                option_string
                for action in parser_for_command._actions
                for option_string in action.option_strings
                if option_string != "-h"
            }
            self.assertTrue(expected_flags[command_name].issubset(option_strings))
            for flag in expected_flags[command_name]:
                self.assertIn(flag, readme_text)


if __name__ == "__main__":
    unittest.main()
