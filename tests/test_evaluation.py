import unittest

import numpy as np
import pandas as pd

from rural_health_anomaly import compare_labeled_score_columns, evaluate_labeled_scores, evaluate_score_columns, summarize_anomaly_scores
from rural_health_anomaly.evaluation import build_evaluation_report, build_unsupervised_analysis


class EvaluationTests(unittest.TestCase):
    def test_evaluate_labeled_scores_returns_label_metrics(self):
        y_true = [0, 0, 1, 1]
        anomaly_scores = [0.1, 0.3, 0.8, 0.9]

        metrics = evaluate_labeled_scores(y_true, anomaly_scores, threshold=0.5)

        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("auprc", metrics)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)

    def test_compare_labeled_score_columns_returns_ranked_table(self):
        y_true = [0, 0, 1, 1]
        scores = pd.DataFrame(
            {
                "model_a": [0.2, 0.1, 0.9, 0.8],
                "model_b": [0.6, 0.6, 0.4, 0.4],
            }
        )

        comparison = compare_labeled_score_columns(y_true, scores, threshold=0.5)

        self.assertEqual(list(comparison["model"]), ["model_a", "model_b"])
        self.assertIn("precision", comparison.columns)
        self.assertIn("recall", comparison.columns)
        self.assertIn("f1", comparison.columns)
        self.assertIn("roc_auc", comparison.columns)
        self.assertIn("auprc", comparison.columns)
        self.assertGreater(comparison.loc[0, "f1"], comparison.loc[1, "f1"])

    def test_evaluate_labeled_scores_accepts_minus_one_plus_one_labels(self):
        y_true = [-1, -1, 1, 1]
        anomaly_scores = [0.9, 0.8, 0.2, 0.1]

        metrics = evaluate_labeled_scores(y_true, anomaly_scores, threshold=0.5)

        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_summarize_anomaly_scores_returns_distribution_metrics(self):
        scores = [0.05, 0.2, 0.4, 0.8, 0.95]

        summary = summarize_anomaly_scores(scores, threshold=0.5, top_fraction=0.2)

        self.assertIn("score_mean", summary)
        self.assertIn("score_std", summary)
        self.assertIn("score_p95", summary)
        self.assertIn("above_threshold_rate", summary)
        self.assertIn("score_spread", summary)
        self.assertAlmostEqual(summary["above_threshold_rate"], 0.4)
        self.assertGreater(summary["score_spread"], 0.0)

    def test_evaluate_score_columns_works_without_labels(self):
        scores = pd.DataFrame(
            {
                "isolation_forest_anomaly_score": [0.1, 0.2, 0.9, 0.95],
                "autoencoder_anomaly_score": [0.05, 0.3, 0.8, 0.7],
            }
        )

        comparison = evaluate_score_columns(scores, score_columns=list(scores.columns), top_fraction=0.25)

        self.assertEqual(list(comparison["model"]), list(scores.columns))
        self.assertIn("score_mean", comparison.columns)
        self.assertIn("score_p95", comparison.columns)
        self.assertIn("above_threshold_rate", comparison.columns)
        self.assertNotIn("precision", comparison.columns)
        self.assertNotIn("f1", comparison.columns)

    def test_evaluate_score_columns_combines_unsupervised_and_label_metrics(self):
        scores = pd.DataFrame(
            {
                "model_a": [0.2, 0.1, 0.9, 0.8],
                "model_b": [0.6, 0.6, 0.4, 0.4],
            }
        )

        comparison = evaluate_score_columns(scores, y_true=[0, 0, 1, 1], threshold=0.5)

        self.assertIn("score_mean", comparison.columns)
        self.assertIn("precision", comparison.columns)
        self.assertIn("recall", comparison.columns)
        self.assertIn("f1", comparison.columns)
        self.assertEqual(list(comparison["model"]), ["model_a", "model_b"])

    def test_build_unsupervised_analysis_tracks_histograms_and_agreement(self):
        scores = pd.DataFrame(
            {
                "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                "autoencoder_anomaly_score": [0.1, 0.2, 0.9, 0.95],
                "deep_svdd_anomaly_score": [0.05, 0.3, 0.8, 0.88],
                "autoencoder_reconstruction_error": [0.05, 0.1, 0.55, 0.6],
            }
        )

        analysis = build_unsupervised_analysis(scores, threshold=0.5, histogram_bins=4)

        self.assertIn("score_distributions", analysis)
        self.assertIn("reconstruction_error_histograms", analysis)
        self.assertIn("agreement", analysis)
        self.assertIn("high_disagreement_rows", analysis)
        self.assertGreaterEqual(len(analysis["score_distributions"]), 4)
        self.assertGreaterEqual(len(analysis["reconstruction_error_histograms"]), 1)
        self.assertIsNotNone(analysis["agreement"])
        self.assertIn("all_three_flag_rate", analysis["agreement"])
        self.assertIn("mean_pairwise_agreement_rate", analysis["agreement"])
        self.assertGreaterEqual(len(analysis["high_disagreement_rows"]), 1)
        self.assertIn("row_index", analysis["high_disagreement_rows"][0])
        self.assertIn("disagreement_count", analysis["high_disagreement_rows"][0])

    def test_build_evaluation_report_includes_unsupervised_sections_without_labels(self):
        scores = pd.DataFrame(
            {
                "anomaly_score": [0.1, 0.2, 0.8, 0.95],
                "isolation_forest_anomaly_score": [0.15, 0.25, 0.85, 0.9],
                "autoencoder_anomaly_score": [0.1, 0.2, 0.9, 0.95],
                "deep_svdd_anomaly_score": [0.05, 0.3, 0.8, 0.88],
                "autoencoder_reconstruction_error": [0.05, 0.1, 0.55, 0.6],
                "training_time_seconds": [4.2, 4.2, 4.2, 4.2],
                "training_time_ms": [4200.0, 4200.0, 4200.0, 4200.0],
                "model_size_bytes": [48_000_000, 48_000_000, 48_000_000, 48_000_000],
                "estimated_ram_usage_bytes": [120_000_000, 120_000_000, 120_000_000, 120_000_000],
                "inference_batch_latency_ms": [120.0, 120.0, 120.0, 120.0],
                "inference_latency_ms_per_patient": [60.0, 60.0, 60.0, 60.0],
                "inference_throughput_rows_per_second": [16.7, 16.7, 16.7, 16.7],
            }
        )

        report = build_evaluation_report(scores, threshold=0.5, top_fraction=0.25)

        self.assertIn("summary_markdown", report)
        self.assertIn("summary_html", report)
        self.assertIn("unsupervised_analysis", report)
        self.assertIn("## Score Distribution", report["summary_markdown"])
        self.assertIn("## Reconstruction Error Histograms", report["summary_markdown"])
        self.assertIn("## Model Agreement", report["summary_markdown"])
        self.assertIn("Agreement Matrix", report["summary_markdown"])
        self.assertIn("## Highest Disagreement Rows", report["summary_markdown"])
        self.assertIn("### Disagreement Notes", report["summary_markdown"])
        self.assertIn("## Model Comparison", report["summary_markdown"])
        self.assertIn("Ensemble", report["summary_markdown"])
        self.assertLess(
            report["summary_markdown"].index("## Model Comparison"),
            report["summary_markdown"].index("## Metrics"),
        )
        self.assertIn("## Runtime Comparison", report["summary_markdown"])
        self.assertIn("### Edge Readiness", report["summary_markdown"])
        self.assertIn("Status: ready", report["summary_markdown"])
        self.assertIn("Labels available: no", report["summary_markdown"])
        self.assertIn("Score Distribution", report["summary_html"])
        self.assertIn("Model Agreement", report["summary_html"])
        self.assertIn("Model Comparison", report["summary_html"])
        self.assertIn("Ensemble", report["summary_html"])
        self.assertIn("Agreement matrix", report["summary_html"])
        self.assertIn("Highest Disagreement Rows", report["summary_html"])
        self.assertIn("Disagreement Notes", report["summary_html"])
        self.assertIn("Runtime Comparison", report["summary_html"])
        self.assertIn("Edge readiness", report["summary_html"])

    def test_build_evaluation_report_includes_best_single_model_and_ensemble(self):
        scores = pd.DataFrame(
            {
                "anomaly_score": [0.2, 0.3, 0.85, 0.9],
                "isolation_forest_anomaly_score": [0.1, 0.2, 0.9, 0.95],
                "autoencoder_anomaly_score": [0.15, 0.25, 0.8, 0.92],
                "deep_svdd_anomaly_score": [0.05, 0.15, 0.88, 0.91],
                "autoencoder_reconstruction_error": [0.04, 0.06, 0.5, 0.57],
            }
        )

        report = build_evaluation_report(scores, y_true=[0, 0, 1, 1], threshold=0.5)

        self.assertIn("Best model:", report["summary_markdown"])
        self.assertIn("Ensemble score: anomaly_score", report["summary_markdown"])
        self.assertIn("## Model Comparison", report["summary_markdown"])

        model_comparison = report["model_comparison_table"]
        self.assertIn("Isolation Forest", list(model_comparison.columns))
        self.assertIn("Autoencoder", list(model_comparison.columns))
        self.assertIn("Deep SVDD", list(model_comparison.columns))
        self.assertIn("Ensemble", list(model_comparison.columns))
        self.assertIn("metric", model_comparison.reset_index().columns)

    def test_build_evaluation_report_executive_summary_is_comparison_only(self):
        scores = pd.DataFrame(
            {
                "anomaly_score": [0.2, 0.8],
                "isolation_forest_anomaly_score": [0.1, 0.9],
                "autoencoder_anomaly_score": [0.15, 0.85],
                "deep_svdd_anomaly_score": [0.05, 0.88],
            }
        )

        report = build_evaluation_report(scores, y_true=[0, 1], executive_summary=True)

        self.assertIn("## Model Comparison", report["summary_markdown"])
        self.assertIn("Best model:", report["summary_markdown"])
        self.assertIn("Ensemble score: anomaly_score", report["summary_markdown"])
        self.assertNotIn("## Metrics", report["summary_markdown"])
        self.assertNotIn("## Score Distribution", report["summary_markdown"])
        self.assertNotIn("## Runtime Comparison", report["summary_markdown"])
        self.assertIn("Anomaly Executive Summary", report["summary_html"])
        self.assertIn("Model Comparison", report["summary_html"])
        self.assertNotIn("Runtime Comparison", report["summary_html"])
        self.assertNotIn("Score Distribution", report["summary_html"])


if __name__ == "__main__":
    unittest.main()
