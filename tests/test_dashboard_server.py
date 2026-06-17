import http.client
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import threading
from unittest.mock import patch

import pandas as pd

import dashboard_server


class DashboardServerTests(unittest.TestCase):
    def test_dashboard_parser_includes_expected_flags(self):
        parser = dashboard_server.build_parser()
        option_strings = {
            option_string
            for action in parser._actions
            for option_string in action.option_strings
            if option_string != "-h"
        }

        self.assertIn("--input", option_strings)
        self.assertIn("--score-column", option_strings)
        self.assertIn("--score-columns", option_strings)
        self.assertIn("--threshold", option_strings)
        self.assertIn("--top-fraction", option_strings)
        self.assertIn("--labels-file", option_strings)
        self.assertIn("--labels-column", option_strings)
        self.assertIn("--label-column", option_strings)
        self.assertIn("--host", option_strings)
        self.assertIn("--port", option_strings)
        self.assertIn("--output", option_strings)
        self.assertIn("--open-browser", option_strings)
        self.assertIn("--no-open-browser", option_strings)

    def test_build_dashboard_payload_without_labels_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"

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

            args = Namespace(
                input=str(scores_path),
                score_column="anomaly_score",
                score_columns=None,
                threshold=0.5,
                top_fraction=0.25,
                labels_file=None,
                labels_column=None,
                label_column=None,
                host="127.0.0.1",
                port=8000,
                output=None,
                open_browser=False,
            )

            payload = dashboard_server._build_dashboard_payload(args)

            self.assertIn("Labels available: no", payload.html)
            self.assertIn("<h2>Model Comparison</h2>", payload.html)
            self.assertIn("<h2>Score Distribution</h2>", payload.html)
            self.assertIn("<h2>Model Agreement</h2>", payload.html)
            self.assertIn("<h2>Runtime Comparison</h2>", payload.html)

    def test_build_dashboard_payload_includes_full_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            labels_path = tmpdir_path / "labels.csv"

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

            args = Namespace(
                input=str(scores_path),
                score_column="anomaly_score",
                score_columns=None,
                threshold=0.5,
                top_fraction=0.25,
                labels_file=str(labels_path),
                labels_column="label",
                label_column=None,
                host="127.0.0.1",
                port=8000,
                output=None,
                open_browser=False,
            )

            payload = dashboard_server._build_dashboard_payload(args)

            self.assertIn("Anomaly Evaluation Report", payload.html)
            self.assertIn("<h2>Metrics</h2>", payload.html)
            self.assertIn("<h2>Score Distribution</h2>", payload.html)
            self.assertIn("<h2>Reconstruction Error Histograms</h2>", payload.html)
            self.assertIn("<h2>Model Agreement</h2>", payload.html)
            self.assertIn("<h2>Runtime Comparison</h2>", payload.html)
            self.assertIn("precision", payload.html)
            self.assertIn("recall", payload.html)
            self.assertIn("f1", payload.html)
            self.assertIn("roc_auc", payload.html)
            self.assertIn("auprc", payload.html)

    def test_main_writes_dashboard_html_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"
            output_path = tmpdir_path / "dashboard.html"

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

            argv = [
                "dashboard_server.py",
                "--input",
                str(scores_path),
                "--output",
                str(output_path),
                "--no-open-browser",
            ]

            with patch("sys.argv", argv):
                with patch("dashboard_server.serve_dashboard") as serve_mock:
                    dashboard_server.main()

            self.assertTrue(output_path.exists())
            html_text = output_path.read_text(encoding="utf-8")
            self.assertIn("Anomaly Evaluation Report", html_text)
            self.assertIn("<h2>Metrics</h2>", html_text)
            self.assertIn("<h2>Runtime Comparison</h2>", html_text)
            serve_mock.assert_called_once()

    def test_main_respects_open_browser_toggle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scores_path = tmpdir_path / "scores.csv"

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
                with patch(
                    "sys.argv",
                    [
                        "anomaly-dashboard",
                        "--input",
                        str(scores_path),
                        "--no-open-browser",
                    ],
                ):
                    dashboard_server.main()
                self.assertEqual(serve_mock.call_args.kwargs["open_browser"], False)

            with patch("dashboard_server.serve_dashboard") as serve_mock:
                with patch(
                    "sys.argv",
                    [
                        "anomaly-dashboard",
                        "--input",
                        str(scores_path),
                        "--open-browser",
                    ],
                ):
                    dashboard_server.main()
                self.assertEqual(serve_mock.call_args.kwargs["open_browser"], True)

    def test_serve_dashboard_returns_http_200_on_root(self):
        payload = dashboard_server.DashboardPayload(
            html="<html><body><h1>Dashboard</h1></body></html>",
            title="Test Dashboard",
        )
        server = dashboard_server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            dashboard_server._make_handler(payload.html),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = server.server_address[:2]
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertIn("<h1>Dashboard</h1>", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_serve_dashboard_returns_http_200_on_index_html(self):
        payload = dashboard_server.DashboardPayload(
            html="<html><body><h1>Dashboard</h1></body></html>",
            title="Test Dashboard",
        )
        server = dashboard_server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            dashboard_server._make_handler(payload.html),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = server.server_address[:2]
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/index.html")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertIn("<h1>Dashboard</h1>", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_serve_dashboard_returns_http_200_on_dashboard_html(self):
        payload = dashboard_server.DashboardPayload(
            html="<html><body><h1>Dashboard</h1></body></html>",
            title="Test Dashboard",
        )
        server = dashboard_server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            dashboard_server._make_handler(payload.html),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = server.server_address[:2]
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/dashboard.html")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertIn("<h1>Dashboard</h1>", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_serve_dashboard_returns_http_404_for_unknown_path(self):
        payload = dashboard_server.DashboardPayload(
            html="<html><body><h1>Dashboard</h1></body></html>",
            title="Test Dashboard",
        )
        server = dashboard_server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            dashboard_server._make_handler(payload.html),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = server.server_address[:2]
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/does-not-exist")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            connection.close()

            self.assertEqual(response.status, 404)
            self.assertIn("Not found", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
