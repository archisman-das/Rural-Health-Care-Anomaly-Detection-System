"""Serve the full rural health anomaly dashboard in a browser."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rural_health_anomaly.cli import _infer_score_columns, _load_labels_from_args
from rural_health_anomaly.evaluation import build_evaluation_report
from rural_health_anomaly.training import load_tabular_data


@dataclass(frozen=True)
class DashboardPayload:
    html: str
    title: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the full anomaly dashboard in a browser.")
    parser.add_argument("--input", required=True, help="Path to scored predictions (.csv or .parquet).")
    parser.add_argument(
        "--score-column",
        default="anomaly_score",
        help="Primary anomaly score column to evaluate.",
    )
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=None,
        help="Optional list of score columns to compare instead of just one column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold used to convert anomaly scores into predicted labels.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Fraction of the highest and lowest scores used for unsupervised spread summaries.",
    )
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional CSV or Parquet file containing binary labels for evaluation.",
    )
    parser.add_argument(
        "--labels-column",
        default=None,
        help="Optional column name to read from --labels-file when it contains more than one column.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional column in the scored input containing binary labels for evaluation.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the local dashboard server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to serve the dashboard on.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the generated HTML dashboard before serving it.",
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the dashboard in the default browser after the server starts.",
    )
    return parser


def _build_dashboard_payload(args: argparse.Namespace) -> DashboardPayload:
    scored = load_tabular_data(args.input)
    labels = _load_labels_from_args(args, scored)

    score_columns = args.score_columns or _infer_score_columns(scored)
    if not score_columns:
        score_columns = [args.score_column]

    report = build_evaluation_report(
        scored,
        y_true=labels,
        score_columns=score_columns,
        threshold=args.threshold,
        top_fraction=args.top_fraction,
        executive_summary=False,
    )
    return DashboardPayload(html=report["summary_html"], title="Anomaly Evaluation Dashboard")


def _make_handler(html_payload: str) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/", "/index.html", "/dashboard", "/dashboard.html"}:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            body = html_payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return DashboardHandler


def serve_dashboard(payload: DashboardPayload, *, host: str, port: int, open_browser: bool) -> None:
    handler_cls = _make_handler(payload.html)
    server = ThreadingHTTPServer((host, port), handler_cls)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Serving {payload.title} at {url}")
    if open_browser:
        webbrowser.open(url)
        print("Opened dashboard in the default browser.")
    print("Press Ctrl+C to stop the server.")

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = _build_dashboard_payload(args)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload.html, encoding="utf-8")
        print(f"Dashboard HTML saved to {args.output}")

    serve_dashboard(payload, host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
