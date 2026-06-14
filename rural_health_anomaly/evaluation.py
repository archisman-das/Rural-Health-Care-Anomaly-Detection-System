"""Evaluation helpers for anomaly detection models."""

from __future__ import annotations

import html
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _coerce_binary_anomaly_labels(y_true: Any) -> np.ndarray:
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    unique_values = set(np.unique(labels).tolist())
    if unique_values.issubset({0, 1}):
        return labels.astype(int)
    if unique_values.issubset({-1, 1}):
        return np.where(labels == -1, 1, 0).astype(int)
    if unique_values.issubset({False, True}):
        return labels.astype(int)

    raise ValueError("Labels must be binary, using 0/1 or -1/1 conventions.")


def _safe_metric(metric_fn, y_true: np.ndarray, y_pred_or_score: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_pred_or_score))
    except ValueError:
        return float("nan")


def summarize_anomaly_scores(
    anomaly_scores: Iterable[float],
    *,
    threshold: float = 0.5,
    top_fraction: float = 0.1,
) -> dict[str, float]:
    """Summarize anomaly score distributions when labels are unavailable."""

    scores = np.asarray(list(anomaly_scores), dtype=float)
    if scores.size == 0:
        raise ValueError("anomaly_scores must not be empty.")

    quantiles = np.percentile(scores, [25, 50, 75, 95, 99])
    fraction = float(np.clip(top_fraction, 0.01, 0.5))
    top_count = max(1, int(np.ceil(scores.size * fraction)))
    bottom_count = max(1, int(np.ceil(scores.size * fraction)))
    ordered_scores = np.sort(scores)
    top_slice = ordered_scores[-top_count:]
    bottom_slice = ordered_scores[:bottom_count]

    return {
        "count": float(scores.size),
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(quantiles[1]),
        "score_std": float(np.std(scores, ddof=0)),
        "score_iqr": float(quantiles[2] - quantiles[0]),
        "score_p95": float(quantiles[3]),
        "score_p99": float(quantiles[4]),
        "above_threshold_rate": float(np.mean(scores >= threshold)),
        "top_fraction_mean": float(np.mean(top_slice)),
        "bottom_fraction_mean": float(np.mean(bottom_slice)),
        "score_spread": float(np.mean(top_slice) - np.mean(bottom_slice)),
    }


def evaluate_labeled_scores(
    y_true: Any,
    anomaly_scores: Iterable[float],
    *,
    threshold: float = 0.5,
    y_pred: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute label-aware anomaly metrics from scores and optional predictions."""

    y_true_binary = _coerce_binary_anomaly_labels(y_true)
    scores = np.asarray(list(anomaly_scores), dtype=float)
    if scores.shape[0] != y_true_binary.shape[0]:
        raise ValueError("y_true and anomaly_scores must have the same length.")

    if y_pred is None:
        y_pred_binary = (scores >= threshold).astype(int)
    else:
        predictions = np.asarray(list(y_pred))
        if predictions.shape[0] != y_true_binary.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        unique_values = set(np.unique(predictions).tolist())
        if unique_values.issubset({0, 1}):
            y_pred_binary = predictions.astype(int)
        elif unique_values.issubset({-1, 1}):
            y_pred_binary = np.where(predictions == -1, 1, 0).astype(int)
        else:
            raise ValueError("Predictions must be binary, using 0/1 or -1/1 conventions.")

    metrics = {
        "precision": _safe_metric(lambda y, p: precision_score(y, p, zero_division=0), y_true_binary, y_pred_binary),
        "recall": _safe_metric(lambda y, p: recall_score(y, p, zero_division=0), y_true_binary, y_pred_binary),
        "f1": _safe_metric(lambda y, p: f1_score(y, p, zero_division=0), y_true_binary, y_pred_binary),
        "accuracy": _safe_metric(lambda y, p: float(np.mean(np.asarray(y) == np.asarray(p))), y_true_binary, y_pred_binary),
        "roc_auc": float("nan"),
        "auprc": float("nan"),
    }

    if len(np.unique(y_true_binary)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, scores))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(y_true_binary, scores))
        except ValueError:
            metrics["auprc"] = float("nan")

    return metrics


def compare_labeled_score_columns(
    y_true: Any,
    scores: pd.DataFrame,
    *,
    score_columns: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute label-aware metrics for multiple score columns."""

    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame.")

    columns = score_columns or list(scores.columns)
    rows: list[dict[str, float | str]] = []
    for column in columns:
        if column not in scores.columns:
            raise ValueError(f"Score column '{column}' was not found.")
        metrics = evaluate_labeled_scores(y_true, scores[column], threshold=threshold)
        rows.append({"model": column, **metrics})

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["f1", "auprc", "roc_auc"], ascending=False).reset_index(drop=True)
    return result


def evaluate_score_columns(
    scores: pd.DataFrame,
    *,
    y_true: Any | None = None,
    score_columns: list[str] | None = None,
    threshold: float = 0.5,
    top_fraction: float = 0.1,
) -> pd.DataFrame:
    """Summarize one or more score columns, optionally with label-aware metrics."""

    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame.")

    columns = score_columns or list(scores.columns)
    rows: list[dict[str, float | str]] = []
    for column in columns:
        if column not in scores.columns:
            raise ValueError(f"Score column '{column}' was not found.")

        metrics = summarize_anomaly_scores(scores[column], threshold=threshold, top_fraction=top_fraction)
        row: dict[str, float | str] = {"model": column, **metrics}
        if y_true is not None:
            row.update(evaluate_labeled_scores(y_true, scores[column], threshold=threshold))
        rows.append(row)

    result = pd.DataFrame(rows)
    if y_true is not None and not result.empty:
        result = result.sort_values(["f1", "auprc", "roc_auc"], ascending=False).reset_index(drop=True)
    return result


def _histogram_summary(values: Iterable[float], *, bins: int = 10) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0.0, "counts": [], "bin_edges": []}

    bin_count = max(1, min(int(bins), int(array.size)))
    counts, bin_edges = np.histogram(array, bins=bin_count)
    return {
        "count": float(array.size),
        "counts": counts.astype(int).tolist(),
        "bin_edges": [float(edge) for edge in bin_edges.tolist()],
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _first_non_null(values: pd.Series) -> Any:
    non_null = values.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def summarize_runtime_metrics(scores: pd.DataFrame) -> dict[str, Any]:
    """Summarize runtime and footprint columns from a scored dataframe."""

    runtime_columns = [
        "training_time_seconds",
        "training_time_ms",
        "model_size_bytes",
        "estimated_ram_usage_bytes",
        "inference_batch_latency_ms",
        "inference_latency_ms_per_patient",
        "inference_throughput_rows_per_second",
    ]
    summary: dict[str, Any] = {}
    for column in runtime_columns:
        if column not in scores.columns:
            continue
        value = _first_non_null(scores[column])
        if value is None:
            continue
        summary[column] = _json_safe_value(value)
    if "inference_latency_ms_per_patient" in summary:
        summary["critical_for_edge_deployment"] = True
        latency_ms = float(summary["inference_latency_ms_per_patient"])
        model_size_bytes = float(summary.get("model_size_bytes", float("nan")))
        ram_usage_bytes = float(summary.get("estimated_ram_usage_bytes", float("nan")))
        latency_ok = latency_ms <= 100.0
        model_size_ok = np.isnan(model_size_bytes) or model_size_bytes <= 100 * 1024 * 1024
        ram_ok = np.isnan(ram_usage_bytes) or ram_usage_bytes <= 512 * 1024 * 1024
        readiness_checks = {
            "latency_ok": latency_ok,
            "model_size_ok": model_size_ok,
            "ram_ok": ram_ok,
        }
        summary["edge_readiness_checks"] = readiness_checks
        if all(readiness_checks.values()):
            summary["edge_readiness_status"] = "ready"
        elif any(readiness_checks.values()):
            summary["edge_readiness_status"] = "needs_optimization"
        else:
            summary["edge_readiness_status"] = "not_ready"
    return summary


def _build_high_disagreement_rows(
    scores: pd.DataFrame,
    agreement_columns: list[str],
    *,
    threshold: float,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if len(agreement_columns) < 2:
        return []

    flag_frame = scores[agreement_columns].ge(threshold)
    pairwise_columns = list(combinations(agreement_columns, 2))
    if not pairwise_columns:
        return []

    disagreement_frame = pd.DataFrame(
        {
            f"{left}__{right}": (flag_frame[left] != flag_frame[right]).astype(int)
            for left, right in pairwise_columns
        },
        index=scores.index,
    )
    disagreement_count = disagreement_frame.sum(axis=1)
    score_range = scores[agreement_columns].max(axis=1) - scores[agreement_columns].min(axis=1)
    flag_pattern = flag_frame.astype(int).astype(str).agg("".join, axis=1)

    preview_columns: list[str] = []
    for column in ("patient_id", "recorded_at", "sample_id", "record_id"):
        if column in scores.columns and column not in preview_columns:
            preview_columns.append(column)
    for column in agreement_columns:
        if column not in preview_columns:
            preview_columns.append(column)
    for column in scores.columns:
        if "reconstruction_error" in column and column not in preview_columns:
            preview_columns.append(column)
    if "anomaly_score" in scores.columns and "anomaly_score" not in preview_columns:
        preview_columns.append("anomaly_score")

    ranked = (
        pd.DataFrame(
            {
                "row_index": scores.index,
                "disagreement_count": disagreement_count,
                "score_range": score_range,
                "flag_pattern": flag_pattern,
            },
            index=scores.index,
        )
        .sort_values(["disagreement_count", "score_range"], ascending=False)
        .head(max(1, int(top_n)))
    )

    rows: list[dict[str, Any]] = []
    for idx, meta in ranked.iterrows():
        row = {"row_index": int(idx)}
        for column in preview_columns:
            if column in scores.columns:
                row[column] = _json_safe_value(scores.at[idx, column])
        row["disagreement_count"] = int(meta["disagreement_count"])
        row["score_range"] = float(meta["score_range"])
        row["flag_pattern"] = str(meta["flag_pattern"])

        row_pairwise_counts = {
            f"{left}__{right}": int(disagreement_frame.at[idx, f"{left}__{right}"])
            for left, right in pairwise_columns
        }
        if row_pairwise_counts:
            max_count = max(row_pairwise_counts.values())
            worst_pairs = [pair for pair, count in row_pairwise_counts.items() if count == max_count]
            row["worst_disagreement_pairs"] = worst_pairs
            row["worst_disagreement_pair"] = worst_pairs[0]
            row["worst_disagreement_pair_count"] = int(max_count)
            row["pairwise_disagreement_counts"] = row_pairwise_counts
        rows.append(row)

    return rows


def build_unsupervised_analysis(
    scores: pd.DataFrame,
    *,
    score_columns: list[str] | None = None,
    threshold: float = 0.5,
    histogram_bins: int = 10,
    top_disagreement_rows: int = 5,
) -> dict[str, Any]:
    """Build label-free analysis summaries for score distributions and agreement."""

    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame.")

    score_columns = score_columns or [column for column in scores.columns if column.endswith("_anomaly_score") or column == "anomaly_score"]
    score_columns = [column for column in score_columns if column in scores.columns]

    distribution_columns = [
        column
        for column in score_columns
        if column == "anomaly_score" or column.endswith("_anomaly_score")
    ]
    score_distributions = [
        {"column": column, **summarize_anomaly_scores(scores[column], threshold=threshold)}
        for column in distribution_columns
    ]

    reconstruction_error_histograms: list[dict[str, Any]] = []
    for column in scores.columns:
        if "reconstruction_error" in column:
            reconstruction_error_histograms.append({"column": column, **_histogram_summary(scores[column], bins=histogram_bins)})

    agreement_columns = [
        column
        for column in (
            "isolation_forest_anomaly_score",
            "one_class_svm_anomaly_score",
            "local_outlier_factor_anomaly_score",
            "autoencoder_anomaly_score",
            "anomaly_transformer_anomaly_score",
            "variational_autoencoder_anomaly_score",
            "ganomaly_anomaly_score",
            "cnn_autoencoder_anomaly_score",
            "deep_svdd_anomaly_score",
        )
        if column in scores.columns
    ]
    agreement: dict[str, Any] | None = None
    high_disagreement_rows: list[dict[str, Any]] = []
    if len(agreement_columns) >= 2:
        flag_frame = scores[agreement_columns].ge(threshold)
        component_count = len(agreement_columns)
        pairwise_rates = {
            f"{left}__{right}": float((flag_frame[left] == flag_frame[right]).mean())
            for left, right in combinations(agreement_columns, 2)
        }
        disagreement_rates = {
            f"{left}__{right}": float((flag_frame[left] != flag_frame[right]).mean())
            for left, right in combinations(agreement_columns, 2)
        }
        matrix = pd.DataFrame(np.eye(len(agreement_columns), dtype=float), index=agreement_columns, columns=agreement_columns)
        for left, right in combinations(agreement_columns, 2):
            rate = pairwise_rates[f"{left}__{right}"]
            matrix.loc[left, right] = rate
            matrix.loc[right, left] = rate
        agreement = {
            "columns": agreement_columns,
            "threshold": float(threshold),
            "pairwise_agreement_rates": pairwise_rates,
            "pairwise_disagreement_rates": disagreement_rates,
            "agreement_matrix": {
                "index": agreement_columns,
                "columns": agreement_columns,
                "data": [[float(matrix.loc[row, col]) for col in agreement_columns] for row in agreement_columns],
            },
            "mean_pairwise_agreement_rate": float(np.mean(list(pairwise_rates.values()))) if pairwise_rates else float("nan"),
            "at_least_two_flag_rate": float((flag_frame.sum(axis=1) >= 2).mean()) if len(agreement_columns) >= 2 else float("nan"),
            "component_count": component_count,
            "all_models_flag_rate": float((flag_frame.sum(axis=1) == component_count).mean()),
            "all_models_flag_count": int((flag_frame.sum(axis=1) == component_count).sum()),
        }
        agreement["all_three_flag_rate"] = agreement["all_models_flag_rate"]
        agreement["all_three_flag_count"] = agreement["all_models_flag_count"]
        high_disagreement_rows = _build_high_disagreement_rows(
            scores,
            agreement_columns,
            threshold=threshold,
            top_n=top_disagreement_rows,
        )

    return {
        "score_distributions": score_distributions,
        "reconstruction_error_histograms": reconstruction_error_histograms,
        "agreement": agreement,
        "high_disagreement_rows": high_disagreement_rows,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without depending on tabulate."""

    if df.empty:
        return "| |\n| --- |\n| (no rows) |"

    frame = df.copy()
    columns = list(frame.columns)
    rows = [columns]
    rows.extend(frame.astype(str).itertuples(index=False, name=None))

    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(columns))]

    def _format_row(values: tuple[Any, ...] | list[Any]) -> str:
        cells = [str(value) for value in values]
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)) + " |"

    header = _format_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [_format_row(row) for row in frame.astype(str).itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def _html_report_table(df: pd.DataFrame) -> str:
    """Render the evaluation table as HTML."""

    if df.empty:
        return "<p>(no rows)</p>"

    return df.to_html(index=False, escape=True, border=0, classes="evaluation-report-table")


def _friendly_model_name(model_name: str) -> str:
    mapping = {
        "isolation_forest_anomaly_score": "Isolation Forest",
        "one_class_svm_anomaly_score": "One-Class SVM",
        "local_outlier_factor_anomaly_score": "Local Outlier Factor",
        "autoencoder_anomaly_score": "Autoencoder",
        "anomaly_transformer_anomaly_score": "Anomaly Transformer",
        "variational_autoencoder_anomaly_score": "Variational Autoencoder",
        "ganomaly_anomaly_score": "GANomaly",
        "cnn_autoencoder_anomaly_score": "CNN Autoencoder",
        "deep_svdd_anomaly_score": "Deep SVDD",
        "anomaly_score": "Ensemble",
    }
    return mapping.get(model_name, model_name.replace("_", " ").title())


def _model_comparison_matrix(comparison: pd.DataFrame, *, model_order: list[str]) -> pd.DataFrame:
    """Pivot the per-model comparison table into a side-by-side metric matrix."""

    if comparison.empty:
        return pd.DataFrame()

    frame = comparison.copy()
    frame = frame[frame["model"].isin(model_order)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["model"] = pd.Categorical(frame["model"], categories=model_order, ordered=True)
    frame = frame.sort_values("model").reset_index(drop=True)

    metric_columns = [
        column
        for column in [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "auprc",
            "score_mean",
            "score_std",
            "score_p95",
            "above_threshold_rate",
            "score_spread",
        ]
        if column in frame.columns
    ]
    if not metric_columns:
        return pd.DataFrame()

    matrix = frame.set_index("model")[metric_columns].T
    matrix.index.name = "metric"
    matrix.columns = [_friendly_model_name(str(column)) for column in matrix.columns]
    return matrix


def build_evaluation_report(
    scores: pd.DataFrame,
    *,
    y_true: Any | None = None,
    score_columns: list[str] | None = None,
    threshold: float = 0.5,
    top_fraction: float = 0.1,
    executive_summary: bool = False,
) -> dict[str, Any]:
    """Build a compact evaluation report for CLI export."""

    unsupervised = build_unsupervised_analysis(
        scores,
        score_columns=score_columns,
        threshold=threshold,
    )
    runtime = summarize_runtime_metrics(scores)
    comparison = evaluate_score_columns(
        scores,
        y_true=y_true,
        score_columns=score_columns,
        threshold=threshold,
        top_fraction=top_fraction,
    )
    has_labels = y_true is not None
    best_model = None
    if has_labels and not comparison.empty and "f1" in comparison.columns:
        component_rows = comparison[comparison["model"] != "anomaly_score"]
        if not component_rows.empty:
            best_model = component_rows.iloc[0]["model"]
    model_comparison = _model_comparison_matrix(
        comparison,
        model_order=[
            "isolation_forest_anomaly_score",
            "one_class_svm_anomaly_score",
            "local_outlier_factor_anomaly_score",
            "autoencoder_anomaly_score",
            "anomaly_transformer_anomaly_score",
            "variational_autoencoder_anomaly_score",
            "ganomaly_anomaly_score",
            "cnn_autoencoder_anomaly_score",
            "deep_svdd_anomaly_score",
            "anomaly_score",
        ],
    )

    summary_lines = [
        "# Anomaly Evaluation Report",
        "",
    ]

    if not model_comparison.empty:
        summary_lines.extend(
            [
                "## Model Comparison",
                "",
                _markdown_table(model_comparison.reset_index()),
                "",
            ]
        )

    summary_lines.extend(
        [
            f"- Labels available: {'yes' if has_labels else 'no'}",
            f"- Threshold: {threshold}",
            f"- Top fraction: {top_fraction}",
            f"- Score columns: {', '.join(score_columns or list(scores.columns))}",
        ]
    )
    if best_model is not None:
        summary_lines.append(f"- Best model: {best_model}")
    if "anomaly_score" in scores.columns:
        summary_lines.append("- Ensemble score: anomaly_score")

    if executive_summary:
        return {
            "summary_markdown": "\n".join(summary_lines).rstrip() + "\n",
            "summary_html": "\n".join(
                [
                    "<!doctype html>",
                    "<html lang=\"en\">",
                    "<head>",
                    "  <meta charset=\"utf-8\" />",
                    "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
                    "  <title>Anomaly Executive Summary</title>",
                    "  <style>",
                    "    body { font-family: Arial, sans-serif; line-height: 1.5; margin: 2rem; color: #1f2937; }",
                    "    h1, h2 { line-height: 1.2; }",
                    "    .meta { list-style: none; padding: 0; }",
                    "    .meta li { margin: 0.25rem 0; }",
                    "    .evaluation-report-table { border-collapse: collapse; width: 100%; margin-top: 1rem; }",
                    "    .evaluation-report-table th, .evaluation-report-table td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }",
                    "    .evaluation-report-table thead th { background: #f9fafb; }",
                    "    .card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1rem 1.25rem; margin: 1rem 0; }",
                    "  </style>",
                    "</head>",
                    "<body>",
                    "  <h1>Anomaly Executive Summary</h1>",
                    "  <ul class=\"meta\">",
                    f"    <li>Labels available: {html.escape('yes' if has_labels else 'no')}</li>",
                    f"    <li>Threshold: {html.escape(str(threshold))}</li>",
                    f"    <li>Top fraction: {html.escape(str(top_fraction))}</li>",
                    f"    <li>Score columns: {html.escape(', '.join(score_columns or list(scores.columns)))}</li>",
                    "  </ul>",
                    *(
                        [f"  <div class=\"card\"><strong>Best model:</strong> {html.escape(str(best_model))}</div>"]
                        if best_model is not None
                        else []
                    ),
                    *(
                        [f"  <div class=\"card\"><strong>Ensemble score:</strong> anomaly_score</div>"]
                        if "anomaly_score" in scores.columns
                        else []
                    ),
                    "  <h2>Model Comparison</h2>",
                    f"  {_html_report_table(model_comparison.reset_index()) if not model_comparison.empty else '<p>(no rows)</p>'}",
                    "</body>",
                    "</html>",
                    "",
                ]
            ),
            "metrics_table": comparison,
            "metrics_records": comparison.to_dict(orient="records"),
            "model_comparison_table": model_comparison,
            "model_comparison_records": model_comparison.reset_index().to_dict(orient="records") if not model_comparison.empty else [],
            "unsupervised_analysis": unsupervised,
            "runtime_metrics": runtime,
        }

    summary_lines.extend(
        [
            "## Metrics",
            "",
            _markdown_table(comparison.reset_index(drop=True)),
            "",
        ]
    )

    if not comparison.empty and "score_spread" in comparison.columns:
        top_row = comparison.iloc[0]
        summary_lines.extend(
            [
                "## Quick Read",
                "",
                f"- Highest-ranked score stream: {top_row['model']}",
                f"- Mean score: {top_row['score_mean']:.4f}",
                f"- 95th percentile score: {top_row['score_p95']:.4f}",
                f"- Score spread: {top_row['score_spread']:.4f}",
            ]
        )
        if has_labels and "f1" in comparison.columns:
            summary_lines.append(f"- F1-score: {top_row['f1']:.4f}")

    if unsupervised["score_distributions"]:
        summary_lines.extend(["", "## Score Distribution"])
        for item in unsupervised["score_distributions"]:
            summary_lines.append(
                f"- {item['column']}: mean={item['score_mean']:.4f}, std={item['score_std']:.4f}, "
                f"p95={item['score_p95']:.4f}, above-threshold={item['above_threshold_rate']:.4f}"
            )

    if unsupervised["reconstruction_error_histograms"]:
        summary_lines.extend(["", "## Reconstruction Error Histograms"])
        for item in unsupervised["reconstruction_error_histograms"]:
            summary_lines.append(
                f"- {item['column']}: counts={item['counts']}, bin_edges={item['bin_edges']}"
            )

    if unsupervised["agreement"]:
        agreement = unsupervised["agreement"]
        component_count = int(agreement.get("component_count", len(agreement["columns"])))
        summary_lines.extend(
            [
                "",
                "## Model Agreement",
                "",
                f"- Compared models: {', '.join(agreement['columns'])}",
                f"- Pairwise agreement rate: {agreement['mean_pairwise_agreement_rate']:.4f}",
                f"- At least two models flag rate: {agreement['at_least_two_flag_rate']:.4f}",
                f"- All {component_count} models flag rate: {agreement['all_models_flag_rate']:.4f}",
                f"- All {component_count} models flag count: {agreement['all_models_flag_count']}",
                "",
                "### Agreement Matrix",
                "",
            ]
        )
        matrix = pd.DataFrame(
            agreement["agreement_matrix"]["data"],
            index=agreement["agreement_matrix"]["index"],
            columns=agreement["agreement_matrix"]["columns"],
        )
        summary_lines.append(_markdown_table(matrix.reset_index().rename(columns={"index": "model"})))

    if unsupervised["high_disagreement_rows"]:
        summary_lines.extend(["", "## Highest Disagreement Rows", ""])
        summary_lines.append(_markdown_table(pd.DataFrame(unsupervised["high_disagreement_rows"])))
        summary_lines.extend(
            [
                "",
                "### Disagreement Notes",
                "",
            ]
        )

    if runtime:
        summary_lines.extend(
            [
                "",
                "## Runtime Comparison",
                "",
                f"- Inference latency (ms per patient): {runtime.get('inference_latency_ms_per_patient', 'n/a')}",
                f"- Batch latency (ms): {runtime.get('inference_batch_latency_ms', 'n/a')}",
                f"- Training time (seconds): {runtime.get('training_time_seconds', 'n/a')}",
                f"- Model size (bytes): {runtime.get('model_size_bytes', 'n/a')}",
                f"- Estimated RAM usage (bytes): {runtime.get('estimated_ram_usage_bytes', 'n/a')}",
                f"- Throughput (rows/sec): {runtime.get('inference_throughput_rows_per_second', 'n/a')}",
            ]
        )
        if runtime.get("edge_readiness_status"):
            summary_lines.extend(
                [
                    "",
                    "### Edge Readiness",
                    "",
                    f"- Status: {runtime['edge_readiness_status']}",
                    f"- Latency check: {'pass' if runtime['edge_readiness_checks']['latency_ok'] else 'fail'}",
                    f"- Model size check: {'pass' if runtime['edge_readiness_checks']['model_size_ok'] else 'fail'}",
                    f"- RAM check: {'pass' if runtime['edge_readiness_checks']['ram_ok'] else 'fail'}",
                ]
            )
        for item in unsupervised["high_disagreement_rows"]:
            summary_lines.append(
                f"- Row {item['row_index']}: worst pair {item.get('worst_disagreement_pair', 'n/a')} "
                f"with {item.get('worst_disagreement_pair_count', 0)} pairwise disagreements"
            )

    return {
        "summary_markdown": "\n".join(summary_lines).rstrip() + "\n",
        "summary_html": "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\">",
                "<head>",
                "  <meta charset=\"utf-8\" />",
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
                "  <title>Anomaly Evaluation Report</title>",
                "  <style>",
                "    body { font-family: Arial, sans-serif; line-height: 1.5; margin: 2rem; color: #1f2937; }",
                "    h1, h2 { line-height: 1.2; }",
                "    .meta { list-style: none; padding: 0; }",
                "    .meta li { margin: 0.25rem 0; }",
                "    .evaluation-report-table { border-collapse: collapse; width: 100%; margin-top: 1rem; }",
                "    .evaluation-report-table th, .evaluation-report-table td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }",
                "    .evaluation-report-table thead th { background: #f9fafb; }",
                "    .card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1rem 1.25rem; margin: 1rem 0; }",
                "  </style>",
                "</head>",
                "<body>",
                "  <h1>Anomaly Evaluation Report</h1>",
                *(
                    [
                        "  <h2>Model Comparison</h2>",
                        "  <div class=\"card\">",
                        _html_report_table(model_comparison.reset_index()) if not model_comparison.empty else "<p>(no rows)</p>",
                        "  </div>",
                        *(
                            [f"  <div class=\"card\"><strong>Best model:</strong> {html.escape(str(best_model))}</div>"]
                            if best_model is not None
                            else []
                        ),
                        *(
                            [f"  <div class=\"card\"><strong>Ensemble score:</strong> anomaly_score</div>"]
                            if "anomaly_score" in scores.columns
                            else []
                        ),
                    ]
                    if not model_comparison.empty
                    else []
                ),
                "  <ul class=\"meta\">",
        f"    <li>Labels available: {html.escape('yes' if has_labels else 'no')}</li>",
        f"    <li>Threshold: {html.escape(str(threshold))}</li>",
        f"    <li>Top fraction: {html.escape(str(top_fraction))}</li>",
        f"    <li>Score columns: {html.escape(', '.join(score_columns or list(scores.columns)))}</li>",
                "  </ul>",
                *(
                    [f"  <div class=\"card\"><strong>Best model:</strong> {html.escape(str(best_model))}</div>"]
                    if best_model is not None
                    else []
                ),
                *(
                    [
                        "  <h2>Score Distribution</h2>",
                        "  <div class=\"card\">",
                        *[
                            (
                                f"    <div><strong>{html.escape(str(item['column']))}:</strong> "
                                f"mean={item['score_mean']:.4f}, std={item['score_std']:.4f}, "
                                f"p95={item['score_p95']:.4f}, above-threshold={item['above_threshold_rate']:.4f}</div>"
                            )
                            for item in unsupervised["score_distributions"]
                        ],
                        "  </div>",
                    ]
                    if unsupervised["score_distributions"]
                    else []
                ),
                *(
                    [
                        "  <h2>Reconstruction Error Histograms</h2>",
                        "  <div class=\"card\">",
                        *[
                            (
                                f"    <div><strong>{html.escape(str(item['column']))}:</strong> "
                                f"counts={html.escape(str(item['counts']))}, "
                                f"bin_edges={html.escape(str(item['bin_edges']))}</div>"
                            )
                            for item in unsupervised["reconstruction_error_histograms"]
                        ],
                        "  </div>",
                    ]
                    if unsupervised["reconstruction_error_histograms"]
                    else []
                ),
                *(
                    [
                        "  <h2>Model Agreement</h2>",
                        "  <div class=\"card\">",
                        f"    <div><strong>Compared models:</strong> {html.escape(', '.join(unsupervised['agreement']['columns']))}</div>",
                        f"    <div><strong>Pairwise agreement rate:</strong> {unsupervised['agreement']['mean_pairwise_agreement_rate']:.4f}</div>",
                        f"    <div><strong>At least two models flag rate:</strong> {unsupervised['agreement']['at_least_two_flag_rate']:.4f}</div>",
                        f"    <div><strong>All {int(unsupervised['agreement'].get('component_count', len(unsupervised['agreement']['columns'])))} models flag rate:</strong> {unsupervised['agreement']['all_models_flag_rate']:.4f}</div>",
                        f"    <div><strong>All {int(unsupervised['agreement'].get('component_count', len(unsupervised['agreement']['columns'])))} models flag count:</strong> {unsupervised['agreement']['all_models_flag_count']}</div>",
                        "    <div><strong>Agreement matrix:</strong></div>",
                        "    <div>",
                        _html_report_table(
                            pd.DataFrame(
                                unsupervised["agreement"]["agreement_matrix"]["data"],
                                index=unsupervised["agreement"]["agreement_matrix"]["index"],
                                columns=unsupervised["agreement"]["agreement_matrix"]["columns"],
                            ).reset_index().rename(columns={"index": "model"})
                        ),
                        "    </div>",
                        "  </div>",
                    ]
                    if unsupervised["agreement"]
                    else []
                ),
                *(
                    [
                        "  <h2>Highest Disagreement Rows</h2>",
                        "  <div class=\"card\">",
                        _html_report_table(pd.DataFrame(unsupervised["high_disagreement_rows"])),
                        "  </div>",
                        "  <h3>Disagreement Notes</h3>",
                        "  <ul>",
                        *[
                            (
                                "    <li>"
                                f"Row {item['row_index']}: worst pair {html.escape(str(item.get('worst_disagreement_pair', 'n/a')))} "
                                f"with {item.get('worst_disagreement_pair_count', 0)} pairwise disagreements"
                                "</li>"
                            )
                            for item in unsupervised["high_disagreement_rows"]
                        ],
                        "  </ul>",
                    ]
                    if unsupervised["high_disagreement_rows"]
                    else []
                ),
                *(
                    [
                        "  <h2>Runtime Comparison</h2>",
                        "  <div class=\"card\">",
                        f"    <div><strong>Inference latency (ms per patient):</strong> {runtime.get('inference_latency_ms_per_patient', 'n/a')}</div>",
                        f"    <div><strong>Batch latency (ms):</strong> {runtime.get('inference_batch_latency_ms', 'n/a')}</div>",
                        f"    <div><strong>Training time (seconds):</strong> {runtime.get('training_time_seconds', 'n/a')}</div>",
                        f"    <div><strong>Model size (bytes):</strong> {runtime.get('model_size_bytes', 'n/a')}</div>",
                        f"    <div><strong>Estimated RAM usage (bytes):</strong> {runtime.get('estimated_ram_usage_bytes', 'n/a')}</div>",
                        f"    <div><strong>Throughput (rows/sec):</strong> {runtime.get('inference_throughput_rows_per_second', 'n/a')}</div>",
                        *(
                            [
                                f"    <div><strong>Edge readiness:</strong> {html.escape(str(runtime['edge_readiness_status']))}</div>",
                                f"    <div><strong>Latency check:</strong> {'pass' if runtime['edge_readiness_checks']['latency_ok'] else 'fail'}</div>",
                                f"    <div><strong>Model size check:</strong> {'pass' if runtime['edge_readiness_checks']['model_size_ok'] else 'fail'}</div>",
                                f"    <div><strong>RAM check:</strong> {'pass' if runtime['edge_readiness_checks']['ram_ok'] else 'fail'}</div>",
                            ]
                            if runtime.get("edge_readiness_status")
                            else []
                        ),
                        "  </div>",
                    ]
                    if runtime
                    else []
                ),
                *(
                    [
                        "  <h2>Quick Read</h2>",
                        "  <div class=\"card\">",
                        f"    <div><strong>Highest-ranked score stream:</strong> {html.escape(str(top_row['model']))}</div>",
                        f"    <div><strong>Mean score:</strong> {top_row['score_mean']:.4f}</div>",
                        f"    <div><strong>95th percentile score:</strong> {top_row['score_p95']:.4f}</div>",
                        f"    <div><strong>Score spread:</strong> {top_row['score_spread']:.4f}</div>",
                *(
                    [f"    <div><strong>F1-score:</strong> {top_row['f1']:.4f}</div>"]
                    if has_labels and "f1" in comparison.columns
                    else []
                ),
                "  </div>",
            ]
            if not comparison.empty and "score_spread" in comparison.columns
            else []
        ),
                "  <h2>Metrics</h2>",
                f"  {_html_report_table(comparison.reset_index(drop=True))}",
                "</body>",
                "</html>",
                "",
            ]
        ),
        "metrics_table": comparison,
        "metrics_records": comparison.to_dict(orient="records"),
        "model_comparison_table": model_comparison,
        "model_comparison_records": model_comparison.reset_index().to_dict(orient="records") if not model_comparison.empty else [],
        "unsupervised_analysis": unsupervised,
        "runtime_metrics": runtime,
    }
