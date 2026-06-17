"""Interactive Streamlit dashboard for scored anomaly outputs."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation import build_unsupervised_analysis
from .training import load_tabular_data

_RISK_BINS = [-float("inf"), 0.4, 0.7, float("inf")]
_RISK_LABELS = ["Low", "Medium", "High"]


def load_dashboard_frame(source: Any) -> pd.DataFrame:
    """Load a dashboard input from a path, dataframe, or uploaded file."""

    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return load_tabular_data(source)
    if hasattr(source, "name") and hasattr(source, "getvalue"):
        suffix = Path(str(source.name)).suffix.lower()
        payload = io.BytesIO(source.getvalue())
        if suffix == ".csv":
            return pd.read_csv(payload, low_memory=False)
        if suffix == ".parquet":
            return pd.read_parquet(payload)
        if suffix == ".json":
            return pd.read_json(payload)
        raise ValueError("Unsupported upload format. Use CSV, Parquet, or JSON.")
    raise TypeError("Unsupported dashboard input type.")


def _coerce_blob(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in {"none", "null", "nan"}:
            return None
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(text)
                except Exception:
                    continue
        return text
    return value


def parse_feature_explanations(blob: Any) -> pd.DataFrame:
    """Normalize a nested explanation payload into a table."""

    blob = _coerce_blob(blob)
    if blob is None:
        return pd.DataFrame(columns=["feature", "shap_value", "absolute_shap_value", "source_columns", "feature_type", "method"])

    if isinstance(blob, dict) and "feature_explanations" in blob:
        blob = blob["feature_explanations"]

    rows: list[dict[str, Any]] = []
    if isinstance(blob, list):
        for item in blob:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "feature": item.get("feature"),
                    "shap_value": float(item.get("shap_value", 0.0) or 0.0),
                    "absolute_shap_value": float(item.get("absolute_shap_value", abs(float(item.get("shap_value", 0.0) or 0.0)))),
                    "source_columns": item.get("source_columns", []),
                    "feature_type": item.get("feature_type"),
                    "method": item.get("method"),
                }
            )
    elif isinstance(blob, dict):
        rows.append(
            {
                "feature": blob.get("feature"),
                "shap_value": float(blob.get("shap_value", 0.0) or 0.0),
                "absolute_shap_value": float(blob.get("absolute_shap_value", abs(float(blob.get("shap_value", 0.0) or 0.0)))),
                "source_columns": blob.get("source_columns", []),
                "feature_type": blob.get("feature_type"),
                "method": blob.get("method"),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty and "absolute_shap_value" in frame.columns:
        frame = frame.sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
    return frame


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return bool(value)


def normalize_explain_batch_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the JSON payload returned by /explain_batch into a dataframe."""

    results = payload.get("results", [])
    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        row: dict[str, Any] = {"patient_index": result.get("patient_index")}
        prediction = result.get("prediction")
        if isinstance(prediction, dict):
            row.update(prediction)
        explanation = result.get("explanation")
        if isinstance(explanation, dict):
            row["feature_explanations"] = explanation.get("feature_explanations")
            row["explanation_method"] = explanation.get("method")
            row["explanation_top_k"] = explanation.get("top_k")
        rows.append(row)
    return pd.DataFrame(rows)


def build_risk_map_frame(frame: pd.DataFrame, *, score_column: str = "anomaly_score") -> pd.DataFrame:
    """Prepare a table for the patient risk map visualization."""

    if frame.empty:
        return pd.DataFrame(columns=["patient_label", "score", "risk_level", "patient_index"])
    if score_column not in frame.columns:
        raise ValueError(f"Score column '{score_column}' is missing.")

    risk_map = frame.copy()
    risk_map["risk_level"] = pd.cut(
        risk_map[score_column].astype(float),
        bins=_RISK_BINS,
        labels=_RISK_LABELS,
        include_lowest=True,
        right=False,
    )
    risk_map["patient_index"] = range(len(risk_map))
    if "patient_id" in risk_map.columns:
        risk_map["patient_label"] = risk_map["patient_id"].astype(str)
    elif "record_id" in risk_map.columns:
        risk_map["patient_label"] = risk_map["record_id"].astype(str)
    else:
        risk_map["patient_label"] = risk_map["patient_index"].astype(str)
    risk_map["score"] = risk_map[score_column].astype(float)
    return risk_map


def build_trend_frame(
    frame: pd.DataFrame,
    *,
    time_column: str = "recorded_at",
    score_column: str = "anomaly_score",
) -> pd.DataFrame:
    """Aggregate anomaly scores over time for trend charts."""

    if frame.empty or time_column not in frame.columns or score_column not in frame.columns:
        return pd.DataFrame(columns=["time_bucket", "mean_score", "max_score", "alert_count", "sample_count"])

    trend = frame.copy()
    trend[time_column] = pd.to_datetime(trend[time_column], errors="coerce")
    trend = trend.dropna(subset=[time_column])
    if trend.empty:
        return pd.DataFrame(columns=["time_bucket", "mean_score", "max_score", "alert_count", "sample_count"])

    trend["time_bucket"] = trend[time_column].dt.floor("D")
    grouped = trend.groupby("time_bucket", dropna=True)
    return grouped.agg(
        mean_score=(score_column, "mean"),
        max_score=(score_column, "max"),
        alert_count=(score_column, lambda values: int((values.astype(float) >= 0.7).sum())),
        sample_count=(score_column, "size"),
    ).reset_index()


def build_agreement_summary(
    frame: pd.DataFrame,
    *,
    score_columns: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Build agreement statistics and a pairwise matrix for display."""

    candidate_columns = score_columns or [
        column
        for column in frame.columns
        if column.endswith("_anomaly_score") and column != "anomaly_score"
    ]
    if len(candidate_columns) < 2:
        return {"columns": candidate_columns, "agreement": None, "matrix": pd.DataFrame()}

    analysis = build_unsupervised_analysis(
        frame,
        score_columns=["anomaly_score", *candidate_columns] if "anomaly_score" in frame.columns else candidate_columns,
        threshold=threshold,
    )
    agreement = analysis.get("agreement")
    matrix = pd.DataFrame()
    if agreement:
        matrix = pd.DataFrame(
            agreement["agreement_matrix"]["data"],
            index=agreement["agreement_matrix"]["index"],
            columns=agreement["agreement_matrix"]["columns"],
        )
    return {"columns": candidate_columns, "agreement": agreement, "matrix": matrix}


def build_top_alert_feature_views(
    frame: pd.DataFrame,
    *,
    score_column: str = "anomaly_score",
    top_alerts: int = 5,
    top_features: int = 5,
) -> list[dict[str, Any]]:
    """Return the highest-scoring alerts with their top feature explanations."""

    if frame.empty or score_column not in frame.columns:
        return []

    ranked = frame.sort_values(score_column, ascending=False).head(max(1, int(top_alerts)))
    views: list[dict[str, Any]] = []
    for row_index, row in ranked.iterrows():
        explanation_blob = None
        for column in ("explanation", "feature_explanations", "explanation_json"):
            if column in row.index and _has_value(row[column]):
                explanation_blob = row[column]
                break

        features = parse_feature_explanations(explanation_blob).head(max(1, int(top_features)))
        views.append(
            {
                "row_index": int(row_index),
                "patient_id": row.get("patient_id"),
                "recorded_at": row.get("recorded_at"),
                "anomaly_score": float(row.get(score_column, 0.0) or 0.0),
                "risk_level": row.get("risk_level"),
                "alert_triggered": bool(row.get("alert_triggered")) if "alert_triggered" in row and pd.notna(row.get("alert_triggered")) else None,
                "features": features.to_dict(orient="records"),
            }
        )
    return views


def render_dashboard() -> None:
    """Render the Streamlit dashboard UI."""

    import streamlit as st

    st.set_page_config(page_title="Rural Health Anomaly Dashboard", layout="wide")
    st.title("Rural Health Anomaly Dashboard")
    st.caption("Interactive risk map, trend view, alert explanations, and model agreement.")

    st.sidebar.header("Data Source")
    scored_upload = st.sidebar.file_uploader(
        "Scored dataset (.csv, .parquet, or .json)",
        type=["csv", "parquet", "json"],
        accept_multiple_files=False,
    )
    explanation_upload = st.sidebar.file_uploader(
        "Optional /explain_batch JSON",
        type=["json"],
        accept_multiple_files=False,
    )

    threshold = st.sidebar.slider("Alert threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    top_alerts = st.sidebar.slider("Top alerts to inspect", min_value=1, max_value=25, value=5, step=1)
    top_features = st.sidebar.slider("Top features per alert", min_value=1, max_value=10, value=5, step=1)

    if scored_upload is None:
        st.info("Upload a scored CSV/Parquet file to view the dashboard.")
        return

    scored_frame = load_dashboard_frame(scored_upload)
    explanation_frame = load_dashboard_frame(explanation_upload) if explanation_upload is not None else pd.DataFrame()

    if not explanation_frame.empty and "patient_index" in explanation_frame.columns:
        explanation_frame = explanation_frame.sort_values("patient_index").reset_index(drop=True)
        merged = scored_frame.reset_index(drop=True).copy()
        if len(explanation_frame) == len(merged):
            merged["feature_explanations"] = explanation_frame.get("feature_explanations")
            if "prediction" in explanation_frame.columns:
                merged["explain_prediction"] = explanation_frame["prediction"]
        else:
            merged = scored_frame.copy()
    else:
        merged = scored_frame.copy()

    if "feature_explanations" not in merged.columns and "explanation" in merged.columns:
        merged["feature_explanations"] = merged["explanation"]

    score_column = "anomaly_score" if "anomaly_score" in merged.columns else next(
        (column for column in merged.columns if column.endswith("_anomaly_score")),
        None,
    )
    if score_column is None:
        st.error("No anomaly score column was found in the uploaded file.")
        return

    risk_map_frame = build_risk_map_frame(merged, score_column=score_column)
    trend_frame = build_trend_frame(merged, score_column=score_column)
    agreement_summary = build_agreement_summary(merged, threshold=threshold)
    alert_views = build_top_alert_feature_views(
        merged,
        score_column=score_column,
        top_alerts=top_alerts,
        top_features=top_features,
    )

    total_patients = len(merged)
    alert_count = int((merged[score_column].astype(float) >= threshold).sum())
    high_risk_count = int((merged[score_column].astype(float) >= 0.7).sum())
    mean_score = float(merged[score_column].astype(float).mean()) if total_patients else float("nan")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Patients", total_patients)
    metric_cols[1].metric("Alerts", alert_count)
    metric_cols[2].metric("High risk", high_risk_count)
    metric_cols[3].metric("Mean score", f"{mean_score:.3f}" if pd.notna(mean_score) else "n/a")

    st.subheader("Patient Risk Map")
    if {"latitude", "longitude"}.issubset(risk_map_frame.columns):
        st.map(risk_map_frame.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
    elif {"lat", "lon"}.issubset(risk_map_frame.columns):
        st.map(risk_map_frame[["lat", "lon"]])
    else:
        scatter_frame = risk_map_frame[["patient_index", "score", "risk_level", "patient_label"]].rename(
            columns={"patient_index": "patient_index", "score": "score"}
        )
        st.scatter_chart(scatter_frame, x="patient_index", y="score", color="risk_level")
        st.dataframe(
            risk_map_frame[["patient_label", score_column, "risk_level"]].sort_values(score_column, ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Anomaly Score Trends Over Time")
    if trend_frame.empty:
        st.info("No usable `recorded_at` timestamps were found for a trend chart.")
    else:
        st.line_chart(trend_frame.set_index("time_bucket")[["mean_score", "max_score"]])
        st.dataframe(trend_frame, use_container_width=True, hide_index=True)

    st.subheader("Top Contributing Features Per Alert")
    if not alert_views:
        st.info(
            "Upload an `/explain_batch` JSON file or include a `feature_explanations` column to view feature attributions."
        )
    else:
        for view in alert_views:
            label = f"Patient {view.get('patient_id', view['row_index'])} - score {view['anomaly_score']:.3f}"
            with st.expander(label, expanded=False):
                st.write(
                    {
                        "patient_id": view.get("patient_id"),
                        "recorded_at": view.get("recorded_at"),
                        "risk_level": view.get("risk_level"),
                        "alert_triggered": view.get("alert_triggered"),
                    }
                )
                features = pd.DataFrame(view["features"])
                if features.empty:
                    st.info("No feature explanations available for this alert.")
                else:
                    st.bar_chart(features.set_index("feature")[["absolute_shap_value"]])
                    st.dataframe(features, use_container_width=True, hide_index=True)

    st.subheader("Model Agreement")
    agreement = agreement_summary["agreement"]
    if agreement is None:
        st.info("At least two model score columns are required to render the agreement view.")
    else:
        agreement_cols = st.columns(4)
        component_count = int(agreement.get("component_count", len(agreement["columns"])))
        agreement_cols[0].metric("Pairwise agreement", f"{agreement['mean_pairwise_agreement_rate']:.3f}")
        agreement_cols[1].metric("At least two flag rate", f"{agreement['at_least_two_flag_rate']:.3f}")
        agreement_cols[2].metric(f"All {component_count} models flag rate", f"{agreement['all_models_flag_rate']:.3f}")
        agreement_cols[3].metric(f"All {component_count} models flag count", agreement["all_models_flag_count"])
        st.dataframe(agreement_summary["matrix"].style.background_gradient(cmap="Blues"), use_container_width=True)

    with st.expander("Raw data preview", expanded=False):
        st.dataframe(merged, use_container_width=True)


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
