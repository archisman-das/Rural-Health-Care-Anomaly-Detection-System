"""FastAPI backend for real-time rural health anomaly predictions."""

from __future__ import annotations

import json
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .feedback import append_feedback_records
from .training import load_pipeline, score_records
from .training import _compute_latent_manifold
from .training import _compute_reconstruction_residual_heatmap


def _score_payload(pipeline, payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    if frame.empty:
        raise HTTPException(status_code=400, detail="Request body must include at least one patient record.")
    return score_records(pipeline, frame)


async def _score_csv_upload(pipeline, file: UploadFile) -> dict[str, Any]:
    filename = file.filename or ""
    if filename and not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported for CSV batch scoring.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        frame = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:  # pragma: no cover - handled by API response
        raise HTTPException(status_code=400, detail=f"Unable to parse CSV upload: {exc}") from exc

    if frame.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file must contain at least one row.")

    scored = score_records(pipeline, frame)
    return {
        "filename": filename,
        "count": int(len(scored)),
        "predictions": jsonable_encoder(scored.to_dict(orient="records")),
    }


def _model_metadata_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".metadata.json")


def _load_model_artifact_metadata(model_path: str | Path) -> dict[str, Any]:
    metadata_path = _model_metadata_path(model_path)
    if not metadata_path.exists():
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return metadata if isinstance(metadata, dict) else {}


def _build_explanation_rows(
    *,
    feature_names: list[str],
    values: list[float],
    top_k: int,
    feature_map: pd.DataFrame,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_lookup = feature_map.set_index("final_feature") if not feature_map.empty and "final_feature" in feature_map.columns else None
    ranked_indices = sorted(range(len(values)), key=lambda idx: abs(values[idx]), reverse=True)[: max(1, int(top_k))]
    for idx in ranked_indices:
        feature_name = feature_names[idx]
        row: dict[str, Any] = {
            "feature": feature_name,
            "shap_value": float(values[idx]),
            "absolute_shap_value": float(abs(values[idx])),
            "method": method,
        }
        if feature_lookup is not None and feature_name in feature_lookup.index:
            meta = feature_lookup.loc[feature_name]
            row["source_columns"] = meta["source_columns"]
            row["feature_type"] = meta.get("feature_type")
        rows.append(row)
    return rows


def _build_interaction_heatmap(
    *,
    feature_names: list[str],
    matrix: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return {
            "method": "unavailable",
            "feature_names": [],
            "matrix": [],
            "top_pairs": [],
            "top_features": [],
        }

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    if matrix.shape[0] == matrix.shape[1]:
        matrix = matrix.copy()
        matrix[np.diag_indices(matrix.shape[0])] = np.diag(matrix)

    feature_strength = np.sum(np.abs(matrix), axis=1)
    ranked_indices = sorted(
        range(len(feature_names)),
        key=lambda idx: float(feature_strength[idx]),
        reverse=True,
    )[: max(1, int(top_k))]
    ranked_names = [feature_names[idx] for idx in ranked_indices]
    ranked_matrix = matrix[np.ix_(ranked_indices, ranked_indices)]

    top_pairs: list[dict[str, Any]] = []
    for i in range(len(ranked_indices)):
        for j in range(i + 1, len(ranked_indices)):
            interaction_value = float(ranked_matrix[i, j])
            top_pairs.append(
                {
                    "feature_i": ranked_names[i],
                    "feature_j": ranked_names[j],
                    "interaction_value": interaction_value,
                    "absolute_interaction_value": float(abs(interaction_value)),
                }
            )

    top_pairs = sorted(top_pairs, key=lambda item: item["absolute_interaction_value"], reverse=True)[: max(1, int(top_k))]
    return {
        "method": "tree_shap_interaction",
        "feature_names": ranked_names,
        "matrix": ranked_matrix.tolist(),
        "top_pairs": top_pairs,
        "top_features": [
            {
                "feature": ranked_names[idx],
                "interaction_strength": float(feature_strength[ranked_indices[idx]]),
            }
            for idx in range(len(ranked_indices))
        ],
    }


def _build_fallback_interaction_heatmap(
    *,
    feature_names: list[str],
    values: list[float],
    transformed_patient: np.ndarray,
    background_transformed: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    transformed_patient = np.asarray(transformed_patient, dtype=float)
    background_transformed = np.asarray(background_transformed, dtype=float)
    if transformed_patient.ndim != 2 or transformed_patient.shape[1] == 0:
        return {
            "method": "unavailable",
            "feature_names": [],
            "matrix": [],
            "top_pairs": [],
            "top_features": [],
        }

    centered = transformed_patient[0] - np.nanmean(background_transformed, axis=0)
    interaction_matrix = np.outer(centered, centered)
    if np.any(np.isfinite(interaction_matrix)) and np.max(np.abs(interaction_matrix)) > 0:
        interaction_matrix = interaction_matrix / float(np.max(np.abs(interaction_matrix)))

    scale = float(max(np.max(np.abs(values)) if values else 0.0, 1e-6))
    interaction_matrix = interaction_matrix * scale
    return {
        "method": "deviation_outer_product_fallback",
        "feature_names": feature_names[: interaction_matrix.shape[0]],
        "matrix": interaction_matrix.tolist(),
        "top_pairs": [],
        "top_features": [
            {
                "feature": feature_name,
                "interaction_strength": float(abs(value)),
            }
            for feature_name, value in zip(feature_names[: len(values)], values, strict=False)
        ][: max(1, int(top_k))],
    }


def _compute_feature_explanation(
    pipeline,
    patient: dict[str, Any],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    frame = pd.DataFrame([patient])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_patient = preprocessor.transform(frame)
    feature_names = list(preprocessor.get_feature_names_out())
    feature_map = preprocessor.export_feature_map()
    background_raw = getattr(pipeline, "explain_background_", frame)
    background_transformed = preprocessor.transform(background_raw)

    method = "ablation_fallback"
    values: list[float]
    interaction_heatmap: dict[str, Any] | None = None

    try:  # pragma: no cover - exercised when shap is available
        import shap  # type: ignore

        tree_model = None
        tree_source = None
        estimators = getattr(model, "estimators_", {})
        if isinstance(estimators, dict):
            isolation_forest = estimators.get("isolation_forest")
            tree_model = getattr(isolation_forest, "model_", None)
            if tree_model is not None:
                tree_source = "isolation_forest"

        if tree_model is not None:
            background_sample = np.asarray(background_transformed, dtype=float)
            if background_sample.ndim == 2 and background_sample.shape[0] > 64:
                background_sample = background_sample[:64]
            explainer = shap.TreeExplainer(tree_model, data=background_sample)
            shap_values = explainer.shap_values(transformed_patient)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            values = [float(value) for value in np.asarray(shap_values, dtype=float)[0].tolist()]
            interaction_values = explainer.shap_interaction_values(transformed_patient)
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0]
            matrix = np.asarray(interaction_values, dtype=float)
            if matrix.ndim == 3:
                matrix = matrix[0]
            interaction_heatmap = _build_interaction_heatmap(
                feature_names=feature_names,
                matrix=matrix,
                top_k=top_k,
            )
            if interaction_heatmap["method"] == "tree_shap_interaction":
                interaction_heatmap["source_model"] = tree_source
            method = "tree_shap_isolation_forest"
        else:
            explainer = shap.KernelExplainer(lambda x: model.score(x), background_transformed)
            shap_values = explainer.shap_values(
                transformed_patient,
                nsamples=min(100, max(20, transformed_patient.shape[1] * 2)),
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            values = [float(value) for value in shap_values[0].tolist()]
            method = "shap_kernel"
    except Exception:
        reference = background_transformed.mean(axis=0, keepdims=True)
        base_score = float(model.score(transformed_patient)[0])
        values = []
        for idx in range(transformed_patient.shape[1]):
            ablated = transformed_patient.copy()
            ablated[0, idx] = reference[0, idx]
            ablated_score = float(model.score(ablated)[0])
            values.append(base_score - ablated_score)
        interaction_heatmap = _build_fallback_interaction_heatmap(
            feature_names=feature_names,
            values=values,
            transformed_patient=transformed_patient,
            background_transformed=background_transformed,
            top_k=top_k,
        )

    rows = _build_explanation_rows(
        feature_names=feature_names,
        values=values,
        top_k=top_k,
        feature_map=feature_map,
        method=method,
    )
    if interaction_heatmap is None:
        interaction_heatmap = _build_fallback_interaction_heatmap(
            feature_names=feature_names,
            values=values,
            transformed_patient=transformed_patient,
            background_transformed=background_transformed,
            top_k=top_k,
        )
    return {
        "method": method,
        "top_k": int(top_k),
        "background_size": int(len(background_raw)),
        "feature_explanations": rows,
        "interaction_heatmap": interaction_heatmap,
    }


def _compute_feature_engineering(
    pipeline,
    patient: dict[str, Any],
    *,
    top_k: int = 25,
) -> dict[str, Any]:
    """Return the engineered feature vector used by the backend model."""

    frame = pd.DataFrame([patient])
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(frame)
    feature_names = list(preprocessor.get_feature_names_out())
    feature_map = preprocessor.export_feature_map()
    feature_lookup = feature_map.set_index("final_feature") if not feature_map.empty and "final_feature" in feature_map.columns else None

    transformed_row = transformed[0].tolist() if getattr(transformed, "shape", (0,))[0] else []
    rows: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(feature_names[: max(1, int(top_k))]):
        value = transformed_row[idx] if idx < len(transformed_row) else None
        row: dict[str, Any] = {
            "feature": feature_name,
            "engineered_value": None if value is None else float(value),
        }
        if feature_lookup is not None and feature_name in feature_lookup.index:
            meta = feature_lookup.loc[feature_name]
            row["source_columns"] = meta["source_columns"]
            row["feature_type"] = meta.get("feature_type")
            row["transformation_path"] = meta.get("transformation_path")
            provenance_depth = meta.get("provenance_depth")
            row["provenance_depth"] = None if provenance_depth is None else int(provenance_depth)
        rows.append(jsonable_encoder(row))

    return jsonable_encoder({
        "feature_count": len(feature_names),
        "top_k": int(top_k),
        "transformed_shape": [int(dim) for dim in getattr(transformed, "shape", ())],
        "engineered_features": rows,
    })


def _compute_data_scaling(
    pipeline,
    patient: dict[str, Any],
    *,
    top_k: int = 25,
) -> dict[str, Any]:
    """Return the scaled numeric inputs that feed the model."""

    frame = pd.DataFrame([patient])
    preprocessor = pipeline.named_steps["preprocessor"]
    engineered = preprocessor._prepare_features(frame, fit=False)
    feature_pipeline = getattr(preprocessor, "feature_pipeline_", None)
    if feature_pipeline is None or not getattr(preprocessor, "fitted_", False):
        return jsonable_encoder({
            "scaler": getattr(preprocessor.config, "scaler", "standard"),
            "feature_count": 0,
            "top_k": int(top_k),
            "scaled_features": [],
        })

    numeric_cols = list(getattr(preprocessor, "numeric_columns_", []))
    if not numeric_cols:
        return jsonable_encoder({
            "scaler": getattr(preprocessor.config, "scaler", "standard"),
            "feature_count": 0,
            "top_k": int(top_k),
            "scaled_features": [],
        })

    column_transformer = feature_pipeline.named_steps["preprocessor"]
    numeric_pipeline = column_transformer.named_transformers_.get("num")
    if numeric_pipeline is None:
        return jsonable_encoder({
            "scaler": getattr(preprocessor.config, "scaler", "standard"),
            "feature_count": 0,
            "top_k": int(top_k),
            "scaled_features": [],
        })

    raw_numeric = engineered.reindex(columns=numeric_cols)
    scaled_numeric = numeric_pipeline.transform(raw_numeric)
    if hasattr(scaled_numeric, "toarray"):
        scaled_numeric = scaled_numeric.toarray()

    feature_map = preprocessor.export_feature_map()
    feature_lookup = feature_map.set_index("final_feature") if not feature_map.empty and "final_feature" in feature_map.columns else None

    rows: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(numeric_cols[: max(1, int(top_k))]):
        raw_value = raw_numeric.iloc[0, idx] if idx < raw_numeric.shape[1] else None
        scaled_value = scaled_numeric[0, idx] if idx < scaled_numeric.shape[1] else None
        row: dict[str, Any] = {
            "feature": feature_name,
            "raw_value": None if pd.isna(raw_value) else float(raw_value),
            "scaled_value": None if scaled_value is None or pd.isna(scaled_value) else float(scaled_value),
            "scaler": getattr(preprocessor.config, "scaler", "standard"),
        }
        if feature_lookup is not None and feature_name in feature_lookup.index:
            meta = feature_lookup.loc[feature_name]
            row["source_columns"] = meta["source_columns"]
            row["feature_type"] = meta.get("feature_type")
        rows.append(jsonable_encoder(row))

    return jsonable_encoder({
        "scaler": getattr(preprocessor.config, "scaler", "standard"),
        "feature_count": len(numeric_cols),
        "top_k": int(top_k),
        "input_shape": [int(dim) for dim in getattr(raw_numeric, "shape", ())],
        "scaled_shape": [int(dim) for dim in getattr(scaled_numeric, "shape", ())],
        "scaled_features": rows,
    })


def _compute_data_encoding(
    pipeline,
    patient: dict[str, Any],
    *,
    top_k: int = 25,
) -> dict[str, Any]:
    """Return the encoded categorical inputs that feed the model."""

    def _split_encoded_feature_name(feature_name: str, source_columns: Any, feature_type: str) -> tuple[str, str]:
        source_feature = ""
        if isinstance(source_columns, list) and source_columns:
            source_feature = str(source_columns[0])
        elif isinstance(source_columns, str) and source_columns:
            source_feature = source_columns.split(",")[0].strip()

        if feature_type == "expanded_multi_value" and "__" in feature_name:
            base, encoded_value = feature_name.split("__", 1)
            return base or source_feature, encoded_value

        if source_feature and feature_name.startswith(f"{source_feature}_"):
            return source_feature, feature_name[len(source_feature) + 1 :]

        if "_" in feature_name:
            base, encoded_value = feature_name.rsplit("_", 1)
            return source_feature or base, encoded_value

        return source_feature or feature_name, ""

    frame = pd.DataFrame([patient])
    preprocessor = pipeline.named_steps["preprocessor"]
    engineered = preprocessor._prepare_features(frame, fit=False)
    feature_pipeline = getattr(preprocessor, "feature_pipeline_", None)
    if feature_pipeline is None or not getattr(preprocessor, "fitted_", False):
        return jsonable_encoder({
            "encoder": "one-hot",
            "feature_count": 0,
            "top_k": int(top_k),
            "encoded_features": [],
        })

    column_transformer = feature_pipeline.named_steps["preprocessor"]
    pre_encoded_matrix = column_transformer.transform(engineered.reindex(columns=getattr(preprocessor, "feature_columns_", [])))
    if hasattr(pre_encoded_matrix, "toarray"):
        pre_encoded_matrix = pre_encoded_matrix.toarray()

    feature_names = list(column_transformer.get_feature_names_out())
    feature_map = preprocessor.export_feature_map()
    feature_lookup = feature_map.set_index("final_feature") if not feature_map.empty and "final_feature" in feature_map.columns else None

    encoded_types = {"one_hot", "expanded_multi_value"}
    rows: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(feature_names):
        if len(rows) >= max(1, int(top_k)):
            break
        meta = None
        if feature_lookup is not None and feature_name in feature_lookup.index:
            meta = feature_lookup.loc[feature_name]
            feature_type = str(meta.get("feature_type") or "")
            if feature_type not in encoded_types:
                continue
        else:
            feature_type = ""
            if "__" not in feature_name and "_" not in feature_name:
                continue

        value = pre_encoded_matrix[0, idx] if idx < pre_encoded_matrix.shape[1] else None
        row: dict[str, Any] = {
            "feature": feature_name,
            "encoded_value": None if value is None or pd.isna(value) else float(value),
            "encoder": "one-hot",
        }
        if meta is not None:
            source_feature, category_value = _split_encoded_feature_name(
                feature_name,
                meta["source_columns"],
                str(meta.get("feature_type") or ""),
            )
            active = bool(value is not None and not pd.isna(value) and float(value) >= 0.5)
            row["source_columns"] = meta["source_columns"]
            row["feature_type"] = meta.get("feature_type")
            row["transformation_path"] = meta.get("transformation_path")
            row["source_feature"] = source_feature
            row["category_value"] = category_value
            row["explanation"] = (
                f"{source_feature} = {category_value} because the input selected {category_value}."
                if active and source_feature and category_value
                else (
                    f"{source_feature} = {category_value} is inactive for this record."
                    if source_feature and category_value
                    else "Categorical value encoded by the backend."
                )
            )
        rows.append(jsonable_encoder(row))

    return jsonable_encoder({
        "encoder": "one-hot",
        "feature_count": len(rows),
        "top_k": int(top_k),
        "encoded_shape": [int(dim) for dim in getattr(pre_encoded_matrix, "shape", ())],
        "encoded_features": rows,
    })


def _explain_most_anomalous_record(
    pipeline,
    frame: pd.DataFrame,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    if frame.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file must contain at least one row.")

    scored = score_records(pipeline, frame)
    best_index = int(scored["anomaly_score"].astype(float).idxmax())
    patient = frame.iloc[best_index].to_dict()
    explanation = _compute_feature_explanation(pipeline, patient, top_k=top_k)
    explanation["row_index"] = best_index
    explanation["anomaly_score"] = float(scored.iloc[best_index]["anomaly_score"])
    explanation["risk_level"] = scored.iloc[best_index].get("risk_level")
    explanation["alert_triggered"] = bool(scored.iloc[best_index].get("alert_triggered"))
    return {
        "selected_row_index": best_index,
        "prediction": jsonable_encoder(scored.iloc[best_index].to_dict()),
        "explanation": explanation,
    }


def _explain_batch_records(
    pipeline,
    patients: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    scored = _score_payload(pipeline, patients)
    results: list[dict[str, Any]] = []
    for index, patient in enumerate(patients):
        explanation = _compute_feature_explanation(pipeline, patient, top_k=top_k)
        results.append(
            {
                "patient_index": index,
                "prediction": jsonable_encoder(scored.iloc[index].to_dict()),
                "explanation": explanation,
            }
        )
    return {"count": len(results), "results": results}


def _model_metadata(app: FastAPI) -> dict[str, Any]:
    model = app.state.pipeline.named_steps["model"]
    preprocessor = app.state.pipeline.named_steps["preprocessor"]
    return {
        "model_name": app.state.model_name,
        "model_version": app.state.model_version,
        "model_path": app.state.model_path,
        "artifact_sha256": getattr(app.state, "artifact_sha256", None),
        "model_type": type(model).__name__,
        "feature_count": len(getattr(preprocessor, "feature_columns_", [])),
        "feature_output_count": len(preprocessor.get_feature_names_out()) if getattr(preprocessor, "fitted_", False) else None,
    }


def _dashboard_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "dashboard-data.json"


def _load_dashboard_data_template() -> dict[str, Any]:
    data_path = _dashboard_data_path()
    if not data_path.exists():
        return {}
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dashboard_last_updated() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%B %d, %Y %I:%M %p")


def _build_dashboard_data(app: FastAPI) -> dict[str, Any]:
    payload = _load_dashboard_data_template()
    metadata = _model_metadata(app)
    feedback_path = Path(app.state.feedback_store_path)
    feedback_count = 0
    if feedback_path.exists():
        with feedback_path.open("r", encoding="utf-8") as handle:
            feedback_count = sum(1 for line in handle if line.strip())

    for key in ("summary", "riskTiles", "riskLegend", "modelRows", "trend", "distribution", "features", "agreement", "feedback", "runtime"):
        payload.setdefault(key, [])

    payload["dashboardState"] = "Clinical dashboard ready"
    payload["lastUpdated"] = _dashboard_last_updated()
    payload["feedbackCount"] = feedback_count
    payload["modelConfig"] = getattr(app.state, "latest_model_config", None)
    return payload


def _normalize_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    hidden_layers = payload.get("stacking_hidden_layer_sizes", [32, 16])
    if isinstance(hidden_layers, str):
        hidden_layers = [item.strip() for item in hidden_layers.split(",") if item.strip()]

    normalized_layers: list[int] = []
    hidden_layer_values = hidden_layers if isinstance(hidden_layers, (list, tuple)) else [hidden_layers]
    for value in hidden_layer_values:
        try:
            layer = int(value)
        except (TypeError, ValueError):
            continue
        if layer > 0:
            normalized_layers.append(layer)

    def _float_value(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if number >= 0 else default

    def _int_value(value: Any, default: int) -> int:
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return default
        return number if number > 0 else default

    meta_model_type = str(payload.get("stacking_meta_model_type") or "mlp").strip().lower()
    if meta_model_type not in {"mlp", "xgboost", "auto"}:
        meta_model_type = "mlp"

    return {
        "stacking_meta_model_type": meta_model_type,
        "stacking_hidden_layer_sizes": normalized_layers or [32, 16],
        "stacking_alpha": _float_value(payload.get("stacking_alpha"), 1e-4),
        "stacking_learning_rate_init": _float_value(payload.get("stacking_learning_rate_init"), 1e-3),
        "stacking_max_iter": _int_value(payload.get("stacking_max_iter"), 500),
        "stacking_random_state": int(_int_value(payload.get("stacking_random_state"), 42)),
        "stacking_verbose": bool(payload.get("stacking_verbose", False)),
    }


def _csv_upload_openapi_example(summary: str) -> dict[str, Any]:
    return {
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                            }
                        },
                        "required": ["file"],
                    },
                    "examples": {
                        "sampleCsv": {
                            "summary": summary,
                            "value": {
                                "file": (
                                    "patient_id,age_years,glucose_fasting_mg_dl,"
                                    "heart_rate_bpm,systolic_bp_mmhg,diastolic_bp_mmhg\n"
                                    "P001,54,118,84,126,80\n"
                                    "P002,61,176,92,148,94\n"
                                )
                            },
                        }
                    },
                }
            }
        }
    }


def _build_token_dependency(expected_token: str | None):
    def _require_token(x_api_token: str | None = Header(default=None, alias="X-API-Token")) -> None:
        if expected_token is None:
            return
        if x_api_token != expected_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Token header.",
            )

    return _require_token


def create_app(
    model_path: str | Path,
    *,
    title: str = "Rural Health Anomaly API",
    auth_token: str | None = None,
    feedback_store: str | Path | None = None,
) -> FastAPI:
    """Create a FastAPI app backed by a saved anomaly pipeline."""

    resolved_model_path = Path(model_path)
    pipeline = load_pipeline(resolved_model_path)
    artifact_metadata = _load_model_artifact_metadata(resolved_model_path)
    pipeline_version = getattr(pipeline, "model_version_", None)
    auth_dependency = _build_token_dependency(auth_token)

    app = FastAPI(
        title=title,
        version="1.0.0",
        description="Real-time anomaly scoring for rural health patient records.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.pipeline = pipeline
    app.state.model_name = str(artifact_metadata.get("model_name") or resolved_model_path.stem)
    app.state.model_version = str(
        artifact_metadata.get("model_version")
        or artifact_metadata.get("artifact_version")
        or pipeline_version
        or "unknown"
    )
    app.state.artifact_sha256 = artifact_metadata.get("artifact_sha256")
    app.state.model_path = str(resolved_model_path)
    app.state.auth_token_enabled = auth_token is not None
    app.state.latest_model_config = _normalize_model_config(
        {
            "stacking_meta_model_type": getattr(pipeline, "stacking_meta_model_type", "mlp"),
            "stacking_hidden_layer_sizes": getattr(pipeline, "stacking_hidden_layer_sizes", (32, 16)),
            "stacking_alpha": getattr(pipeline, "stacking_alpha", 1e-4),
            "stacking_learning_rate_init": getattr(pipeline, "stacking_learning_rate_init", 1e-3),
            "stacking_max_iter": getattr(pipeline, "stacking_max_iter", 500),
            "stacking_random_state": getattr(pipeline, "stacking_random_state", 42),
            "stacking_verbose": getattr(pipeline, "stacking_verbose", False),
        }
    )
    default_feedback_store = resolved_model_path.with_name("feedback_ledger.jsonl")
    app.state.feedback_store_path = str(Path(feedback_store) if feedback_store is not None else default_feedback_store)

    @app.get("/health", dependencies=[Depends(auth_dependency)])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            **_model_metadata(app),
        }

    @app.get("/models", dependencies=[Depends(auth_dependency)])
    def models() -> dict[str, Any]:
        return {
            "count": 1,
            "models": [_model_metadata(app)],
        }

    @app.get("/dashboard-data.json", dependencies=[Depends(auth_dependency)])
    def dashboard_data() -> dict[str, Any]:
        return _build_dashboard_data(app)

    @app.get("/api/model-config", dependencies=[Depends(auth_dependency)])
    def get_model_config() -> dict[str, Any]:
        return {
            "model_info": _model_metadata(app),
            "config": getattr(app.state, "latest_model_config", None),
        }

    @app.post("/api/model-config", dependencies=[Depends(auth_dependency)])
    def set_model_config(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_model_config(payload)
        app.state.latest_model_config = normalized
        return {
            "model_info": _model_metadata(app),
            "config": normalized,
            "message": "Model configuration saved for the current backend session.",
        }

    @app.get("/feedback", dependencies=[Depends(auth_dependency)])
    def feedback_overview() -> dict[str, Any]:
        store_path = Path(app.state.feedback_store_path)
        if not store_path.exists():
            return {"path": str(store_path), "count": 0, "exists": False}
        with store_path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        return {"path": str(store_path), "count": count, "exists": True}

    @app.post("/predict", dependencies=[Depends(auth_dependency)])
    def predict(patient: dict[str, Any]) -> dict[str, Any]:
        scored = _score_payload(app.state.pipeline, patient)
        record = jsonable_encoder(scored.iloc[0].to_dict())
        return {
            "input": jsonable_encoder(patient),
            "prediction": record,
            "model_config": getattr(app.state, "latest_model_config", None),
            "model_info": {
                "model_name": app.state.model_name,
                "model_version": app.state.model_version,
                "model_type": type(app.state.pipeline.named_steps["model"]).__name__,
            },
            "conformal_p_value": record.get("conformal_p_value"),
            "conformal_assessment": record.get("conformal_assessment"),
            "anomaly_score": record.get("anomaly_score"),
            "risk_score": record.get("risk_score"),
            "risk_category": record.get("risk_category", record.get("risk_level")),
            "risk_level": record.get("risk_level"),
            "alert_triggered": record.get("alert_triggered"),
            "is_anomaly": record.get("is_anomaly"),
            "explanation": _compute_feature_explanation(app.state.pipeline, patient, top_k=10),
            "latent_manifold": _compute_latent_manifold(app.state.pipeline, patient),
            "reconstruction_residual_heatmap": _compute_reconstruction_residual_heatmap(app.state.pipeline, patient),
            "feature_engineering": _compute_feature_engineering(app.state.pipeline, patient, top_k=25),
            "data_scaling": _compute_data_scaling(app.state.pipeline, patient, top_k=25),
            "data_encoding": _compute_data_encoding(app.state.pipeline, patient, top_k=25),
        }

    @app.post("/batch-predict", dependencies=[Depends(auth_dependency)])
    def batch_predict(patients: list[dict[str, Any]]) -> dict[str, Any]:
        scored = _score_payload(app.state.pipeline, patients)
        return {
            "count": int(len(scored)),
            "predictions": jsonable_encoder(scored.to_dict(orient="records")),
        }

    @app.post(
        "/predict_file",
        dependencies=[Depends(auth_dependency)],
        openapi_extra=_csv_upload_openapi_example("CSV file for batch scoring"),
    )
    async def predict_file(file: UploadFile = File(...)) -> dict[str, Any]:
        return await _score_csv_upload(app.state.pipeline, file)

    @app.post("/batch", dependencies=[Depends(auth_dependency)])
    async def batch(file: UploadFile = File(...)) -> dict[str, Any]:
        return await _score_csv_upload(app.state.pipeline, file)

    @app.post("/explain", dependencies=[Depends(auth_dependency)])
    async def explain(patient: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
        scored = _score_payload(app.state.pipeline, patient)
        record = jsonable_encoder(scored.iloc[0].to_dict())
        explanation = _compute_feature_explanation(app.state.pipeline, patient, top_k=top_k)
        feature_engineering = _compute_feature_engineering(app.state.pipeline, patient, top_k=max(25, top_k))
        data_scaling = _compute_data_scaling(app.state.pipeline, patient, top_k=max(25, top_k))
        data_encoding = _compute_data_encoding(app.state.pipeline, patient, top_k=max(25, top_k))
        return {
            "input": jsonable_encoder(patient),
            "prediction": record,
            "model_config": getattr(app.state, "latest_model_config", None),
            "model_info": {
                "model_name": app.state.model_name,
                "model_version": app.state.model_version,
                "model_type": type(app.state.pipeline.named_steps["model"]).__name__,
            },
            "conformal_p_value": record.get("conformal_p_value"),
            "conformal_assessment": record.get("conformal_assessment"),
            "explanation": explanation,
            "latent_manifold": _compute_latent_manifold(app.state.pipeline, patient),
            "reconstruction_residual_heatmap": _compute_reconstruction_residual_heatmap(app.state.pipeline, patient),
            "feature_engineering": feature_engineering,
            "data_scaling": data_scaling,
            "data_encoding": data_encoding,
            "risk_score": record.get("risk_score"),
            "risk_category": record.get("risk_category", record.get("risk_level")),
        }

    @app.post("/feature-engineering", dependencies=[Depends(auth_dependency)])
    def feature_engineering(patient: dict[str, Any], top_k: int = 25) -> dict[str, Any]:
        return {
            "input": jsonable_encoder(patient),
            "model_info": {
                "model_name": app.state.model_name,
                "model_version": app.state.model_version,
                "model_type": type(app.state.pipeline.named_steps["model"]).__name__,
            },
            "feature_engineering": _compute_feature_engineering(app.state.pipeline, patient, top_k=top_k),
            "data_scaling": _compute_data_scaling(app.state.pipeline, patient, top_k=top_k),
            "data_encoding": _compute_data_encoding(app.state.pipeline, patient, top_k=top_k),
        }

    @app.post(
        "/explain_file",
        dependencies=[Depends(auth_dependency)],
        openapi_extra=_csv_upload_openapi_example("CSV file for explanation"),
    )
    async def explain_file(file: UploadFile = File(...), top_k: int = 10) -> dict[str, Any]:
        filename = file.filename or ""
        if filename and not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV uploads are supported for /explain_file.")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        try:
            frame = pd.read_csv(io.BytesIO(contents))
        except Exception as exc:  # pragma: no cover - handled by API response
            raise HTTPException(status_code=400, detail=f"Unable to parse CSV upload: {exc}") from exc

        result = _explain_most_anomalous_record(app.state.pipeline, frame, top_k=top_k)
        result["filename"] = filename
        return result

    @app.post("/explain_batch", dependencies=[Depends(auth_dependency)])
    async def explain_batch(
        patients: list[dict[str, Any]],
        top_k: int = 10,
    ) -> dict[str, Any]:
        if not patients:
            raise HTTPException(status_code=400, detail="At least one patient record is required.")
        result = _explain_batch_records(app.state.pipeline, patients, top_k=top_k)
        for item, patient in zip(result["results"], patients, strict=False):
            item["feature_engineering"] = _compute_feature_engineering(app.state.pipeline, patient, top_k=max(25, top_k))
            item["data_scaling"] = _compute_data_scaling(app.state.pipeline, patient, top_k=max(25, top_k))
            item["data_encoding"] = _compute_data_encoding(app.state.pipeline, patient, top_k=max(25, top_k))
        return result

    @app.post("/feedback", dependencies=[Depends(auth_dependency)])
    async def feedback(review: dict[str, Any]) -> dict[str, Any]:
        store_path = Path(app.state.feedback_store_path)
        count = append_feedback_records(store_path, [review])
        return {
            "count": count,
            "path": str(store_path),
            "message": "Feedback recorded for periodic retraining.",
        }

    @app.post("/api/clinician-feedback", dependencies=[Depends(auth_dependency)])
    async def clinician_feedback_alias(review: dict[str, Any]) -> dict[str, Any]:
        return await feedback(review)

    @app.post("/feedback_batch", dependencies=[Depends(auth_dependency)])
    async def feedback_batch(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        if not reviews:
            raise HTTPException(status_code=400, detail="At least one feedback record is required.")
        store_path = Path(app.state.feedback_store_path)
        count = append_feedback_records(store_path, reviews)
        return {
            "count": count,
            "path": str(store_path),
            "message": "Feedback batch recorded for periodic retraining.",
        }

    frontend_dist = Path(os.getenv("FRONTEND_DIST_DIR", "/app/web/dist"))
    frontend_index = frontend_dist / "index.html"
    if frontend_index.exists():
        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(frontend_index)

        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the rural health anomaly FastAPI backend.")
    parser.add_argument("--model", required=True, help="Path to a saved anomaly pipeline (.joblib).")
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Optional shared secret required in the X-API-Token header for every request.",
    )
    parser.add_argument(
        "--feedback-store",
        default=None,
        help="Optional JSONL path to append clinician feedback records for periodic retraining.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind the API server to.")
    parser.add_argument("--port", type=int, default=8001, help="TCP port to serve the API on.")
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False, help="Enable Uvicorn auto-reload.")
    args = parser.parse_args()

    auth_token = args.auth_token or os.getenv("API_AUTH_TOKEN")
    app = create_app(args.model, auth_token=auth_token, feedback_store=args.feedback_store)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
