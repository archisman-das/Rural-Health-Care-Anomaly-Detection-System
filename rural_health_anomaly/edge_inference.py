"""Offline inference for exported ONNX edge bundles."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .training import _clinical_risk_component, _risk_category_from_score, load_tabular_data


@dataclass(frozen=True)
class EdgeBundle:
    manifest: dict[str, Any]
    preprocessor: Any
    sessions: dict[str, Any]
    python_models: dict[str, Any]


def _require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised only when deps are missing
        raise RuntimeError(
            "onnxruntime is required for edge inference. Install the project dependencies first."
        ) from exc
    return ort


def load_edge_bundle(bundle_dir: str | Path) -> EdgeBundle:
    """Load the exported bundle from disk."""

    ort = _require_onnxruntime()
    bundle_path = Path(bundle_dir)
    manifest = json.loads((bundle_path / "edge_bundle_manifest.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(bundle_path / "preprocessor.joblib")

    sessions: dict[str, Any] = {}
    python_models: dict[str, Any] = {}
    for name, artifact in manifest.get("artifacts", {}).items():
        onnx_name = artifact.get("onnx")
        joblib_name = artifact.get("joblib")
        if onnx_name:
            sessions[name] = ort.InferenceSession(
                str(bundle_path / onnx_name),
                providers=["CPUExecutionProvider"],
            )
            continue
        if joblib_name:
            python_models[name] = joblib.load(bundle_path / joblib_name)

    return EdgeBundle(manifest=manifest, preprocessor=preprocessor, sessions=sessions, python_models=python_models)


def _to_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _select_output_array(session: Any, raw_outputs: list[np.ndarray], preferred_names: list[str]) -> np.ndarray:
    output_names = [output.name for output in session.get_outputs()]
    output_map = {name: np.asarray(value) for name, value in zip(output_names, raw_outputs)}

    for preferred in preferred_names:
        for name, value in output_map.items():
            if preferred in name.lower():
                return np.asarray(value)

    first_value = np.asarray(raw_outputs[0])
    if first_value.ndim == 2 and first_value.shape[1] > 1:
        return first_value[:, -1]
    return first_value


def _extract_raw_component_score(session: Any, transformed: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    outputs = session.run(None, {session.get_inputs()[0].name: transformed})
    raw_score = _select_output_array(
        session,
        outputs,
        preferred_names=["reconstruction_error", "latent_distance", "scores", "score", "decision"],
    )
    raw_score = np.asarray(raw_score, dtype=float).reshape(-1)

    mode = artifact.get("raw_score_mode", "raw_output")
    if mode == "negative_decision_function":
        raw_score = -raw_score

    raw_mean = float(artifact.get("raw_score_mean", 0.0))
    raw_std = float(artifact.get("raw_score_std", 1.0))
    if not np.isfinite(raw_std) or raw_std == 0.0:
        raw_std = 1.0
    normalized = (raw_score - raw_mean) / raw_std
    return normalized


def _minmax_scale(values: np.ndarray, *, minimum: float, maximum: float) -> np.ndarray:
    scale = maximum - minimum
    if not np.isfinite(scale) or scale == 0.0:
        return np.full_like(values, 0.5, dtype=float)
    return (values - minimum) / scale


def _fuse_scores(
    matrix: np.ndarray,
    manifest: dict[str, Any],
    component_names: list[str],
    *,
    transformed: np.ndarray | None = None,
    gate_model: Any | None = None,
    stacking_meta_model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    strategy = manifest.get("fusion_strategy", "weighted_average")
    weights_map = {str(key): float(value) for key, value in manifest.get("fusion_weights", {}).items()}
    threshold = float(manifest.get("decision_threshold", 0.5))

    if strategy == "weighted_average":
        if not weights_map:
            weights = np.full(matrix.shape[1], 1.0 / max(matrix.shape[1], 1), dtype=float)
        else:
            weights = np.array([weights_map.get(name, 0.0) for name in component_names], dtype=float)
            total = float(weights.sum())
            if total <= 0.0 or not np.isfinite(total):
                weights = np.full(matrix.shape[1], 1.0 / max(matrix.shape[1], 1), dtype=float)
            else:
                weights = weights / total
        fused = matrix @ weights
    elif strategy == "max_score_voting":
        fused = np.max(matrix, axis=1)
    elif strategy == "stacking":
        stacking = manifest.get("stacking_meta_model") or {}
        feature_order = list(stacking.get("feature_order", component_names))
        feature_index_map = {name: idx for idx, name in enumerate(component_names)}
        if not feature_order:
            raise RuntimeError("Stacking metadata is missing component feature ordering.")
        feature_columns: list[np.ndarray] = []
        for name in feature_order:
            if name in feature_index_map:
                feature_columns.append(matrix[:, feature_index_map[name]])
            else:
                feature_columns.append(np.full(matrix.shape[0], 0.5, dtype=float))
        features = np.column_stack(feature_columns)
        expected_features = getattr(stacking_meta_model, "n_features_in_", None)
        if expected_features is not None and int(expected_features) != features.shape[1]:
            expected_features = int(expected_features)
            if features.shape[1] > expected_features:
                features = features[:, :expected_features]
            else:
                pad_width = expected_features - features.shape[1]
                if pad_width > 0:
                    padding = np.full((features.shape[0], pad_width), 0.5, dtype=float)
                    features = np.column_stack([features, padding])
        if stacking_meta_model is not None and hasattr(stacking_meta_model, "predict_proba"):
            fused = np.asarray(stacking_meta_model.predict_proba(features), dtype=float)[:, 1]
        else:
            coef = np.asarray(stacking.get("coef", [[0.0] * features.shape[1]]), dtype=float)
            intercept = np.asarray(stacking.get("intercept", [0.0]), dtype=float)
            logit = features @ coef.reshape(-1, 1)
            logit = logit.reshape(-1) + float(intercept.reshape(-1)[0])
            fused = 1.0 / (1.0 + np.exp(-logit))
    elif strategy == "moe":
        if gate_model is not None and transformed is not None:
            gate_weights = np.asarray(gate_model.predict_proba(transformed), dtype=float)
            if gate_weights.ndim != 2 or gate_weights.shape[1] != matrix.shape[1]:
                raise RuntimeError("MoE gate produced incompatible routing weights.")
        else:
            gate_weights = np.full_like(matrix, 1.0 / max(matrix.shape[1], 1), dtype=float)
        fused = np.sum(matrix * gate_weights, axis=1)
    else:
        raise RuntimeError(f"Unsupported fusion strategy '{strategy}'.")

    decision_margin = threshold - fused
    return fused, decision_margin


def score_bundle_frame(bundle: EdgeBundle, data: pd.DataFrame) -> pd.DataFrame:
    """Score a dataframe using the exported ONNX bundle."""

    if data.empty:
        return data.copy()

    transformed = bundle.preprocessor.transform(data)
    transformed_array = _to_2d_float_array(transformed)

    component_scores: dict[str, np.ndarray] = {}
    for name in bundle.manifest.get("component_names", []):
        session = bundle.sessions.get(name)
        artifact = bundle.manifest.get("artifacts", {}).get(name, {})
        if session is not None:
            raw_component_score = _extract_raw_component_score(session, transformed_array, artifact)
        else:
            python_model = bundle.python_models.get(name)
            if python_model is None or not hasattr(python_model, "score"):
                continue
            raw_component_score = np.asarray(python_model.score(transformed_array), dtype=float).reshape(-1)
            mode = artifact.get("raw_score_mode", "raw_output")
            if mode == "negative_decision_function":
                raw_component_score = -raw_component_score
            raw_mean = float(artifact.get("raw_score_mean", 0.0))
            raw_std = float(artifact.get("raw_score_std", 1.0))
            if not np.isfinite(raw_std) or raw_std == 0.0:
                raw_std = 1.0
            raw_component_score = (raw_component_score - raw_mean) / raw_std
        stats = bundle.manifest.get("component_stats", {}).get(name, {})
        normalized_component_score = _minmax_scale(
            raw_component_score,
            minimum=float(stats.get("min", 0.0)),
            maximum=float(stats.get("max", 1.0)),
        )
        component_scores[name] = normalized_component_score

    if not component_scores:
        raise RuntimeError("No ONNX component sessions were loaded from the bundle.")

    component_names = [name for name in bundle.manifest.get("component_names", []) if name in component_scores]
    component_matrix = np.column_stack([component_scores[name] for name in component_names])
    fused_score, decision_margin = _fuse_scores(
        component_matrix,
        bundle.manifest,
        component_names,
        transformed=transformed_array,
        gate_model=bundle.python_models.get("moe_gate"),
        stacking_meta_model=bundle.python_models.get("stacking_meta_model"),
    )

    output = data.copy().reset_index(drop=True)
    for name in component_names:
        output[f"{name}_anomaly_score"] = component_scores[name]
    moe_gate = bundle.python_models.get("moe_gate")
    if moe_gate is not None and hasattr(moe_gate, "predict_proba"):
        gate_weights = np.asarray(moe_gate.predict_proba(transformed_array), dtype=float)
        if gate_weights.ndim == 2 and gate_weights.shape[1] == len(component_names):
            for idx, name in enumerate(component_names):
                output[f"{name}_gate_weight"] = gate_weights[:, idx]

    output["raw_anomaly_score"] = fused_score
    output["anomaly_score"] = fused_score
    risk_scoring_weights = bundle.manifest.get("risk_scoring_weights", {})
    output["clinical_risk_score"] = output.apply(
        lambda row: _clinical_risk_component(row, anomaly_score=float(row["anomaly_score"]), weights=risk_scoring_weights),
        axis=1,
    )
    output["risk_score"] = output["clinical_risk_score"]
    output["risk_category"] = output["risk_score"].apply(lambda value: _risk_category_from_score(float(value) / 100.0))
    output["risk_level"] = output["risk_category"]
    output["alert_triggered"] = output["risk_category"].isin(["High", "Critical"])
    output["decision_margin"] = decision_margin
    output["anomaly_flag"] = np.where(decision_margin >= 0.0, 1, -1)
    output["is_anomaly"] = output["anomaly_flag"] == -1
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline inference from an exported edge bundle, including autoencoder, Anomaly Transformer, GANomaly, VAE, CNN autoencoder, and Deep SVDD artifacts."
    )
    parser.add_argument("--bundle-dir", required=True, help="Directory containing the exported ONNX bundle.")
    parser.add_argument("--input", required=True, help="Input data (.csv or .parquet).")
    parser.add_argument("--output", required=True, help="Path to write the scored CSV output.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = time.perf_counter()
    bundle = load_edge_bundle(args.bundle_dir)
    data = load_tabular_data(args.input)
    scored = score_bundle_frame(bundle, data)
    elapsed = time.perf_counter() - start
    scored["inference_batch_latency_ms"] = elapsed * 1000.0
    scored["inference_latency_ms_per_patient"] = (elapsed / max(len(scored), 1)) * 1000.0
    scored["inference_throughput_rows_per_second"] = float(len(scored) / elapsed) if elapsed > 0 else float("inf")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Scored edge predictions saved to {args.output}")


if __name__ == "__main__":
    main()
