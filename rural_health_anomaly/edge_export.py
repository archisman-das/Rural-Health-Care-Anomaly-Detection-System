"""Export trained anomaly models to offline-friendly edge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .training import load_pipeline


def _require_onnx_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only when deps are missing
        raise RuntimeError(
            "onnx is required for edge export. Install the package with the project extras."
        ) from exc
    return onnx, TensorProto, helper, numpy_helper


def _require_skl2onnx() -> Any:
    try:
        from skl2onnx import to_onnx
    except ImportError as exc:  # pragma: no cover - exercised only when deps are missing
        raise RuntimeError(
            "skl2onnx is required to export the Isolation Forest component."
        ) from exc
    return to_onnx


def _as_float32(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _scalar_initializer(numpy_helper: Any, name: str, value: float) -> Any:
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name=name)


def _safe_scale(value: float) -> float:
    if not np.isfinite(value) or value == 0.0:
        return 1.0
    return float(value)


def _build_mlp_score_graph(
    *,
    input_name: str,
    hidden_layers: list[tuple[np.ndarray, np.ndarray]],
    output_layer: tuple[np.ndarray, np.ndarray],
    score_mean: float,
    score_std: float,
    output_prefix: str,
    onnx: Any,
    helper: Any,
    TensorProto: Any,
    numpy_helper: Any,
    score_kind: str,
    extra_inputs: list[Any] | None = None,
    extra_initializers: list[Any] | None = None,
    extra_nodes: list[Any] | None = None,
    raw_score_name: str = "raw_score",
    score_name: str = "normalized_score",
) -> Any:
    nodes = list(extra_nodes or [])
    initializers = list(extra_initializers or [])
    current = input_name

    for index, (weights, bias) in enumerate(hidden_layers):
        weight_name = f"{output_prefix}_W{index}"
        bias_name = f"{output_prefix}_B{index}"
        hidden_name = f"{output_prefix}_Z{index}"
        current_name = f"{output_prefix}_H{index}"
        initializers.extend(
            [
                numpy_helper.from_array(_as_float32(weights), name=weight_name),
                numpy_helper.from_array(_as_float32(bias), name=bias_name),
            ]
        )
        nodes.append(helper.make_node("Gemm", [current, weight_name, bias_name], [hidden_name], alpha=1.0, beta=1.0))
        nodes.append(helper.make_node("Relu", [hidden_name], [current_name]))
        current = current_name

    weight_name = f"{output_prefix}_W_out"
    bias_name = f"{output_prefix}_B_out"
    output_name = f"{output_prefix}_output"
    initializers.extend(
        [
            numpy_helper.from_array(_as_float32(output_layer[0]), name=weight_name),
            numpy_helper.from_array(_as_float32(output_layer[1]), name=bias_name),
            _scalar_initializer(numpy_helper, f"{output_prefix}_score_mean", score_mean),
            _scalar_initializer(numpy_helper, f"{output_prefix}_score_std", _safe_scale(score_std)),
        ]
    )
    nodes.append(helper.make_node("Gemm", [current, weight_name, bias_name], [output_name], alpha=1.0, beta=1.0))

    if score_kind == "reconstruction_error":
        diff_name = f"{output_prefix}_diff"
        abs_name = f"{output_prefix}_abs"
        raw_name = raw_score_name
        nodes.append(helper.make_node("Sub", [output_name, input_name], [diff_name]))
        nodes.append(helper.make_node("Abs", [diff_name], [abs_name]))
        nodes.append(helper.make_node("ReduceMean", [abs_name], [raw_name], axes=[1], keepdims=0))
    else:
        raise ValueError(f"Unknown score_kind '{score_kind}'.")

    mean_name = f"{output_prefix}_score_mean_out"
    std_name = f"{output_prefix}_score_std_out"
    nodes.append(helper.make_node("Sub", [raw_score_name, f"{output_prefix}_score_mean"], [mean_name]))
    nodes.append(helper.make_node("Div", [mean_name, f"{output_prefix}_score_std"], [score_name]))

    graph_inputs = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", None])]
    if extra_inputs:
        graph_inputs.extend(extra_inputs)
    graph_outputs = [
        helper.make_tensor_value_info(raw_score_name, TensorProto.FLOAT, ["batch"]),
        helper.make_tensor_value_info(score_name, TensorProto.FLOAT, ["batch"]),
    ]

    return helper.make_graph(
        nodes,
        f"{output_prefix}_graph",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )


def _build_autoencoder_model(estimator: Any, *, opset: int, onnx: Any, helper: Any, TensorProto: Any, numpy_helper: Any) -> Any:
    hidden_layers = list(zip(estimator.weights_[:-1], estimator.biases_[:-1]))
    output_layer = (estimator.weights_[-1], estimator.biases_[-1])
    graph = _build_mlp_score_graph(
        input_name="X",
        hidden_layers=hidden_layers,
        output_layer=output_layer,
        score_mean=float(estimator._training_raw_score_mean_),
        score_std=float(estimator._training_raw_score_std_),
        output_prefix="autoencoder",
        onnx=onnx,
        helper=helper,
        TensorProto=TensorProto,
        numpy_helper=numpy_helper,
        score_kind="reconstruction_error",
        raw_score_name="reconstruction_error",
        score_name="normalized_score",
    )
    custom_opset = min(opset, 12)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", custom_opset)])


def _build_variational_autoencoder_model(estimator: Any, *, opset: int, onnx: Any, helper: Any, TensorProto: Any, numpy_helper: Any) -> Any:
    """Export the deterministic reconstruction path of the VAE for edge inference."""

    input_name = "X"
    nodes: list[Any] = []
    initializers: list[Any] = []

    hidden1 = "variational_autoencoder_hidden1"
    hidden1_relu = "variational_autoencoder_hidden1_relu"
    latent_mu = "variational_autoencoder_latent_mu"
    hidden2 = "variational_autoencoder_hidden2"
    hidden2_relu = "variational_autoencoder_hidden2_relu"
    output_name = "variational_autoencoder_output"
    diff_name = "variational_autoencoder_diff"
    abs_name = "variational_autoencoder_abs"
    raw_name = "reconstruction_error"
    score_name = "normalized_score"

    initializers.extend(
        [
            numpy_helper.from_array(_as_float32(estimator.weights_[0]), name="variational_autoencoder_W0"),
            numpy_helper.from_array(_as_float32(estimator.biases_[0]), name="variational_autoencoder_B0"),
            numpy_helper.from_array(_as_float32(estimator.weights_[1]), name="variational_autoencoder_W_mu"),
            numpy_helper.from_array(_as_float32(estimator.biases_[1]), name="variational_autoencoder_B_mu"),
            numpy_helper.from_array(_as_float32(estimator.weights_[3]), name="variational_autoencoder_W2"),
            numpy_helper.from_array(_as_float32(estimator.biases_[3]), name="variational_autoencoder_B2"),
            numpy_helper.from_array(_as_float32(estimator.weights_[4]), name="variational_autoencoder_W_out"),
            numpy_helper.from_array(_as_float32(estimator.biases_[4]), name="variational_autoencoder_B_out"),
            _scalar_initializer(numpy_helper, "variational_autoencoder_score_mean", float(estimator._training_raw_score_mean_)),
            _scalar_initializer(numpy_helper, "variational_autoencoder_score_std", _safe_scale(float(estimator._training_raw_score_std_))),
        ]
    )

    nodes.append(helper.make_node("Gemm", [input_name, "variational_autoencoder_W0", "variational_autoencoder_B0"], [hidden1], alpha=1.0, beta=1.0))
    nodes.append(helper.make_node("Relu", [hidden1], [hidden1_relu]))
    nodes.append(helper.make_node("Gemm", [hidden1_relu, "variational_autoencoder_W_mu", "variational_autoencoder_B_mu"], [latent_mu], alpha=1.0, beta=1.0))
    nodes.append(helper.make_node("Gemm", [latent_mu, "variational_autoencoder_W2", "variational_autoencoder_B2"], [hidden2], alpha=1.0, beta=1.0))
    nodes.append(helper.make_node("Relu", [hidden2], [hidden2_relu]))
    nodes.append(helper.make_node("Gemm", [hidden2_relu, "variational_autoencoder_W_out", "variational_autoencoder_B_out"], [output_name], alpha=1.0, beta=1.0))
    nodes.append(helper.make_node("Sub", [output_name, input_name], [diff_name]))
    nodes.append(helper.make_node("Abs", [diff_name], [abs_name]))
    nodes.append(helper.make_node("ReduceMean", [abs_name], [raw_name], axes=[1], keepdims=0))
    nodes.append(helper.make_node("Sub", [raw_name, "variational_autoencoder_score_mean"], ["variational_autoencoder_centered_score"]))
    nodes.append(helper.make_node("Div", ["variational_autoencoder_centered_score", "variational_autoencoder_score_std"], [score_name]))

    graph = helper.make_graph(
        nodes,
        "variational_autoencoder_graph",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", None])],
        [
            helper.make_tensor_value_info(raw_name, TensorProto.FLOAT, ["batch"]),
            helper.make_tensor_value_info(score_name, TensorProto.FLOAT, ["batch"]),
        ],
        initializer=initializers,
    )
    custom_opset = min(opset, 12)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", custom_opset)])


def _build_deep_svdd_mlp_model(estimator: Any, *, opset: int, onnx: Any, helper: Any, TensorProto: Any, numpy_helper: Any) -> Any:
    input_name = "X"
    nodes: list[Any] = []
    initializers: list[Any] = []
    current = input_name
    for index, (weights, bias) in enumerate(zip(estimator.weights_[:-1], estimator.biases_[:-1])):
        weight_name = f"deep_svdd_mlp_W{index}"
        bias_name = f"deep_svdd_mlp_B{index}"
        hidden_name = f"deep_svdd_mlp_Z{index}"
        output_name = f"deep_svdd_mlp_H{index}"
        initializers.extend(
            [
                numpy_helper.from_array(_as_float32(weights), name=weight_name),
                numpy_helper.from_array(_as_float32(bias), name=bias_name),
            ]
        )
        nodes.append(helper.make_node("Gemm", [current, weight_name, bias_name], [hidden_name], alpha=1.0, beta=1.0))
        nodes.append(helper.make_node("Relu", [hidden_name], [output_name]))
        current = output_name

    final_weight = numpy_helper.from_array(_as_float32(estimator.weights_[-1]), name="deep_svdd_mlp_W_out")
    final_bias = numpy_helper.from_array(_as_float32(estimator.biases_[-1]), name="deep_svdd_mlp_B_out")
    latent_name = "deep_svdd_mlp_latent"
    initializers.extend(
        [
            final_weight,
            final_bias,
            _scalar_initializer(numpy_helper, "deep_svdd_mlp_score_mean", float(estimator._training_raw_score_mean_)),
            _scalar_initializer(numpy_helper, "deep_svdd_mlp_score_std", float(estimator._training_raw_score_std_)),
            _scalar_initializer(numpy_helper, "deep_svdd_mlp_center", 0.0),
        ]
    )
    nodes.append(helper.make_node("Gemm", [current, "deep_svdd_mlp_W_out", "deep_svdd_mlp_B_out"], [latent_name], alpha=1.0, beta=1.0))

    center_name = "deep_svdd_mlp_center_vector"
    initializers[-1] = numpy_helper.from_array(_as_float32(estimator.center_), name=center_name)
    diff_name = "deep_svdd_mlp_diff"
    sq_name = "deep_svdd_mlp_sq"
    raw_name = "latent_distance"
    norm_name = "normalized_score"
    nodes.append(helper.make_node("Sub", [latent_name, center_name], [diff_name]))
    nodes.append(helper.make_node("Mul", [diff_name, diff_name], [sq_name]))
    nodes.append(helper.make_node("ReduceSum", [sq_name], [raw_name], axes=[1], keepdims=0))
    nodes.append(helper.make_node("Sub", [raw_name, "deep_svdd_mlp_score_mean"], ["deep_svdd_mlp_centered_score"]))
    nodes.append(helper.make_node("Div", ["deep_svdd_mlp_centered_score", "deep_svdd_mlp_score_std"], [norm_name]))

    graph = helper.make_graph(
        nodes,
        "deep_svdd_mlp_graph",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", None])],
        [
            helper.make_tensor_value_info(raw_name, TensorProto.FLOAT, ["batch"]),
            helper.make_tensor_value_info(norm_name, TensorProto.FLOAT, ["batch"]),
        ],
        initializer=initializers,
    )
    custom_opset = min(opset, 12)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", custom_opset)])


def _build_deep_svdd_cnn_model(estimator: Any, *, opset: int, onnx: Any, helper: Any, TensorProto: Any, numpy_helper: Any) -> Any:
    input_name = "X"
    nodes: list[Any] = []
    initializers: list[Any] = []

    expanded_name = "deep_svdd_cnn_input"
    nodes.append(helper.make_node("Unsqueeze", [input_name], [expanded_name], axes=[1]))

    initializers.extend(
        [
            numpy_helper.from_array(_as_float32(estimator.conv1_weights_), name="deep_svdd_cnn_W1"),
            numpy_helper.from_array(_as_float32(estimator.conv1_biases_), name="deep_svdd_cnn_B1"),
            numpy_helper.from_array(_as_float32(estimator.conv2_weights_), name="deep_svdd_cnn_W2"),
            numpy_helper.from_array(_as_float32(estimator.conv2_biases_), name="deep_svdd_cnn_B2"),
            numpy_helper.from_array(_as_float32(estimator.dense_weights_), name="deep_svdd_cnn_W3"),
            numpy_helper.from_array(_as_float32(estimator.dense_biases_), name="deep_svdd_cnn_B3"),
            numpy_helper.from_array(_as_float32(estimator.center_), name="deep_svdd_cnn_center"),
            _scalar_initializer(numpy_helper, "deep_svdd_cnn_score_mean", float(estimator._training_raw_score_mean_)),
            _scalar_initializer(numpy_helper, "deep_svdd_cnn_score_std", float(estimator._training_raw_score_std_)),
        ]
    )

    conv1 = "deep_svdd_cnn_conv1"
    relu1 = "deep_svdd_cnn_relu1"
    conv2 = "deep_svdd_cnn_conv2"
    relu2 = "deep_svdd_cnn_relu2"
    pooled = "deep_svdd_cnn_pooled"
    latent = "deep_svdd_cnn_latent"
    diff = "deep_svdd_cnn_diff"
    sq = "deep_svdd_cnn_sq"
    raw = "latent_distance"
    norm = "normalized_score"

    nodes.append(
        helper.make_node(
            "Conv",
            [expanded_name, "deep_svdd_cnn_W1", "deep_svdd_cnn_B1"],
            [conv1],
            pads=[1, 1],
            strides=[1],
        )
    )
    nodes.append(helper.make_node("Relu", [conv1], [relu1]))
    nodes.append(
        helper.make_node(
            "Conv",
            [relu1, "deep_svdd_cnn_W2", "deep_svdd_cnn_B2"],
            [conv2],
            pads=[1, 1],
            strides=[1],
        )
    )
    nodes.append(helper.make_node("Relu", [conv2], [relu2]))
    nodes.append(helper.make_node("ReduceMean", [relu2], [pooled], axes=[2], keepdims=0))
    nodes.append(helper.make_node("Gemm", [pooled, "deep_svdd_cnn_W3", "deep_svdd_cnn_B3"], [latent], alpha=1.0, beta=1.0))
    nodes.append(helper.make_node("Sub", [latent, "deep_svdd_cnn_center"], [diff]))
    nodes.append(helper.make_node("Mul", [diff, diff], [sq]))
    nodes.append(helper.make_node("ReduceSum", [sq], [raw], axes=[1], keepdims=0))
    nodes.append(helper.make_node("Sub", [raw, "deep_svdd_cnn_score_mean"], ["deep_svdd_cnn_centered_score"]))
    nodes.append(helper.make_node("Div", ["deep_svdd_cnn_centered_score", "deep_svdd_cnn_score_std"], [norm]))

    graph = helper.make_graph(
        nodes,
        "deep_svdd_cnn_graph",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", None])],
        [
            helper.make_tensor_value_info(raw, TensorProto.FLOAT, ["batch"]),
            helper.make_tensor_value_info(norm, TensorProto.FLOAT, ["batch"]),
        ],
        initializer=initializers,
    )
    custom_opset = min(opset, 12)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", custom_opset)])


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def export_edge_bundle(pipeline: Any, output_dir: str | Path, *, opset: int = 13) -> dict[str, str]:
    """Export the fitted ensemble to edge-friendly artifacts."""

    onnx, TensorProto, helper, numpy_helper = _require_onnx_runtime()
    to_onnx = _require_skl2onnx()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    preprocessor_path = output_path / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    feature_map = preprocessor.export_feature_map()
    feature_map_csv = output_path / "feature_map.csv"
    feature_map_json = output_path / "feature_map.json"
    feature_map.to_csv(feature_map_csv, index=False)
    feature_map.to_json(feature_map_json, orient="records", indent=2)

    if not hasattr(model, "estimators_"):
        raise RuntimeError("The loaded pipeline must be fit before exporting edge artifacts.")

    feature_names = list(getattr(preprocessor, "get_feature_names_out")())
    input_dim = max(1, len(feature_names))
    sample_input = np.zeros((1, input_dim), dtype=np.float32)

    exported_files: dict[str, str] = {}
    manifest: dict[str, Any] = {
        "component_names": list(getattr(model, "component_names_", [])),
        "fusion_strategy": getattr(model, "fusion_strategy_", getattr(model, "fusion_strategy", None)),
        "decision_threshold": float(getattr(model, "offset_", 0.5)),
        "feature_count": int(sample_input.shape[1]),
        "feature_names": feature_names,
        "component_stats": getattr(model, "component_stats_", {}),
        "fusion_weights": getattr(model, "fusion_weights_", {}),
        "risk_scoring_weights": getattr(pipeline, "risk_scoring_weights_", {}),
        "stacking_meta_model": None,
        "moe_gate": None,
        "artifacts": {},
    }

    if "isolation_forest" in model.estimators_:
        isolation_forest = model.estimators_["isolation_forest"].model_
        isolation_path = output_path / "isolation_forest.onnx"
        onnx_model = to_onnx(
            isolation_forest,
            sample_input[:1],
            target_opset={"": opset, "ai.onnx.ml": 3},
        )
        _save_model(onnx_model, isolation_path)
        exported_files["isolation_forest_onnx"] = str(isolation_path)
        manifest["artifacts"]["isolation_forest"] = {"onnx": isolation_path.name}

    if "autoencoder" in model.estimators_:
        autoencoder = model.estimators_["autoencoder"]
        autoencoder_path = output_path / "autoencoder.onnx"
        autoencoder_model = _build_autoencoder_model(autoencoder, opset=opset, onnx=onnx, helper=helper, TensorProto=TensorProto, numpy_helper=numpy_helper)
        _save_model(autoencoder_model, autoencoder_path)
        exported_files["autoencoder_onnx"] = str(autoencoder_path)
        manifest["artifacts"]["autoencoder"] = {
            "onnx": autoencoder_path.name,
            "threshold": float(getattr(autoencoder, "threshold_", float("nan"))),
        }

    if "anomaly_transformer" in model.estimators_:
        anomaly_transformer = model.estimators_["anomaly_transformer"]
        anomaly_transformer_path = output_path / "anomaly_transformer.joblib"
        joblib.dump(anomaly_transformer, anomaly_transformer_path)
        exported_files["anomaly_transformer_joblib"] = str(anomaly_transformer_path)
        manifest["artifacts"]["anomaly_transformer"] = {
            "joblib": anomaly_transformer_path.name,
            "threshold": float(getattr(anomaly_transformer, "threshold_", float("nan"))),
            "raw_score_mean": float(getattr(anomaly_transformer, "_training_raw_score_mean_", float("nan"))),
            "raw_score_std": float(getattr(anomaly_transformer, "_training_raw_score_std_", float("nan"))),
            "raw_score_mode": "raw_output",
        }

    if "variational_autoencoder" in model.estimators_:
        variational_autoencoder = model.estimators_["variational_autoencoder"]
        variational_autoencoder_path = output_path / "variational_autoencoder.onnx"
        variational_autoencoder_model = _build_variational_autoencoder_model(
            variational_autoencoder,
            opset=opset,
            onnx=onnx,
            helper=helper,
            TensorProto=TensorProto,
            numpy_helper=numpy_helper,
        )
        _save_model(variational_autoencoder_model, variational_autoencoder_path)
        exported_files["variational_autoencoder_onnx"] = str(variational_autoencoder_path)
        manifest["artifacts"]["variational_autoencoder"] = {
            "onnx": variational_autoencoder_path.name,
            "threshold": float(getattr(variational_autoencoder, "threshold_", float("nan"))),
        }

    if "ganomaly" in model.estimators_:
        ganomaly = model.estimators_["ganomaly"]
        ganomaly_path = output_path / "ganomaly.joblib"
        joblib.dump(ganomaly, ganomaly_path)
        exported_files["ganomaly_joblib"] = str(ganomaly_path)
        manifest["artifacts"]["ganomaly"] = {
            "joblib": ganomaly_path.name,
            "threshold": float(getattr(ganomaly, "threshold_", float("nan"))),
            "raw_score_mean": float(getattr(ganomaly, "_training_raw_score_mean_", float("nan"))),
            "raw_score_std": float(getattr(ganomaly, "_training_raw_score_std_", float("nan"))),
            "raw_score_mode": "raw_output",
        }

    if "deep_svdd" in model.estimators_:
        deep_svdd = model.estimators_["deep_svdd"]
        deep_svdd_path = output_path / "deep_svdd.onnx"
        if getattr(deep_svdd, "architecture", "mlp") == "1d_cnn":
            deep_svdd_model = _build_deep_svdd_cnn_model(deep_svdd, opset=opset, onnx=onnx, helper=helper, TensorProto=TensorProto, numpy_helper=numpy_helper)
        else:
            deep_svdd_model = _build_deep_svdd_mlp_model(deep_svdd, opset=opset, onnx=onnx, helper=helper, TensorProto=TensorProto, numpy_helper=numpy_helper)
        _save_model(deep_svdd_model, deep_svdd_path)
        exported_files["deep_svdd_onnx"] = str(deep_svdd_path)
        manifest["artifacts"]["deep_svdd"] = {
            "onnx": deep_svdd_path.name,
            "radius": float(getattr(deep_svdd, "radius_", float("nan"))),
            "architecture": getattr(deep_svdd, "architecture", "mlp"),
        }
        manifest["artifacts"]["deep_svdd"]["raw_score_mean"] = float(getattr(deep_svdd, "_training_raw_score_mean_", float("nan")))
        manifest["artifacts"]["deep_svdd"]["raw_score_std"] = float(getattr(deep_svdd, "_training_raw_score_std_", float("nan")))

    if "isolation_forest" in model.estimators_:
        isolation_model = model.estimators_["isolation_forest"]
        manifest["artifacts"]["isolation_forest"]["raw_score_mean"] = float(getattr(isolation_model, "_training_raw_score_mean_", float("nan")))
        manifest["artifacts"]["isolation_forest"]["raw_score_std"] = float(getattr(isolation_model, "_training_raw_score_std_", float("nan")))
        manifest["artifacts"]["isolation_forest"]["raw_score_mode"] = "negative_decision_function"

    if "autoencoder" in model.estimators_:
        autoencoder = model.estimators_["autoencoder"]
        manifest["artifacts"]["autoencoder"]["raw_score_mean"] = float(getattr(autoencoder, "_training_raw_score_mean_", float("nan")))
        manifest["artifacts"]["autoencoder"]["raw_score_std"] = float(getattr(autoencoder, "_training_raw_score_std_", float("nan")))

    if "anomaly_transformer" in model.estimators_:
        anomaly_transformer = model.estimators_["anomaly_transformer"]
        manifest["artifacts"]["anomaly_transformer"]["raw_score_mean"] = float(
            getattr(anomaly_transformer, "_training_raw_score_mean_", float("nan"))
        )
        manifest["artifacts"]["anomaly_transformer"]["raw_score_std"] = float(
            getattr(anomaly_transformer, "_training_raw_score_std_", float("nan"))
        )

    if "variational_autoencoder" in model.estimators_:
        variational_autoencoder = model.estimators_["variational_autoencoder"]
        manifest["artifacts"]["variational_autoencoder"]["raw_score_mean"] = float(
            getattr(variational_autoencoder, "_training_raw_score_mean_", float("nan"))
        )
        manifest["artifacts"]["variational_autoencoder"]["raw_score_std"] = float(
            getattr(variational_autoencoder, "_training_raw_score_std_", float("nan"))
        )

    if getattr(model, "fusion_strategy_", getattr(model, "fusion_strategy", None)) == "stacking" and hasattr(model, "stacking_meta_model_"):
        meta_model = model.stacking_meta_model_
        stacking_meta_model_path = output_path / "stacking_meta_model.joblib"
        joblib.dump(meta_model, stacking_meta_model_path)
        exported_files["stacking_meta_model_joblib"] = str(stacking_meta_model_path)
        manifest["stacking_meta_model"] = {
            "joblib": stacking_meta_model_path.name,
            "feature_order": list(getattr(model, "component_names_", [])),
            "model_type": type(meta_model).__name__,
            "stacking_meta_model_type": getattr(model, "stacking_meta_model_type_", getattr(model, "stacking_meta_model_type", "mlp")),
        }
        manifest["artifacts"]["stacking_meta_model"] = {
            "joblib": stacking_meta_model_path.name,
            "model_type": type(meta_model).__name__,
        }

    if getattr(model, "fusion_strategy_", getattr(model, "fusion_strategy", None)) == "moe" and hasattr(model, "moe_gate_"):
        moe_gate = model.moe_gate_
        moe_gate_path = output_path / "moe_gate.joblib"
        joblib.dump(moe_gate, moe_gate_path)
        exported_files["moe_gate_joblib"] = str(moe_gate_path)
        manifest["moe_gate"] = {
            "joblib": moe_gate_path.name,
            "training_signal": getattr(moe_gate, "training_signal_", getattr(model, "moe_gate_training_signal_", "unknown")),
            "feature_order": list(getattr(model, "component_names_", [])),
        }
        manifest["artifacts"]["moe_gate"] = {
            "joblib": moe_gate_path.name,
            "training_signal": getattr(moe_gate, "training_signal_", getattr(model, "moe_gate_training_signal_", "unknown")),
        }

    manifest_path = output_path / "edge_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    exported_files["manifest"] = str(manifest_path)
    exported_files["preprocessor"] = str(preprocessor_path)
    exported_files["feature_map_csv"] = str(feature_map_csv)
    exported_files["feature_map_json"] = str(feature_map_json)
    return exported_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the trained anomaly ensemble for edge inference, including autoencoder, Anomaly Transformer, GANomaly, VAE, CNN autoencoder, and Deep SVDD artifacts."
    )
    parser.add_argument("--model", required=True, help="Path to the trained pipeline (.joblib).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the ONNX bundle into.")
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to target for the exported models.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = load_pipeline(args.model)
    exported = export_edge_bundle(pipeline, args.output_dir, opset=args.opset)
    print(json.dumps(exported, indent=2))


if __name__ == "__main__":
    main()
