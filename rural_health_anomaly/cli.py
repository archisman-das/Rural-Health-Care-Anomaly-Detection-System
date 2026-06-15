"""CLI entry points for the rural health anomaly package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import PreprocessingConfig
from .edge_export import export_edge_bundle
from .feedback import build_retraining_dataset, load_feedback_ledger
from .evaluation import build_evaluation_report
from .example import build_large_training_data
from .training import (
    load_pipeline,
    load_tabular_data,
    save_pipeline,
    score_records,
    split_tabular_dataset,
    save_dataset_splits,
    train_anomaly_pipeline,
    train_anomaly_pipeline_from_split,
)

_CONFIG_HELP = (
    "Optional JSON file with PreprocessingConfig overrides. "
    "Common keys include autoencoder_latent_dim, autoencoder_threshold_percentile, "
    "autoencoder_dropout, autoencoder_learning_rate, vae_hidden_dim, vae_latent_dim, "
    "vae_beta, anomaly_transformer_hidden_dim, anomaly_transformer_latent_dim, "
    "anomaly_transformer_attention_weight, ganomaly_hidden_dim, ganomaly_latent_dim, "
    "ganomaly_consistency_weight, ensemble_fusion_strategy, ensemble_max_score_threshold, "
    "cnn_autoencoder_weight, anomaly_transformer_weight, ganomaly_weight, vae_weight, "
    "calibrate_threshold, calibration_min_samples, "
    "stacking_meta_model_type, stacking_hidden_layer_sizes, stacking_alpha, stacking_learning_rate_init, "
    "stacking_max_iter, stacking_random_state, stacking_verbose, "
    "moe_gate_hidden_dim, moe_gate_dropout, moe_gate_learning_rate, moe_gate_batch_size, "
    "moe_gate_max_epochs, moe_gate_patience, moe_gate_l2, moe_gate_random_state, moe_gate_verbose, "
    "deep_svdd_nu, and deep_svdd_architecture."
)

_MODEL_HELP = (
    "Model tuning highlights:\n"
    "  Autoencoder:\n"
    "    - autoencoder_latent_dim: latent space size (typically 8 to 16)\n"
    "    - autoencoder_threshold_percentile: validation reconstruction cutoff (typically 95.0 to 99.0)\n"
    "    - autoencoder_dropout: dropout rate for hidden layers (typically 0.1 to 0.3)\n"
    "    - autoencoder_learning_rate: optimizer step size (typically 1e-3 to 1e-4)\n"
    "    - autoencoder_batch_size: minibatch size used during training\n"
    "    - autoencoder_validation_fraction: holdout fraction used for early stopping\n"
    "    - autoencoder_max_epochs: maximum autoencoder training epochs\n"
    "    - autoencoder_patience: early stopping patience in epochs\n"
    "    - autoencoder_l2: L2 regularization strength\n"
    "  Deep SVDD:\n"
    "    - deep_svdd_nu: hypersphere tightness (typically 0.01 to 0.10)\n"
    "    - deep_svdd_center_fixed: keep the center fixed after initialization\n"
    "    - deep_svdd_architecture: encoder type ('mlp' or '1d_cnn')\n"
    "    - deep_svdd_latent_dim: latent size for the encoder\n"
    "    - deep_svdd_learning_rate: optimizer step size (typically 1e-3 to 1e-4)\n"
    "    - deep_svdd_batch_size: minibatch size used during training\n"
    "    - deep_svdd_max_epochs: maximum Deep SVDD training epochs\n"
    "    - deep_svdd_validation_fraction: holdout fraction used for early stopping\n"
    "    - deep_svdd_pretrain_autoencoder: initialize encoder weights from the autoencoder"
    "\n  Anomaly Transformer:\n"
    "    - anomaly_transformer_hidden_dim: hidden width in the encoder and decoder\n"
    "    - anomaly_transformer_latent_dim: latent bottleneck size\n"
    "    - anomaly_transformer_attention_weight: attention discrepancy contribution\n"
    "    - anomaly_transformer_attention_temperature: softmax temperature for feature attention\n"
    "    - anomaly_transformer_threshold_percentile: validation score cutoff percentile\n"
    "    - anomaly_transformer_dropout: dropout rate for transformer layers\n"
    "    - anomaly_transformer_learning_rate: optimizer step size\n"
    "    - anomaly_transformer_batch_size: minibatch size used during training\n"
    "    - anomaly_transformer_validation_fraction: holdout fraction used for early stopping\n"
    "    - anomaly_transformer_max_epochs: maximum training epochs\n"
    "    - anomaly_transformer_patience: early stopping patience in epochs\n"
    "    - anomaly_transformer_l2: L2 regularization strength\n"
    "    - anomaly_transformer_verbose: enable verbose transformer training logs"
    "\n  GANomaly:\n"
    "    - ganomaly_hidden_dim: hidden width in the encoder and decoder\n"
    "    - ganomaly_latent_dim: latent bottleneck size\n"
    "    - ganomaly_consistency_weight: latent consistency penalty weight\n"
    "    - ganomaly_threshold_percentile: validation score cutoff percentile\n"
    "    - ganomaly_dropout: dropout rate for GANomaly layers\n"
    "    - ganomaly_learning_rate: optimizer step size\n"
    "    - ganomaly_batch_size: minibatch size used during training\n"
    "    - ganomaly_validation_fraction: holdout fraction used for early stopping\n"
    "    - ganomaly_max_epochs: maximum GANomaly training epochs\n"
    "    - ganomaly_patience: early stopping patience in epochs\n"
    "    - ganomaly_l2: L2 regularization strength\n"
    "    - ganomaly_verbose: enable verbose GANomaly training logs"
    "\n  Variational Autoencoder:\n"
    "    - vae_hidden_dim: encoder/decoder hidden width (typically 32 to 128)\n"
    "    - vae_latent_dim: latent space size (typically 8 to 16)\n"
    "    - vae_beta: KL-divergence regularization strength\n"
    "    - vae_dropout: dropout rate for hidden layers (typically 0.1 to 0.3)\n"
    "    - vae_learning_rate: optimizer step size (typically 1e-3 to 1e-4)\n"
    "    - vae_batch_size: minibatch size used during training\n"
    "    - vae_validation_fraction: holdout fraction used for early stopping\n"
    "    - vae_max_epochs: maximum VAE training epochs\n"
    "    - vae_patience: early stopping patience in epochs\n"
    "    - vae_l2: L2 regularization strength"
    "\n  Ensemble Fusion:\n"
    "    - ensemble_fusion_strategy: choose 'weighted_average', 'max_score_voting', 'stacking', or 'moe'\n"
    "    - ensemble_max_score_threshold: component score cutoff used by max-score voting\n"
    "    - stacking_meta_model_type: choose 'mlp', 'xgboost', or 'auto' for stacking fusion\n"
    "    - stacking_hidden_layer_sizes: hidden layer layout used by the MLP stacking meta-model\n"
    "    - stacking_alpha: L2 regularization strength for the MLP stacking meta-model\n"
    "    - stacking_learning_rate_init: learning rate used by the MLP stacking meta-model\n"
    "    - stacking_max_iter: maximum training iterations for the stacking meta-model\n"
    "    - stacking_random_state: random seed for the stacking meta-model\n"
    "    - stacking_verbose: enable or disable verbose stacking meta-model training logs\n"
    "    - cnn_autoencoder_weight: optional custom weight for the CNN autoencoder in weighted fusion\n"
    "    - anomaly_transformer_weight: optional custom weight for the Anomaly Transformer in weighted fusion\n"
    "    - ganomaly_weight: optional custom weight for GANomaly in weighted fusion\n"
    "    - vae_weight: optional custom weight for the variational autoencoder in weighted fusion\n"
    "    - calibrate_threshold: enable label-aware threshold calibration during training\n"
    "    - calibration_min_samples: minimum labeled rows required before calibration runs\n"
    "    - moe_gate_hidden_dim: hidden width for the neural routing gate\n"
    "    - moe_gate_dropout: dropout rate used by the gating network\n"
    "    - moe_gate_learning_rate: learning rate for the gating network\n"
    "    - moe_gate_batch_size: batch size used for gate training\n"
    "    - moe_gate_max_epochs: maximum gate training epochs\n"
    "    - moe_gate_patience: early stopping patience for the gate\n"
    "    - moe_gate_l2: L2 regularization strength for the gate\n"
    "    - moe_gate_random_state: random seed for the gate\n"
    "    - moe_gate_verbose: enable or disable verbose gate training logs\n"
    "    - stacking uses a labeled set to train a logistic regression meta-classifier"
)


def _add_autoencoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--autoencoder-latent-dim",
        type=int,
        default=None,
        help="Override the autoencoder latent dimension.",
    )
    parser.add_argument(
        "--autoencoder-dropout",
        type=float,
        default=None,
        help="Override the dropout used in the autoencoder.",
    )
    parser.add_argument(
        "--autoencoder-learning-rate",
        type=float,
        default=None,
        help="Override the autoencoder learning rate.",
    )
    parser.add_argument(
        "--autoencoder-batch-size",
        type=int,
        default=None,
        help="Override the autoencoder batch size.",
    )
    parser.add_argument(
        "--autoencoder-threshold-percentile",
        type=float,
        default=None,
        help="Override the autoencoder reconstruction threshold percentile.",
    )
    parser.add_argument(
        "--autoencoder-validation-fraction",
        type=float,
        default=None,
        help="Override the autoencoder validation split.",
    )
    parser.add_argument(
        "--autoencoder-max-epochs",
        type=int,
        default=None,
        help="Override the autoencoder training epoch budget.",
    )
    parser.add_argument(
        "--autoencoder-patience",
        type=int,
        default=None,
        help="Override the autoencoder early stopping patience.",
    )
    parser.add_argument(
        "--autoencoder-l2",
        type=float,
        default=None,
        help="Override the autoencoder L2 regularization strength.",
    )
    parser.add_argument(
        "--autoencoder-random-state",
        type=int,
        default=None,
        help="Override the autoencoder random seed.",
    )
    parser.add_argument(
        "--autoencoder-verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable verbose autoencoder training logs.",
    )


def _apply_autoencoder_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    field_names = [
        "autoencoder_latent_dim",
        "autoencoder_dropout",
        "autoencoder_learning_rate",
        "autoencoder_batch_size",
        "autoencoder_threshold_percentile",
        "autoencoder_validation_fraction",
        "autoencoder_max_epochs",
        "autoencoder_patience",
        "autoencoder_l2",
        "autoencoder_random_state",
        "autoencoder_verbose",
    ]
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _add_anomaly_transformer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--anomaly-transformer-hidden-dim",
        type=int,
        default=None,
        help="Override the Anomaly Transformer hidden layer width.",
    )
    parser.add_argument(
        "--anomaly-transformer-latent-dim",
        type=int,
        default=None,
        help="Override the Anomaly Transformer latent dimension.",
    )
    parser.add_argument(
        "--anomaly-transformer-dropout",
        type=float,
        default=None,
        help="Override the dropout used in the Anomaly Transformer.",
    )
    parser.add_argument(
        "--anomaly-transformer-learning-rate",
        type=float,
        default=None,
        help="Override the Anomaly Transformer learning rate.",
    )
    parser.add_argument(
        "--anomaly-transformer-batch-size",
        type=int,
        default=None,
        help="Override the Anomaly Transformer batch size.",
    )
    parser.add_argument(
        "--anomaly-transformer-attention-weight",
        type=float,
        default=None,
        help="Override the attention discrepancy weight.",
    )
    parser.add_argument(
        "--anomaly-transformer-attention-temperature",
        type=float,
        default=None,
        help="Override the attention softmax temperature.",
    )
    parser.add_argument(
        "--anomaly-transformer-threshold-percentile",
        type=float,
        default=None,
        help="Override the Anomaly Transformer reconstruction threshold percentile.",
    )
    parser.add_argument(
        "--anomaly-transformer-validation-fraction",
        type=float,
        default=None,
        help="Override the Anomaly Transformer validation split.",
    )
    parser.add_argument(
        "--anomaly-transformer-max-epochs",
        type=int,
        default=None,
        help="Override the Anomaly Transformer training epoch budget.",
    )
    parser.add_argument(
        "--anomaly-transformer-patience",
        type=int,
        default=None,
        help="Override the Anomaly Transformer early stopping patience.",
    )
    parser.add_argument(
        "--anomaly-transformer-l2",
        type=float,
        default=None,
        help="Override the Anomaly Transformer L2 regularization strength.",
    )
    parser.add_argument(
        "--anomaly-transformer-random-state",
        type=int,
        default=None,
        help="Override the Anomaly Transformer random seed.",
    )
    parser.add_argument(
        "--anomaly-transformer-verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable verbose Anomaly Transformer training logs.",
    )


def _apply_anomaly_transformer_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    field_names = [
        "anomaly_transformer_hidden_dim",
        "anomaly_transformer_latent_dim",
        "anomaly_transformer_dropout",
        "anomaly_transformer_learning_rate",
        "anomaly_transformer_batch_size",
        "anomaly_transformer_attention_weight",
        "anomaly_transformer_attention_temperature",
        "anomaly_transformer_threshold_percentile",
        "anomaly_transformer_validation_fraction",
        "anomaly_transformer_max_epochs",
        "anomaly_transformer_patience",
        "anomaly_transformer_l2",
        "anomaly_transformer_random_state",
        "anomaly_transformer_verbose",
    ]
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _add_ganomaly_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ganomaly-hidden-dim",
        type=int,
        default=None,
        help="Override the GANomaly hidden layer width.",
    )
    parser.add_argument(
        "--ganomaly-latent-dim",
        type=int,
        default=None,
        help="Override the GANomaly latent dimension.",
    )
    parser.add_argument(
        "--ganomaly-dropout",
        type=float,
        default=None,
        help="Override the dropout used in GANomaly.",
    )
    parser.add_argument(
        "--ganomaly-learning-rate",
        type=float,
        default=None,
        help="Override the GANomaly learning rate.",
    )
    parser.add_argument(
        "--ganomaly-batch-size",
        type=int,
        default=None,
        help="Override the GANomaly batch size.",
    )
    parser.add_argument(
        "--ganomaly-consistency-weight",
        type=float,
        default=None,
        help="Override the GANomaly latent consistency weight.",
    )
    parser.add_argument(
        "--ganomaly-threshold-percentile",
        type=float,
        default=None,
        help="Override the GANomaly reconstruction threshold percentile.",
    )
    parser.add_argument(
        "--ganomaly-validation-fraction",
        type=float,
        default=None,
        help="Override the GANomaly validation split.",
    )
    parser.add_argument(
        "--ganomaly-max-epochs",
        type=int,
        default=None,
        help="Override the GANomaly training epoch budget.",
    )
    parser.add_argument(
        "--ganomaly-patience",
        type=int,
        default=None,
        help="Override the GANomaly early stopping patience.",
    )
    parser.add_argument(
        "--ganomaly-l2",
        type=float,
        default=None,
        help="Override the GANomaly L2 regularization strength.",
    )
    parser.add_argument(
        "--ganomaly-random-state",
        type=int,
        default=None,
        help="Override the GANomaly random seed.",
    )
    parser.add_argument(
        "--ganomaly-verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable verbose GANomaly training logs.",
    )


def _apply_ganomaly_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    field_names = [
        "ganomaly_hidden_dim",
        "ganomaly_latent_dim",
        "ganomaly_dropout",
        "ganomaly_learning_rate",
        "ganomaly_batch_size",
        "ganomaly_consistency_weight",
        "ganomaly_threshold_percentile",
        "ganomaly_validation_fraction",
        "ganomaly_max_epochs",
        "ganomaly_patience",
        "ganomaly_l2",
        "ganomaly_random_state",
        "ganomaly_verbose",
    ]
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _add_variational_autoencoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vae-hidden-dim",
        type=int,
        default=None,
        help="Override the variational autoencoder hidden layer width.",
    )
    parser.add_argument(
        "--vae-latent-dim",
        type=int,
        default=None,
        help="Override the variational autoencoder latent dimension.",
    )
    parser.add_argument(
        "--vae-dropout",
        type=float,
        default=None,
        help="Override the dropout used in the variational autoencoder.",
    )
    parser.add_argument(
        "--vae-learning-rate",
        type=float,
        default=None,
        help="Override the variational autoencoder learning rate.",
    )
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=None,
        help="Override the variational autoencoder batch size.",
    )
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=None,
        help="Override the KL regularization strength used by the variational autoencoder.",
    )
    parser.add_argument(
        "--vae-threshold-percentile",
        type=float,
        default=None,
        help="Override the variational autoencoder reconstruction threshold percentile.",
    )
    parser.add_argument(
        "--vae-validation-fraction",
        type=float,
        default=None,
        help="Override the variational autoencoder validation split.",
    )
    parser.add_argument(
        "--vae-max-epochs",
        type=int,
        default=None,
        help="Override the variational autoencoder training epoch budget.",
    )
    parser.add_argument(
        "--vae-patience",
        type=int,
        default=None,
        help="Override the variational autoencoder early stopping patience.",
    )
    parser.add_argument(
        "--vae-l2",
        type=float,
        default=None,
        help="Override the variational autoencoder L2 regularization strength.",
    )
    parser.add_argument(
        "--vae-random-state",
        type=int,
        default=None,
        help="Override the variational autoencoder random seed.",
    )
    parser.add_argument(
        "--vae-verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable verbose variational autoencoder training logs.",
    )


def _apply_variational_autoencoder_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    field_names = [
        "vae_hidden_dim",
        "vae_latent_dim",
        "vae_dropout",
        "vae_learning_rate",
        "vae_batch_size",
        "vae_beta",
        "vae_threshold_percentile",
        "vae_validation_fraction",
        "vae_max_epochs",
        "vae_patience",
        "vae_l2",
        "vae_random_state",
        "vae_verbose",
    ]
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _add_deep_svdd_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deep-svdd-nu", type=float, default=None, help="Override the Deep SVDD nu value.")
    parser.add_argument(
        "--deep-svdd-center-fixed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the Deep SVDD center fixed after initialization.",
    )
    parser.add_argument(
        "--deep-svdd-architecture",
        choices=("mlp", "1d_cnn"),
        default=None,
        help="Override the Deep SVDD encoder architecture for this run.",
    )
    parser.add_argument(
        "--deep-svdd-latent-dim",
        type=int,
        default=None,
        help="Override the Deep SVDD latent dimension.",
    )
    parser.add_argument(
        "--deep-svdd-learning-rate",
        type=float,
        default=None,
        help="Override the Deep SVDD learning rate.",
    )
    parser.add_argument(
        "--deep-svdd-batch-size",
        type=int,
        default=None,
        help="Override the Deep SVDD batch size.",
    )
    parser.add_argument(
        "--deep-svdd-max-epochs",
        type=int,
        default=None,
        help="Override the Deep SVDD training epoch budget.",
    )
    parser.add_argument(
        "--deep-svdd-validation-fraction",
        type=float,
        default=None,
        help="Override the validation split used by Deep SVDD.",
    )
    parser.add_argument(
        "--deep-svdd-pretrain-autoencoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable autoencoder pretraining for Deep SVDD.",
    )
    parser.add_argument(
        "--deep-svdd-pretrain-epochs",
        type=int,
        default=None,
        help="Override the number of autoencoder pretraining epochs.",
    )
    parser.add_argument(
        "--deep-svdd-pretrain-dropout",
        type=float,
        default=None,
        help="Override the dropout used during autoencoder pretraining.",
    )
    parser.add_argument(
        "--deep-svdd-pretrain-learning-rate",
        type=float,
        default=None,
        help="Override the learning rate used during autoencoder pretraining.",
    )
    parser.add_argument(
        "--deep-svdd-pretrain-batch-size",
        type=int,
        default=None,
        help="Override the batch size used during autoencoder pretraining.",
    )
    parser.add_argument(
        "--deep-svdd-random-state",
        type=int,
        default=None,
        help="Override the Deep SVDD random seed.",
    )
    parser.add_argument(
        "--deep-svdd-verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable verbose Deep SVDD training logs.",
    )


def _apply_deep_svdd_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    field_names = [
        "deep_svdd_nu",
        "deep_svdd_center_fixed",
        "deep_svdd_architecture",
        "deep_svdd_latent_dim",
        "deep_svdd_learning_rate",
        "deep_svdd_batch_size",
        "deep_svdd_max_epochs",
        "deep_svdd_validation_fraction",
        "deep_svdd_pretrain_autoencoder",
        "deep_svdd_pretrain_epochs",
        "deep_svdd_pretrain_dropout",
        "deep_svdd_pretrain_learning_rate",
        "deep_svdd_pretrain_batch_size",
        "deep_svdd_random_state",
        "deep_svdd_verbose",
    ]
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _add_ensemble_fusion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ensemble-fusion-strategy",
        choices=("weighted_average", "max_score_voting", "stacking", "moe"),
        default=None,
        help="Override how component scores are fused into the final ensemble score.",
    )
    parser.add_argument(
        "--ensemble-max-score-threshold",
        type=float,
        default=None,
        help="Override the per-model threshold used by max-score voting.",
    )
    parser.add_argument(
        "--cnn-autoencoder-weight",
        type=float,
        default=None,
        help="Override the CNN autoencoder contribution inside weighted ensemble fusion.",
    )
    parser.add_argument(
        "--anomaly-transformer-weight",
        type=float,
        default=None,
        help="Override the Anomaly Transformer contribution inside weighted ensemble fusion.",
    )
    parser.add_argument(
        "--ganomaly-weight",
        type=float,
        default=None,
        help="Override the GANomaly contribution inside weighted ensemble fusion.",
    )
    parser.add_argument(
        "--vae-weight",
        type=float,
        default=None,
        help="Override the variational autoencoder contribution inside weighted ensemble fusion.",
    )


def _add_label_arguments(parser: argparse.ArgumentParser) -> None:
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


def _add_training_data_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default=None,
        help="Path to training data (.csv or .parquet). Required unless --synthetic-demo-data is set.",
    )
    parser.add_argument(
        "--synthetic-demo-data",
        action="store_true",
        help="Generate a larger synthetic training cohort instead of loading an input file.",
    )
    parser.add_argument(
        "--synthetic-demo-rows",
        type=int,
        default=9600,
        help="Number of rows to generate when --synthetic-demo-data is enabled.",
    )
    parser.add_argument(
        "--synthetic-demo-seed",
        type=int,
        default=42,
        help="Random seed used for synthetic demo data generation.",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Optional directory containing train, validation, and test subfolders with CSV or Parquet data.",
    )


def _add_split_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train, validation, and test folders will be written.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of rows assigned to the train split.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows assigned to the validation split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows assigned to the test split.",
    )
    parser.add_argument(
        "--group-column",
        default="patient_id",
        help="Column used to keep related rows in the same split when available.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when shuffling rows or groups for the split.",
    )


def _add_threshold_calibration_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibrate-threshold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable label-aware threshold calibration during training.",
    )
    parser.add_argument(
        "--calibration-min-samples",
        type=int,
        default=None,
        help="Minimum labeled rows required before threshold calibration runs.",
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


def _add_executive_summary_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--executive-summary",
        dest="executive_summary",
        action="store_const",
        const=True,
        default=None,
        help="Emit a short comparison-only report with the model table, best model, and ensemble score.",
    )
    group.add_argument(
        "--no-executive-summary",
        dest="executive_summary",
        action="store_const",
        const=False,
        help="Force the full report sections even when --report-prefix is set.",
    )


def _load_labels_from_args(args: argparse.Namespace, data: pd.DataFrame | None = None) -> pd.Series | None:
    labels = None

    if getattr(args, "labels_file", None):
        labels_frame = load_tabular_data(args.labels_file)
        if args.labels_column:
            if args.labels_column not in labels_frame.columns:
                raise ValueError(f"Labels column '{args.labels_column}' was not found in the labels file.")
            labels = labels_frame[args.labels_column]
        elif labels_frame.shape[1] == 1:
            labels = labels_frame.iloc[:, 0]
        else:
            raise ValueError(
                "labels-file contains more than one column; pass --labels-column to choose the label field."
            )

    if getattr(args, "label_column", None):
        if data is None:
            raise ValueError("A scored input frame is required when --label-column is used.")
        if args.label_column not in data.columns:
            raise ValueError(f"Label column '{args.label_column}' was not found in the scored input.")
        labels = data[args.label_column]

    return labels


def _infer_score_columns(scored: pd.DataFrame) -> list[str]:
    score_columns = [
        column
        for column in scored.columns
        if any(token in column.lower() for token in ("score", "error", "distance", "margin"))
    ]
    return score_columns


def _derive_report_paths(report_prefix: str | None, args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    if not report_prefix:
        return args.output, args.report_md, args.report_html

    prefix_path = Path(report_prefix)
    derived_json = str(prefix_path.parent / f"{prefix_path.name}.json")
    derived_md = str(prefix_path.parent / f"{prefix_path.name}.md")
    derived_html = str(prefix_path.parent / f"{prefix_path.name}.html")
    return (
        args.output or derived_json,
        args.report_md or derived_md,
        args.report_html or derived_html,
    )


def _apply_ensemble_fusion_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    field_names = [
        "ensemble_fusion_strategy",
        "ensemble_max_score_threshold",
        "cnn_autoencoder_weight",
        "anomaly_transformer_weight",
        "ganomaly_weight",
        "vae_weight",
    ]
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _apply_threshold_calibration_overrides(config: PreprocessingConfig, args: argparse.Namespace) -> None:
    if getattr(args, "calibrate_threshold", None) is not None:
        config.calibrate_threshold = bool(args.calibrate_threshold)
    if getattr(args, "calibration_min_samples", None) is not None:
        config.calibration_min_samples = int(args.calibration_min_samples)


def load_config(path: str | None) -> PreprocessingConfig:
    if not path:
        return PreprocessingConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        overrides = json.load(f)

    return PreprocessingConfig(**overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run the rural health anomaly detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MODEL_HELP,
        conflict_handler="resolve",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from tabular data with Isolation Forest, SVM, LOF, autoencoder, Anomaly Transformer, GANomaly, VAE, CNN autoencoder, and Deep SVDD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MODEL_HELP,
        conflict_handler="resolve",
    )
    _add_training_data_source_arguments(train_parser)
    _add_threshold_calibration_argument(train_parser)
    train_parser.add_argument("--output", required=True, help="Path to save the trained pipeline (.joblib).")
    train_parser.add_argument(
        "--feature-map",
        default=None,
        help="Optional path to save the exported feature map as CSV.",
    )
    train_parser.add_argument(
        "--config-json",
        default=None,
        help=_CONFIG_HELP,
    )
    train_parser.add_argument(
        "--label-column",
        default=None,
        help="Optional column in the training file containing binary labels for stacking.",
    )
    train_parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional separate CSV or Parquet file containing binary labels for stacking.",
    )
    train_parser.add_argument(
        "--labels-column",
        default=None,
        help="Optional column name to read from --labels-file when it contains more than one column.",
    )
    _add_autoencoder_arguments(train_parser)
    _add_anomaly_transformer_arguments(train_parser)
    _add_ganomaly_arguments(train_parser)
    _add_variational_autoencoder_arguments(train_parser)
    _add_deep_svdd_arguments(train_parser)
    _add_ensemble_fusion_arguments(train_parser)

    split_parser = subparsers.add_parser(
        "split-data",
        help="Split a raw dataset into train, validation, and test folders.",
    )
    split_parser.add_argument("--input", required=True, help="Path to the raw CSV or Parquet dataset.")
    _add_split_dataset_arguments(split_parser)

    predict_parser = subparsers.add_parser("predict", help="Run inference from a saved model.")
    predict_parser.add_argument("--model", required=True, help="Path to a saved pipeline (.joblib).")
    predict_parser.add_argument("--input", required=True, help="Path to inference data (.csv or .parquet).")
    predict_parser.add_argument("--output", required=True, help="Path to save scored predictions (.csv).")

    export_edge_parser = subparsers.add_parser(
        "export-edge",
        help="Export the trained ensemble to ONNX artifacts for offline edge inference, including autoencoder, VAE, CNN autoencoder, and Deep SVDD artifacts.",
    )
    export_edge_parser.add_argument("--model", required=True, help="Path to a saved pipeline (.joblib).")
    export_edge_parser.add_argument("--output-dir", required=True, help="Directory to write the edge bundle.")
    export_edge_parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to target for the exported models.",
    )

    retrain_feedback_parser = subparsers.add_parser(
        "retrain-feedback",
        help="Retrain the ensemble from a raw dataset plus clinician feedback labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MODEL_HELP,
    )
    retrain_feedback_parser.add_argument("--input", required=True, help="Path to the raw base training data (.csv or .parquet).")
    retrain_feedback_parser.add_argument("--feedback-file", required=True, help="Path to the clinician feedback JSONL ledger.")
    retrain_feedback_parser.add_argument("--output", required=True, help="Path to save the retrained pipeline (.joblib).")
    retrain_feedback_parser.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON file with PreprocessingConfig overrides used for retraining.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate labeled anomaly scores and compare score columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MODEL_HELP,
    )
    evaluate_parser.add_argument(
        "--input",
        required=True,
        help="Path to scored predictions (.csv or .parquet).",
    )
    evaluate_parser.add_argument(
        "--score-column",
        default="anomaly_score",
        help="Primary anomaly score column to evaluate.",
    )
    evaluate_parser.add_argument(
        "--score-columns",
        nargs="+",
        default=None,
        help="Optional list of score columns to compare instead of just one column.",
    )
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold used to convert anomaly scores into predicted labels.",
    )
    evaluate_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the metrics as JSON.",
    )
    evaluate_parser.add_argument(
        "--report-prefix",
        default=None,
        help="Optional base path that defaults the JSON, Markdown, and HTML report names. Markdown and HTML outputs use the executive summary unless you pass --no-executive-summary.",
    )
    evaluate_parser.add_argument(
        "--report-md",
        default=None,
        help="Optional path to save a human-readable Markdown report. Markdown outputs use the executive summary unless you pass --no-executive-summary.",
    )
    evaluate_parser.add_argument(
        "--report-html",
        default=None,
        help="Optional path to save a browser-friendly HTML report. HTML outputs use the executive summary unless you pass --no-executive-summary.",
    )
    evaluate_parser.add_argument(
        "--dashboard-html",
        default=None,
        help="Optional path to save the full dashboard HTML report with metrics, score distributions, agreement analysis, and runtime comparison.",
    )
    evaluate_parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Fraction of the highest and lowest scores used for unsupervised spread summaries.",
    )
    _add_executive_summary_arguments(evaluate_parser)
    _add_label_arguments(evaluate_parser)

    return parser


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args.config_json)
    _apply_autoencoder_overrides(config, args)
    _apply_anomaly_transformer_overrides(config, args)
    _apply_ganomaly_overrides(config, args)
    _apply_variational_autoencoder_overrides(config, args)
    _apply_deep_svdd_overrides(config, args)
    _apply_ensemble_fusion_overrides(config, args)
    _apply_threshold_calibration_overrides(config, args)
    if getattr(args, "synthetic_demo_data", False):
        if getattr(args, "input", None):
            print("Synthetic demo data enabled; ignoring --input.")
        data = build_large_training_data(
            target_rows=int(getattr(args, "synthetic_demo_rows", 9600)),
            seed=int(getattr(args, "synthetic_demo_seed", 42)),
        )
        pipeline = train_anomaly_pipeline(data, config=config)
    else:
        if getattr(args, "split_dir", None):
            split_path = Path(args.split_dir)
            if not split_path.exists():
                raise FileNotFoundError(f"Split directory not found: {split_path}")
            pipeline, split_metrics = train_anomaly_pipeline_from_split(
                split_path,
                label_column=getattr(args, "label_column", None),
                config=config,
            )
            if split_metrics:
                print(json.dumps(split_metrics, indent=2))
        else:
            if not getattr(args, "input", None):
                raise ValueError("Training input is required unless --synthetic-demo-data or --split-dir is set.")
            data = load_tabular_data(args.input)
            labels = None
            if getattr(args, "labels_file", None):
                labels_frame = load_tabular_data(args.labels_file)
                if args.labels_column:
                    if args.labels_column not in labels_frame.columns:
                        raise ValueError(
                            f"Labels column '{args.labels_column}' was not found in the labels file."
                        )
                    labels = labels_frame[args.labels_column]
                elif labels_frame.shape[1] == 1:
                    labels = labels_frame.iloc[:, 0]
                else:
                    raise ValueError(
                        "labels-file contains more than one column; pass --labels-column to choose the label field."
                    )
            if getattr(args, "label_column", None):
                if args.label_column not in data.columns:
                    raise ValueError(f"Label column '{args.label_column}' was not found in the training data.")
                labels = data[args.label_column]
                data = data.drop(columns=[args.label_column])
            pipeline = train_anomaly_pipeline(data, y=labels, config=config)
    save_pipeline(pipeline, args.output)

    if args.feature_map:
        feature_map = pipeline.named_steps["preprocessor"].export_feature_map()
        feature_map.to_csv(args.feature_map, index=False)

    print(f"Trained pipeline saved to {args.output}")
    if args.feature_map:
        print(f"Feature map saved to {args.feature_map}")


def run_split_data(args: argparse.Namespace) -> None:
    splits = split_tabular_dataset(
        load_tabular_data(args.input),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        group_column=args.group_column,
        random_state=args.random_state,
    )
    manifest = save_dataset_splits(splits, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "files": manifest}, indent=2))


def run_predict(args: argparse.Namespace) -> None:
    pipeline = load_pipeline(args.model)
    data = load_tabular_data(args.input)
    scored = score_records(pipeline, data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Scored predictions saved to {args.output}")


def run_export_edge(args: argparse.Namespace) -> None:
    pipeline = load_pipeline(args.model)
    exported = export_edge_bundle(pipeline, args.output_dir, opset=args.opset)
    print(json.dumps(exported, indent=2))


def run_retrain_feedback(args: argparse.Namespace) -> None:
    base_data = load_tabular_data(args.input)
    feedback_ledger = load_feedback_ledger(args.feedback_file)
    if feedback_ledger.empty:
        raise ValueError(f"No clinician feedback records were found in {args.feedback_file}.")

    combined_data, labels = build_retraining_dataset(base_data, feedback_ledger)
    config = load_config(args.config_json)
    pipeline = train_anomaly_pipeline(combined_data, y=labels, config=config)
    save_pipeline(pipeline, args.output)
    print(f"Retrained pipeline saved to {args.output}")


def run_evaluate(args: argparse.Namespace) -> None:
    scored = load_tabular_data(args.input)
    labels = _load_labels_from_args(args, scored)

    score_columns = args.score_columns or _infer_score_columns(scored)
    if not score_columns:
        score_columns = [args.score_column]

    auto_selected_executive_summary = getattr(args, "executive_summary", None) is None and bool(
        args.report_prefix or args.report_md or args.report_html
    )
    executive_summary = getattr(args, "executive_summary", None)
    if executive_summary is None:
        executive_summary = bool(args.report_prefix or args.report_md or args.report_html)

    report = build_evaluation_report(
        scored,
        y_true=labels,
        score_columns=score_columns,
        threshold=args.threshold,
        top_fraction=args.top_fraction,
        executive_summary=bool(executive_summary),
    )
    dashboard_report = None
    if getattr(args, "dashboard_html", None):
        dashboard_report = build_evaluation_report(
            scored,
            y_true=labels,
            score_columns=score_columns,
            threshold=args.threshold,
            top_fraction=args.top_fraction,
            executive_summary=False,
        )
    comparison = report["metrics_table"]
    json_output_path, markdown_report_path, html_report_path = _derive_report_paths(args.report_prefix, args)

    comparison = comparison.reset_index(drop=True)
    print(comparison.to_string(index=False))
    if auto_selected_executive_summary:
        print("Executive summary mode enabled for report output.")

    if json_output_path:
        output_path = Path(json_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")
        print(f"Metrics saved to {json_output_path}")

    if markdown_report_path:
        report_path = Path(markdown_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report["summary_markdown"], encoding="utf-8")
        print(f"Markdown report saved to {markdown_report_path}")

    if html_report_path:
        report_path = Path(html_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report["summary_html"], encoding="utf-8")
        print(f"HTML report saved to {html_report_path}")

    if getattr(args, "dashboard_html", None):
        if dashboard_report is None:
            dashboard_report = report
        dashboard_path = Path(args.dashboard_html)
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(dashboard_report["summary_html"], encoding="utf-8")
        print(f"Dashboard report saved to {args.dashboard_html}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
        return

    if args.command == "split-data":
        run_split_data(args)
        return

    if args.command == "predict":
        run_predict(args)
        return

    if args.command == "export-edge":
        run_export_edge(args)
        return

    if args.command == "retrain-feedback":
        run_retrain_feedback(args)
        return

    if args.command == "evaluate":
        run_evaluate(args)
        return

    parser.error("Unknown command.")


def train_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the rural health anomaly detection pipeline with Isolation Forest, SVM, LOF, autoencoder, Anomaly Transformer, GANomaly, VAE, CNN autoencoder, and Deep SVDD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MODEL_HELP,
        conflict_handler="resolve",
    )
    _add_training_data_source_arguments(parser)
    _add_threshold_calibration_argument(parser)
    parser.add_argument("--output", required=True, help="Path to save the trained pipeline (.joblib).")
    parser.add_argument(
        "--feature-map",
        default=None,
        help="Optional path to save the exported feature map as CSV.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help=_CONFIG_HELP,
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional column in the training file containing binary labels for stacking.",
    )
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional separate CSV or Parquet file containing binary labels for stacking.",
    )
    parser.add_argument(
        "--labels-column",
        default=None,
        help="Optional column name to read from --labels-file when it contains more than one column.",
    )
    _add_autoencoder_arguments(parser)
    _add_anomaly_transformer_arguments(parser)
    _add_ganomaly_arguments(parser)
    _add_variational_autoencoder_arguments(parser)
    _add_deep_svdd_arguments(parser)
    _add_ensemble_fusion_arguments(parser)
    run_train(parser.parse_args())


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a saved rural health anomaly pipeline.")
    parser.add_argument("--model", required=True, help="Path to a saved pipeline (.joblib).")
    parser.add_argument("--input", required=True, help="Path to inference data (.csv or .parquet).")
    parser.add_argument("--output", required=True, help="Path to save scored predictions (.csv).")
    run_predict(parser.parse_args())


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate labeled anomaly scores.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to scored predictions (.csv or .parquet).",
    )
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
        "--output",
        default=None,
        help="Optional path to save the metrics as JSON.",
    )
    parser.add_argument(
        "--report-prefix",
        default=None,
        help="Optional base path that defaults the JSON, Markdown, and HTML report names.",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        help="Optional path to save a human-readable Markdown report.",
    )
    parser.add_argument(
        "--report-html",
        default=None,
        help="Optional path to save a browser-friendly HTML report.",
    )
    parser.add_argument(
        "--dashboard-html",
        default=None,
        help="Optional path to save the full dashboard HTML report with metrics, score distributions, agreement analysis, and runtime comparison.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Fraction of the highest and lowest scores used for unsupervised spread summaries.",
    )
    _add_executive_summary_arguments(parser)
    _add_label_arguments(parser)
    run_evaluate(parser.parse_args())
