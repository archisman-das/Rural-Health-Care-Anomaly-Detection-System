"""Backward-compatible wrapper for the package example."""

from rural_health_anomaly.example import (
    build_inference_data,
    build_large_inference_data,
    build_large_training_data,
    build_training_data,
    main,
)

__all__ = [
    "build_training_data",
    "build_inference_data",
    "build_large_training_data",
    "build_large_inference_data",
    "main",
]


if __name__ == "__main__":
    main()
