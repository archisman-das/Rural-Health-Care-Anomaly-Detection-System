"""Backward-compatible redirect to the package training entry point."""

from rural_health_anomaly.cli import train_main as main


if __name__ == "__main__":
    main()
