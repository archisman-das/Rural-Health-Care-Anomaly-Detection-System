"""Backward-compatible redirect to the FastAPI backend entry point."""

from rural_health_anomaly.backend import main


if __name__ == "__main__":
    main()
