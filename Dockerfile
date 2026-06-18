FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY rural_health_anomaly /app/rural_health_anomaly
COPY anomaly_cli.py backend_server.py dashboard_server.py example_training_inference.py preprocessing_pipeline.py train_pipeline.py /app/
COPY artifacts /app/artifacts

RUN pip install --no-cache-dir .

EXPOSE 8001

# Use the bundled demo model so the container can start without a mounted volume.
CMD ["anomaly-api", "--model", "/app/artifacts/web-demo-model.joblib", "--host", "0.0.0.0", "--port", "8001"]
