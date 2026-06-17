FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY rural_health_anomaly /app/rural_health_anomaly
COPY anomaly_cli.py backend_server.py dashboard_server.py example_training_inference.py preprocessing_pipeline.py train_pipeline.py /app/

RUN pip install --no-cache-dir .

EXPOSE 8001

# Mount a trained model at /models/model.joblib when running the container.
CMD ["anomaly-api", "--model", "/models/model.joblib", "--host", "0.0.0.0", "--port", "8001"]
