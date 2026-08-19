# =============================================================================
# 5G-NIDD attack-classifier inference API (FastAPI + MLflow-served XGBoost).
#
# Build from the REPO ROOT (build context = ".") so it can copy contracts/ and
# modeling/ :
#     docker compose build api
# or standalone:
#     docker build -f modeling/serving/Dockerfile -t nidd-api .
#
# At runtime it loads models:/5g-nidd-attack-classifier@staging from the MLflow
# tracking server given by MLFLOW_TRACKING_URI -- so that server must be up AND
# the model must already be registered there (see register_model.py).
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# libgomp1: OpenMP runtime that XGBoost's shared library links against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so this layer is cached unless requirements change.
COPY modeling/serving/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only the code + artifacts the API actually needs at runtime.
#  - contracts/                       : the data contract (schema, sentinel...)
#  - modeling/serving/                : the FastAPI app + pydantic schemas
#  - modeling/monitoring/             : drift.py, imported by the /drift endpoint
#  - modeling/artifacts/splits/       : y_train (label encoder) + X_train (columns)
#  - .../monitoring/reference.parquet : Evidently reference set for /drift
COPY contracts/ ./contracts/
COPY modeling/serving/ ./modeling/serving/
COPY modeling/monitoring/ ./modeling/monitoring/
COPY modeling/artifacts/splits/ ./modeling/artifacts/splits/
COPY modeling/artifacts/monitoring/reference.parquet ./modeling/artifacts/monitoring/reference.parquet

EXPOSE 8000

# api.py resolves SPLITS_DIR relative to the working dir, so run from /app.
CMD ["uvicorn", "modeling.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
