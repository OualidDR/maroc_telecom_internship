# The official MLflow image does NOT include boto3, so the tracking server
# can't write artifacts to MinIO (S3) -- every artifact upload 500s with
# "ModuleNotFoundError: No module named 'boto3'". This thin layer adds it.
FROM ghcr.io/mlflow/mlflow:v3.14.0

RUN pip install --no-cache-dir boto3==1.43.67
