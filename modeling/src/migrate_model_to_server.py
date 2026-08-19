"""
Migrate the locally-trained production model onto a remote MLflow tracking
server (e.g. the Dockerized one in compose.yml).

WHY THIS EXISTS:
register_model.py registers `runs:/<run_id>/model`, which only works when that
run lives in the tracking store you're pointing at. The winning run was created
in the LOCAL file store (mlruns/), so it does NOT exist in the fresh Dockerized
server's SQLite store. This script bridges that gap: it loads the already-
trained model from the local artifacts and re-logs it into a new run ON THE
SERVER (uploading the artifacts to the server's MinIO bucket via the
--serve-artifacts proxy), then registers it and sets the `staging` alias.

Run once after bringing up the Dockerized MLflow server:
    MLFLOW_TRACKING_URI=http://localhost:5000 python modeling/src/migrate_model_to_server.py
"""

import os

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# The local artifacts of the winning run (see register_model.py WINNING_RUN_ID
# = 7742eed4...; its registered model artifact dir is m-623fad3...).
LOCAL_MODEL_PATH = "mlruns/1/models/m-623fad3337d14c7fb7f6d5d9a9b69c0b/artifacts"

MODEL_NAME = "5g-nidd-attack-classifier"
EXPERIMENT_NAME = "5g-nidd-attack-classification"
SERVER_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def migrate():
    # 1. Load the trained model from the LOCAL store (standalone, no server).
    print(f"Loading local model from {LOCAL_MODEL_PATH} ...")
    model = mlflow.xgboost.load_model(LOCAL_MODEL_PATH)
    print(f"  loaded {type(model).__name__}: {model.n_classes_} classes, "
          f"{model.n_features_in_} features")

    # 2. Point at the server and re-log the model inside a fresh run. With the
    #    server running --serve-artifacts, the client uploads through the server
    #    to its MinIO bucket -- no local S3 credentials needed here.
    mlflow.set_tracking_uri(SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Re-logging model to server {SERVER_URI} ...")

    with mlflow.start_run(run_name="migrated_production_model") as run:
        mlflow.set_tag("migrated_from_local_run", "7742eed4fa9d45a1a43d0d09f686e1db")
        mlflow.xgboost.log_model(model, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"  registered {MODEL_NAME} as version {result.version}")

    # 3. Point the `staging` alias (what the API loads) at this version.
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "staging", result.version)
    print(f"  alias 'staging' -> version {result.version}")

    client.update_registered_model(
        name=MODEL_NAME,
        description=(
            "XGBoost multiclass classifier for 5G-NIDD attack detection. "
            "9 classes: Benign + 8 attack types. Migrated from the local "
            "training run 7742eed4 onto the Dockerized MLflow server."
        ),
    )

    print("\nDone. Load it with:")
    print(f"  mlflow.xgboost.load_model('models:/{MODEL_NAME}@staging')")


if __name__ == "__main__":
    migrate()
