"""
Register the production model in the MLflow Model Registry.

This script promotes a specific run's logged model to a named, versioned
entry in the registry, then assigns aliases (staging, production) that
downstream code (FastAPI serving, drift monitoring) will use to load the
model without hardcoding a version.

Run this ONCE after you've picked a winning run. Re-running creates a new
version, which is what you want for genuine model updates but not for
re-registration of the same model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "5g-nidd-attack-classifier"

WINNING_RUN_ID = "7742eed4fa9d45a1a43d0d09f686e1db"


def register():
    client = MlflowClient()

    # Register the model artifact from that run
    model_uri = f"runs:/{WINNING_RUN_ID}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Registered {MODEL_NAME} as version {result.version}")

    # Set the "staging" alias to point at this new version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="staging",
        version=result.version,
    )
    print(f"Alias 'staging' now points at version {result.version}")

    # Add a description explaining what this model is
    client.update_registered_model(
        name=MODEL_NAME,
        description=(
            "XGBoost multiclass classifier for 5G-NIDD attack detection. "
            "9 classes: Benign + 8 attack types. Trained on 70/15/15 stratified "
            "split of Combined.csv (schema v1.0.1, Offset dropped). "
            "77 features, sample_weight='balanced' for class imbalance. "
            "Achieves macro-F1 0.92 with near-perfect attack recall (98-100%) "
            "and Benign recall 0.41 (documented false-positive trade-off, "
            "not addressable within per-flow feature representation)."
        ),
    )

    # Description on the specific version — what makes THIS version distinct
    client.update_model_version(
        name=MODEL_NAME,
        version=result.version,
        description=(
            f"Initial production candidate. Trained from run {WINNING_RUN_ID}. "
            "Features: 77 baseline features (52 raw + is_tcp + has_dst_reply, "
            "after 9 dropped columns per schema v1.0.1). "
            "Engineered features tested separately, no measurable improvement, "
            "not included in production pipeline (see notes.md). "
            "Model: XGBClassifier, tree_method='hist', "
            "sample_weight='balanced', n_estimators=200, max_depth=6."
        ),
    )

    print(f"\nRegistered model summary:")
    print(f"  Name:       {MODEL_NAME}")
    print(f"  Version:    {result.version}")
    print(f"  Alias:      staging")
    print(f"  Load with:  mlflow.xgboost.load_model('models:/{MODEL_NAME}@staging')")


if __name__ == "__main__":
    if WINNING_RUN_ID == "PASTE_YOUR_RUN_ID_HERE":
        print("Error: paste your winning run_id into WINNING_RUN_ID first.")
        sys.exit(1)
    register()