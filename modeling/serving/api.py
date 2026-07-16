import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager

from contracts.schemas import SCHEMA_VERSION
from modeling.serving.schemas import (
    FlowFeatures,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    TopFeature,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "5g-nidd-attack-classifier"
MODEL_ALIAS = "staging"  # switch to "production" once end-to-end verified
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# When the API starts, we need the training-time label encoding to map
# predicted class indices back to strings ("UDPFlood", "Benign", etc.).
# We refit the encoder on the y_train parquet at startup.
SPLITS_DIR = Path("modeling/artifacts/splits")


# ---------------------------------------------------------------------------
# App and model loading (at startup)
# ---------------------------------------------------------------------------


# These get populated at startup — see startup handler below
state: dict = {
    "model": None,
    "label_encoder": None,
    "feature_columns": None,
    "model_version": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: run startup logic before serving, cleanup after."""
    # === startup ===
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print(f"Loading model from {MODEL_URI}...")
    state["model"] = mlflow.xgboost.load_model(MODEL_URI)

    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()
    le = LabelEncoder().fit(y_train)
    state["label_encoder"] = le

    X_train_sample = pd.read_parquet(SPLITS_DIR / "X_train.parquet").head(1)
    state["feature_columns"] = list(X_train_sample.columns)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    state["model_version"] = str(mv.version)

    print(f"Model loaded: {MODEL_NAME} v{mv.version} (alias '{MODEL_ALIAS}')")
    print(f"Classes: {list(le.classes_)}")
    print(f"Features: {len(state['feature_columns'])}")

    yield  # === app runs here ===

    # === shutdown ===
    # (nothing to clean up for now; MLflow closes gracefully on its own)
    print("Shutting down.")


app = FastAPI(
    title="5G-NIDD Attack Classifier API",
    description="Real-time flow classification for 5G intrusion detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check."""
    return HealthResponse(
        status="ok",
        model_loaded=state["model"] is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Metadata about the currently loaded model."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=str(state["model_version"]),
        model_alias=MODEL_ALIAS,
        schema_version=SCHEMA_VERSION,
        n_features=len(state["feature_columns"]),
        classes=list(state["label_encoder"].classes_),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: FlowFeatures, explain: bool = False):
    """Classify a single flow.

    Query parameter `explain=true` also returns the top-5 SHAP features
    driving the prediction. Adds ~10-50ms per request.
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert incoming feature dict to a DataFrame in the correct column order
    try:
        X = pd.DataFrame(
            [[flow.features.get(col, 0.0) for col in state["feature_columns"]]],
            columns=state["feature_columns"],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature preparation failed: {e}")

    # Predict
    proba = state["model"].predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = state["label_encoder"].classes_[pred_idx]

    all_probs = {
        cls: float(proba[i])
        for i, cls in enumerate(state["label_encoder"].classes_)
    }

    response = PredictionResponse(
        predicted_class=str(pred_class),
        probability=float(proba[pred_idx]),
        all_probabilities=all_probs,
        model_version=state["model_version"],
        schema_version=SCHEMA_VERSION,
    )

    # SHAP explanation, only if requested (it's not free)
    if explain:
        booster = state["model"].get_booster()
        dmatrix = xgb.DMatrix(X)
        contribs = booster.predict(dmatrix, pred_contribs=True)
        # Shape: (1, n_classes, n_features + 1) — strip bias column
        class_shap = contribs[0, pred_idx, :-1]

        top_k_idx = np.argsort(np.abs(class_shap))[-5:][::-1]
        response.top_features = [
            TopFeature(
                feature=state["feature_columns"][i],
                value=float(X.iloc[0, i]),
                shap_contribution=float(class_shap[i]),
            )
            for i in top_k_idx
        ]

    return response