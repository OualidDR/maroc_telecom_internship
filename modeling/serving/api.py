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
    BatchFlowRequest,           # new
    BatchPredictionResponse,    # new
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

from modeling.monitoring.drift import compute_drift


@app.post("/drift")
def check_drift(batch: list[dict]):
    """Compute drift for a batch of flows against the reference distribution.

    Body: list of feature dicts (like /predict payloads), optionally with
    a 'prediction' key. Returns drift summary + saves an HTML report.
    """
    if not batch:
        raise HTTPException(status_code=422, detail="Empty batch")

    # Build the current DataFrame
    df = pd.DataFrame(batch)
    if "prediction" not in df.columns:
        df["prediction"] = "unknown"

    summary = compute_drift(df, save_html=True)

    # Extract just the interesting numbers for the response
    return {
        "n_flows_analyzed": len(df),
        "summary": summary,
    }
    
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchFlowRequest, explain: bool = False):
    """Classify a batch of flows in one vectorized inference call.

    Designed for the Spark streaming pipeline: sends batches of ~100-500
    flows per micro-batch, gets back predictions in the same order.

    Query parameter `explain=true` computes SHAP top-5 for every flow.
    Adds significant latency; only use when the pipeline actually needs
    explanations (e.g. writing to the Gold layer for analyst review).
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.flows:
        raise HTTPException(status_code=422, detail="Empty batch")

    # Build the batch DataFrame in the model's expected feature order
    try:
        X = pd.DataFrame(
            [
                [flow.get(col, 0.0) for col in state["feature_columns"]]
                for flow in request.flows
            ],
            columns=state["feature_columns"],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature preparation failed: {e}")

    # One vectorized prediction call — this is where the batch speedup comes from
    proba_matrix = state["model"].predict_proba(X)   # shape (n_flows, n_classes)
    pred_indices = np.argmax(proba_matrix, axis=1)   # shape (n_flows,)

    classes = state["label_encoder"].classes_

    # Optional SHAP — computed once for the whole batch, not per row
    top_features_batch = None
    if explain:
        booster = state["model"].get_booster()
        dmatrix = xgb.DMatrix(X)
        contribs = booster.predict(dmatrix, pred_contribs=True)
        # Shape: (n_flows, n_classes, n_features + 1) — strip bias column
        contribs = contribs[:, :, :-1]

    # Build one response per flow
    responses = []
    for i, pred_idx in enumerate(pred_indices):
        proba_row = proba_matrix[i]
        pred_class = str(classes[pred_idx])

        all_probs = {
            str(cls): float(proba_row[j])
            for j, cls in enumerate(classes)
        }

        top_features = None
        if explain:
            class_shap = contribs[i, pred_idx, :]
            top_k_idx = np.argsort(np.abs(class_shap))[-5:][::-1]
            top_features = [
                TopFeature(
                    feature=state["feature_columns"][j],
                    value=float(X.iloc[i, j]),
                    shap_contribution=float(class_shap[j]),
                )
                for j in top_k_idx
            ]

        responses.append(PredictionResponse(
            predicted_class=pred_class,
            probability=float(proba_row[pred_idx]),
            all_probabilities=all_probs,
            top_features=top_features,
            model_version=str(state["model_version"]),
            schema_version=SCHEMA_VERSION,
        ))

    return BatchPredictionResponse(
        predictions=responses,
        n_flows=len(responses),
        model_version=str(state["model_version"]),
        schema_version=SCHEMA_VERSION,
    )