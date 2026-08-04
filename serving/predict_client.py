"""
serving/predict_client.py
============================
Client for the DS colleague's FastAPI /predict/raw endpoint (not built yet
at the time this was written). Assumed contract, per rapport_DE_TODO
response point 2 -- confirm with him once he builds it:

    POST /predict/raw
    body:  {"flows": [{<39 raw contract features>, "event_id": "..."}]}
    resp:  {"predictions": [{"event_id": "...", "predicted_class": "...",
                              "probability": 0.97, "model_version": "3"}]}

MOCK MODE (default on, via PREDICT_API_MOCK=true in .env): returns
deterministic fake predictions instead of calling a real API, so the rest
of the pipeline (join, Gold writes, Snowflake load) can be built and tested
end-to-end right now. Flip PREDICT_API_MOCK=false once the real endpoint
exists -- no other code changes needed.
"""

import os
import random

import pandas as pd
import requests

PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict/raw")
PREDICT_API_MOCK = os.getenv("PREDICT_API_MOCK", "true").lower() == "true"
PREDICT_API_TIMEOUT = float(os.getenv("PREDICT_API_TIMEOUT", "10"))

_MOCK_CLASSES = [
    "Benign", "UDPFlood", "HTTPFlood", "SlowrateDoS",
    "TCPConnectScan", "SYNScan", "UDPScan", "SYNFlood", "ICMPFlood",
]


def _mock_predict(batch_pdf: pd.DataFrame) -> pd.DataFrame:
    """Deterministic-ish fake predictions, weighted toward Benign like the
    real class distribution, so downstream alert-rate logic isn't absurd."""
    rng = random.Random(42)
    rows = []
    for event_id in batch_pdf["event_id"]:
        predicted_class = rng.choices(
            _MOCK_CLASSES,
            weights=[50, 25, 8, 4, 1, 1, 1, 0.5, 0.5],  # roughly mirrors real imbalance
            k=1,
        )[0]
        rows.append({
            "event_id": event_id,
            "predicted_class": predicted_class,
            "probability": round(rng.uniform(0.55, 0.99), 4),
            "model_version": "mock",
        })
    return pd.DataFrame(rows)


def call_predict_api(batch_pdf: pd.DataFrame) -> pd.DataFrame:
    """Send a batch of raw flows to /predict/raw, return predictions keyed by event_id.

    On any failure (timeout, connection error, non-200), returns an empty
    DataFrame rather than raising -- a prediction-service hiccup should not
    take down the Spark stream. Callers should treat missing predictions as
    "not yet scored", not as an error to propagate.
    """
    if PREDICT_API_MOCK:
        return _mock_predict(batch_pdf)

    feature_cols = [c for c in batch_pdf.columns if c != "event_id"] + ["event_id"]
    payload = {"flows": batch_pdf[feature_cols].to_dict(orient="records")}

    try:
        resp = requests.post(PREDICT_API_URL, json=payload, timeout=PREDICT_API_TIMEOUT)
        resp.raise_for_status()
        predictions = resp.json().get("predictions", [])
        return pd.DataFrame(predictions)
    except requests.RequestException as e:
        print(f"[predict_client] WARNING: prediction API call failed ({e}), "
              f"skipping predictions for this batch ({len(batch_pdf)} rows).")
        return pd.DataFrame(columns=["event_id", "predicted_class", "probability", "model_version"])