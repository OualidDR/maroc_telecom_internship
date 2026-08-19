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

import math
import os
import random

import pandas as pd
import requests

# NOTE: these are read at CALL time (inside call_predict_api), NOT at import
# time. Reading them at import time was a bug: spark_predict_and_gold.py imports
# this module BEFORE it calls load_dotenv(), so PREDICT_API_MOCK defaulted to
# "true" and the whole pipeline silently ran in mock mode. Reading per-call makes
# it immune to import/.env-load ordering.
def _config():
    return {
        "url": os.getenv("PREDICT_API_URL", "http://localhost:8000/predict/raw"),
        "mock": os.getenv("PREDICT_API_MOCK", "true").lower() == "true",
        "timeout": float(os.getenv("PREDICT_API_TIMEOUT", "30")),
        # The API caps each request at 1000 flows (RawFlowRequest.max_length), so
        # we chunk larger Spark batches. 500 keeps requests small/memory-bounded.
        "max_rows": int(os.getenv("PREDICT_API_MAX_ROWS", "500")),
    }

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


_EMPTY_PREDICTIONS = pd.DataFrame(
    columns=["event_id", "predicted_class", "probability", "model_version"]
)


def _json_safe(v):
    """Make one value JSON-transportable for requests (allow_nan=False):
      - numpy scalar   -> native Python scalar
      - NaN / NaT / inf / pd.NA / None -> None  (JSON has no NaN; requests rejects it)
      - datetime / Timestamp / anything non-native -> str  (these are non-feature
        columns like ingestion_timestamp; the API drops them on reindex anyway,
        but they must still serialize).
    """
    if hasattr(v, "item"):          # numpy scalar -> python scalar
        try:
            v = v.item()
        except Exception:
            pass
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass                        # pd.isna on non-scalar -> keep the value
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)                   # datetime, Timestamp, etc.


def call_predict_api(batch_pdf: pd.DataFrame) -> pd.DataFrame:
    """Send a batch of raw flows to /predict/raw, return predictions keyed by event_id.

    Splits the batch into chunks of PREDICT_API_MAX_ROWS (the API caps each
    request at 1000 flows) and concatenates the results. Values are sanitized
    to be JSON-safe first (structural NaNs and the datetime ingestion_timestamp
    would otherwise make requests raise before anything is sent).

    On failure of a chunk (timeout, connection error, non-200), that chunk is
    skipped with a warning rather than raising -- a prediction-service hiccup
    should not take down the Spark stream. Callers treat missing predictions as
    "not yet scored", not as an error to propagate.
    """
    cfg = _config()
    if cfg["mock"]:
        return _mock_predict(batch_pdf)

    feature_cols = [c for c in batch_pdf.columns if c != "event_id"] + ["event_id"]
    records = [
        {k: _json_safe(v) for k, v in row.items()}
        for row in batch_pdf[feature_cols].to_dict(orient="records")
    ]

    all_predictions = []
    n_failed = 0
    for start in range(0, len(records), cfg["max_rows"]):
        chunk = records[start:start + cfg["max_rows"]]
        try:
            resp = requests.post(
                cfg["url"], json={"flows": chunk}, timeout=cfg["timeout"]
            )
            resp.raise_for_status()
            all_predictions.extend(resp.json().get("predictions", []))
        except requests.RequestException as e:
            n_failed += len(chunk)
            print(f"[predict_client] WARNING: chunk at row {start} failed ({e}); "
                  f"skipping {len(chunk)} rows.")

    if n_failed:
        print(f"[predict_client] {n_failed}/{len(records)} rows had no prediction "
              f"(API errors) -- treated as not-yet-scored.")

    return pd.DataFrame(all_predictions) if all_predictions else _EMPTY_PREDICTIONS