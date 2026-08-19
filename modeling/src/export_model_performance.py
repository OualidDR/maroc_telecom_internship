"""
modeling/src/export_model_performance.py
==========================================
Compute the production model's OFFLINE performance (per-class precision/recall/
F1 on the held-out TEST set) and load it into Snowflake GOLD_MODEL_PERFORMANCE
for the Power BI "Model & Data Quality" page.

Why the test set (not the 150k Gold predictions): the streamed 150k is the full
dataset (train+val+test mixed), so scoring it would report inflated, leaked
metrics. X_test / y_test is the honest held-out evaluation.

Run warehouse/create_gold_model_performance.sql once first (or via
warehouse/run_gold_ddl-style execution). Then:
    python modeling/src/export_model_performance.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

MODEL_NAME = "5g-nidd-attack-classifier"
MODEL_ALIAS = "staging"
LOCAL_MODEL_PATH = "mlruns/1/models/m-623fad3337d14c7fb7f6d5d9a9b69c0b/artifacts"
SPLITS_DIR = Path("modeling/artifacts/splits")
TABLE_NAME = "GOLD_MODEL_PERFORMANCE"


def load_model_and_version():
    """Load the served model + its registry version. Falls back to the local
    artifact (version '1') if the MLflow server isn't reachable."""
    try:
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
        print(f"Loaded {MODEL_NAME}@{MODEL_ALIAS} (v{version}) from MLflow server.")
    except Exception as e:
        print(f"MLflow server not reachable ({e}); loading local artifact instead.")
        model = mlflow.xgboost.load_model(LOCAL_MODEL_PATH)
        version = "1"
    return model, str(version)


def compute_metrics() -> pd.DataFrame:
    model, version = load_model_and_version()

    X_test = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet").squeeze()
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()

    # The model outputs encoded ints; reproduce the training-time encoding.
    le = LabelEncoder().fit(y_train)
    y_pred = le.inverse_transform(model.predict(X_test))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])
    now = datetime.now(timezone.utc)

    rows = []
    for cls in le.classes_:
        m = report[cls]
        rows.append({
            "MODEL_VERSION": version,
            "CLASS": str(cls),
            "PRECISION_SCORE": float(m["precision"]),
            "RECALL_SCORE": float(m["recall"]),
            "F1_SCORE": float(m["f1-score"]),
            "SUPPORT": int(m["support"]),
            "ACCURACY": accuracy,
            "MACRO_F1": macro_f1,
            "WEIGHTED_F1": weighted_f1,
            "EVALUATED_AT": now,
        })
    df = pd.DataFrame(rows)
    print("\nPer-class test-set performance:")
    print(df[["CLASS", "PRECISION_SCORE", "RECALL_SCORE", "F1_SCORE", "SUPPORT"]].to_string(index=False))
    print(f"\nOverall: accuracy={accuracy:.3f}  macro_f1={macro_f1:.3f}  weighted_f1={weighted_f1:.3f}")
    return df


def load_to_snowflake(df: pd.DataFrame):
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Missing .env values: {missing}"); sys.exit(1)

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"), user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "NIDD_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "NIDD_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "SILVER"),
    )
    conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {TABLE_NAME}")
    ok, _, n, _ = write_pandas(conn, df, TABLE_NAME, auto_create_table=False,
                               quote_identifiers=True, use_logical_type=True)
    print(f"\n{TABLE_NAME}: success={ok} rows_loaded={n}")
    conn.close()


if __name__ == "__main__":
    load_to_snowflake(compute_metrics())
