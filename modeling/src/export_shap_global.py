"""
modeling/src/export_shap_global.py
====================================
Compute global SHAP feature importance (mean |SHAP| per feature, per class +
Overall) from the production model and load it into GOLD_SHAP_GLOBAL for the
Power BI "Explainability" page (Page 3).

Uses XGBoost's exact TreeSHAP via booster.predict(pred_contribs=True) on a
sample of the test set -- same mechanism the /predict?explain endpoint uses.

Run warehouse/create_gold_explainability.sql once first, then:
    python modeling/src/export_shap_global.py
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

MODEL_NAME = "5g-nidd-attack-classifier"
MODEL_ALIAS = "staging"
LOCAL_MODEL_PATH = "mlruns/1/models/m-623fad3337d14c7fb7f6d5d9a9b69c0b/artifacts"
SPLITS_DIR = Path("modeling/artifacts/splits")
TABLE_NAME = "GOLD_SHAP_GLOBAL"
SAMPLE_SIZE = 3000  # enough for stable global importance; keeps SHAP fast


def load_model_and_version():
    try:
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
        print(f"Loaded {MODEL_NAME}@{MODEL_ALIAS} (v{version}) from MLflow server.")
    except Exception as e:
        print(f"MLflow server not reachable ({e}); loading local artifact.")
        model = mlflow.xgboost.load_model(LOCAL_MODEL_PATH)
        version = "1"
    return model, str(version)


def compute_shap() -> pd.DataFrame:
    model, version = load_model_and_version()
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()
    classes = list(LabelEncoder().fit(y_train).classes_)

    X = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    if len(X) > SAMPLE_SIZE:
        X = X.sample(SAMPLE_SIZE, random_state=42)
    features = list(X.columns)
    n_feat, n_class = len(features), len(classes)

    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    # Multiclass shape is (n_samples, n_class, n_feat+1); some builds flatten it.
    if contribs.ndim == 2:
        contribs = contribs.reshape(contribs.shape[0], n_class, n_feat + 1)
    contribs = contribs[:, :, :-1]  # strip the bias column

    rows = []
    per_class_imp = {}
    for ci, cls in enumerate(classes):
        imp = np.abs(contribs[:, ci, :]).mean(axis=0)  # (n_feat,)
        per_class_imp[cls] = imp
        for rank, fi in enumerate(np.argsort(imp)[::-1], start=1):
            rows.append({"MODEL_VERSION": version, "CLASS": str(cls),
                         "FEATURE": features[fi], "MEAN_ABS_SHAP": float(imp[fi]),
                         "FEATURE_RANK": int(rank)})

    # 'Overall' = mean of the per-class importances.
    overall = np.mean(list(per_class_imp.values()), axis=0)
    for rank, fi in enumerate(np.argsort(overall)[::-1], start=1):
        rows.append({"MODEL_VERSION": version, "CLASS": "Overall",
                     "FEATURE": features[fi], "MEAN_ABS_SHAP": float(overall[fi]),
                     "FEATURE_RANK": int(rank)})

    df = pd.DataFrame(rows)
    print(f"\nTop 10 features overall (mean |SHAP|):")
    print(df[df.CLASS == "Overall"].head(10)[["FEATURE", "MEAN_ABS_SHAP"]].to_string(index=False))
    return df


def load_to_snowflake(df: pd.DataFrame):
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
    load_to_snowflake(compute_shap())
